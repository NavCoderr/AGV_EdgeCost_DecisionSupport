# -*- coding: utf-8 -*-
"""
alpha_sensitivity_operation_to_all_nodes.py

Purpose:
- Reviewer #4 alpha-sensitivity support for ESWA revision.
- Tests whether changing alpha affects route selection.
- Starts from operation nodes 1..6 and tries all reachable graph destination nodes.
- Uses learned predicted edge costs from edge_costs_pred.csv.
- Does NOT rerun model training.
- Produces CSV proof files for manuscript/rebuttal.

Inputs expected in project folder or inductive_folder_new_data:
    edge_costs_pred.csv
    Node_F3.csv or Node_F3(3).csv
    Edge_Distances3_.csv or Edge_Distances3_(4).csv

Outputs:
    inductive_folder_new_data/alpha_operation_to_all_nodes/
        alpha_all_dest_detail.csv
        alpha_all_dest_summary.csv
        alpha_changed_only.csv
        alpha_representative_examples.csv
"""

from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np
import networkx as nx


# =========================
# USER SETTINGS
# =========================

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

# Main operation/station nodes in your layout
START_NODES = [1, 2, 3, 4, 5, 6]

# These are operation nodes. Used only to mark destination type in output.
OPERATION_NODES = {1, 2, 3, 4, 5, 6}

# File/folder setup
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "inductive_folder_new_data"
OUT_DIR = OUT_ROOT / "alpha_operation_to_all_nodes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# FILE RESOLUTION
# =========================

def find_file(candidates: list[str]) -> Path:
    """
    Find file either in project root or inductive_folder_new_data.
    """
    search_dirs = [
        SCRIPT_DIR,
        OUT_ROOT,
    ]

    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                return p

    raise FileNotFoundError(
        "Could not find any of these files: "
        + ", ".join(candidates)
        + f"\nSearched in: {SCRIPT_DIR} and {OUT_ROOT}"
    )


def read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Reads comma or semicolon CSV automatically.
    """
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=";")


# =========================
# LOAD DATA
# =========================

def load_predicted_edges() -> pd.DataFrame:
    pred_path = find_file([
        "edge_costs_pred.csv",
        "edge_costs_pred(1).csv",
        "edge_costs_pred(2).csv",
    ])
    print(f"[load] predicted costs: {pred_path}")

    df = read_csv_auto(pred_path)

    # Normalize column names if needed
    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"u", "from", "source", "src", "u_node", "u_node_id"}:
            rename_map[c] = "u"
        elif lc in {"v", "to", "target", "dst", "v_node", "v_node_id"}:
            rename_map[c] = "v"
        elif lc in {"pred_time_s", "time", "time_s", "t_pred_s"}:
            rename_map[c] = "pred_time_s"
        elif lc in {"pred_energy_j", "energy", "energy_j", "e_pred_j"}:
            rename_map[c] = "pred_energy_J"

    df = df.rename(columns=rename_map)

    required = {"u", "v", "pred_time_s", "pred_energy_J"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {pred_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[["u", "v", "pred_time_s", "pred_energy_J"]].copy()
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    df["pred_time_s"] = pd.to_numeric(df["pred_time_s"], errors="coerce")
    df["pred_energy_J"] = pd.to_numeric(df["pred_energy_J"], errors="coerce")
    df = df.dropna(subset=["pred_time_s", "pred_energy_J"])

    if df.empty:
        raise ValueError("edge_costs_pred.csv is empty after cleaning.")

    return df


def normalize_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalize predicted time and energy for alpha scalarization.
    This matches the common planner logic:
        cost = alpha * normalized_time + (1-alpha) * normalized_energy
    """
    out = df.copy()

    t_min, t_max = out["pred_time_s"].min(), out["pred_time_s"].max()
    e_min, e_max = out["pred_energy_J"].min(), out["pred_energy_J"].max()

    if math.isclose(t_max, t_min):
        out["t_norm"] = 0.0
    else:
        out["t_norm"] = (out["pred_time_s"] - t_min) / (t_max - t_min)

    if math.isclose(e_max, e_min):
        out["e_norm"] = 0.0
    else:
        out["e_norm"] = (out["pred_energy_J"] - e_min) / (e_max - e_min)

    corr = out["pred_time_s"].corr(out["pred_energy_J"])
    out.attrs["time_energy_corr"] = float(corr)
    print(f"[info] predicted time-energy correlation = {corr:.6f}")

    return out


# =========================
# GRAPH + PLANNING
# =========================

def build_graph(pred_df: pd.DataFrame, alpha: float) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, r in pred_df.iterrows():
        u = int(r["u"])
        v = int(r["v"])

        cost = float(alpha * r["t_norm"] + (1.0 - alpha) * r["e_norm"])

        G.add_edge(
            u,
            v,
            weight=cost,
            pred_time_s=float(r["pred_time_s"]),
            pred_energy_J=float(r["pred_energy_J"]),
        )

    return G


def route_metrics(G: nx.DiGraph, route: list[int]) -> tuple[float, float, float]:
    """
    Return scalar cost, total predicted time, total predicted energy.
    """
    total_cost = 0.0
    total_time = 0.0
    total_energy = 0.0

    for u, v in zip(route[:-1], route[1:]):
        d = G[u][v]
        total_cost += float(d["weight"])
        total_time += float(d["pred_time_s"])
        total_energy += float(d["pred_energy_J"])

    return total_cost, total_time, total_energy


def route_to_str(route: list[int]) -> str:
    return " -> ".join(str(x) for x in route)


def get_all_nodes(pred_df: pd.DataFrame) -> list[int]:
    nodes = sorted(set(pred_df["u"].astype(int)).union(set(pred_df["v"].astype(int))))
    return nodes


# =========================
# MAIN ANALYSIS
# =========================

def main() -> None:
    pred = load_predicted_edges()
    pred = normalize_costs(pred)
    time_energy_corr = pred.attrs.get("time_energy_corr", np.nan)


    all_nodes = get_all_nodes(pred)
    print(f"[graph] nodes from predicted edges = {len(all_nodes)}")
    print(f"[analysis] starts = {START_NODES}")
    print(f"[analysis] alphas = {ALPHAS}")

    detail_rows = []

    for start in START_NODES:
        for goal in all_nodes:
            if int(goal) == int(start):
                continue

            # Check reachability with alpha=0.5 graph
            G_check = build_graph(pred, alpha=0.5)
            if not nx.has_path(G_check, int(start), int(goal)):
                continue

            for alpha in ALPHAS:
                G = build_graph(pred, alpha=alpha)

                try:
                    route = nx.shortest_path(G, int(start), int(goal), weight="weight")
                    total_cost, total_time, total_energy = route_metrics(G, route)
                except nx.NetworkXNoPath:
                    route = []
                    total_cost = np.nan
                    total_time = np.nan
                    total_energy = np.nan

                detail_rows.append({
                    "start_node": int(start),
                    "goal_node": int(goal),
                    "mission": f"{int(start)}->{int(goal)}",
                    "goal_is_operation_node": int(int(goal) in OPERATION_NODES),
                    "alpha": float(alpha),
                    "route_nodes": route_to_str(route) if route else "",
                    "n_edges": int(len(route) - 1) if route else 0,
                    "scalar_cost": total_cost,
                    "total_pred_time_s": total_time,
                    "total_pred_energy_J": total_energy,
                })

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        raise RuntimeError("No routes were generated. Check input graph/cost files.")

    # Save detailed alpha-route output
    detail_path = OUT_DIR / "alpha_all_dest_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    # Summary per start-goal pair
    summary_rows = []

    for mission, g in detail_df.groupby("mission"):
        routes = list(g["route_nodes"].dropna().unique())
        routes = [r for r in routes if str(r).strip()]

        n_unique_routes = len(routes)
        route_changed = int(n_unique_routes > 1)

        min_t = float(g["total_pred_time_s"].min())
        max_t = float(g["total_pred_time_s"].max())
        min_e = float(g["total_pred_energy_J"].min())
        max_e = float(g["total_pred_energy_J"].max())

        first = g.iloc[0]

        # Low-alpha and high-alpha route names
        low_alpha_row = g.sort_values("alpha").iloc[0]
        high_alpha_row = g.sort_values("alpha").iloc[-1]

        summary_rows.append({
            "start_node": int(first["start_node"]),
            "goal_node": int(first["goal_node"]),
            "mission": str(mission),
            "goal_is_operation_node": int(first["goal_is_operation_node"]),
            "n_alpha_values": int(len(g)),
            "n_unique_routes": int(n_unique_routes),
            "route_changed_across_alpha": int(route_changed),
            "min_pred_time_s": min_t,
            "max_pred_time_s": max_t,
            "range_pred_time_s": max_t - min_t,
            "min_pred_energy_J": min_e,
            "max_pred_energy_J": max_e,
            "range_pred_energy_J": max_e - min_e,
            "low_alpha": float(low_alpha_row["alpha"]),
            "high_alpha": float(high_alpha_row["alpha"]),
            "low_alpha_route": str(low_alpha_row["route_nodes"]),
            "high_alpha_route": str(high_alpha_row["route_nodes"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["route_changed_across_alpha", "range_pred_time_s"],
        ascending=[False, False]
    )

    summary_path = OUT_DIR / "alpha_all_dest_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    changed_df = summary_df[summary_df["route_changed_across_alpha"] == 1].copy()
    changed_path = OUT_DIR / "alpha_changed_only.csv"
    changed_df.to_csv(changed_path, index=False)

    # Representative examples:
    # Prefer operation-node destinations if available, then highest time-range cases.
    op_changed = changed_df[changed_df["goal_is_operation_node"] == 1].copy()
    nonop_changed = changed_df[changed_df["goal_is_operation_node"] == 0].copy()

    reps = pd.concat([
        op_changed.sort_values("range_pred_time_s", ascending=False).head(8),
        nonop_changed.sort_values("range_pred_time_s", ascending=False).head(8),
    ], ignore_index=True)

    reps_path = OUT_DIR / "alpha_representative_examples.csv"
    reps.to_csv(reps_path, index=False)

    total_pairs = len(summary_df)
    changed_pairs = int(changed_df.shape[0])
    changed_rate = changed_pairs / total_pairs if total_pairs else 0.0

    op_pairs = int((summary_df["goal_is_operation_node"] == 1).sum())
    op_changed_pairs = int(((summary_df["goal_is_operation_node"] == 1) &
                            (summary_df["route_changed_across_alpha"] == 1)).sum())

    print("=" * 90)
    print("[DONE] Alpha sensitivity operation-to-all-destinations")
    print(f"[saved] {detail_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {changed_path}")
    print(f"[saved] {reps_path}")
    print("-" * 90)
    print(f"Total reachable start-destination pairs: {total_pairs}")
    print(f"Pairs with route change across alpha: {changed_pairs}")
    print(f"Route-change rate: {changed_rate * 100:.2f}%")
    print(f"Operation-destination pairs: {op_pairs}")
    print(f"Operation-destination pairs with route change: {op_changed_pairs}")
    print("-" * 90)

    if not changed_df.empty:
        print("[Top changed examples]")
        cols = [
            "mission",
            "goal_is_operation_node",
            "n_unique_routes",
            "range_pred_time_s",
            "range_pred_energy_J",
            "low_alpha_route",
            "high_alpha_route",
        ]
        print(changed_df[cols].head(12).to_string(index=False))
    else:
        print("[info] No route changes found across alpha values.")


if __name__ == "__main__":
    main()