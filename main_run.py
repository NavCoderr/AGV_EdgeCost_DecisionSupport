# -*- coding: utf-8 -*-
from pathlib import Path
import inspect
import pandas as pd

import config as C
from utils_io import safe_mkdir, abs_path, set_seed
from graph_data import load_graph
from preprocess import apply_snap, build_edge_observations, snap_xy_to_node_with_dist
from preprocess_nav2_to_1hz import run_preprocess_nav2
from templates import (
    extract_geom_templates,
    build_tau_templates,
    load_geom_templates,
    load_tau_templates,
    global_tau_fallback,
)
from edge_cost_train import train_edge_cost_model, predict_all_edge_costs
from temporal_mlp import build_temporal_dataset, train_temporal_model, train_temporal_all, load_temporal_model
from planner import build_full_cost_maps, dijkstra_route, astar_route_safe, build_edge_schedule
from trajectory import simulate_tau_warp, simulate_temporal_mlp, simulate_temporal_mlp_closed_loop
from model import EdgeCostModel

def read_any_table(path):
    path = Path(path)
    suf = path.suffix.lower()

    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")

    # CSV encoding fallback
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def train_pipeline(G, out_dir):
    """
    TRAIN pipeline (robust):
      - Prefer cached data_1s_global.csv if present (repeatable runs)
      - Otherwise read C.DATA_1HZ_CSV (your move-only 1 Hz input)
      - Always re-snap (safe) and overwrite data_1s_global.csv
      - Generate edge_samples + templates + train edge-cost + train temporal
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Choose input source
    data1s_csv = out_dir / "data_1s_global.csv"
    use_cached = bool(getattr(C, "USE_CACHED_GLOBAL_1S", True))

    if use_cached and data1s_csv.exists():
        in_path = data1s_csv
        print(f"[data] Using cached global 1s: {abs_path(in_path)}")
    else:
        in_path = Path(C.DATA_1HZ_CSV)
        print(f"[data] Using DATA_1HZ_CSV={abs_path(in_path)}")

    # 1) Load 1Hz CSV/table
    df_1s = read_any_table(in_path)

    # 2) Snap to graph nodes (safe even if already snapped)
    df_1s_snapped, radius = apply_snap(df_1s, G)
    print(f"[snap] radius_used_m={radius:.3f} fallback={C.SNAP_FALLBACK} ffill_max_gap={C.FFILL_MAX_GAP_S}s")

    # Optional: goal filter
    if "goal_id" in df_1s_snapped.columns and getattr(C, "GOAL_KEEP_LIST", None):
        df_1s_snapped["goal_id"] = pd.to_numeric(df_1s_snapped["goal_id"], errors="coerce")
        df_1s_snapped = df_1s_snapped[df_1s_snapped["goal_id"].isin(list(C.GOAL_KEEP_LIST))].copy()
        df_1s_snapped = df_1s_snapped.reset_index(drop=True)

    # 2.1) Save/overwrite snapped global (always keep fresh)
    df_1s_snapped.to_csv(data1s_csv, index=False)
    print(f"[saved] {abs_path(data1s_csv)} rows={len(df_1s_snapped)}")

    # 3) Edge samples
    edge_samples, _energy_per_m = build_edge_observations(df_1s_snapped, G)
    edge_samples_csv = out_dir / "edge_samples.csv"
    edge_samples.to_csv(edge_samples_csv, index=False)
    print(f"[saved] {abs_path(edge_samples_csv)} rows={len(edge_samples)}")

    # 4) Geom templates
    geom_csv = out_dir / "geom_templates.csv"
    geom_templates = extract_geom_templates(df_1s_snapped, G, geom_csv, K=C.GEOM_K)
    print(f"[saved] geom_templates -> {abs_path(geom_csv)} edges={len(geom_templates)}")

    # 5) Tau templates
    tau_csv = out_dir / "tau_templates.csv"
    tau_templates = build_tau_templates(df_1s_snapped, G, geom_templates, tau_csv, M=C.TAU_M)
    print(f"[saved] tau_templates -> {abs_path(tau_csv)} edges={len(tau_templates)}")

    # 6) Train edge-cost model 
    model_path, _pred_train = train_edge_cost_model(
    G, edge_samples, out_dir, EdgeCostModel=EdgeCostModel
)
    print(f"[edge_cost_model] saved={abs_path(Path(model_path))}")

    # 7) Predict all edges
    predict_all_edge_costs(G, model_path, edge_samples, out_dir, EdgeCostModel)
    print(f"[edge_cost_pred] saved={abs_path(out_dir / 'edge_costs_pred.csv')}")

    # 8) Train temporal models (MLP/GRU/LSTM/Transformer/Physics)
    if C.USE_TEMPORAL_MLP:
        X, y, groups, meta = build_temporal_dataset(df_1s_snapped, G, geom_templates, edge_samples, out_dir)

        temp_model = str(getattr(C, "TEMP_MODEL", "mlp")).lower().strip()
        kinds = getattr(C, "TEMP_MODELS", ("mlp", "gru", "lstm", "transformer_lite", "physics_delta"))

        if temp_model in ("all", "*"):
            train_temporal_all(X, y, groups, meta, out_dir, kinds=kinds)
        else:
            train_temporal_model(X, y, groups, meta, out_dir, kind=temp_model)

    print("[done] TRAIN finished. Now set MODE='plan' in config.py and run again.")


def plan_pipeline(G, out_dir):
    pred_csv = out_dir / "edge_costs_pred.csv"
    geom_csv = out_dir / "geom_templates.csv"
    tau_csv  = out_dir / "tau_templates.csv"
    edge_samples_csv = out_dir / "edge_samples.csv"
    best_kind_path = out_dir / "best_temporal_kind.txt"

    if best_kind_path.exists():
        best_kind = best_kind_path.read_text(encoding="utf-8").strip()
    else:
        best_kind = str(getattr(C, "TEMP_MODEL", "mlp")).lower().strip()
        if best_kind in ("all", "*"):
            best_kind = "mlp"

    temporal_path = out_dir / f"temporal_{best_kind}.pt"

    # Geom templates: load OR build (PLAN mode)
    df_1s_snapped = None  # will be created only if needed

    if not geom_csv.exists():
        print(f"[geom] Missing {abs_path(geom_csv)} -> building from 1Hz data by snapping to graph...")

        # 1) Load 1Hz real data (CSV or Excel)
        df_1s = read_any_table(C.DATA_1HZ_CSV)

        # 2) Snap to graph nodes (adds node_id, snap_dist_m, etc.)
        df_1s_snapped, radius = apply_snap(df_1s, G)
        print(f"[snap] radius_used_m={radius:.3f} fallback={C.SNAP_FALLBACK} ffill_max_gap={C.FFILL_MAX_GAP_S}s")

        # Optional: goal filter (same as train)
        if "goal_id" in df_1s_snapped.columns and getattr(C, "GOAL_KEEP_LIST", None):
            df_1s_snapped["goal_id"] = pd.to_numeric(df_1s_snapped["goal_id"], errors="coerce")
            df_1s_snapped = df_1s_snapped[df_1s_snapped["goal_id"].isin(list(C.GOAL_KEEP_LIST))].copy()
            df_1s_snapped = df_1s_snapped.reset_index(drop=True)

        # 3) Save snapped file (so you can inspect)
        data1s_csv = out_dir / "data_1s_global.csv"
        df_1s_snapped.to_csv(data1s_csv, index=False)
        print(f"[saved] {abs_path(data1s_csv)} rows={len(df_1s_snapped)}")

        # 4) Build geom templates from snapped data
        geom_templates = extract_geom_templates(df_1s_snapped, G, geom_csv, K=C.GEOM_K)
        print(f"[saved] geom_templates -> {abs_path(geom_csv)} edges={len(geom_templates)}")
    else:
        geom_templates = load_geom_templates(geom_csv, G)
        print(f"[geom] loaded -> {abs_path(geom_csv)} edges={len(geom_templates)}")

    # Tau templates: load OR build (PLAN mode)
    if not tau_csv.exists():
        print(f"[tau] Missing {abs_path(tau_csv)} -> building from snapped data...")

        # ensure snapped exists
        if df_1s_snapped is None:
            df_1s = read_any_table(C.DATA_1HZ_CSV)
            df_1s_snapped, _ = apply_snap(df_1s, G)

            # save for inspection
            data1s_csv = out_dir / "data_1s_global.csv"
            if not data1s_csv.exists():
                df_1s_snapped.to_csv(data1s_csv, index=False)
                print(f"[saved] {abs_path(data1s_csv)} rows={len(df_1s_snapped)}")

        tau_templates = build_tau_templates(df_1s_snapped, G, geom_templates, tau_csv, M=getattr(C, "TAU_M", 101))
        print(f"[saved] tau_templates -> {abs_path(tau_csv)} edges={len(tau_templates)}")
    else:
        tau_templates = load_tau_templates(tau_csv, G)
        print(f"[tau] loaded -> {abs_path(tau_csv)} edges={len(tau_templates)}")

    tau_fallback = global_tau_fallback(tau_templates, getattr(C, "TAU_M", 101))

    pred = pd.read_csv(pred_csv) if pred_csv.exists() else None
    edge_samples = pd.read_csv(edge_samples_csv) if edge_samples_csv.exists() else None

    time_cost, energy_cost, combo_cost, slowdown_map = build_full_cost_maps(G, pred, edge_samples)

    # Start / Goal
    if C.START_XY is None:
        start_id = int(C.START_NODE)
    else:
        start_id = int(snap_xy_to_node_with_dist(float(C.START_XY[0]), float(C.START_XY[1]), G)[0])
    goal_id = int(C.GOAL_NODE)

    main_map = {"time": time_cost, "energy": energy_cost, "combo": combo_cost}[C.MAIN_COST]

    # Route
    if C.PLANNER.lower() == "dijkstra":
        route_nodes, route_cost = dijkstra_route(G, start_id, goal_id, main_map, C.BLOCKED_NODES, C.BLOCKED_EDGES)
    else:
        route_nodes, route_cost = astar_route_safe(G, start_id, goal_id, main_map, C.BLOCKED_NODES, C.BLOCKED_EDGES, w=C.ASTAR_W)

    print(f"[route] {route_nodes}")
    print(f"[route_cost] {route_cost:.6f} main_cost={C.MAIN_COST} planner={C.PLANNER}")

    # Build edge schedule + save
    sched_list = build_edge_schedule(G, route_nodes, time_cost, energy_cost, slowdown_map)
    sched_csv = out_dir / "planned_edge_schedule.csv"
    pd.DataFrame(sched_list).to_csv(sched_csv, index=False)
    print(f"[saved] {abs_path(sched_csv)} edges={len(sched_list)}")

    sched_df = pd.DataFrame(sched_list)

    # 1) TAU-WARP trajectory
    simulate_tau_warp(
        G,
        sched_df,
        geom_templates,
        tau_templates,
        tau_fallback,
        out_dir / "planned_trajectory_1hz_CURVED_TAU_WARP.csv"
    )

    # 2) Temporal trajectories (OPEN + CLOSED) for ALL models
    if C.USE_TEMPORAL_MLP:
        kinds = getattr(C, "TEMP_MODELS", ("mlp", "gru", "lstm", "transformer_lite", "physics_delta"))
        for k in kinds:
            k = str(k).lower().strip()
            temporal_path_k = out_dir / f"temporal_{k}.pt"
            if not temporal_path_k.exists():
                print(f"[plan] skip {k} (missing): {abs_path(temporal_path_k)}")
                continue

            tm, meta_k = load_temporal_model(str(temporal_path_k))
            meta_k["kind"] = k  # used inside trajectory simulation for correct input shape

            # Open-loop
            out_open = out_dir / f"planned_trajectory_1hz_CURVED_TEMPORAL_{k.upper()}.csv"
            simulate_temporal_mlp(
                G,
                sched_df,
                geom_templates,
                tm,
                meta_k,
                out_open,
            )
            print(f"[plan] saved open-loop {k} -> {abs_path(out_open)}")

            # Closed-loop
            out_closed = out_dir / f"planned_trajectory_1hz_CURVED_TEMPORAL_{k.upper()}_CLOSED_LOOP.csv"
            simulate_temporal_mlp_closed_loop(
                G,
                sched_df,
                geom_templates,
                tm,
                meta_k,
                real_1hz_csv=C.REAL_1HZ_CSV,
                out_csv=out_closed,
            )
            print(f"[plan] saved closed-loop {k} -> {abs_path(out_closed)}")
    else:
        print("[plan] USE_TEMPORAL_MLP=False -> skipping temporal trajectories.")

    print("[done] PLAN finished.")



def ensure_1hz_csv(out_dir: Path) -> Path:
    raw = Path(getattr(C, "DATA_1HZ_CSV", ""))

    if raw.exists() and raw.suffix.lower() in [".xlsx", ".xls"]:
        subdir = safe_mkdir(out_dir / "_1hz_from_excel")
        move_csv = subdir / "nav_1hz_move_only.csv"

        if not move_csv.exists():
            print(f"[1hz] preprocessing RAW Excel -> 1Hz move-only CSV: {abs_path(raw)}")
            run_preprocess_nav2(raw, out_dir=subdir)
        else:
            print(f"[1hz] using existing 1Hz move-only CSV: {abs_path(move_csv)}")

        if not move_csv.exists():
            raise FileNotFoundError(f"Expected 1Hz output not found: {move_csv}")

        return move_csv

    if not raw.exists():
        raise FileNotFoundError(f"DATA_1HZ_CSV not found: {raw}")

    return raw


def main():
    set_seed(C.SEED)
    out_dir = safe_mkdir(C.OUT_DIR)
    print(f"[out_dir] {abs_path(out_dir)}")

    # Resolve 1Hz CSV (auto-convert Excel if needed)
    data_1hz_csv = ensure_1hz_csv(out_dir)
    C.DATA_1HZ_CSV = data_1hz_csv
    C.REAL_1HZ_CSV = data_1hz_csv
    print(f"[data] Using DATA_1HZ_CSV={abs_path(data_1hz_csv)}")

    G = load_graph(C.NODE_FILE, C.EDGE_FILE)
    print(f"[graph] nodes={len(G.node_ids)} edges={G.edge_index.shape[1]} (DIRECTED) node_feat_dim={G.node_feat.shape[1]}")

    if C.MODE == "train":
        train_pipeline(G, out_dir)
    elif C.MODE == "plan":
        plan_pipeline(G, out_dir)
    else:
        raise ValueError("MODE must be 'train' or 'plan'")


if __name__ == "__main__":
    main()
