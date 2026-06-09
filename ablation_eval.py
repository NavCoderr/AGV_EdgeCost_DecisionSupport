# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import math
import shutil
import numpy as np
import pandas as pd
import torch

import config as C
from graph_data import load_graph
from planner import build_full_cost_maps, dijkstra_route, astar_route_safe, build_edge_schedule
from templates import load_geom_templates
from edge_cost_train import train_edge_cost_model, predict_all_edge_costs
from temporal_mlp import load_temporal_model
from trajectory import simulate_temporal_mlp, simulate_temporal_mlp_closed_loop
from model import EdgeCostModel


# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "inductive_folder_new_data"

OUT_DIR = DATA_DIR
ABL_DIR = DATA_DIR / "ablation_reports"
ABL_DIR.mkdir(parents=True, exist_ok=True)

EDGE_SAMPLES_CSV = DATA_DIR / "edge_samples.csv"
EDGE_COSTS_PRED_CSV = DATA_DIR / "edge_costs_pred.csv"
GEOM_CSV = DATA_DIR / "geom_templates.csv"
TEMP_SUMMARY_CSV = DATA_DIR / "temporal_all_summary.csv"
BEST_TEMP_KIND_TXT = DATA_DIR / "best_temporal_kind.txt"

REAL_1HZ = Path(C.REAL_1HZ_CSV)


# ============================================================
# HELPERS
# ============================================================
def _pick_col(df: pd.DataFrame, *names):
    norm = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for n in names:
        k = str(n).strip().lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    return None


def _read_auto(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _route_nodes_to_edges(route_nodes):
    return [(int(route_nodes[i]), int(route_nodes[i + 1])) for i in range(len(route_nodes) - 1)]


def _route_to_str(route_nodes):
    return " -> ".join(map(str, route_nodes))


def _load_best_temporal_kind() -> str:
    if BEST_TEMP_KIND_TXT.exists():
        return BEST_TEMP_KIND_TXT.read_text(encoding="utf-8").strip().lower()
    if TEMP_SUMMARY_CSV.exists():
        d = pd.read_csv(TEMP_SUMMARY_CSV)
        if "kind" in d.columns:
            if "MAE" in d.columns:
                d = d.sort_values("MAE", ascending=True)
            return str(d.iloc[0]["kind"]).strip().lower()
    return "physics_delta"


def _load_best_temporal():
    kind = _load_best_temporal_kind()
    model_path = OUT_DIR / f"temporal_{kind}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Best temporal model not found: {model_path}")
    tm, meta = load_temporal_model(str(model_path))
    return kind, tm, meta


def _aggregate_baseline(edge_samples: pd.DataFrame) -> pd.DataFrame:
    d = edge_samples.copy()
    out = (
        d.groupby(["u_node_id", "v_node_id"], as_index=False)
        .agg(
            time_mean=("time_s", "mean"),
            energy_mean=("energy_J", "mean"),
            n_samples=("samples", "max"),
        )
    )
    out["u"] = out["u_node_id"].astype(int)
    out["v"] = out["v_node_id"].astype(int)
    return out[["u", "v", "time_mean", "energy_mean", "n_samples"]]


def _summarize_route_against_baseline(route_nodes, sched_df: pd.DataFrame, edge_samples: pd.DataFrame):
    base = _aggregate_baseline(edge_samples)
    base_t = {(int(r.u), int(r.v)): float(r.time_mean) for r in base.itertuples(index=False)}
    base_e = {(int(r.u), int(r.v)): float(r.energy_mean) for r in base.itertuples(index=False)}

    edges = _route_nodes_to_edges(route_nodes)

    baseline_time = 0.0
    baseline_energy = 0.0
    miss_base = 0

    for u, v in edges:
        if (u, v) in base_t:
            baseline_time += float(base_t[(u, v)])
        else:
            miss_base += 1
        if (u, v) in base_e:
            baseline_energy += float(base_e[(u, v)])

    pred_time = float(pd.to_numeric(sched_df["t_pred_s"], errors="coerce").fillna(0).sum())
    pred_energy = float(pd.to_numeric(sched_df["e_pred_J"], errors="coerce").fillna(0).sum())

    return {
        "route_nodes": _route_to_str(route_nodes),
        "n_edges": int(len(edges)),
        "baseline_time_s_sum": float(baseline_time),
        "pred_time_s_sum": float(pred_time),
        "planning_time_err_s": float(pred_time - baseline_time),
        "planning_time_err_pct": float(((pred_time - baseline_time) / baseline_time) * 100.0) if baseline_time > 1e-9 else np.nan,
        "baseline_energy_J_sum": float(baseline_energy),
        "pred_energy_J_sum": float(pred_energy),
        "planning_energy_err_J": float(pred_energy - baseline_energy),
        "missing_edges_in_baseline": int(miss_base),
    }


def _load_real_for_eval(path: Path) -> pd.DataFrame:
    df = _read_auto(path)
    cols = set(df.columns)

    if {"x", "y"}.issubset(cols):
        tcol = "t_sec_work" if "t_sec_work" in cols else ("t_sec" if "t_sec" in cols else None)
        if tcol is None:
            df["t"] = np.arange(len(df), dtype=int)
        else:
            df["t"] = pd.to_numeric(df[tcol], errors="coerce").ffill().fillna(0).astype(int)

        out = pd.DataFrame({
            "t": df["t"].astype(int),
            "x_real": pd.to_numeric(df["x"], errors="coerce"),
            "y_real": pd.to_numeric(df["y"], errors="coerce"),
            "speed_real": pd.to_numeric(df["Speed"], errors="coerce") if "Speed" in cols else np.nan,
        })
        out = out.dropna(subset=["t", "x_real", "y_real"]).sort_values("t").reset_index(drop=True)
        out["t"] = out["t"] - int(out["t"].iloc[0])
        out = out.groupby("t", as_index=False).first()
        return out

    if {"X-coordinate", "Y-coordinate"}.issubset(cols):
        if "t_sec_work" in cols:
            t = pd.to_numeric(df["t_sec_work"], errors="coerce")
        elif "t_sec" in cols:
            t = pd.to_numeric(df["t_sec"], errors="coerce")
        else:
            t = pd.Series(np.arange(len(df), dtype=int))

        t = t.ffill().fillna(0).astype(int)

        out = pd.DataFrame({
            "t": t.astype(int),
            "x_real": pd.to_numeric(df["X-coordinate"], errors="coerce"),
            "y_real": pd.to_numeric(df["Y-coordinate"], errors="coerce"),
            "speed_real": pd.to_numeric(df["Speed"], errors="coerce") if "Speed" in cols else np.nan,
        })
        out = out.dropna(subset=["t", "x_real", "y_real"]).sort_values("t").reset_index(drop=True)
        out["t"] = out["t"] - int(out["t"].iloc[0])
        out = out.groupby("t", as_index=False).first()
        return out

    raise ValueError(f"Unsupported real trajectory format: {path}")


def _load_plan_for_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "t_global_s" not in df.columns:
        raise ValueError(f"{path.name} missing t_global_s")

    if "note" in df.columns:
        bad = df["note"].astype(str).str.contains("hold_after_finish", na=False)
        df = df.loc[~bad].copy()

    if "u" in df.columns and "v" in df.columns:
        uu = pd.to_numeric(df["u"], errors="coerce")
        vv = pd.to_numeric(df["v"], errors="coerce")
        df = df.loc[~((uu == -1) & (vv == -1))].copy()

    t = pd.to_numeric(df["t_global_s"], errors="coerce").fillna(0.0)
    t = np.round(t).astype(int)

    out = pd.DataFrame({
        "t": t,
        "x_pred": pd.to_numeric(df["x"], errors="coerce") if "x" in df.columns else np.nan,
        "y_pred": pd.to_numeric(df["y"], errors="coerce") if "y" in df.columns else np.nan,
        "speed_pred": pd.to_numeric(df["Speed"], errors="coerce") if "Speed" in df.columns else np.nan,
    })

    out = out.dropna(subset=["t", "x_pred", "y_pred"]).sort_values("t").reset_index(drop=True)
    out["t"] = out["t"] - int(out["t"].iloc[0])
    out = out.groupby("t", as_index=False).first()
    return out


def _best_window_in_real_by_pos_rmse(
    real_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    max_shift_s: int = 3,
) -> dict:
    """
    FIX (ablation_eval): The original _trajectory_metrics merged plan and real
    both starting from t=0, but the plan starts at the planned route's first node
    while the real recording starts at a completely different position in the shop
    floor.  This caused ~10-11 m position RMSE in Table 8 — a meaningless number.

    This function slides the plan window across the full real recording to find
    the time offset where plan and real positions best agree (minimum RMSE), then
    returns that offset so _trajectory_metrics can cut the correct real segment.
    """
    r = real_df[["t", "x_real", "y_real"]].copy().sort_values("t").reset_index(drop=True)
    p = plan_df[["t", "x_pred", "y_pred"]].copy().sort_values("t").reset_index(drop=True)

    T = int(p["t"].max())
    L = T + 1

    # If the real recording is shorter than the plan, fall back to t=0
    if len(r) < L:
        return {"real_start_t": int(r["t"].iloc[0]), "shift_s": 0, "rmse": float("inf")}

    rx = r["x_real"].to_numpy(dtype=float)
    ry = r["y_real"].to_numpy(dtype=float)
    px = p["x_pred"].to_numpy(dtype=float)
    py = p["y_pred"].to_numpy(dtype=float)
    rt = r["t"].to_numpy(dtype=int)

    best_rmse = float("inf")
    best_start_t = int(rt[0])
    best_shift = 0

    for start_idx in range(0, len(r) - L + 1):
        wx = rx[start_idx:start_idx + L]
        wy = ry[start_idx:start_idx + L]

        for sh in range(-int(max_shift_s), int(max_shift_s) + 1):
            a0 = max(0, sh)
            a1 = min(L, L + sh)
            b0 = max(0, -sh)
            b1 = min(L, L - sh)

            if (a1 - a0) < 5:
                continue

            dx = px[b0:b1] - wx[a0:a1]
            dy = py[b0:b1] - wy[a0:a1]
            rmse = float(np.sqrt(np.mean(dx * dx + dy * dy)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_start_t = int(rt[start_idx])
                best_shift = int(sh)

    return {"real_start_t": best_start_t, "shift_s": best_shift, "rmse": best_rmse}


def _trajectory_metrics(real_path: Path, plan_path: Path):
    """
    FIX: replaced naive t=0 merge with window-aligned comparison.

    Original code merged real (whole recording, t=0..N) with plan (t=0..M)
    directly on the 't' column.  Because the plan trajectory starts at the
    planned route's first node while the real recording starts wherever the
    AGV happened to be at the beginning of the log, the two position sequences
    were spatially offset by ~15-20 m from the very first second.  This made
    pos_RMSE ~10-11 m for every ablation variant — an artefact of misalignment,
    not a real quality difference.

    The fix: use _best_window_in_real_by_pos_rmse to locate the real-recording
    segment that best corresponds to the planned route, then compute errors only
    over that aligned window.  This is consistent with the approach used in
    holdout_trajectory_eval.py and produces physically meaningful metrics.
    """
    if not real_path.exists() or not plan_path.exists():
        return {"traj_pos_mae_m": np.nan, "traj_pos_rmse_m": np.nan, "traj_speed_mae_mps": np.nan}

    real = _load_real_for_eval(real_path)
    plan = _load_plan_for_eval(plan_path)

    # --- interpolate real onto a uniform 1-Hz grid ---
    t_end_real = int(real["t"].max())
    t_grid_real = np.arange(0, t_end_real + 1, 1, dtype=int)
    real_i = (
        real.set_index("t")
        .reindex(t_grid_real)
        .interpolate(limit_direction="both")
        .reset_index()
        .rename(columns={"index": "t"})
    )
    real_i = real_i.dropna(subset=["x_real", "y_real"]).copy()
    real_i["t"] = real_i["t"].astype(int)
    real_i = real_i.groupby("t", as_index=False).first()

    # --- interpolate plan onto a uniform 1-Hz grid ---
    T = int(plan["t"].max())
    t_grid_plan = np.arange(0, T + 1, 1, dtype=int)
    plan_i = (
        plan.set_index("t")
        .reindex(t_grid_plan)
        .interpolate(limit_area="inside")
        .reset_index()
        .rename(columns={"index": "t"})
    )
    plan_i["t"] = plan_i["t"].astype(int)
    plan_i = plan_i.dropna(subset=["x_pred", "y_pred"]).copy()
    plan_i = plan_i.groupby("t", as_index=False).first()

    if len(plan_i) < 3:
        return {"traj_pos_mae_m": np.nan, "traj_pos_rmse_m": np.nan, "traj_speed_mae_mps": np.nan}

    # --- find the best-matching window in real for this plan ---
    best = _best_window_in_real_by_pos_rmse(real_i, plan_i, max_shift_s=3)
    t0 = int(best["real_start_t"])
    sh = int(best["shift_s"])

    # cut the matching real segment
    t1 = t0 + T
    real_cut = real_i[(real_i["t"] >= t0) & (real_i["t"] <= t1)].copy()
    if len(real_cut) < 3:
        return {"traj_pos_mae_m": np.nan, "traj_pos_rmse_m": np.nan, "traj_speed_mae_mps": np.nan}

    real_cut = real_cut.copy()
    real_cut["t"] = real_cut["t"] - int(real_cut["t"].iloc[0])
    real_cut = real_cut.groupby("t", as_index=False).first()

    # apply sub-second shift to plan if found
    plan_cut = plan_i.copy()
    plan_cut["t"] = plan_cut["t"] + sh
    plan_cut = plan_cut[plan_cut["t"] >= 0].copy()
    plan_cut = plan_cut.groupby("t", as_index=False).first()

    # --- merge and compute errors ---
    m = pd.merge(real_cut, plan_cut, on="t", how="inner")
    if len(m) < 3:
        return {"traj_pos_mae_m": np.nan, "traj_pos_rmse_m": np.nan, "traj_speed_mae_mps": np.nan}

    pos_err = np.sqrt((m["x_pred"] - m["x_real"]) ** 2 + (m["y_pred"] - m["y_real"]) ** 2)
    pos_mae = float(np.mean(np.abs(pos_err)))
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))

    if "speed_real" in m.columns and "speed_pred" in m.columns and m["speed_pred"].notna().any():
        speed_err = np.abs(
            pd.to_numeric(m["speed_pred"], errors="coerce")
            - pd.to_numeric(m["speed_real"], errors="coerce")
        )
        speed_mae = float(np.nanmean(speed_err))
    else:
        speed_mae = np.nan

    return {
        "traj_pos_mae_m": pos_mae,
        "traj_pos_rmse_m": pos_rmse,
        "traj_speed_mae_mps": speed_mae,
    }


def _polyline_cumlen(pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) < 2:
        return np.array([0.0], dtype=np.float32)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg).astype(np.float32)])


def _interp_xy_on_polyline_frac(pts: np.ndarray, frac: float):
    frac = float(np.clip(frac, 0.0, 1.0))
    if pts is None or len(pts) == 0:
        return float("nan"), float("nan")
    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1])

    cum = _polyline_cumlen(pts)
    total = float(cum[-1])
    if total <= 1e-9:
        return float(pts[0, 0]), float(pts[0, 1])

    s = frac * total
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 2))
    s0, s1 = float(cum[idx]), float(cum[idx + 1])
    p0 = pts[idx]
    p1 = pts[idx + 1]
    if s1 <= s0:
        return float(p0[0]), float(p0[1])

    a = (s - s0) / (s1 - s0)
    xy = (1.0 - a) * p0 + a * p1
    return float(xy[0]), float(xy[1])


def _constant_speed_rollout(G, sched_df: pd.DataFrame, geom_templates, out_csv: Path):
    rows = []
    t_global = 0
    e_cum = 0.0

    for ei, r in enumerate(sched_df.itertuples(index=False)):
        u = int(r.u)
        v = int(r.v)
        dt_s = max(int(getattr(r, "dt_s", 1)), 1)
        dist = float(getattr(r, "edge_distance", 0.0))
        t_edge = float(getattr(r, "t_pred_s", dt_s))
        e_edge = float(getattr(r, "e_pred_J", 0.0))
        slow = float(getattr(r, "slowdown_idx", 0.0))

        pts = geom_templates.get((u, v), None)
        if pts is None:
            ui = G.node_id_to_idx[u]
            vi = G.node_id_to_idx[v]
            pts = np.array([
                [float(G.node_x[ui]), float(G.node_y[ui])],
                [float(G.node_x[vi]), float(G.node_y[vi])],
            ], dtype=np.float32)

        for s in range(dt_s):
            frac = float((s + 1) / dt_s)
            x, y = _interp_xy_on_polyline_frac(pts, frac)
            e_cum += float(e_edge / dt_s)

            rows.append({
                "t_global_s": int(t_global),
                "edge_idx": int(ei),
                "u": int(u),
                "v": int(v),
                "tau": frac,
                "frac": frac,
                "x": float(x),
                "y": float(y),
                "edge_distance": float(dist),
                "slowdown_idx": float(slow),
                "t_pred_s": float(t_edge),
                "e_pred_J": float(e_edge),
                "e_cum_J": float(e_cum),
            })
            t_global += 1

    df = pd.DataFrame(rows)
    if len(df):
        dx = pd.to_numeric(df["x"], errors="coerce").diff()
        dy = pd.to_numeric(df["y"], errors="coerce").diff()
        step_dist = np.sqrt(dx * dx + dy * dy).fillna(0.0)
        df["step_dist_m"] = step_dist
        df["Speed"] = step_dist.astype(float)
        df["accel_mps2"] = df["Speed"].diff().fillna(0.0)
    df.to_csv(out_csv, index=False)


def _plan_route(G, pred_df, edge_samples_df):
    time_cost, energy_cost, combo_cost, slowdown_map = build_full_cost_maps(G, pred_df, edge_samples_df)
    main_map = {"time": time_cost, "energy": energy_cost, "combo": combo_cost}[C.MAIN_COST]

    if str(C.PLANNER).lower() == "dijkstra":
        route_nodes, route_cost = dijkstra_route(G, int(C.START_NODE), int(C.GOAL_NODE), main_map, C.BLOCKED_NODES, C.BLOCKED_EDGES)
    else:
        route_nodes, route_cost = astar_route_safe(
            G, int(C.START_NODE), int(C.GOAL_NODE), main_map, C.BLOCKED_NODES, C.BLOCKED_EDGES, w=float(C.ASTAR_W)
        )
    if not route_nodes:
        raise RuntimeError("Planner failed to find route")

    sched_list = build_edge_schedule(G, route_nodes, time_cost, energy_cost, slowdown_map)
    sched_df = pd.DataFrame(sched_list)
    return route_nodes, sched_df


def _run_variant(name: str, G, pred_df, edge_samples_df, geom_templates, temporal_mode: str, tm=None, meta=None):
    route_nodes, sched_df = _plan_route(G, pred_df, edge_samples_df)

    sched_csv = ABL_DIR / f"{name}_planned_edge_schedule.csv"
    sched_df.to_csv(sched_csv, index=False)

    traj_csv = None
    if temporal_mode == "closed_loop":
        traj_csv = ABL_DIR / f"{name}_trajectory.csv"
        simulate_temporal_mlp_closed_loop(
            G, sched_df, geom_templates, tm, meta, real_1hz_csv=REAL_1HZ, out_csv=traj_csv
        )
    elif temporal_mode == "open_loop":
        traj_csv = ABL_DIR / f"{name}_trajectory.csv"
        simulate_temporal_mlp(
            G, sched_df, geom_templates, tm, meta, out_csv=traj_csv
        )
    elif temporal_mode == "constant":
        traj_csv = ABL_DIR / f"{name}_trajectory.csv"
        _constant_speed_rollout(G, sched_df, geom_templates, traj_csv)
    else:
        traj_csv = None

    route_metrics = _summarize_route_against_baseline(route_nodes, sched_df, edge_samples_df)
    traj_metrics = _trajectory_metrics(REAL_1HZ, traj_csv) if traj_csv else {
        "traj_pos_mae_m": np.nan, "traj_pos_rmse_m": np.nan, "traj_speed_mae_mps": np.nan
    }

    return {
        "variant": name,
        **route_metrics,
        **traj_metrics,
        "schedule_csv": str(sched_csv),
        "trajectory_csv": str(traj_csv) if traj_csv else "",
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"[ablation_out] {ABL_DIR}")

    G = load_graph(C.NODE_FILE, C.EDGE_FILE)
    print(f"[graph] nodes={len(G.node_ids)} edges={G.edge_index.shape[1]}")

    edge_samples = pd.read_csv(EDGE_SAMPLES_CSV)
    pred_base = pd.read_csv(EDGE_COSTS_PRED_CSV)
    geom_templates = load_geom_templates(G, GEOM_CSV)

    best_kind, tm, meta = _load_best_temporal()
    print(f"[best_temporal] kind={best_kind}")

    rows = []

    # 1) Full model
    print("\n[run] full_model")
    rows.append(_run_variant(
        "full_model", G, pred_base, edge_samples, geom_templates,
        temporal_mode="closed_loop", tm=tm, meta=meta
    ))

    # 2) No learned edge cost
    print("\n[run] no_learned_edge_cost")
    pred_empty = pd.DataFrame(columns=["u_node_id", "v_node_id", "pred_time_s", "pred_energy_J"])
    rows.append(_run_variant(
        "no_learned_edge_cost", G, pred_empty, edge_samples, geom_templates,
        temporal_mode="closed_loop", tm=tm, meta=meta
    ))

    # 3) No slowdown features (retrain)
    print("\n[run] no_slowdown_features")
    edge_samples_no_slow = edge_samples.copy()
    if "slowdown_idx" in edge_samples_no_slow.columns:
        edge_samples_no_slow["slowdown_idx"] = 0.0

    no_slow_dir = ABL_DIR / "no_slowdown_train"
    no_slow_dir.mkdir(parents=True, exist_ok=True)
    model_path, train_df = train_edge_cost_model(G, edge_samples_no_slow, no_slow_dir, EdgeCostModel)
    pred_no_slow_csv = predict_all_edge_costs(G, model_path, edge_samples_no_slow, no_slow_dir, EdgeCostModel)
    pred_no_slow = pd.read_csv(pred_no_slow_csv)

    rows.append(_run_variant(
        "no_slowdown_features", G, pred_no_slow, edge_samples_no_slow, geom_templates,
        temporal_mode="closed_loop", tm=tm, meta=meta
    ))

    # 4) No temporal model
    print("\n[run] no_temporal_model")
    rows.append(_run_variant(
        "no_temporal_model", G, pred_base, edge_samples, geom_templates,
        temporal_mode="constant", tm=None, meta=None
    ))

    # 5) Open-loop only
    print("\n[run] open_loop_only")
    rows.append(_run_variant(
        "open_loop_only", G, pred_base, edge_samples, geom_templates,
        temporal_mode="open_loop", tm=tm, meta=meta
    ))

    out = pd.DataFrame(rows)

    # nicer ordering
    pref = [
        "variant", "route_nodes", "n_edges",
        "baseline_time_s_sum", "pred_time_s_sum", "planning_time_err_s", "planning_time_err_pct",
        "baseline_energy_J_sum", "pred_energy_J_sum", "planning_energy_err_J",
        "traj_pos_mae_m", "traj_pos_rmse_m", "traj_speed_mae_mps",
        "missing_edges_in_baseline", "schedule_csv", "trajectory_csv",
    ]
    cols = [c for c in pref if c in out.columns] + [c for c in out.columns if c not in pref]
    out = out[cols]

    summary_csv = ABL_DIR / "ablation_summary.csv"
    out.to_csv(summary_csv, index=False)

    print("\n[done] Ablation finished")
    print(f"[saved] {summary_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
