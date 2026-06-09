# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import config as C
from graph_data import load_graph 
from planner import dijkstra_route, astar_route_safe, build_edge_schedule
from templates import load_geom_templates
from trajectory import simulate_temporal_mlp_closed_loop
from temporal_mlp import load_temporal_model


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "inductive_folder_new_data"
OUT_DIR = DATA_DIR / "distance_only_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_SAMPLES_CSV = DATA_DIR / "edge_samples.csv"
GEOM_CSV = DATA_DIR / "geom_templates.csv"
TEMP_SUMMARY_CSV = DATA_DIR / "temporal_all_summary.csv"
REAL_1HZ = Path(C.REAL_1HZ_CSV)


def load_best_temporal():
    summ = pd.read_csv(TEMP_SUMMARY_CSV).sort_values("MAE", ascending=True)
    kind = str(summ.iloc[0]["kind"]).strip().lower()
    model_path = DATA_DIR / f"temporal_{kind}.pt"
    tm, meta = load_temporal_model(str(model_path))
    return kind, tm, meta


def build_distance_only_maps(G):
    time_map = {}
    energy_map = {}
    for u, v, dist in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance):
        u = int(u); v = int(v)
        dist = float(dist)
        time_map[(u, v)] = dist
        energy_map[(u, v)] = dist
    return time_map, energy_map


def build_combo(time_map, energy_map, alpha):
    tvals = np.array(list(time_map.values()), dtype=float)
    evals = np.array(list(energy_map.values()), dtype=float)
    tscale = float(np.median(tvals)) if len(tvals) else 1.0
    escale = float(np.median(evals)) if len(evals) else 1.0
    combo = {}
    for k in time_map:
        combo[k] = float(alpha * (time_map[k] / max(tscale, 1e-9)) + (1.0 - alpha) * (energy_map[k] / max(escale, 1e-9)))
    return combo


def main():
    G = load_graph(C.NODE_FILE, C.EDGE_FILE)
    kind, tm, meta = load_best_temporal()
    print("[best_temporal]", kind)

    time_map, energy_map = build_distance_only_maps(G)
    combo = build_combo(time_map, energy_map, float(C.ALPHA))

    if str(C.PLANNER).lower() == "dijkstra":
        route_nodes, route_cost = dijkstra_route(G, int(C.START_NODE), int(C.GOAL_NODE), combo, C.BLOCKED_NODES, C.BLOCKED_EDGES)
    else:
        route_nodes, route_cost = astar_route_safe(G, int(C.START_NODE), int(C.GOAL_NODE), combo, C.BLOCKED_NODES, C.BLOCKED_EDGES, w=float(C.ASTAR_W))

    slowdown_map = {k: 0.0 for k in combo.keys()}
    sched = pd.DataFrame(build_edge_schedule(G, route_nodes, time_map, energy_map, slowdown_map))
    sched_csv = OUT_DIR / "distance_only_planned_edge_schedule.csv"
    sched.to_csv(sched_csv, index=False)

    geom_templates = load_geom_templates(G, GEOM_CSV)
    traj_csv = OUT_DIR / "distance_only_trajectory.csv"
    simulate_temporal_mlp_closed_loop(G, sched, geom_templates, tm, meta, real_1hz_csv=REAL_1HZ, out_csv=traj_csv)

    summary = pd.DataFrame([{
        "variant": "distance_only_without_gnn",
        "route": " -> ".join(map(str, route_nodes)),
        "pred_time_s": float(pd.to_numeric(sched["t_pred_s"], errors="coerce").sum()),
        "pred_energy_J": float(pd.to_numeric(sched["e_pred_J"], errors="coerce").sum()),
        "n_edges": int(len(sched)),
    }])
    summary.to_csv(OUT_DIR / "distance_only_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("[saved]", sched_csv)
    print("[saved]", traj_csv)
    print("[saved]", OUT_DIR / "distance_only_summary.csv")


if __name__ == "__main__":
    main()
