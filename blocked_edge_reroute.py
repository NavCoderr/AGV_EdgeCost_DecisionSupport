# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import heapq
import pandas as pd
import numpy as np

import config as C
from graph_data import load_graph
from planner import build_full_cost_maps


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(C.OUT_DIR)
BLOCK_OUT_DIR = OUT_DIR / "blocked_edge_reroute_reports"
BLOCK_OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = float(getattr(C, "ALPHA", 0.7))

SCENARIOS = [
    {
        "mission": "5->9",
        "start": 5,
        "goal": 9,
        "blocked_edge": (20, 22),
    },
    {
        "mission": "1->5",
        "start": 1,
        "goal": 5,
        "blocked_edge": (26, 29),
    },
    {
        "mission": "6->2",
        "start": 6,
        "goal": 2,
        "blocked_edge": (20, 21),
    },
    {
        "mission": "5->6",
        "start": 5,
        "goal": 6,
        "blocked_edge": (29, 28),
    },
    {
        "mission": "4->1",
        "start": 4,
        "goal": 1,
        "blocked_edge": (10, 7),
    },
]


def as_edge_list(G):
    edges = []
    for u, v, d in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance):
        edges.append((int(u), int(v), float(d)))
    return edges


def build_combo_costs(G, time_cost, energy_cost, alpha: float):
    edge_list = as_edge_list(G)

    t_vals = []
    e_vals = []

    for u, v, _ in edge_list:
        if (u, v) in time_cost and (u, v) in energy_cost:
            t_vals.append(float(time_cost[(u, v)]))
            e_vals.append(float(energy_cost[(u, v)]))

    t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
    e_min, e_max = float(np.min(e_vals)), float(np.max(e_vals))

    t_den = max(t_max - t_min, 1e-9)
    e_den = max(e_max - e_min, 1e-9)

    combo = {}

    for u, v, dist in edge_list:
        t = float(time_cost.get((u, v), dist / 0.15))
        e = float(energy_cost.get((u, v), dist * 300.0))

        t_norm = (t - t_min) / t_den
        e_norm = (e - e_min) / e_den

        combo[(u, v)] = float(alpha * t_norm + (1.0 - alpha) * e_norm)

    return combo


def shortest_path(G, start: int, goal: int, edge_cost: dict, blocked_edge=None):
    blocked_edge = tuple(blocked_edge) if blocked_edge is not None else None

    adj = {}

    for u, v, _ in as_edge_list(G):
        u = int(u)
        v = int(v)

        if blocked_edge is not None and (u, v) == blocked_edge:
            continue

        c = float(edge_cost.get((u, v), 1e6))
        adj.setdefault(u, []).append((v, c))

    pq = [(0.0, int(start), [int(start)])]
    seen = {}

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == int(goal):
            return path, float(cost)

        if node in seen and seen[node] <= cost:
            continue

        seen[node] = cost

        for nxt, c in adj.get(node, []):
            if nxt in path:
                continue
            heapq.heappush(pq, (cost + c, nxt, path + [nxt]))

    return [], float("inf")


def route_edge_metrics(route, time_cost, energy_cost):
    total_t = 0.0
    total_e = 0.0

    for u, v in zip(route[:-1], route[1:]):
        total_t += float(time_cost.get((int(u), int(v)), 0.0))
        total_e += float(energy_cost.get((int(u), int(v)), 0.0))

    return total_t, total_e


def route_str(route):
    if not route:
        return ""
    return " -> ".join(map(str, route))


def main():
    print(f"[out] {BLOCK_OUT_DIR}")

    G = load_graph(C.NODE_FILE, C.EDGE_FILE)

    pred_path = OUT_DIR / "edge_costs_pred.csv"
    samples_path = OUT_DIR / "edge_samples.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_path}. Run main_run.py training first.")

    if not samples_path.exists():
        raise FileNotFoundError(f"Missing file: {samples_path}. Run main_run.py training first.")

    pred_df = pd.read_csv(pred_path)

    edge_samples_df = pd.read_csv(samples_path)

    cost_maps = build_full_cost_maps(G, pred_df, edge_samples_df)

    time_cost = cost_maps[0]

    energy_cost = cost_maps[1]
    combo_cost = build_combo_costs(G, time_cost, energy_cost, alpha=ALPHA)

    rows = []

    for sc in SCENARIOS:
        mission = sc["mission"]
        start = int(sc["start"])
        goal = int(sc["goal"])
        blocked_edge = tuple(sc["blocked_edge"])

        print("=" * 80)
        print(f"[scenario] mission={mission} blocked_edge={blocked_edge}")
        print("=" * 80)

        normal_route, normal_combo = shortest_path(
            G=G,
            start=start,
            goal=goal,
            edge_cost=combo_cost,
            blocked_edge=None,
        )

        blocked_route, blocked_combo = shortest_path(
            G=G,
            start=start,
            goal=goal,
            edge_cost=combo_cost,
            blocked_edge=blocked_edge,
        )

        normal_t, normal_e = route_edge_metrics(normal_route, time_cost, energy_cost)
        blocked_t, blocked_e = route_edge_metrics(blocked_route, time_cost, energy_cost)

        route_changed = int(normal_route != blocked_route and len(blocked_route) > 0)

        row = {
            "mission": mission,
            "start_node": start,
            "goal_node": goal,
            "blocked_edge": f"{blocked_edge[0]}->{blocked_edge[1]}",
            "normal_route": route_str(normal_route),
            "blocked_route": route_str(blocked_route),
            "route_changed": route_changed,
            "normal_n_edges": max(len(normal_route) - 1, 0),
            "blocked_n_edges": max(len(blocked_route) - 1, 0),
            "normal_pred_time_s": normal_t,
            "blocked_pred_time_s": blocked_t,
            "delta_time_s": blocked_t - normal_t,
            "normal_pred_energy_J": normal_e,
            "blocked_pred_energy_J": blocked_e,
            "delta_energy_J": blocked_e - normal_e,
            "normal_combo_cost": normal_combo,
            "blocked_combo_cost": blocked_combo,
            "delta_combo_cost": blocked_combo - normal_combo,
        }

        rows.append(row)

        print(f"[normal]  {row['normal_route']}")
        print(f"[blocked] {row['blocked_route']}")
        print(
            f"[delta] route_changed={route_changed} "
            f"dt={row['delta_time_s']:.3f}s "
            f"dE={row['delta_energy_J']:.3f}J"
        )

    detail_df = pd.DataFrame(rows)
    detail_csv = BLOCK_OUT_DIR / "blocked_edge_reroute_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    summary = {
        "n_scenarios": int(len(detail_df)),
        "n_route_changed": int(detail_df["route_changed"].sum()),
        "route_change_rate": float(detail_df["route_changed"].mean()),
        "mean_delta_time_s": float(detail_df["delta_time_s"].mean()),
        "mean_delta_energy_J": float(detail_df["delta_energy_J"].mean()),
        "mean_delta_combo_cost": float(detail_df["delta_combo_cost"].mean()),
    }

    summary_df = pd.DataFrame([summary])
    summary_csv = BLOCK_OUT_DIR / "blocked_edge_reroute_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("=" * 80)
    print("[done] Blocked-edge rerouting experiment finished")
    print(f"[saved] {detail_csv}")
    print(f"[saved] {summary_csv}")
    print(detail_df.to_string(index=False))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()