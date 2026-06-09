# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List
import numpy as np
import math
import pandas as pd

import config as C
from graph_data import GraphData
from utils_io import abs_path


def build_full_cost_maps(G: GraphData, pred_df, edge_samples_df):
    global LAST_PREDICTED_EDGES
    LAST_PREDICTED_EDGES = set()

    time_cost: Dict[Tuple[int, int], float] = {}
    energy_cost: Dict[Tuple[int, int], float] = {}

    if pred_df is not None and len(pred_df) > 0:
        for r in pred_df.itertuples(index=False):
            u = int(r.u_node_id); v = int(r.v_node_id)
            t = float(r.pred_time_s); e = float(r.pred_energy_J)
            if np.isfinite(t) and t > 0:
                time_cost[(u, v)] = max(t, C.PRED_TIME_MIN_S)
                LAST_PREDICTED_EDGES.add((u, v))
            if np.isfinite(e) and e > 0:
                energy_cost[(u, v)] = max(e, C.PRED_ENERGY_MIN_J)
                LAST_PREDICTED_EDGES.add((u, v))

    samples_time = {}
    samples_energy = {}
    samples_slow = {}
    samples_speed = []
    samples_power = []
    samples_jpm = []

    if edge_samples_df is not None and len(edge_samples_df) > 0:
        for r in edge_samples_df.itertuples(index=False):
            u = int(r.u_node_id); v = int(r.v_node_id)
            d = float(r.edge_distance)
            t = float(r.time_s); e = float(r.energy_J)
            ms = float(getattr(r, "mean_speed", np.nan))
            mp = float(getattr(r, "mean_power_W", np.nan))
            sl = float(getattr(r, "slowdown_idx", np.nan))

            if np.isfinite(t) and t > 0: samples_time[(u, v)] = t
            if np.isfinite(e) and e > 0: samples_energy[(u, v)] = e
            if np.isfinite(sl): samples_slow[(u, v)] = float(np.clip(sl, 0, 1))
            if np.isfinite(ms) and ms > 0: samples_speed.append(ms)
            if np.isfinite(mp) and mp > 0: samples_power.append(mp)
            if np.isfinite(e) and e > 0 and np.isfinite(d) and d > 1e-6: samples_jpm.append(e / d)

    speed_fallback = float(np.percentile(samples_speed, 95)) if samples_speed else float(C.DEFAULT_FALLBACK_SPEED_MPS)
    speed_fallback = max(speed_fallback, 1e-3)
    time_per_m = 1.0 / speed_fallback

    energy_per_m = float(np.median(samples_jpm)) if samples_jpm else float(C.DEFAULT_FALLBACK_ENERGY_J_PER_M)
    slow_global = float(np.mean(list(samples_slow.values()))) if samples_slow else 0.0

    slowdown_map = {}

    for u, v, d in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance):
        uu, vv = int(u), int(v)
        dist = float(d)

        if (uu, vv) not in time_cost:
            if (uu, vv) in samples_time:
                time_cost[(uu, vv)] = max(float(samples_time[(uu, vv)]), C.PRED_TIME_MIN_S)
            else:
                time_cost[(uu, vv)] = max(dist * time_per_m, C.PRED_TIME_MIN_S)

        if (uu, vv) not in energy_cost:
            if (uu, vv) in samples_energy:
                energy_cost[(uu, vv)] = max(float(samples_energy[(uu, vv)]), C.PRED_ENERGY_MIN_J)
            else:
                energy_cost[(uu, vv)] = max(dist * energy_per_m, C.PRED_ENERGY_MIN_J)

        slowdown_map[(uu, vv)] = float(np.clip(samples_slow.get((uu, vv), slow_global), 0, 1))

    # combo cost normalize
    t_vals = np.array(list(time_cost.values()), dtype=float)
    e_wh = np.array([v / 3600.0 for v in energy_cost.values()], dtype=float)

    t_scale = float(np.median(t_vals[np.isfinite(t_vals)]) + 1e-6)
    e_scale = float(np.median(e_wh[np.isfinite(e_wh)]) + 1e-9)

    combo_cost = {}
    for (u, v) in time_cost.keys():
        t_n = time_cost[(u, v)] / t_scale
        e_n = (energy_cost[(u, v)] / 3600.0) / e_scale
        combo_cost[(u, v)] = float(C.ALPHA * t_n + (1.0 - C.ALPHA) * e_n)

    return time_cost, energy_cost, combo_cost, slowdown_map


def dijkstra_route(G: GraphData, start_id: int, goal_id: int, edge_cost: Dict, blocked_nodes, blocked_edges):
    import heapq
    blocked_nodes = set(map(int, blocked_nodes))
    blocked_edges = set((int(a), int(b)) for a, b in blocked_edges)

    adj = {}
    for u, v in zip(G.edge_u_ids, G.edge_v_ids):
        u = int(u); v = int(v)
        if u in blocked_nodes or v in blocked_nodes: 
            continue
        if (u, v) in blocked_edges:
            continue
        adj.setdefault(u, []).append(v)

    dist = {start_id: 0.0}
    prev = {}
    pq = [(0.0, start_id)]

    while pq:
        dcur, u = heapq.heappop(pq)
        if u == goal_id:
            break
        if dcur != dist.get(u, float("inf")):
            continue
        for v in adj.get(u, []):
            w = float(edge_cost.get((u, v), float("inf")))
            if not np.isfinite(w):
                continue
            nd = dcur + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal_id not in dist:
        return [], float("inf")

    path = [goal_id]
    cur = goal_id
    while cur != start_id:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path, float(dist[goal_id])


def astar_route_safe(G: GraphData, start_id: int, goal_id: int, edge_cost: Dict, blocked_nodes, blocked_edges, w=1.0):
    import heapq
    blocked_nodes = set(map(int, blocked_nodes))
    blocked_edges = set((int(a), int(b)) for a, b in blocked_edges)

    adj = {}
    for u, v in zip(G.edge_u_ids, G.edge_v_ids):
        u = int(u); v = int(v)
        if u in blocked_nodes or v in blocked_nodes:
            continue
        if (u, v) in blocked_edges:
            continue
        adj.setdefault(u, []).append(v)

    # heuristic: euclid * median(cost/m)
    dist_map = {(int(u), int(v)): float(d) for u, v, d in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance)}
    ratios = []
    for (u, v), c in edge_cost.items():
        d = dist_map.get((int(u), int(v)), 0.0)
        if np.isfinite(c) and d and d > 1e-6:
            ratios.append(float(c) / d)
    cpm = float(np.median(ratios)) if ratios else 1.0

    gi = G.node_id_to_idx[int(goal_id)]
    xg, yg = float(G.node_x[gi]), float(G.node_y[gi])

    def h(nid: int) -> float:
        i = G.node_id_to_idx[int(nid)]
        xn, yn = float(G.node_x[i]), float(G.node_y[i])
        return math.hypot(xn - xg, yn - yg) * cpm

    gscore = {start_id: 0.0}
    prev = {}
    pq = []
    heapq.heappush(pq, (w * h(start_id), 0.0, start_id))

    while pq:
        fcur, gcur, u = heapq.heappop(pq)
        if gcur != gscore.get(u, float("inf")):
            continue
        if u == goal_id:
            break
        for v in adj.get(u, []):
            wuv = float(edge_cost.get((u, v), float("inf")))
            if not np.isfinite(wuv):
                continue
            ng = gcur + wuv
            if ng < gscore.get(v, float("inf")):
                gscore[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + w * h(v), ng, v))

    if goal_id not in gscore:
        return [], float("inf")

    path = [goal_id]
    cur = goal_id
    while cur != start_id:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path, float(gscore[goal_id])


def build_edge_schedule(G, route_nodes, time_cost, energy_cost, slowdown_map, out_csv=None):
    """
    If GGNN predicted time exists for an edge, trust it.
    Only apply v_cap-based dt_cap when we are using fallback time.
    """
    import math
    import pandas as pd

    # ---- auto-fix wrong order ----
    if isinstance(G, list) and hasattr(route_nodes, "edge_u_ids"):
        G, route_nodes = route_nodes, G

    if not hasattr(G, "edge_u_ids"):
        raise TypeError("build_edge_schedule: first argument must be GraphData (G).")

    dist_map = {(int(u), int(v)): float(d) for u, v, d in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance)}
    predicted_edges = set(globals().get("LAST_PREDICTED_EDGES", set()))

    sched = []
    for i in range(len(route_nodes) - 1):
        u = int(route_nodes[i])
        v = int(route_nodes[i + 1])

        dist = float(dist_map.get((u, v), 0.0))

        # predicted or fallback
        t_pred_s = float(time_cost.get((u, v), max(C.PRED_TIME_MIN_S, dist * C.DEFAULT_FALLBACK_TIME_S_PER_M)))
        e_pred_J = float(energy_cost.get((u, v), max(C.PRED_ENERGY_MIN_J, dist * C.DEFAULT_FALLBACK_ENERGY_J_PER_M)))

        dt_s = int(max(1, round(t_pred_s)))
        slow = float(slowdown_map.get((u, v), 0.0))

        # ONLY if this edge is NOT predicted (fallback case)
        if (u, v) not in predicted_edges:
            v_max = float(getattr(C, "V_MAX_MPS", 0.30))
            v_turn = float(getattr(C, "V_TURN_MPS", 0.12))
            gain = float(getattr(C, "SLOWDOWN_GAIN", 0.70))
            v_cap = max(v_turn, v_max * (1.0 - gain * max(0.0, min(1.0, slow))))
            if dist > 0 and v_cap > 1e-6:
                dt_cap = int(math.ceil(dist / v_cap))
                dt_s = max(dt_s, dt_cap)

        sched.append({
            "edge_idx": i,
            "u": u,
            "v": v,
            "edge_distance": dist,
            "t_pred_s": t_pred_s,
            "e_pred_J": e_pred_J,
            "dt_s": dt_s,
            "slowdown_idx": slow,
            "used_pred": int((u, v) in predicted_edges),
        })

    if out_csv is not None:
        pd.DataFrame(sched).to_csv(out_csv, index=False)

    return sched
