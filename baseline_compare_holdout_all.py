from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import config as C
from graph_data import load_graph
from planner import build_full_cost_maps, dijkstra_route, astar_route_safe
from holdout_trajectory_eval import build_mission_ids, build_graph_edge_df, extract_traversals, aggregate_train_edges

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'inductive_folder_new_data'
OUT_DIR = DATA_DIR / 'baseline_reports_holdout_all'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_1S_GLOBAL = DATA_DIR / 'data_1s_global.csv'
PRED_EDGE_CSV = DATA_DIR / 'holdout_traj_reports' / 'edge_costs_pred.csv'
HOLDOUT_SEED = 11
TRAIN_RATIO = 0.70

NODE_FILE = SCRIPT_DIR / 'Node_F3.csv'
EDGE_FILE = SCRIPT_DIR / 'Edge_Distances3_.csv'


def aggregate_edge_baselines(edge_samples: pd.DataFrame, G):
    d = edge_samples.copy()
    agg = d.groupby(['u_node_id', 'v_node_id'], as_index=False).agg(
        mean_speed=('mean_speed', 'mean'),
        energy_per_m=('energy_J_per_m', 'mean'),
    )

    dist_map = {(int(u), int(v)): float(dist) for u, v, dist in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance)}

    global_speed = float(pd.to_numeric(d['mean_speed'], errors='coerce').dropna().mean())
    global_speed = global_speed if np.isfinite(global_speed) and global_speed > 1e-6 else 0.2

    global_epm = float(pd.to_numeric(d['energy_J'], errors='coerce').sum() / np.maximum(pd.to_numeric(d['edge_distance'], errors='coerce').sum(), 1e-6))
    global_epm = global_epm if np.isfinite(global_epm) and global_epm > 1e-6 else 50.0

    uniform_time = {}
    median_time = {}
    static_energy = {}

    for (u, v), dist in dist_map.items():
        uniform_time[(u, v)] = float(dist / global_speed)
        row = agg[(agg['u_node_id'] == u) & (agg['v_node_id'] == v)]
        if len(row):
            sp = float(row['mean_speed'].iloc[0])
            sp = global_speed if (not np.isfinite(sp) or sp <= 1e-6) else sp
            median_time[(u, v)] = float(dist / sp)
            epm = float(row['energy_per_m'].iloc[0])
            epm = global_epm if (not np.isfinite(epm) or epm <= 1e-6) else epm
            static_energy[(u, v)] = float(dist * epm)
        else:
            median_time[(u, v)] = float(dist / global_speed)
            static_energy[(u, v)] = float(dist * global_epm)
    return uniform_time, median_time, static_energy


def build_combo(time_map, energy_map, alpha=0.7):
    tvals = np.array(list(time_map.values()), dtype=float)
    evals = np.array(list(energy_map.values()), dtype=float)
    tscale = float(np.median(tvals)) if len(tvals) else 1.0
    escale = float(np.median(evals)) if len(evals) else 1.0
    tscale = tscale if tscale > 1e-9 else 1.0
    escale = escale if escale > 1e-9 else 1.0

    combo = {}
    keys = set(time_map.keys()) & set(energy_map.keys())
    for k in keys:
        combo[k] = float(alpha * (time_map[k] / tscale) + (1.0 - alpha) * (energy_map[k] / escale))
    return combo


def plan_route(G, start_node, goal_node, cost_map):
    if str(C.PLANNER).lower() == 'dijkstra':
        route_nodes, _ = dijkstra_route(G, int(start_node), int(goal_node), cost_map, C.BLOCKED_NODES, C.BLOCKED_EDGES)
    else:
        route_nodes, _ = astar_route_safe(G, int(start_node), int(goal_node), cost_map, C.BLOCKED_NODES, C.BLOCKED_EDGES, w=float(C.ASTAR_W))
    return route_nodes


def summarize_route(route_nodes, time_map, energy_map):
    t = 0.0
    e = 0.0
    for i in range(len(route_nodes) - 1):
        k = (int(route_nodes[i]), int(route_nodes[i + 1]))
        t += float(time_map.get(k, np.nan))
        e += float(energy_map.get(k, np.nan))
    return float(t), float(e)


def abs_pct_err(pred, true):
    return float(abs(pred - true) / max(abs(true), 1e-9) * 100.0)


def summarize_variant(df, variant):
    x = df[df['variant'] == variant].copy()
    return {
        'variant': variant,
        'n_missions': int(len(x)),
        'mean_abs_time_error_s': float(x['abs_time_err_s'].mean()),
        'std_abs_time_error_s': float(x['abs_time_err_s'].std(ddof=1)) if len(x) > 1 else 0.0,
        'mean_abs_time_error_pct': float(x['abs_time_err_pct'].mean()),
        'std_abs_time_error_pct': float(x['abs_time_err_pct'].std(ddof=1)) if len(x) > 1 else 0.0,
        'mean_abs_energy_error_J': float(x['abs_energy_err_J'].mean()),
        'std_abs_energy_error_J': float(x['abs_energy_err_J'].std(ddof=1)) if len(x) > 1 else 0.0,
        'mean_signed_time_error_s': float(x['time_err_s'].mean()),
        'mean_signed_energy_error_J': float(x['energy_err_J'].mean()),
    }


def main():
    G = load_graph(NODE_FILE, EDGE_FILE)
    graph_edges = build_graph_edge_df(G)
    df = pd.read_csv(DATA_1S_GLOBAL)
    df = build_mission_ids(df)
    trav_df = extract_traversals(df, G.edge_set)

    rng = np.random.RandomState(HOLDOUT_SEED)
    missions = np.array(sorted(trav_df['mission_id'].unique()))
    rng.shuffle(missions)
    n_tr = max(1, int(len(missions) * TRAIN_RATIO))
    tr_ids = set(missions[:n_tr])
    te_ids = set(missions[n_tr:])

    train_trav = trav_df[trav_df['mission_id'].isin(tr_ids)].reset_index(drop=True)
    test_trav = trav_df[trav_df['mission_id'].isin(te_ids)].reset_index(drop=True)
    train_edge_agg = aggregate_train_edges(train_trav, graph_edges)

    pred_df = pd.read_csv(PRED_EDGE_CSV)
    learned_time, learned_energy, _, _ = build_full_cost_maps(G, pred_df, train_edge_agg)
    uniform_time, median_time, static_energy = aggregate_edge_baselines(train_edge_agg, G)

    variants = {
        'learned_model': (learned_time, learned_energy),
        'uniform_speed_plus_static_energy': (uniform_time, static_energy),
        'median_speed_plus_static_energy': (median_time, static_energy),
    }

    rows = []
    for mid in sorted(test_trav['mission_id'].unique()):
        g = test_trav[test_trav['mission_id'] == mid].copy()
        start_node = int(g.iloc[0]['u_node_id'])
        goal_node = int(g.iloc[-1]['v_node_id'])
        true_time = float(g['time_s'].sum())
        true_energy = float(g['energy_J'].sum())
        mission_goal = int(g.iloc[0]['mission_goal'])

        for variant, (tmap, emap) in variants.items():
            combo = build_combo(tmap, emap, alpha=float(C.ALPHA))
            route_nodes = plan_route(G, start_node, goal_node, combo)
            pred_time, pred_energy = summarize_route(route_nodes, tmap, emap)
            rows.append({
                'mission_id': int(mid),
                'mission_goal': mission_goal,
                'start_node': start_node,
                'goal_node': goal_node,
                'variant': variant,
                'route_nodes': ' -> '.join(map(str, route_nodes)),
                'true_time_s': true_time,
                'true_energy_J': true_energy,
                'pred_time_s': pred_time,
                'pred_energy_J': pred_energy,
                'time_err_s': pred_time - true_time,
                'energy_err_J': pred_energy - true_energy,
                'abs_time_err_s': abs(pred_time - true_time),
                'abs_energy_err_J': abs(pred_energy - true_energy),
                'abs_time_err_pct': abs_pct_err(pred_time, true_time),
                'abs_energy_err_pct': abs_pct_err(pred_energy, true_energy),
            })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([summarize_variant(detail, v) for v in variants.keys()])

    detail.to_csv(OUT_DIR / 'baseline_holdout_all_detail.csv', index=False)
    summary.to_csv(OUT_DIR / 'baseline_holdout_all_summary.csv', index=False)

    print(summary.to_string(index=False))
    print('[saved]', OUT_DIR / 'baseline_holdout_all_detail.csv')
    print('[saved]', OUT_DIR / 'baseline_holdout_all_summary.csv')


if __name__ == '__main__':
    main()
