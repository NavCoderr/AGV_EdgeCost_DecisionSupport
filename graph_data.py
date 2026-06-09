# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from utils_io import read_csv_auto, norm_colname, to_float_series


@dataclass
class GraphData:
    node_ids: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    node_feat: np.ndarray
    edge_u_ids: np.ndarray
    edge_v_ids: np.ndarray
    edge_distance: np.ndarray
    node_id_to_idx: Dict[int, int]
    edge_index: np.ndarray
    edge_set: set


def load_graph(node_csv, edge_csv) -> GraphData:
    nodes = read_csv_auto(node_csv)
    edges = read_csv_auto(edge_csv)

    cols_nodes = {norm_colname(c): c for c in nodes.columns}

    def pick_node(*cands):
        for c in cands:
            k = norm_colname(c)
            if k in cols_nodes:
                return cols_nodes[k]
        return None

    nid_c = pick_node("node", "nodeid", "node_id", "id")
    x_c = pick_node("xcoordinate", "x", "xcoord")
    y_c = pick_node("ycoordinate", "y", "ycoord")
    if nid_c is None or x_c is None or y_c is None:
        raise ValueError(f"Node CSV columns not recognized. Found={list(nodes.columns)}")

    tmp = pd.DataFrame({
        "nid": pd.to_numeric(nodes[nid_c], errors="coerce"),
        "x": to_float_series(nodes[x_c]),
        "y": to_float_series(nodes[y_c]),
    }).dropna(subset=["nid", "x", "y"]).copy()

    tmp["nid"] = tmp["nid"].astype(int)
    tmp = tmp.drop_duplicates(subset=["nid"]).sort_values("nid").reset_index(drop=True)

    node_ids = tmp["nid"].to_numpy(dtype=int)
    node_x = tmp["x"].to_numpy(dtype=np.float32)
    node_y = tmp["y"].to_numpy(dtype=np.float32)
    node_id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    cols_edges = {norm_colname(c): c for c in edges.columns}

    def pick_edge(*cands):
        for c in cands:
            k = norm_colname(c)
            if k in cols_edges:
                return cols_edges[k]
        return None

    u_c = pick_edge("from", "u", "u_node_id", "start", "source")
    v_c = pick_edge("to", "v", "v_node_id", "end", "target", "dest")
    d_c = pick_edge("distance", "edge_distance", "dist", "length", "d")
    if u_c is None or v_c is None or d_c is None:
        raise ValueError(f"Edge CSV columns not recognized. Found={list(edges.columns)}")

    e_tmp = pd.DataFrame({
        "u": pd.to_numeric(edges[u_c], errors="coerce"),
        "v": pd.to_numeric(edges[v_c], errors="coerce"),
        "d": to_float_series(edges[d_c]).fillna(0.0),
    }).dropna(subset=["u", "v"]).copy()

    e_tmp["u"] = e_tmp["u"].astype(int)
    e_tmp["v"] = e_tmp["v"].astype(int)
    e_tmp["d"] = e_tmp["d"].astype(float)

    for uu, vv in zip(e_tmp["u"].to_numpy(), e_tmp["v"].to_numpy()):
        if int(uu) not in node_id_to_idx or int(vv) not in node_id_to_idx:
            raise ValueError(f"Edge has unknown node id: ({int(uu)},{int(vv)})")

    edge_u_ids = e_tmp["u"].to_numpy(dtype=np.int64)
    edge_v_ids = e_tmp["v"].to_numpy(dtype=np.int64)
    edge_distance = e_tmp["d"].to_numpy(dtype=np.float32)

    u_idx = np.array([node_id_to_idx[int(u)] for u in edge_u_ids], dtype=np.int64)
    v_idx = np.array([node_id_to_idx[int(v)] for v in edge_v_ids], dtype=np.int64)
    edge_index = np.stack([u_idx, v_idx], axis=0)

    # Node features: x_norm, y_norm, degree_norm, is_charger
    deg = np.zeros(len(node_ids), dtype=np.float32)
    for uu, vv in zip(edge_u_ids, edge_v_ids):
        deg[node_id_to_idx[int(uu)]] += 1.0
        deg[node_id_to_idx[int(vv)]] += 1.0
    deg_norm = deg / (deg.max() + 1e-6)

    charge_col = None
    for c in nodes.columns:
        nc = norm_colname(c)
        if ("charge" in nc) or ("charger" in nc):
            charge_col = c
            break

    if charge_col is None:
        is_charger = np.zeros(len(node_ids), dtype=np.float32)
    else:
        ch_tmp = pd.DataFrame({
            "nid": pd.to_numeric(nodes[nid_c], errors="coerce"),
            "ch": pd.to_numeric(nodes[charge_col], errors="coerce"),
        }).dropna(subset=["nid"]).copy()
        ch_tmp["nid"] = ch_tmp["nid"].astype(int)
        ch_map = {int(r.nid): float(r.ch) if np.isfinite(r.ch) else 0.0 for r in ch_tmp.itertuples(index=False)}
        is_charger = np.array([(1.0 if ch_map.get(int(n), 0.0) > 0 else 0.0) for n in node_ids], dtype=np.float32)

    x_min, x_max = float(node_x.min()), float(node_x.max())
    y_min, y_max = float(node_y.min()), float(node_y.max())
    node_x_n = (node_x - x_min) / (x_max - x_min + 1e-6)
    node_y_n = (node_y - y_min) / (y_max - y_min + 1e-6)

    node_feat = np.stack([node_x_n, node_y_n, deg_norm, is_charger], axis=1).astype(np.float32)
    edge_set = set((int(u), int(v)) for u, v in zip(edge_u_ids, edge_v_ids))

    return GraphData(
        node_ids=node_ids,
        node_x=node_x,
        node_y=node_y,
        node_feat=node_feat,
        edge_u_ids=edge_u_ids,
        edge_v_ids=edge_v_ids,
        edge_distance=edge_distance,
        node_id_to_idx=node_id_to_idx,
        edge_index=edge_index,
        edge_set=edge_set,
    )
