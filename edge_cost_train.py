# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from utils_io import abs_path
from graph_data import GraphData


def _split_df(df):
    df = df.copy()
    df["edge_id"] = df["u_node_id"].astype(str) + "_" + df["v_node_id"].astype(str)

    rng = np.random.RandomState(int(C.EDGE_SPLIT_SEED))
    uniq = df["edge_id"].unique()
    rng.shuffle(uniq)

    n = len(uniq)
    n_tr = int(n * float(C.EDGE_SPLIT_TRAIN))
    n_va = int(n * float(C.EDGE_SPLIT_VAL))

    tr_ids = set(uniq[:n_tr])
    va_ids = set(uniq[n_tr:n_tr + n_va])
    te_ids = set(uniq[n_tr + n_va:])

    tr = df[df["edge_id"].isin(tr_ids)].reset_index(drop=True)
    va = df[df["edge_id"].isin(va_ids)].reset_index(drop=True)
    te = df[df["edge_id"].isin(te_ids)].reset_index(drop=True)

    return tr, va, te


def train_edge_cost_model(G: GraphData, edge_samples: pd.DataFrame, out_dir: Path, EdgeCostModel) -> Tuple[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = edge_samples.copy()
    need = {"u_node_id", "v_node_id", "edge_distance", "time_s", "energy_J", "slowdown_idx", "mean_speed", "samples"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"edge_samples missing columns: {sorted(miss)}")

    df = df.dropna(subset=["u_node_id", "v_node_id", "time_s", "energy_J"]).copy()
    df["u_node_id"] = df["u_node_id"].astype(int)
    df["v_node_id"] = df["v_node_id"].astype(int)

    df_tr, df_va, df_te = _split_df(df)

    def build_xy(dff: pd.DataFrame):
        u_ids = dff["u_node_id"].to_numpy(np.int64)
        v_ids = dff["v_node_id"].to_numpy(np.int64)

        u_idx = np.array([G.node_id_to_idx[int(u)] for u in u_ids], dtype=np.int64)
        v_idx = np.array([G.node_id_to_idx[int(v)] for v in v_ids], dtype=np.int64)

        dist = dff["edge_distance"].to_numpy(np.float32)
        dist_norm = dist / (dist.max() + 1e-6)

        slow = dff["slowdown_idx"].to_numpy(np.float32)
        ms = dff["mean_speed"].to_numpy(np.float32)

        # mean power (W) = J/s if not present
        if "mean_power_W" in dff.columns:
            mp = dff["mean_power_W"].to_numpy(np.float32)
        else:
            mp = (dff["energy_J"].to_numpy(np.float32) / np.maximum(dff["time_s"].to_numpy(np.float32), 1e-6)).astype(np.float32)

        mp_norm = mp / (mp.max() + 1e-6)

        samples = dff["samples"].to_numpy(np.float32)
        samples_norm = samples / (samples.max() + 1e-6)

        edge_feat = np.stack([dist_norm, slow, ms, mp_norm, samples_norm], axis=1).astype(np.float32)

        y_time = dff["time_s"].to_numpy(np.float32)
        y_energy = dff["energy_J"].to_numpy(np.float32)

        return u_idx, v_idx, edge_feat, y_time, y_energy

    u_tr, v_tr, ef_tr, yt_tr, ye_tr = build_xy(df_tr)
    u_va, v_va, ef_va, yt_va, ye_va = build_xy(df_va)
    u_te, v_te, ef_te, yt_te, ye_te = build_xy(df_te)

    # scales (train only)
    time_scale = float(np.median(yt_tr) + 1e-6)
    energy_scale = float(np.median(ye_tr) + 1e-6)

    def pack(u_idx, v_idx, ef, yt, ye):
        y = np.stack([yt / time_scale, ye / energy_scale], axis=1).astype(np.float32)
        return (
            torch.tensor(u_idx, dtype=torch.long),
            torch.tensor(v_idx, dtype=torch.long),
            torch.tensor(ef, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    uTr, vTr, eTr, yTr = pack(u_tr, v_tr, ef_tr, yt_tr, ye_tr)
    uVa, vVa, eVa, yVa = pack(u_va, v_va, ef_va, yt_va, ye_va)
    uTe, vTe, eTe, yTe = pack(u_te, v_te, ef_te, yt_te, ye_te)

    device = torch.device(C.DEVICE)

    model = EdgeCostModel(
        num_nodes=len(G.node_ids),
        node_feat_dim=int(G.node_feat.shape[1]),
        edge_feat_dim=int(eTr.shape[1]),
        hidden_dim=int(C.EDGE_HIDDEN),
        gnn_layers=int(C.EDGE_GNN_LAYERS),
        backbone=str(C.EDGE_BACKBONE),
        use_edge_weight=True,
        dropout=float(C.EDGE_DROPOUT),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(C.EDGE_LR), weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    node_feat = torch.tensor(G.node_feat, dtype=torch.float32, device=device)
    edge_index = torch.tensor(G.edge_index, dtype=torch.long, device=device)
    edge_w = torch.tensor(G.edge_distance, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_path = out_dir / "edge_cost_model.pt"

    for ep in range(1, int(C.EDGE_EPOCHS) + 1):
        model.train()
        pred = model(node_feat, edge_index, edge_w, uTr.to(device), vTr.to(device), eTr.to(device))
        loss = loss_fn(pred, yTr.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss = float(loss.detach().cpu().item())

        model.eval()
        with torch.no_grad():
            pv = model(node_feat, edge_index, edge_w, uVa.to(device), vVa.to(device), eVa.to(device))
            vloss = loss_fn(pv, yVa.to(device))
            val_loss = float(vloss.detach().cpu().item())

        if ep == 1 or ep % 10 == 0 or ep == int(C.EDGE_EPOCHS):
            print(f"[edge_train] ep={ep:03d}/{int(C.EDGE_EPOCHS)} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "time_scale": time_scale,
                "energy_scale": energy_scale,
                "edge_feat_dim": int(eTr.shape[1]),
                "node_feat_dim": int(G.node_feat.shape[1]),
                "hidden_dim": int(C.EDGE_HIDDEN),
                "gnn_layers": int(C.EDGE_GNN_LAYERS),
                "backbone": str(C.EDGE_BACKBONE),
            }, best_path)

    # test metrics
    model.eval()
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    with torch.no_grad():
        pt = model(node_feat, edge_index, edge_w, uTe.to(device), vTe.to(device), eTe.to(device)).cpu().numpy()

    y_true = yTe.cpu().numpy()
    t_pred = pt[:, 0] * time_scale
    e_pred = pt[:, 1] * energy_scale
    t_true = y_true[:, 0] * time_scale
    e_true = y_true[:, 1] * energy_scale

    def reg_metrics(yhat, y):
        yhat = np.asarray(yhat).reshape(-1)
        y = np.asarray(y).reshape(-1)
        mae = float(np.mean(np.abs(yhat - y)))
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return mae, rmse, r2

    t_mae, t_rmse, t_r2 = reg_metrics(t_pred, t_true)
    e_mae, e_rmse, e_r2 = reg_metrics(e_pred, e_true)

    print(f"[edge_test] TIME   MAE={t_mae:.4f} RMSE={t_rmse:.4f} R2={t_r2:.4f}")
    print(f"[edge_test] ENERGY MAE={e_mae:.4f} RMSE={e_rmse:.4f} R2={e_r2:.4f}")

    return str(best_path), df


def predict_all_edge_costs(G: GraphData, model_path: str, edge_samples_agg: pd.DataFrame, out_dir: Path, EdgeCostModel) -> str:
    out_dir = Path(out_dir)
    device = torch.device(C.DEVICE)

    ckpt = torch.load(model_path, map_location=device)

    model = EdgeCostModel(
        num_nodes=len(G.node_ids),
        node_feat_dim=int(ckpt.get("node_feat_dim", G.node_feat.shape[1])),
        edge_feat_dim=int(ckpt.get("edge_feat_dim", 5)),
        hidden_dim=int(ckpt.get("hidden_dim", C.EDGE_HIDDEN)),
        gnn_layers=int(ckpt.get("gnn_layers", C.EDGE_GNN_LAYERS)),
        backbone=str(ckpt.get("backbone", C.EDGE_BACKBONE)),
        use_edge_weight=True,
        dropout=float(C.EDGE_DROPOUT),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    time_scale = float(ckpt["time_scale"])
    energy_scale = float(ckpt["energy_scale"])

    # build edge_feat for ALL graph edges using fallback stats from edge_samples_agg
    dist = G.edge_distance.astype(np.float32)
    dist_norm = dist / (dist.max() + 1e-6)

    slow_map = {}
    ms_map = {}
    mp_map = {}

    if edge_samples_agg is not None and len(edge_samples_agg) > 0:
        for r in edge_samples_agg.itertuples(index=False):
            slow_map[(int(r.u_node_id), int(r.v_node_id))] = float(getattr(r, "slowdown_idx", 0.0))
            ms_map[(int(r.u_node_id), int(r.v_node_id))] = float(getattr(r, "mean_speed", 0.0))
            if hasattr(r, "mean_power_W"):
                mp_map[(int(r.u_node_id), int(r.v_node_id))] = float(getattr(r, "mean_power_W", 0.0))
            else:
                mp_map[(int(r.u_node_id), int(r.v_node_id))] = float(getattr(r, "energy_J", 0.0)) / max(float(getattr(r, "time_s", 1.0)), 1e-6)

    slow_global = float(np.mean(list(slow_map.values()))) if slow_map else 0.0
    ms_global = float(np.median(list(ms_map.values()))) if ms_map else float(C.DEFAULT_FALLBACK_SPEED_MPS)
    mp_global = float(np.median(list(mp_map.values()))) if mp_map else float(max(C.DEFAULT_FALLBACK_ENERGY_J_PER_M * ms_global, 1.0))

    # samples_norm can be set constant=1 for full graph
    samples_norm = np.ones_like(dist_norm, dtype=np.float32)

    # edge_feat dim=5 same as training
    slow_all = np.array([slow_map.get((int(u), int(v)), slow_global) for u, v in zip(G.edge_u_ids, G.edge_v_ids)], dtype=np.float32)
    ms_all = np.array([ms_map.get((int(u), int(v)), ms_global) for u, v in zip(G.edge_u_ids, G.edge_v_ids)], dtype=np.float32)
    mp_all = np.array([mp_map.get((int(u), int(v)), mp_global) for u, v in zip(G.edge_u_ids, G.edge_v_ids)], dtype=np.float32)
    mp_all = mp_all / (mp_all.max() + 1e-6)

    edge_feat_all = np.stack([dist_norm, slow_all, ms_all, mp_all, samples_norm], axis=1).astype(np.float32)

    edge_u = np.array([G.node_id_to_idx[int(u)] for u in G.edge_u_ids], dtype=np.int64)
    edge_v = np.array([G.node_id_to_idx[int(v)] for v in G.edge_v_ids], dtype=np.int64)

    node_feat = torch.tensor(G.node_feat, dtype=torch.float32, device=device)
    edge_index = torch.tensor(G.edge_index, dtype=torch.long, device=device)
    edge_w = torch.tensor(G.edge_distance, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(
            node_feat, edge_index, edge_w,
            torch.tensor(edge_u, dtype=torch.long, device=device),
            torch.tensor(edge_v, dtype=torch.long, device=device),
            torch.tensor(edge_feat_all, dtype=torch.float32, device=device),
        ).cpu().numpy()

    pred_time_s = pred[:, 0] * time_scale
    pred_energy_J = pred[:, 1] * energy_scale

    out = pd.DataFrame({
        "u_node_id": G.edge_u_ids.astype(int),
        "v_node_id": G.edge_v_ids.astype(int),
        "pred_time_s": pred_time_s.astype(float),
        "pred_energy_J": pred_energy_J.astype(float),
    })

    out_csv = out_dir / "edge_costs_pred.csv"
    out.to_csv(out_csv, index=False)
    print(f"[saved] edge_costs_pred -> {abs_path(out_csv)}")
    return str(out_csv)