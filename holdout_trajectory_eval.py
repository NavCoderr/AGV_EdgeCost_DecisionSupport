# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

import config as C
from graph_data import load_graph
from model import EdgeCostModel
from edge_cost_train import predict_all_edge_costs
from planner import build_full_cost_maps, build_edge_schedule
from templates import load_geom_templates
from trajectory import simulate_temporal_mlp_closed_loop
from temporal_mlp import load_temporal_model


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "inductive_folder_new_data"
OUT_DIR = DATA_DIR / "holdout_traj_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_1S_GLOBAL = DATA_DIR / "data_1s_global.csv"
NODE_FILE = SCRIPT_DIR / "Node_F3.csv"
EDGE_FILE = SCRIPT_DIR / "Edge_Distances3_.csv"
GEOM_CSV = DATA_DIR / "geom_templates.csv"
TEMP_SUMMARY_CSV = DATA_DIR / "temporal_all_summary.csv"

HOLDOUT_SEED = 11
TRAIN_RATIO = 0.70

def set_all_seeds(seed: int = 11):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def build_mission_ids(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Target reached"] = _to_num(d["Target reached"]).fillna(0).astype(int)
    d["Going to ID"] = _to_num(d["Going to ID"]).fillna(-1).astype(int)

    mission_id = []
    mid = 0
    for i in range(len(d)):
        mission_id.append(mid)
        if int(d.iloc[i]["Target reached"]) == 1:
            mid += 1

    d["mission_id"] = np.asarray(mission_id, dtype=int)
    d["mission_goal"] = d.groupby("mission_id")["Going to ID"].transform(
        lambda s: int(s.dropna().iloc[-1]) if len(s.dropna()) else -1
    )
    return d


def build_graph_edge_df(G) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_node_id": G.edge_u_ids.astype(int),
            "v_node_id": G.edge_v_ids.astype(int),
            "edge_distance": G.edge_distance.astype(float),
        }
    )


def extract_traversals(df: pd.DataFrame, edge_set: set) -> pd.DataFrame:
    d = df.copy()

    tcol = "t_sec_work" if "t_sec_work" in d.columns else "t_sec"

    d[tcol] = _to_num(d[tcol])
    d["node_id"] = _to_num(d["node_id"])
    d["Speed"] = _to_num(d["Speed"]).fillna(0.0)

    if "energy_J_1s" in d.columns:
        d["energy_J_1s"] = _to_num(d["energy_J_1s"]).fillna(0.0)

    elif "step_energy" in d.columns:
        d["energy_J_1s"] = _to_num(d["step_energy"]).fillna(0.0)

    elif "energy_cum_from_start" in d.columns:
        d["energy_J_1s"] = (
            _to_num(d["energy_cum_from_start"])
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )

    elif "Cumulative energy consumption" in d.columns:
        d["energy_J_1s"] = (
            _to_num(d["Cumulative energy consumption"])
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )

    else:
        d["energy_J_1s"] = 0.0

    x_col = "x" if "x" in d.columns else ("X-coordinate" if "X-coordinate" in d.columns else None)
    y_col = "y" if "y" in d.columns else ("Y-coordinate" if "Y-coordinate" in d.columns else None)

    rows = []
    n = len(d)
    i = 0
    trav_id = 0

    while i < n - 2:
        if not np.isfinite(d.iloc[i]["node_id"]):
            i += 1
            continue

        u = int(d.iloc[i]["node_id"])
        mid = int(d.iloc[i]["mission_id"])
        goal = int(d.iloc[i]["mission_goal"])

        j = i
        while (
            j < n
            and np.isfinite(d.iloc[j]["node_id"])
            and int(d.iloc[j]["node_id"]) == u
            and int(d.iloc[j]["mission_id"]) == mid
        ):
            j += 1

        if j >= n or not np.isfinite(d.iloc[j]["node_id"]):
            i = j
            continue

        v = int(d.iloc[j]["node_id"])

        if v == u or int(d.iloc[j]["mission_id"]) != mid or (u, v) not in edge_set:
            i = j
            continue

        k = j
        while (
            k < n
            and np.isfinite(d.iloc[k]["node_id"])
            and int(d.iloc[k]["node_id"]) == v
            and int(d.iloc[k]["mission_id"]) == mid
        ):
            k += 1

        seg = d.iloc[i:k].copy()

        if len(seg) <= 2:
            i = j
            continue

        rec = {
            "trav_id": trav_id,
            "mission_id": mid,
            "mission_goal": goal,
            "route_key": f"m{mid}_g{goal}",
            "u_node_id": u,
            "v_node_id": v,
            "time_s": float(len(seg)),
            "energy_J": float(seg["energy_J_1s"].sum()),
            "mean_speed": float(seg["Speed"].mean()),
            "t_start_idx": int(i),
            "t_end_idx": int(k - 1),
        }

        if x_col and y_col:
            rec["x_start"] = float(_to_num(seg.iloc[0][x_col]))
            rec["y_start"] = float(_to_num(seg.iloc[0][y_col]))
            rec["x_end"] = float(_to_num(seg.iloc[-1][x_col]))
            rec["y_end"] = float(_to_num(seg.iloc[-1][y_col]))

        rows.append(rec)
        trav_id += 1
        i = j

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise ValueError("No traversals extracted")

    return out


def aggregate_train_edges(trav_df: pd.DataFrame, graph_edges: pd.DataFrame) -> pd.DataFrame:
    agg = (
        trav_df.groupby(["u_node_id", "v_node_id"], as_index=False)
        .agg(
            time_s=("time_s", "mean"),
            energy_J=("energy_J", "mean"),
            mean_speed=("mean_speed", "mean"),
            samples=("trav_id", "count"),
        )
    )
    out = agg.merge(graph_edges, on=["u_node_id", "v_node_id"], how="left")
    out["mean_power_W"] = out["energy_J"] / np.maximum(out["time_s"], 1e-6)
    nominal = out["edge_distance"] / np.maximum(out["time_s"], 1e-6)
    slowdown = 1.0 - (nominal / np.maximum(out["mean_speed"], 1e-6))
    out["slowdown_idx"] = np.clip(np.nan_to_num(slowdown, nan=0.0), 0.0, 1.0)
    out["energy_J_per_m"] = out["energy_J"] / np.maximum(out["edge_distance"], 1e-6)
    return out


def fit_model_on_train_edges(G, train_edge_agg: pd.DataFrame):
    df = train_edge_agg.copy()
    df["u_node_id"] = df["u_node_id"].astype(int)
    df["v_node_id"] = df["v_node_id"].astype(int)

    def build_xy(dff: pd.DataFrame):
        u_idx = np.array([G.node_id_to_idx[int(u)] for u in dff["u_node_id"]], dtype=np.int64)
        v_idx = np.array([G.node_id_to_idx[int(v)] for v in dff["v_node_id"]], dtype=np.int64)
        edge_feat = np.stack([
            dff["edge_distance"].to_numpy(np.float32) / (dff["edge_distance"].max() + 1e-6),
            dff["slowdown_idx"].to_numpy(np.float32),
            dff["mean_speed"].to_numpy(np.float32),
            dff["mean_power_W"].to_numpy(np.float32) / (dff["mean_power_W"].max() + 1e-6),
            dff["samples"].to_numpy(np.float32) / (dff["samples"].max() + 1e-6),
        ], axis=1)
        y_time = dff["time_s"].to_numpy(np.float32)
        y_energy = dff["energy_J"].to_numpy(np.float32)
        return u_idx, v_idx, edge_feat, y_time, y_energy

    rng = np.random.RandomState(HOLDOUT_SEED)
    df["edge_id"] = df["u_node_id"].astype(str) + "_" + df["v_node_id"].astype(str)
    uniq = df["edge_id"].unique()
    rng.shuffle(uniq)
    n_tr = max(1, int(len(uniq) * 0.85))
    tr_ids = set(uniq[:n_tr])
    va_ids = set(uniq[n_tr:])

    df_tr = df[df["edge_id"].isin(tr_ids)].reset_index(drop=True)
    df_va = df[df["edge_id"].isin(va_ids)].reset_index(drop=True)
    if len(df_va) == 0:
        df_va = df_tr.sample(min(len(df_tr), max(1, len(df_tr)//5)), random_state=HOLDOUT_SEED).reset_index(drop=True)

    u_tr, v_tr, ef_tr, yt_tr, ye_tr = build_xy(df_tr)
    u_va, v_va, ef_va, yt_va, ye_va = build_xy(df_va)

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
    loss_fn = torch.nn.SmoothL1Loss()

    node_feat = torch.tensor(G.node_feat, dtype=torch.float32, device=device)
    edge_index = torch.tensor(G.edge_index, dtype=torch.long, device=device)
    edge_w = torch.tensor(G.edge_distance, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_path = OUT_DIR / "holdout_edge_cost_model.pt"

    for ep in range(1, int(C.EDGE_EPOCHS) + 1):
        model.train()
        pred = model(node_feat, edge_index, edge_w, uTr.to(device), vTr.to(device), eTr.to(device))
        loss = loss_fn(pred, yTr.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(node_feat, edge_index, edge_w, uVa.to(device), vVa.to(device), eVa.to(device))
            vloss = loss_fn(pv, yVa.to(device))
            val_loss = float(vloss.detach().cpu().item())

        if ep == 1 or ep % 10 == 0 or ep == int(C.EDGE_EPOCHS):
            print(f"[holdout_edge_train] ep={ep:03d}/{int(C.EDGE_EPOCHS)} train={float(loss.item()):.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "time_scale": time_scale,
                    "energy_scale": energy_scale,
                    "edge_feat_dim": int(eTr.shape[1]),
                    "node_feat_dim": int(G.node_feat.shape[1]),
                    "hidden_dim": int(C.EDGE_HIDDEN),
                    "gnn_layers": int(C.EDGE_GNN_LAYERS),
                    "backbone": str(C.EDGE_BACKBONE),
                },
                best_path,
            )

    pred_csv = predict_all_edge_costs(G, str(best_path), train_edge_agg, OUT_DIR, EdgeCostModel)
    return pd.read_csv(pred_csv)


def load_best_temporal():
    summ = pd.read_csv(TEMP_SUMMARY_CSV)
    summ = summ.sort_values("MAE", ascending=True)
    kind = str(summ.iloc[0]["kind"]).strip().lower()
    model_path = DATA_DIR / f"temporal_{kind}.pt"
    tm, meta = load_temporal_model(str(model_path))
    return kind, tm, meta


def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def get_real_mission_segment(df_full: pd.DataFrame, mission_id: int):
    seg = df_full[df_full["mission_id"] == mission_id].copy()
    x_col = "x" if "x" in seg.columns else ("X-coordinate" if "X-coordinate" in seg.columns else None)
    y_col = "y" if "y" in seg.columns else ("Y-coordinate" if "Y-coordinate" in seg.columns else None)
    t_col = "t_sec_work" if "t_sec_work" in seg.columns else "t_sec"
    out = pd.DataFrame({
        "t": _to_num(seg[t_col]).fillna(0.0),
        "x_real": _to_num(seg[x_col]),
        "y_real": _to_num(seg[y_col]),
    }).dropna()
    if len(out) == 0:
        return out
    out["t"] = np.round(out["t"] - float(out["t"].iloc[0])).astype(int)
    out = out.groupby("t", as_index=False).first()
    return out


def plan_for_mission(G, pred_df, edge_samples, start_node, goal_node, real_seg, tm, meta):
    # temporarily override config
    old_start, old_goal = C.START_NODE, C.GOAL_NODE
    C.START_NODE, C.GOAL_NODE = int(start_node), int(goal_node)

    try:
        time_cost, energy_cost, combo_cost, slowdown_map = build_full_cost_maps(G, pred_df, edge_samples)
        main_map = {"time": time_cost, "energy": energy_cost, "combo": combo_cost}[C.MAIN_COST]

        from planner import astar_route_safe, dijkstra_route
        if str(C.PLANNER).lower() == "dijkstra":
            route_nodes, route_cost = dijkstra_route(G, int(C.START_NODE), int(C.GOAL_NODE), main_map, C.BLOCKED_NODES, C.BLOCKED_EDGES)
        else:
            route_nodes, route_cost = astar_route_safe(G, int(C.START_NODE), int(C.GOAL_NODE), main_map, C.BLOCKED_NODES, C.BLOCKED_EDGES, w=float(C.ASTAR_W))

        sched = pd.DataFrame(build_edge_schedule(G, route_nodes, time_cost, energy_cost, slowdown_map))
        geom_templates = load_geom_templates(G, GEOM_CSV)

        tmp_real = OUT_DIR / "_tmp_real.csv"
        tmp_pred = OUT_DIR / "_tmp_pred.csv"
        real_for_rollout = real_seg.copy()
        real_for_rollout["x"] = real_for_rollout["x_real"]
        real_for_rollout["y"] = real_for_rollout["y_real"]
        real_for_rollout["t_sec_work"] = real_for_rollout["t"]
        real_for_rollout.to_csv(tmp_real, index=False)

        simulate_temporal_mlp_closed_loop(G, sched, geom_templates, tm, meta, real_1hz_csv=tmp_real, out_csv=tmp_pred)

        pred = pd.read_csv(tmp_pred)
        pred = pred[["t_global_s", "x", "y"]].copy()
        pred["t"] = np.round(pred["t_global_s"]).astype(int)
        pred = pred.groupby("t", as_index=False).first().rename(columns={"x": "x_pred", "y": "y_pred"})

        m = real_seg.merge(pred[["t", "x_pred", "y_pred"]], on="t", how="inner")
        if len(m) >= 2:
            xy = np.sqrt((m["x_pred"] - m["x_real"]) ** 2 + (m["y_pred"] - m["y_real"]) ** 2)
            xy_mae = float(np.mean(np.abs(xy)))
            xy_rmse = float(np.sqrt(np.mean(xy ** 2)))
        else:
            xy_mae, xy_rmse = np.nan, np.nan

        return {
            "route_nodes": " -> ".join(map(str, route_nodes)),
            "pred_time_s": float(pd.to_numeric(sched["t_pred_s"], errors="coerce").sum()),
            "pred_energy_J": float(pd.to_numeric(sched["e_pred_J"], errors="coerce").sum()),
            "xy_mae_m": xy_mae,
            "xy_rmse_m": xy_rmse,
        }
    finally:
        C.START_NODE, C.GOAL_NODE = old_start, old_goal


def main():
    set_all_seeds(HOLDOUT_SEED)
    print("[load] graph")
    G = load_graph(NODE_FILE, EDGE_FILE)
    graph_edges = build_graph_edge_df(G)

    print("[load] data_1s_global")
    df = pd.read_csv(DATA_1S_GLOBAL)
    df = build_mission_ids(df)

    print("[extract] traversals")
    trav_df = extract_traversals(df, G.edge_set)

    rng = np.random.RandomState(HOLDOUT_SEED)
    missions = np.array(sorted(trav_df["mission_id"].unique()))
    rng.shuffle(missions)

    n = len(missions)
    n_tr = max(1, int(n * TRAIN_RATIO))
    tr_ids = set(missions[:n_tr])
    te_ids = set(missions[n_tr:])

    train_trav = trav_df[trav_df["mission_id"].isin(tr_ids)].reset_index(drop=True)
    test_trav = trav_df[trav_df["mission_id"].isin(te_ids)].reset_index(drop=True)

    print(f"[holdout_split] train_missions={len(tr_ids)} test_missions={len(te_ids)}")
    print(f"[holdout_split] train_traversals={len(train_trav)} test_traversals={len(test_trav)}")

    train_edge_agg = aggregate_train_edges(train_trav, graph_edges)
    pred_df = fit_model_on_train_edges(G, train_edge_agg)

    print("[debug] pred_df columns:", list(pred_df.columns))

    time_pred_col = None
    energy_pred_col = None

    for c in ["t_pred_s", "pred_time_s", "time_pred_s", "time_s_pred", "time_s"]:
        if c in pred_df.columns:
            time_pred_col = c
            break

    for c in ["e_pred_J", "pred_energy_J", "energy_pred_J", "energy_J_pred", "energy_J"]:
        if c in pred_df.columns:
            energy_pred_col = c
            break

    if time_pred_col is None or energy_pred_col is None:
        raise KeyError(
            f"Prediction columns not found. Available columns are: {list(pred_df.columns)}"
        )

    pred_for_eval = pred_df[
        ["u_node_id", "v_node_id", time_pred_col, energy_pred_col]
    ].copy()

    pred_for_eval = pred_for_eval.rename(
        columns={
            time_pred_col: "t_pred_s",
            energy_pred_col: "e_pred_J",
        }
    )

    test_eval = test_trav.merge(
        pred_for_eval,
        on=["u_node_id", "v_node_id"],
        how="left",
    )

    time_m = reg_metrics(test_eval["time_s"], test_eval["t_pred_s"])
    energy_m = reg_metrics(test_eval["energy_J"], test_eval["e_pred_J"])

    edge_metrics = pd.DataFrame(
        [
            {
                "target": "Traversal time",
                "MAE": time_m["MAE"],
                "RMSE": time_m["RMSE"],
                "R2": time_m["R2"],
            },
            {
                "target": "Energy",
                "MAE": energy_m["MAE"],
                "RMSE": energy_m["RMSE"],
                "R2": energy_m["R2"],
            },
        ]
    )

    edge_metrics.to_csv(OUT_DIR / "holdout_edge_metrics.csv", index=False)
    print("[saved]", OUT_DIR / "holdout_edge_metrics.csv")
    print(edge_metrics)

    kind, tm, meta = load_best_temporal()
    print(f"[best_temporal] {kind}")

    per_mission = []
    for mid in sorted(test_trav["mission_id"].unique()):
        g = test_trav[test_trav["mission_id"] == mid].copy()
        start_node = int(g.iloc[0]["u_node_id"])
        goal_node = int(g.iloc[-1]["v_node_id"])
        real_seg = get_real_mission_segment(df, mid)
        if len(real_seg) == 0:
            continue

        res = plan_for_mission(G, pred_df, train_edge_agg, start_node, goal_node, real_seg, tm, meta)
        res["mission_id"] = int(mid)
        res["mission_goal"] = int(g.iloc[0]["mission_goal"])
        res["true_time_s"] = float(g["time_s"].sum())
        res["true_energy_J"] = float(g["energy_J"].sum())
        res["time_err_s"] = res["pred_time_s"] - res["true_time_s"]
        res["energy_err_J"] = res["pred_energy_J"] - res["true_energy_J"]
        per_mission.append(res)

    mission_df = pd.DataFrame(per_mission)
    mission_df.to_csv(OUT_DIR / "holdout_mission_traj_metrics.csv", index=False)

    summary = {
        "n_test_missions": int(len(mission_df)),
        "xy_mae_mean_m": float(mission_df["xy_mae_m"].mean()) if len(mission_df) else np.nan,
        "xy_rmse_mean_m": float(mission_df["xy_rmse_m"].mean()) if len(mission_df) else np.nan,
        "route_time_err_mean_s": float(mission_df["time_err_s"].mean()) if len(mission_df) else np.nan,
        "route_energy_err_mean_J": float(mission_df["energy_err_J"].mean()) if len(mission_df) else np.nan,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "holdout_traj_summary.csv", index=False)

    print("[saved]", OUT_DIR / "holdout_mission_traj_metrics.csv")
    print("[saved]", OUT_DIR / "holdout_traj_summary.csv")


if __name__ == "__main__":
    main()
