# -*- coding: utf-8 -*-
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import config as C
from utils_io import abs_path
from graph_data import GraphData


def add_speed_from_xy(df: pd.DataFrame, x_col="x", y_col="y", dt_sec=1.0) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        return df
    dx = pd.to_numeric(df[x_col], errors="coerce").diff()
    dy = pd.to_numeric(df[y_col], errors="coerce").diff()
    step_dist = np.sqrt(dx * dx + dy * dy).fillna(0.0)
    speed = (step_dist / float(dt_sec)).astype(float)
    accel = speed.diff().fillna(0.0)
    df["step_dist_m"] = step_dist
    df["Speed"] = speed
    df["accel_mps2"] = accel
    return df


def _read_table_auto(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # try utf-8 then latin-1
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

# Helpers
def _pick_col(df: pd.DataFrame, *cands: str):
    cols = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in cands:
        k = str(cand).strip().lower().replace(" ", "").replace("_", "")
        if k in cols:
            return cols[k]
    return None


def _polyline_cumlen(pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) < 2:
        return np.array([0.0], dtype=np.float32)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg).astype(np.float32)])


def _interp_xy_on_polyline_frac(pts: np.ndarray, frac: float) -> Tuple[float, float]:
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


def _turn_rate_norm_from_polyline(pts: np.ndarray) -> float:
    # mean abs turn angle between segments, normalized by pi => [0..1]
    if pts is None or len(pts) < 3:
        return 0.0
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    ang = np.arctan2(dy, dx)
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    val = float(np.mean(np.abs(d))) if len(d) else 0.0
    return float(np.clip(val / np.pi, 0.0, 1.0))


def _get_pts_for_edge(
    G: GraphData, geom_templates: Dict[Tuple[int, int], np.ndarray], u: int, v: int
) -> np.ndarray:
    pts = geom_templates.get((u, v), None)
    if pts is None:
        ui = G.node_id_to_idx[u]
        vi = G.node_id_to_idx[v]
        pts = np.array(
            [
                [float(G.node_x[ui]), float(G.node_y[ui])],
                [float(G.node_x[vi]), float(G.node_y[vi])],
            ],
            dtype=np.float32,
        )
    return pts


def _hard_speed_cap_xy(rows, x, y, slow):
    V_MAX = float(getattr(C, "V_MAX_MPS", 0.30))
    V_TURN = float(getattr(C, "V_TURN_MPS", 0.12))
    GAIN = float(getattr(C, "SLOWDOWN_GAIN", 0.70))
    slow01 = max(0.0, min(1.0, float(slow)))
    v_cap = max(V_TURN, V_MAX * (1.0 - GAIN * slow01))

    if rows:
        x_prev = float(rows[-1]["x"])
        y_prev = float(rows[-1]["y"])
        step = float(((x - x_prev) ** 2 + (y - y_prev) ** 2) ** 0.5)
        max_step = v_cap * 1.0  # DT=1s
        if step > max_step and step > 1e-9:
            sc = max_step / step
            x = x_prev + (x - x_prev) * sc
            y = y_prev + (y - y_prev) * sc
    return x, y


def _delta_stability_clamp(deltas: np.ndarray, dt_s: int) -> np.ndarray:
    dt_s = max(int(dt_s), 1)
    uni = 1.0 / float(dt_s)
    min_r = float(getattr(C, "DELTA_FRAC_MIN_RATIO", 0.20))
    max_r = float(getattr(C, "DELTA_FRAC_MAX_RATIO", 4.00))
    dmin = max(0.0, min_r * uni)
    dmax = max(dmin + 1e-9, max_r * uni)
    return np.clip(deltas.astype(np.float32), dmin, dmax).astype(np.float32)

# TAU-WARP
def simulate_tau_warp(
    G: GraphData,
    sched_df: pd.DataFrame,
    geom_templates: Dict[Tuple[int, int], np.ndarray],
    tau_templates: Dict[Tuple[int, int], np.ndarray],
    tau_fallback: np.ndarray,
    out_csv,
):
    out_csv = Path(out_csv)
    rows = []
    t_global = 0
    e_cum = 0.0

    for ei, r in enumerate(sched_df.itertuples(index=False)):
        u = int(r.u)
        v = int(r.v)

        dt_s = int(getattr(r, "dt_s", 0))
        dt_s = max(dt_s, 1)

        dist = float(getattr(r, "edge_distance", 0.0))
        slow = float(getattr(r, "slowdown_idx", 0.0))
        t_edge = float(getattr(r, "t_pred_s", float(dt_s)))
        e_edge = float(getattr(r, "e_pred_J", 0.0))

        pts = _get_pts_for_edge(G, geom_templates, u, v)

        curve = tau_templates.get((u, v), None)
        if curve is None or len(curve) < 2:
            curve = tau_fallback

        M = len(curve)

        for s in range(dt_s):
            tau = (s + 1) / float(dt_s)
            idx = int(round(tau * (M - 1)))
            idx = max(0, min(idx, M - 1))
            frac = float(curve[idx])

            if s == dt_s - 1:
                frac = 1.0

            x, y = _interp_xy_on_polyline_frac(pts, frac)
            x, y = _hard_speed_cap_xy(rows, x, y, slow)

            e_per_s = (e_edge / float(dt_s))
            e_cum += e_per_s

            rows.append(
                {
                    "t_global_s": int(t_global),
                    "edge_idx": int(ei),
                    "u": int(u),
                    "v": int(v),
                    "tau": float(tau),
                    "frac": float(frac),
                    "x": float(x),
                    "y": float(y),
                    "edge_distance": float(dist),
                    "slowdown_idx": float(slow),
                    "t_pred_s": float(t_edge),
                    "e_pred_J": float(e_edge),
                    "e_cum_J": float(e_cum),
                }
            )
            t_global += 1

    df_out = pd.DataFrame(rows)
    df_out = add_speed_from_xy(df_out, x_col="x", y_col="y", dt_sec=1.0)
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] traj_tau_warp -> {abs_path(out_csv)} steps={len(rows)}")

# Temporal-MLP / GRU / LSTM / Transformer / physics_delta
def _build_feature_vector(
    tau: float,
    dist: float,
    slow: float,
    t_pred_s: float,
    e_pred_J: float,
    meta: dict,
    turn_rate_norm: float,
    ref_speed: float = 0.0,
    ref_power: float = 0.0,
    extras: dict = None,
) -> np.ndarray:
    ex = extras or {}

    use_tau = bool(meta.get("use_tau_feature", getattr(C, "USE_TAU_FEATURE", True)))
    use_lookahead = bool(meta.get("use_lookahead", getattr(C, "USE_LOOKAHEAD", False)))
    base_dim = int(meta.get("temp_base_dim", getattr(C, "TEMP_BASE_DIM", 9)))
    dim = int(meta.get("temp_in_dim", getattr(C, "TEMP_IN_DIM", base_dim)))

    if base_dim not in (7, 9, 13, 17):
        raise ValueError(f"_build_feature_vector: TEMP_BASE_DIM must be 7/9/13/17, got {base_dim}")

    base_no_tau = {7: 6, 9: 7, 13: 11, 17: 15}[base_dim]
    expected_dim = (base_dim if use_tau else base_no_tau) + (2 if use_lookahead else 0)
    if dim != expected_dim:
        raise ValueError(
            f"_build_feature_vector: TEMP_IN_DIM mismatch. Expected {expected_dim} "
            f"(USE_TAU_FEATURE={use_tau}, USE_LOOKAHEAD={use_lookahead}, TEMP_BASE_DIM={base_dim}) "
            f"but got dim={dim}"
        )

    dist_max = float(meta.get("dist_max", max(dist, 1.0)))
    time_scale = float(meta.get("time_scale", 10.0))
    energy_scale = float(meta.get("energy_scale", 1000.0))
    speed_scale = float(meta.get("speed_scale", max(getattr(C, "DEFAULT_FALLBACK_SPEED_MPS", 0.2), 1e-6)))
    power_scale = float(meta.get("power_scale", 100.0))

    dist_norm = float(dist) / max(dist_max, 1e-6)
    rem_dist_norm = 1.0 - float(tau)
    t_norm = float(t_pred_s) / max(time_scale, 1e-6)
    e_norm = float(e_pred_J) / max(energy_scale, 1e-6)

    sp_norm = float(ref_speed) / max(speed_scale, 1e-6)
    pw_norm = float(ref_power) / max(power_scale, 1e-6)

    mean_speed = float(dist) / max(float(t_pred_s), 1e-6)
    mean_power = float(e_pred_J) / max(float(t_pred_s), 1e-6)
    mean_sp_norm = float(mean_speed) / max(speed_scale, 1e-6)
    mean_pw_norm = float(mean_power) / max(power_scale, 1e-6)

    accel_scale = float(meta.get("accel_scale", 0.5))
    jerk_scale  = float(meta.get("jerk_scale", 1.0))
    yaw_scale   = float(meta.get("yaw_scale", 1.0))
    wheel_diff_scale = float(meta.get("wheel_diff_scale", 100.0))
    conf_scale  = float(meta.get("conf_scale", 100.0))
    batt_scale  = float(meta.get("batt_scale", 100.0))

    accel_norm = float(ex.get("accel", 0.0)) / max(accel_scale, 1e-6)
    jerk_norm  = float(ex.get("jerk", 0.0)) / max(jerk_scale, 1e-6)
    yaw_norm   = float(ex.get("yaw_rate", 0.0)) / max(yaw_scale, 1e-6)
    wdiff_norm = float(ex.get("wheel_diff", 0.0)) / max(wheel_diff_scale, 1e-6)
    conf_norm  = float(ex.get("pos_conf", 0.0)) / max(conf_scale, 1e-6)
    batt_norm  = float(ex.get("battery", 0.0)) / max(batt_scale, 1e-6)
    idle_flag  = float(ex.get("idle_flag", 0.0))
    tr_flag    = float(ex.get("target_reached", 0.0))

    next_turn = float(ex.get("next_turn_angle_norm", 0.0))
    next_tr = float(ex.get("next_edge_turn_rate_norm", 0.0))

    if use_tau:
        if base_dim == 17:
            feat = [
                float(tau), float(dist_norm), float(slow), float(rem_dist_norm),
                float(t_norm), float(e_norm),
                float(sp_norm), float(pw_norm), float(turn_rate_norm),
                float(accel_norm), float(jerk_norm), float(yaw_norm),
                float(wdiff_norm), float(conf_norm), float(batt_norm),
                float(idle_flag), float(tr_flag),
            ]
        elif base_dim == 13:
            feat = [
                float(tau), float(dist_norm), float(slow), float(rem_dist_norm),
                float(t_norm), float(e_norm),
                float(sp_norm), float(pw_norm), float(turn_rate_norm),
                float(accel_norm), float(yaw_norm), float(wdiff_norm), float(conf_norm),
            ]
        elif base_dim == 9:
            feat = [
                float(tau), float(dist_norm), float(slow), float(rem_dist_norm),
                float(t_norm), float(e_norm),
                float(mean_sp_norm), float(mean_pw_norm), float(turn_rate_norm),
            ]
        else:  # 7
            feat = [
                float(tau), float(dist_norm), float(slow),
                float(t_norm), float(e_norm),
                float(mean_sp_norm), float(mean_pw_norm),
            ]
    else:
        if base_dim == 17:
            feat = [
                float(dist_norm), float(slow),
                float(t_norm), float(e_norm),
                float(sp_norm), float(pw_norm), float(turn_rate_norm),
                float(accel_norm), float(jerk_norm), float(yaw_norm),
                float(wdiff_norm), float(conf_norm), float(batt_norm),
                float(idle_flag), float(tr_flag),
            ]
        elif base_dim == 13:
            feat = [
                float(dist_norm), float(slow),
                float(t_norm), float(e_norm),
                float(sp_norm), float(pw_norm), float(turn_rate_norm),
                float(accel_norm), float(yaw_norm), float(wdiff_norm), float(conf_norm),
            ]
        elif base_dim == 9:
            feat = [
                float(dist_norm), float(slow),
                float(t_norm), float(e_norm),
                float(mean_sp_norm), float(mean_pw_norm), float(turn_rate_norm),
            ]
        else:  # 7
            feat = [
                float(dist_norm), float(slow),
                float(t_norm), float(e_norm),
                float(mean_sp_norm), float(mean_pw_norm),
            ]

    if use_lookahead:
        feat += [float(next_turn), float(next_tr)]

    if len(feat) != dim:
        raise ValueError(f"_build_feature_vector: len(feat)={len(feat)} but dim={dim}")

    return np.array(feat, dtype=np.float32)


def simulate_temporal_mlp(
    G: GraphData,
    sched_df: pd.DataFrame,
    geom_templates: Dict[Tuple[int, int], np.ndarray],
    tm,
    meta: dict,
    out_csv,
):
    out_csv = Path(out_csv)
    device = torch.device(C.DEVICE)
    tm.eval()

    kind = str(meta.get("kind", "mlp")).lower().strip()
    seq_len = int(meta.get("seq_len", 1))
    use_lookahead = bool(meta.get("use_lookahead", getattr(C, "USE_LOOKAHEAD", False)))

    x_buf = []
    rows = []
    t_global = 0
    e_cum = 0.0

    sched = list(sched_df.itertuples(index=False))

    def _angle_norm(u_xy, v_xy, w_xy) -> float:
        ux, uy = u_xy
        vx, vy = v_xy
        wx, wy = w_xy
        a = np.array([vx - ux, vy - uy], dtype=float)
        b = np.array([wx - vx, wy - vy], dtype=float)
        na = float(np.linalg.norm(a) + 1e-9)
        nb = float(np.linalg.norm(b) + 1e-9)
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return float(ang / np.pi)

    def _lookahead_for_edge(edge_idx: int) -> dict:
        if not use_lookahead:
            return {}
        if edge_idx < 0 or edge_idx >= len(sched):
            return {"next_turn_angle_norm": 0.0, "next_edge_turn_rate_norm": 0.0}

        u = int(getattr(sched[edge_idx], "u"))
        v = int(getattr(sched[edge_idx], "v"))

        if edge_idx + 1 >= len(sched):
            return {"next_turn_angle_norm": 0.0, "next_edge_turn_rate_norm": 0.0}

        v2 = int(getattr(sched[edge_idx + 1], "u"))
        w = int(getattr(sched[edge_idx + 1], "v"))
        if v2 != v:
            return {"next_turn_angle_norm": 0.0, "next_edge_turn_rate_norm": 0.0}

        ui = G.node_id_to_idx.get(u, None)
        vi = G.node_id_to_idx.get(v, None)
        wi = G.node_id_to_idx.get(w, None)
        next_turn = 0.0
        if ui is not None and vi is not None and wi is not None:
            u_xy = (float(G.node_x[ui]), float(G.node_y[ui]))
            v_xy = (float(G.node_x[vi]), float(G.node_y[vi]))
            w_xy = (float(G.node_x[wi]), float(G.node_y[wi]))
            next_turn = _angle_norm(u_xy, v_xy, w_xy)

        pts_next = geom_templates.get((v, w), None)
        next_tr = float(_turn_rate_norm_from_polyline(pts_next)) if pts_next is not None else 0.0
        return {"next_turn_angle_norm": float(next_turn), "next_edge_turn_rate_norm": float(next_tr)}

    def _model_forward(x_in_vec: np.ndarray) -> float:
        nonlocal x_buf

        if kind in ("gru", "lstm", "transformer_lite"):
            x_buf.append(x_in_vec.astype(np.float32))
            if len(x_buf) > seq_len:
                x_buf = x_buf[-seq_len:]
            x_arr = np.stack(x_buf, axis=0)
            if x_arr.shape[0] < seq_len:
                pad = np.zeros((seq_len - x_arr.shape[0], x_arr.shape[1]), dtype=np.float32)
                x_arr = np.concatenate([pad, x_arr], axis=0)
            xin_t = torch.tensor(x_arr[None, :, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                y = tm(xin_t)
            return float(y.detach().cpu().numpy().reshape(-1)[0])

        xin_t = torch.tensor(x_in_vec[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            y = tm(xin_t)
        return float(y.detach().cpu().numpy().reshape(-1)[0])

    for ei, r in enumerate(sched):
        u = int(r.u)
        v = int(r.v)

        dt_s = int(getattr(r, "dt_s", 0))
        dt_s = max(dt_s, 1)

        dist = float(getattr(r, "edge_distance", 0.0))
        slow = float(getattr(r, "slowdown_idx", 0.0))
        t_pred_s = float(getattr(r, "t_pred_s", float(dt_s)))
        e_pred_J = float(getattr(r, "e_pred_J", 0.0))

        pts = _get_pts_for_edge(G, geom_templates, u, v)
        turn_rate_norm = _turn_rate_norm_from_polyline(pts)
        look_ex = _lookahead_for_edge(ei)

        deltas = np.zeros(dt_s, dtype=np.float32)
        taus = np.zeros(dt_s, dtype=np.float32)

        for s in range(dt_s):
            tau = (s + 1) / float(dt_s)
            taus[s] = tau

            x_in = _build_feature_vector(
                tau, dist, slow, t_pred_s, e_pred_J, meta, turn_rate_norm,
                ref_speed=0.0, ref_power=0.0,
                extras=look_ex if use_lookahead else None,
            ).astype(np.float32)

            d_raw = _model_forward(x_in)
            deltas[s] = max(0.0, float(d_raw))

        deltas = _delta_stability_clamp(deltas, dt_s)

        ssum = float(deltas.sum())
        if (not np.isfinite(ssum)) or ssum <= 1e-9:
            deltas[:] = 1.0 / float(dt_s)
        else:
            deltas[:] = deltas / ssum

        frac = 0.0
        for s in range(dt_s):
            delta = float(np.clip(deltas[s], 0.0, 1.0))
            frac = float(np.clip(frac + delta, 0.0, 1.0))
            if s == dt_s - 1:
                frac = 1.0

            x, y = _interp_xy_on_polyline_frac(pts, frac)
            x, y = _hard_speed_cap_xy(rows, x, y, slow)

            e_per_s = (e_pred_J / float(dt_s))
            e_cum += e_per_s

            rows.append(
                {
                    "t_global_s": int(t_global),
                    "edge_idx": int(ei),
                    "u": int(u),
                    "v": int(v),
                    "tau": float(taus[s]),
                    "delta_frac": float(delta),
                    "frac": float(frac),
                    "x": float(x),
                    "y": float(y),
                    "edge_distance": float(dist),
                    "slowdown_idx": float(slow),
                    "t_pred_s": float(t_pred_s),
                    "e_pred_J": float(e_pred_J),
                    "e_cum_J": float(e_cum),
                }
            )
            t_global += 1

    df_out = pd.DataFrame(rows)
    df_out = add_speed_from_xy(df_out, x_col="x", y_col="y", dt_sec=1.0)
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] traj_temporal_model(open_loop) -> {abs_path(out_csv)} steps={len(rows)} kind={kind}")


def simulate_temporal_mlp_closed_loop(
    G: GraphData,
    sched_df: pd.DataFrame,
    geom_templates: Dict[Tuple[int, int], np.ndarray],
    tm,
    meta: dict,
    real_1hz_csv,
    out_csv,
):
    out_csv = Path(out_csv)
    device = torch.device(C.DEVICE)
    tm.eval()

    kind = str(meta.get("kind", "mlp")).lower().strip()
    seq_len = int(meta.get("seq_len", 1))
    use_lookahead = bool(meta.get("use_lookahead", getattr(C, "USE_LOOKAHEAD", False)))
    base_dim = int(meta.get("temp_base_dim", getattr(C, "TEMP_BASE_DIM", 9)))

    x_buf = []

    planned_steps = int(np.sum(pd.to_numeric(sched_df["dt_s"], errors="coerce").fillna(0).to_numpy()))
    planned_steps = max(planned_steps, 1)

    real_df = pd.DataFrame()
    real_path = Path(real_1hz_csv)
    if real_path.exists():
        real_df = _read_table_auto(real_path)

    col_speed = _pick_col(real_df, "speed", "speed_mps")
    col_power = _pick_col(real_df, "power_w", "power", "powerw")
    col_heading = _pick_col(real_df, "heading", "yaw", "theta")
    col_lsp = _pick_col(real_df, "left_speed", "leftspeed", "wheel_left_speed")
    col_rsp = _pick_col(real_df, "right_speed", "rightspeed", "wheel_right_speed")
    col_conf = _pick_col(real_df, "pos_confidence", "confidence", "pose_confidence")
    col_batt = _pick_col(real_df, "battery", "battery_pct", "battery_percent")
    col_idle = _pick_col(real_df, "idle_flag", "idle")
    col_tr = _pick_col(real_df, "target_reached", "goal_reached", "reached")

    ref_speed = (
        pd.to_numeric(real_df[col_speed], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_speed)
        else None
    )
    ref_power = (
        pd.to_numeric(real_df[col_power], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_power)
        else None
    )
    ref_heading = (
        pd.to_numeric(real_df[col_heading], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_heading)
        else None
    )
    ref_lsp = (
        pd.to_numeric(real_df[col_lsp], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_lsp)
        else None
    )
    ref_rsp = (
        pd.to_numeric(real_df[col_rsp], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_rsp)
        else None
    )
    ref_conf = (
        pd.to_numeric(real_df[col_conf], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_conf)
        else None
    )
    ref_batt = (
        pd.to_numeric(real_df[col_batt], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_batt)
        else None
    )
    ref_idle = (
        pd.to_numeric(real_df[col_idle], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_idle)
        else None
    )
    ref_tr = (
        pd.to_numeric(real_df[col_tr], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (len(real_df) > 0 and col_tr)
        else None
    )

    sched = list(sched_df.itertuples(index=False))

    def _angle_norm(u_xy, v_xy, w_xy) -> float:
        ux, uy = u_xy
        vx, vy = v_xy
        wx, wy = w_xy
        a = np.array([vx - ux, vy - uy], dtype=float)
        b = np.array([wx - vx, wy - vy], dtype=float)
        na = float(np.linalg.norm(a) + 1e-9)
        nb = float(np.linalg.norm(b) + 1e-9)
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return float(ang / np.pi)

    def _lookahead_for_edge(edge_idx: int) -> dict:
        if not use_lookahead:
            return {}
        if edge_idx < 0 or edge_idx >= len(sched):
            return {"next_turn_angle_norm": 0.0, "next_edge_turn_rate_norm": 0.0}
        u = int(getattr(sched[edge_idx], "u"))
        v = int(getattr(sched[edge_idx], "v"))
        if edge_idx + 1 >= len(sched):
            return {"next_turn_angle_norm": 0.0, "next_edge_turn_rate_norm": 0.0}
        v2 = int(getattr(sched[edge_idx + 1], "u"))
        w = int(getattr(sched[edge_idx + 1], "v"))
        if v2 != v:
            return {"next_turn_angle_norm": 0.0, "next_edge_turn_rate_norm": 0.0}

        ui = G.node_id_to_idx.get(u, None)
        vi = G.node_id_to_idx.get(v, None)
        wi = G.node_id_to_idx.get(w, None)
        next_turn = 0.0
        if ui is not None and vi is not None and wi is not None:
            u_xy = (float(G.node_x[ui]), float(G.node_y[ui]))
            v_xy = (float(G.node_x[vi]), float(G.node_y[vi]))
            w_xy = (float(G.node_x[wi]), float(G.node_y[wi]))
            next_turn = _angle_norm(u_xy, v_xy, w_xy)

        pts_next = geom_templates.get((v, w), None)
        next_tr = float(_turn_rate_norm_from_polyline(pts_next)) if pts_next is not None else 0.0
        return {"next_turn_angle_norm": float(next_turn), "next_edge_turn_rate_norm": float(next_tr)}

    def _forward_seq(x_in_vec: np.ndarray) -> float:
        nonlocal x_buf
        if kind in ("gru", "lstm", "transformer_lite"):
            x_buf.append(x_in_vec.astype(np.float32))
            if len(x_buf) > seq_len:
                x_buf = x_buf[-seq_len:]
            x_arr = np.stack(x_buf, axis=0)
            if x_arr.shape[0] < seq_len:
                pad = np.zeros((seq_len - x_arr.shape[0], x_arr.shape[1]), dtype=np.float32)
                x_arr = np.concatenate([pad, x_arr], axis=0)
            with torch.no_grad():
                xin_t = torch.tensor(x_arr[None, :, :], dtype=torch.float32, device=device)
                y = tm(xin_t)
            return float(y.detach().cpu().numpy().reshape(-1)[0])

        with torch.no_grad():
            xin_t = torch.tensor(x_in_vec[None, :], dtype=torch.float32, device=device)
            y = tm(xin_t)
        return float(y.detach().cpu().numpy().reshape(-1)[0])

    rows = []
    e_cum = 0.0
    t_global = 0

    edge_ptr = 0
    last_x, last_y = (float("nan"), float("nan"))

    edge_deltas = None
    edge_taus = None
    edge_dt_s = 0
    s_in_edge = 0
    frac = 0.0

    while t_global < planned_steps and edge_ptr < len(sched):
        r = sched[edge_ptr]
        u = int(r.u)
        v = int(r.v)

        dt_s = int(getattr(r, "dt_s", 0))
        dt_s = max(dt_s, 1)

        dist = float(getattr(r, "edge_distance", 0.0))
        slow = float(getattr(r, "slowdown_idx", 0.0))
        t_pred_s = float(getattr(r, "t_pred_s", float(dt_s)))
        e_pred_J = float(getattr(r, "e_pred_J", 0.0))

        pts = _get_pts_for_edge(G, geom_templates, u, v)
        turn_rate_norm = _turn_rate_norm_from_polyline(pts)
        look_ex = _lookahead_for_edge(edge_ptr)

        if s_in_edge == 0:
            edge_dt_s = dt_s
            edge_deltas = np.zeros(edge_dt_s, dtype=np.float32)
            edge_taus = np.zeros(edge_dt_s, dtype=np.float32)

            for s in range(edge_dt_s):
                tau = (s + 1) / float(edge_dt_s)
                edge_taus[s] = tau

                rs = float(ref_speed[t_global + s]) if (ref_speed is not None and (t_global + s) < len(ref_speed)) else 0.0
                rp = float(ref_power[t_global + s]) if (ref_power is not None and (t_global + s) < len(ref_power)) else 0.0

                extras = dict(look_ex) if use_lookahead else {}

                if base_dim == 17:
                    idx = t_global + s

                    prev_rs = float(ref_speed[idx - 1]) if (ref_speed is not None and idx - 1 >= 0 and idx - 1 < len(ref_speed)) else rs
                    prev_prev_rs = float(ref_speed[idx - 2]) if (ref_speed is not None and idx - 2 >= 0 and idx - 2 < len(ref_speed)) else prev_rs
                    accel = rs - prev_rs
                    prev_accel = prev_rs - prev_prev_rs
                    jerk = accel - prev_accel

                    yaw_rate = 0.0
                    if ref_heading is not None and idx < len(ref_heading):
                        h_now = float(ref_heading[idx])
                        h_prev = float(ref_heading[idx - 1]) if idx - 1 >= 0 else h_now
                        dh = h_now - h_prev
                        if np.nanmax(np.abs(ref_heading)) <= 6.5:
                            dh = (dh + np.pi) % (2 * np.pi) - np.pi
                        else:
                            dh = (dh + 180.0) % 360.0 - 180.0
                        yaw_rate = float(dh)

                    wheel_diff = 0.0
                    if ref_lsp is not None and ref_rsp is not None and idx < len(ref_lsp) and idx < len(ref_rsp):
                        wheel_diff = float(ref_lsp[idx] - ref_rsp[idx])

                    pos_conf = float(ref_conf[idx]) if (ref_conf is not None and idx < len(ref_conf)) else 0.0
                    battery = float(ref_batt[idx]) if (ref_batt is not None and idx < len(ref_batt)) else 0.0
                    idle_flag = float(ref_idle[idx]) if (ref_idle is not None and idx < len(ref_idle)) else 0.0
                    target_reached = float(ref_tr[idx]) if (ref_tr is not None and idx < len(ref_tr)) else 0.0

                    extras.update({
                        "accel": accel,
                        "jerk": jerk,
                        "yaw_rate": yaw_rate,
                        "wheel_diff": wheel_diff,
                        "pos_conf": pos_conf,
                        "battery": battery,
                        "idle_flag": idle_flag,
                        "target_reached": target_reached,
                    })

                x_in = _build_feature_vector(
                    tau, dist, slow, t_pred_s, e_pred_J, meta, turn_rate_norm,
                    ref_speed=rs, ref_power=rp,
                    extras=extras if (use_lookahead or base_dim == 17) else None,
                ).astype(np.float32)

                d_raw = _forward_seq(x_in)
                edge_deltas[s] = max(0.0, float(d_raw))

            edge_deltas = _delta_stability_clamp(edge_deltas, edge_dt_s)
            ssum = float(edge_deltas.sum())
            if (not np.isfinite(ssum)) or ssum <= 1e-9:
                edge_deltas[:] = 1.0 / float(edge_dt_s)
            else:
                edge_deltas[:] = edge_deltas / ssum

            frac = 0.0

        tau = float(edge_taus[s_in_edge])
        delta = float(np.clip(edge_deltas[s_in_edge], 0.0, 1.0))
        frac = float(np.clip(frac + delta, 0.0, 1.0))
        if s_in_edge == edge_dt_s - 1:
            frac = 1.0

        x, y = _interp_xy_on_polyline_frac(pts, frac)
        x, y = _hard_speed_cap_xy(rows, x, y, slow)
        last_x, last_y = x, y

        e_per_s = (e_pred_J / float(edge_dt_s))
        e_cum += e_per_s

        rs_now = float(ref_speed[t_global]) if (ref_speed is not None and t_global < len(ref_speed)) else 0.0
        rp_now = float(ref_power[t_global]) if (ref_power is not None and t_global < len(ref_power)) else 0.0

        rows.append(
            {
                "t_global_s": int(t_global),
                "edge_idx": int(edge_ptr),
                "u": int(u),
                "v": int(v),
                "tau": float(tau),
                "delta_frac": float(delta),
                "frac": float(frac),
                "x": float(x),
                "y": float(y),
                "edge_distance": float(dist),
                "slowdown_idx": float(slow),
                "t_pred_s": float(t_pred_s),
                "e_pred_J": float(e_pred_J),
                "e_cum_J": float(e_cum),
                "ref_speed": float(rs_now),
                "ref_power_W": float(rp_now),
            }
        )

        t_global += 1
        s_in_edge += 1

        if s_in_edge >= edge_dt_s:
            edge_ptr += 1
            s_in_edge = 0
            frac = 0.0

    while t_global < planned_steps:
        rows.append(
            {
                "t_global_s": int(t_global),
                "edge_idx": int(edge_ptr),
                "u": int(-1),
                "v": int(-1),
                "tau": 1.0,
                "delta_frac": 0.0,
                "frac": 1.0,
                "x": float(last_x),
                "y": float(last_y),
                "e_cum_J": float(e_cum),
                "note": "hold_after_finish",
            }
        )
        t_global += 1

    df_out = pd.DataFrame(rows)
    df_out = add_speed_from_xy(df_out, x_col="x", y_col="y", dt_sec=1.0)
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] traj_temporal_mlp_closed_loop -> {abs_path(out_csv)} steps={len(rows)}")