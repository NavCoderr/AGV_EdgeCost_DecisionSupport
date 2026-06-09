# -*- coding: utf-8 -*-
from typing import Tuple, List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from utils_io import abs_path
from graph_data import GraphData
from templates import progress_on_polyline

# small helpers

def _pick_col(df: pd.DataFrame, *cands: str):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in cands:
        k = str(cand).strip().lower()
        if k in cols:
            return cols[k]
    return None


def _turn_rate_norm_from_polyline(pts: np.ndarray) -> float:
    """mean abs turn angle between segments, normalized by pi => [0..1]"""
    if pts is None or len(pts) < 3:
        return 0.0
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    ang = np.arctan2(dy, dx)
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    val = float(np.mean(np.abs(d))) if len(d) else 0.0
    return float(np.clip(val / np.pi, 0.0, 1.0))

# Models

class TemporalMLP(nn.Module):
    """Per-step delta predictor (B,F) -> (B,1)"""
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3, dropout: float = 0.10):
        super().__init__()
        layers = max(1, int(layers))
        dims = [in_dim] + [hidden] * layers + [1]
        mods = []
        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.ReLU())
            if dropout and dropout > 0:
                mods.append(nn.Dropout(float(dropout)))
        mods.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)                 # (B,1)
        return torch.nn.functional.softplus(out)  # positive


class TemporalGRU(nn.Module):
    """Sequence delta predictor (B,T,F) OR (B,F) -> (B,1)"""
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = max(1, int(layers))
        self.dropout = float(dropout)

        self.rnn = nn.GRU(
            input_size=self.in_dim,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=(self.dropout if self.layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # Accept both (B,F) and (B,T,F)
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)  # (B,1,F)

        out, _ = self.rnn(x_seq)        # (B,T,H)
        h_last = out[:, -1, :]          # (B,H)
        y = self.head(h_last)           # (B,1)
        return torch.nn.functional.softplus(y)    # positive


class TemporalLSTM(nn.Module):
    """Sequence delta predictor (B,T,F) OR (B,F) -> (B,1)"""
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = max(1, int(layers))
        self.dropout = float(dropout)

        self.rnn = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=(self.dropout if self.layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)  # (B,1,F)

        out, _ = self.rnn(x_seq)        # (B,T,H)
        h_last = out[:, -1, :]          # (B,H)
        y = self.head(h_last)           # (B,1)
        return torch.nn.functional.softplus(y)    # positive


class TransformerLite(nn.Module):
    """Sequence delta predictor (B,T,F) OR (B,F) -> (B,1)"""
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, heads: int = 2,
                 dropout: float = 0.10, max_len: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = max(1, int(layers))
        self.heads = max(1, int(heads))
        self.dropout = float(dropout)
        self.max_len = int(max_len)

        self.in_proj = nn.Linear(self.in_dim, self.hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=self.heads,
            dropout=self.dropout,
            batch_first=True,
            activation="relu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=self.layers)
        self.out = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

        # fixed sinusoidal positional encoding
        pe = torch.zeros(self.max_len, self.hidden, dtype=torch.float32)
        pos = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.hidden, 2, dtype=torch.float32) * (-np.log(10000.0) / self.hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,L,H)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)  # (B,1,F)

        B, T, _ = x_seq.shape
        T2 = min(T, self.max_len)

        x = self.in_proj(x_seq[:, -T2:, :])     # (B,T2,H)
        x = x + self.pe[:, :T2, :]              # (B,T2,H)
        h = self.enc(x)                         # (B,T2,H)
        h_last = h[:, -1, :]                    # (B,H)
        y = self.out(h_last)                    # (B,1)
        return torch.nn.functional.softplus(y)  # positive


class PhysicsDeltaNet(nn.Module):
    """
    Hybrid: base_delta ~= 1 / t_pred_s (from t_norm * time_scale), plus learned residual.
    Output clamped to [0,1].
    """
    def __init__(self, in_dim: int, time_scale: float, t_norm_index: int,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = max(1, int(layers))
        self.dropout = float(dropout)
        self.time_scale = float(time_scale)
        self.t_norm_index = int(t_norm_index)

        dims = [self.in_dim] + [self.hidden] * self.layers + [1]
        mods = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            if self.dropout > 0:
                mods += [nn.Dropout(self.dropout)]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.res_net = nn.Sequential(*mods)

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        # x_last: (B,F)
        if x_last.dim() == 3:
            x_last = x_last[:, -1, :]  # (B,F)

        t_norm = x_last[:, self.t_norm_index].clamp(min=0.0)
        t_pred_s = t_norm * max(self.time_scale, 1e-6)
        base = 1.0 / t_pred_s.clamp(min=1.0)  # base fraction per second

        res = torch.tanh(self.res_net(x_last)).squeeze(1) * 0.25
        out = (base + res).clamp(min=0.0, max=1.0)
        return out.unsqueeze(1)  # (B,1)


# Dataset builder

def build_temporal_dataset(df_1s_snapped: pd.DataFrame,
                           G: GraphData,
                           geom_templates: Dict,
                           edge_samples_agg: pd.DataFrame,
                           out_dir) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
  
    d = df_1s_snapped.copy()

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

    # Pick columns
    col_node = _pick_col(d, "node_id", "nodeid", "node")
    col_x = _pick_col(d, "x", "x_coordinate", "xcoordinate", "x-coordinate", "x-coordinate ")
    col_y = _pick_col(d, "y", "y_coordinate", "ycoordinate", "y-coordinate", "y-coordinate ")
    col_speed = _pick_col(d, "speed")
    col_power = _pick_col(d, "power_w", "power", "powerconsumption", "powerconsumptionw")

    col_heading = _pick_col(d, "heading", "yaw", "theta")
    col_lsp = _pick_col(d, "left_speed", "leftspeed", "wheel_left_speed")
    col_rsp = _pick_col(d, "right_speed", "rightspeed", "wheel_right_speed")
    col_conf = _pick_col(d, "pos_confidence", "confidence", "pose_confidence")
    col_batt = _pick_col(d, "battery", "battery_pct", "battery_percent")
    col_idle = _pick_col(d, "idle_flag", "idle")
    col_tr = _pick_col(d, "target_reached", "goal_reached", "reached")

    if col_node is None or col_x is None or col_y is None:
        raise ValueError("Temporal dataset needs node_id, x, y columns (after snap).")

    # Config toggles
    use_tau = bool(getattr(C, "USE_TAU_FEATURE", True))
    use_lookahead = bool(getattr(C, "USE_LOOKAHEAD", False))
    base_dim = int(getattr(C, "TEMP_BASE_DIM", 9))
    eff_dim = int(getattr(C, "TEMP_IN_DIM", base_dim))

    if base_dim not in (7, 9, 13, 17):
        raise ValueError(f"TEMP_BASE_DIM must be 7/9/13/17. Got {base_dim}")

    base_no_tau = {7: 6, 9: 7, 13: 11, 17: 15}[base_dim]
    expected_dim = (base_dim if use_tau else base_no_tau) + (2 if use_lookahead else 0)
    if eff_dim != expected_dim:
        raise ValueError(
            f"TEMP_IN_DIM mismatch. Expected {expected_dim} "
            f"(USE_TAU_FEATURE={use_tau}, USE_LOOKAHEAD={use_lookahead}, TEMP_BASE_DIM={base_dim}) "
            f"but got TEMP_IN_DIM={eff_dim}"
        )

    # Read columns
    node_ids = pd.to_numeric(d[col_node], errors="coerce").to_numpy()
    xs = pd.to_numeric(d[col_x], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(d[col_y], errors="coerce").to_numpy(dtype=float)

    sp = pd.to_numeric(d[col_speed], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_speed else np.zeros(len(d), dtype=float)
    if getattr(C, "DROP_NEG_SPEED", True):
        sp = np.where(sp < 0, 0.0, sp)
    sp = np.where(sp < getattr(C, "MIN_MOVE_SPEED", 0.09), 0.0, sp)

    pw = pd.to_numeric(d[col_power], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_power else np.zeros(len(d), dtype=float)
    pw = np.where(pw < 0, 0.0, pw)

    # extra signals (optional)
    hd = pd.to_numeric(d[col_heading], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_heading else np.zeros(len(d), dtype=float)
    lsp = pd.to_numeric(d[col_lsp], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_lsp else np.zeros(len(d), dtype=float)
    rsp = pd.to_numeric(d[col_rsp], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_rsp else np.zeros(len(d), dtype=float)
    conf = pd.to_numeric(d[col_conf], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_conf else np.zeros(len(d), dtype=float)
    batt = pd.to_numeric(d[col_batt], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_batt else np.zeros(len(d), dtype=float)
    idle = pd.to_numeric(d[col_idle], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_idle else np.zeros(len(d), dtype=float)
    tr = pd.to_numeric(d[col_tr], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col_tr else np.zeros(len(d), dtype=float)

    # Robust scales
    accel_all = np.diff(sp, prepend=sp[:1])
    jerk_all = np.diff(accel_all, prepend=accel_all[:1])

    dh = np.diff(hd, prepend=hd[:1])
    if np.nanmax(np.abs(hd)) <= 6.5:
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        yaw_all = dh
        yaw_scale = float(np.percentile(np.abs(yaw_all), 95) + 1e-6) if np.any(np.isfinite(yaw_all)) else 1.0
    else:
        dh = (dh + 180.0) % 360.0 - 180.0
        yaw_all = dh
        yaw_scale = float(np.percentile(np.abs(yaw_all), 95) + 1e-6) if np.any(np.isfinite(yaw_all)) else 30.0

    accel_scale = float(np.percentile(np.abs(accel_all), 95) + 1e-6) if np.any(np.isfinite(accel_all)) else 0.5
    jerk_scale  = float(np.percentile(np.abs(jerk_all), 95) + 1e-6) if np.any(np.isfinite(jerk_all)) else 1.0

    wheel_diff_all = lsp - rsp
    wheel_diff_scale = float(np.percentile(np.abs(wheel_diff_all), 95) + 1e-6) if np.any(np.isfinite(wheel_diff_all)) else 100.0

    conf_scale = float(np.percentile(conf[conf > 0], 95) + 1e-6) if np.any(conf > 0) else 100.0
    batt_scale = float(np.percentile(batt[batt > 0], 95) + 1e-6) if np.any(batt > 0) else 100.0

    dist_max = float(np.max(G.edge_distance) + 1e-6)

    if edge_samples_agg is not None and len(edge_samples_agg):
        time_scale = float(np.median(edge_samples_agg["time_s"].to_numpy(dtype=float)) + 1e-6)
        energy_scale = float(np.median(edge_samples_agg["energy_J"].to_numpy(dtype=float)) + 1e-6)
    else:
        time_scale = 10.0
        energy_scale = 1000.0

    speed_scale = float(np.percentile(sp[sp > 0], 95) + 1e-6) if np.any(sp > 0) else float(getattr(C, "DEFAULT_FALLBACK_SPEED_MPS", 0.2))
    power_scale = float(np.percentile(pw[pw > 0], 95) + 1e-6) if np.any(pw > 0) else 100.0

    slow_map = {}
    if edge_samples_agg is not None and len(edge_samples_agg) > 0 and "slowdown_idx" in edge_samples_agg.columns:
        for r in edge_samples_agg.itertuples(index=False):
            slow_map[(int(r.u_node_id), int(r.v_node_id))] = float(getattr(r, "slowdown_idx"))
    slow_global = float(np.mean(list(slow_map.values()))) if slow_map else 0.0

    dist_map = {(int(u), int(v)): float(dv) for u, v, dv in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance)}

    X_list: List[List[float]] = []
    y_list: List[List[float]] = []
    g_list: List[int] = []

    n = len(d)
    i = 0
    trav_id = 0

    while i < n - 2:
        if not np.isfinite(node_ids[i]):
            i += 1
            continue

        u = int(node_ids[i])
        j = i
        while j < n and np.isfinite(node_ids[j]) and int(node_ids[j]) == u:
            j += 1
        if j >= n or not np.isfinite(node_ids[j]):
            i = j
            continue

        v = int(node_ids[j])
        if v == u:
            i = j
            continue
        if (u, v) not in dist_map:
            i = j
            continue

        # FIXED: take full run on node v
        k = j
        while k < n and np.isfinite(node_ids[k]) and int(node_ids[k]) == v:
            k += 1

        dt = k - i
        if dt <= 2 or dt > getattr(C, "MAX_SEG_SECONDS", 60):
            i = j
            continue

        pts = geom_templates.get((u, v), None)
        if pts is None:
            ui = G.node_id_to_idx[u]
            vi = G.node_id_to_idx[v]
            pts = np.array([[float(G.node_x[ui]), float(G.node_y[ui])],
                            [float(G.node_x[vi]), float(G.node_y[vi])]], dtype=np.float32)

        frac_seq = []
        idx_move = []
        for t in range(i, k):
            if (not np.isfinite(xs[t])) or (not np.isfinite(ys[t])) or (sp[t] < getattr(C, "MIN_MOVE_SPEED", 0.09)):
                frac_seq.append(np.nan)
            else:
                frac_seq.append(float(progress_on_polyline((float(xs[t]), float(ys[t])), pts)))
                idx_move.append(t)

        frac_seq = np.array(frac_seq, dtype=float)
        if np.all(~np.isfinite(frac_seq)):
            i = j
            continue

        frac_seq = pd.Series(frac_seq).interpolate().ffill().bfill().to_numpy(dtype=np.float32)
        frac_seq = pd.Series(frac_seq).rolling(window=3, center=True, min_periods=1).median().to_numpy(dtype=np.float32)
        frac_seq = np.clip(frac_seq, 0.0, 1.0)
        frac_seq = np.maximum.accumulate(frac_seq)
        frac_seq[-1] = 1.0

        delta = np.diff(np.concatenate([[0.0], frac_seq.astype(np.float64)]))
        delta = np.clip(delta, 0.0, 1.0)
        cap = float(np.percentile(delta, 95))
        cap = max(cap, 1e-6)
        delta = np.minimum(delta, cap)
        ssum = float(np.sum(delta))
        if ssum <= 1e-9:
            i = j
            continue
        delta = (delta / ssum).astype(np.float32)

        t_move = float(np.sum(sp[i:k] >= getattr(C, "MIN_MOVE_SPEED", 0.09)))
        e_move = float(np.sum(pw[i:k][sp[i:k] >= getattr(C, "MIN_MOVE_SPEED", 0.09)]))
        mean_speed_move = float(np.mean(sp[np.array(idx_move, dtype=int)])) if len(idx_move) else 0.0
        mean_power_move = float(e_move / max(t_move, 1e-6))

        dist = float(dist_map[(u, v)])
        dist_norm = dist / max(dist_max, 1e-6)
        slowdown_mean = float(slow_map.get((u, v), slow_global))

        t_norm = t_move / max(time_scale, 1e-6)
        e_norm = e_move / max(energy_scale, 1e-6)

        turn_rate_norm = float(_turn_rate_norm_from_polyline(pts))

        next_turn_angle_norm = 0.0
        next_edge_turn_rate_norm = 0.0
        if use_lookahead:
            w = None
            t2 = k
            if t2 < n and np.isfinite(node_ids[t2]):
                w = int(node_ids[t2])
            if w is not None and w != v:
                ui = G.node_id_to_idx.get(u, None)
                vi = G.node_id_to_idx.get(v, None)
                wi = G.node_id_to_idx.get(w, None)
                if ui is not None and vi is not None and wi is not None:
                    u_xy = (float(G.node_x[ui]), float(G.node_y[ui]))
                    v_xy = (float(G.node_x[vi]), float(G.node_y[vi]))
                    w_xy = (float(G.node_x[wi]), float(G.node_y[wi]))
                    next_turn_angle_norm = _angle_norm(u_xy, v_xy, w_xy)

                pts_next = geom_templates.get((v, w), None)
                if pts_next is not None:
                    next_edge_turn_rate_norm = float(_turn_rate_norm_from_polyline(pts_next))

        seg_sp = sp[i:k]
        seg_pw = pw[i:k]
        seg_accel = np.diff(seg_sp, prepend=seg_sp[:1])
        seg_jerk  = np.diff(seg_accel, prepend=seg_accel[:1])

        seg_h = hd[i:k]
        seg_dh = np.diff(seg_h, prepend=seg_h[:1])
        if np.nanmax(np.abs(hd)) <= 6.5:
            seg_dh = (seg_dh + np.pi) % (2 * np.pi) - np.pi
        else:
            seg_dh = (seg_dh + 180.0) % 360.0 - 180.0
        seg_yaw = seg_dh

        seg_wdiff = (lsp[i:k] - rsp[i:k])
        seg_conf  = conf[i:k]
        seg_batt  = batt[i:k]
        seg_idle  = idle[i:k]
        seg_tr    = tr[i:k]

        mean_sp_norm = float(mean_speed_move / max(speed_scale, 1e-6))
        mean_pw_norm = float(mean_power_move / max(power_scale, 1e-6))

        for s in range(dt):
            tau = (s + 1) / float(dt)
            rem_dist_norm = 1.0 - float(tau)

            sp_norm = float(float(seg_sp[s]) / max(speed_scale, 1e-6))
            pw_norm = float(float(seg_pw[s]) / max(power_scale, 1e-6))

            accel_norm = float(float(seg_accel[s]) / max(accel_scale, 1e-6))
            jerk_norm  = float(float(seg_jerk[s]) / max(jerk_scale, 1e-6))
            yaw_norm   = float(float(seg_yaw[s]) / max(yaw_scale, 1e-6))
            wdiff_norm = float(float(seg_wdiff[s]) / max(wheel_diff_scale, 1e-6))
            conf_norm  = float(float(seg_conf[s]) / max(conf_scale, 1e-6))
            batt_norm  = float(float(seg_batt[s]) / max(batt_scale, 1e-6))
            idle_flag  = float(float(seg_idle[s]))
            tr_flag    = float(float(seg_tr[s]))

            if use_tau:
                if base_dim == 17:
                    feat = [tau, dist_norm, slowdown_mean, rem_dist_norm, t_norm, e_norm,
                            sp_norm, pw_norm, turn_rate_norm,
                            accel_norm, jerk_norm, yaw_norm, wdiff_norm, conf_norm, batt_norm, idle_flag, tr_flag]
                elif base_dim == 13:
                    feat = [tau, dist_norm, slowdown_mean, rem_dist_norm, t_norm, e_norm,
                            sp_norm, pw_norm, turn_rate_norm,
                            accel_norm, yaw_norm, wdiff_norm, conf_norm]
                elif base_dim == 9:
                    feat = [tau, dist_norm, slowdown_mean, rem_dist_norm, t_norm, e_norm,
                            mean_sp_norm, mean_pw_norm, turn_rate_norm]
                else:
                    feat = [tau, dist_norm, slowdown_mean, t_norm, e_norm, mean_sp_norm, mean_pw_norm]
            else:
                if base_dim == 17:
                    feat = [dist_norm, slowdown_mean, t_norm, e_norm,
                            sp_norm, pw_norm, turn_rate_norm,
                            accel_norm, jerk_norm, yaw_norm, wdiff_norm, conf_norm, batt_norm, idle_flag, tr_flag]
                elif base_dim == 13:
                    feat = [dist_norm, slowdown_mean, t_norm, e_norm,
                            sp_norm, pw_norm, turn_rate_norm,
                            accel_norm, yaw_norm, wdiff_norm, conf_norm]
                elif base_dim == 9:
                    feat = [dist_norm, slowdown_mean, t_norm, e_norm, mean_sp_norm, mean_pw_norm, turn_rate_norm]
                else:
                    feat = [dist_norm, slowdown_mean, t_norm, e_norm, mean_sp_norm, mean_pw_norm]

            if use_lookahead:
                feat += [float(next_turn_angle_norm), float(next_edge_turn_rate_norm)]

            if len(feat) != eff_dim:
                raise ValueError(f"Feature dim mismatch: len(feat)={len(feat)} but TEMP_IN_DIM={eff_dim}")

            X_list.append([float(x) for x in feat])
            y_list.append([float(delta[s])])
            g_list.append(trav_id)

        trav_id += 1
        i = j

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(g_list, dtype=np.int64)

    meta = {
        "dist_max": float(dist_max),
        "time_scale": float(time_scale),
        "energy_scale": float(energy_scale),
        "speed_scale": float(speed_scale),
        "power_scale": float(power_scale),
        "accel_scale": float(accel_scale),
        "jerk_scale": float(jerk_scale),
        "yaw_scale": float(yaw_scale),
        "wheel_diff_scale": float(wheel_diff_scale),
        "conf_scale": float(conf_scale),
        "batt_scale": float(batt_scale),
        "use_tau_feature": bool(use_tau),
        "use_lookahead": bool(use_lookahead),
        "temp_base_dim": int(base_dim),
        "temp_in_dim": int(eff_dim),
    }

    meta_csv = Path(out_dir) / "temporal_mlp_meta.csv"
    pd.DataFrame([meta]).to_csv(meta_csv, index=False)
    print(f"[temporal_ds] rows={len(X)} traversals={trav_id} saved_meta={abs_path(meta_csv)}")
    return X, y, groups, meta
# Splits / Metrics / Windowing

def _split_groups(groups: np.ndarray):
    rng = np.random.RandomState(int(C.TEMP_SPLIT_SEED))
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    n = len(uniq)
    n_tr = int(n * float(C.TEMP_SPLIT_TRAIN))
    n_va = int(n * float(C.TEMP_SPLIT_VAL))

    tr_g = set(uniq[:n_tr])
    va_g = set(uniq[n_tr:n_tr + n_va])
    te_g = set(uniq[n_tr + n_va:])

    idx = np.arange(len(groups))
    tr = idx[np.isin(groups, list(tr_g))]
    va = idx[np.isin(groups, list(va_g))]
    te = idx[np.isin(groups, list(te_g))]
    print(f"[temp_split] train={len(tr)} val={len(va)} test={len(te)} (samples)")
    return tr, va, te


def _reg_metrics(y_pred, y_true, p_for_adj: int):
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    n = len(y_true)
    p = int(p_for_adj)
    if n - p - 1 <= 0:
        adj = float("nan")
    else:
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    return mae, rmse, float(r2), float(adj)


def _windowize_by_group(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seq_len: int):
    seq_len = int(seq_len)
    if seq_len <= 1:
        return X.astype(np.float32), y.reshape(-1).astype(np.float32), groups.astype(np.int64)

    Xw, yw, gw = [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx.sort()
        if len(idx) < seq_len:
            continue
        for j in range(seq_len - 1, len(idx)):
            sl = idx[j - seq_len + 1: j + 1]
            Xw.append(X[sl, :])
            yw.append(float(y[idx[j]]))
            gw.append(int(g))
    if not Xw:
        return (np.zeros((0, seq_len, X.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.stack(Xw, axis=0).astype(np.float32), np.asarray(yw, dtype=np.float32), np.asarray(gw, dtype=np.int64)


# Train / Load

def _make_temporal_model(kind: str, in_dim: int, meta: dict):
    kind = (kind or "mlp").lower().strip()
    dropout = float(getattr(C, "TEMP_DROPOUT", 0.10))

    if kind == "gru":
        return TemporalGRU(in_dim=in_dim, hidden=int(C.TEMP_HIDDEN), layers=max(1, int(C.TEMP_LAYERS)), dropout=dropout)
    if kind == "lstm":
        return TemporalLSTM(in_dim=in_dim, hidden=int(C.TEMP_HIDDEN), layers=max(1, int(C.TEMP_LAYERS)), dropout=dropout)
    if kind in ("transformer_lite", "transformer"):
        heads = int(getattr(C, "TEMP_HEADS", 2))
        layers = max(1, int(getattr(C, "TEMP_TF_LAYERS", 2)))
        return TransformerLite(in_dim=in_dim, hidden=int(C.TEMP_HIDDEN), layers=layers, heads=heads, dropout=dropout, max_len=int(getattr(C, "TEMP_SEQ_LEN", 5)))
    if kind in ("physics", "physics_delta", "physics_nn"):
        time_scale = float(meta.get("time_scale", 10.0))
        # for 9-dim: t_norm index = 4; for 17-dim we still keep t_norm at index 4 (same layout in this project)
        t_idx = 4
        return PhysicsDeltaNet(in_dim=in_dim, time_scale=time_scale, t_norm_index=t_idx, hidden=int(C.TEMP_HIDDEN), layers=max(1, int(C.TEMP_LAYERS)), dropout=dropout)

    return TemporalMLP(in_dim=in_dim, hidden=int(C.TEMP_HIDDEN), layers=max(1, int(C.TEMP_LAYERS)), dropout=dropout)


def train_temporal_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, meta: dict, out_dir: Path, kind: str = "mlp") -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(C.DEVICE)

    kind = (kind or "mlp").lower().strip()
    in_dim = int(X.shape[-1])

    seq_len = int(getattr(C, "TEMP_SEQ_LEN", 5)) if kind in ("gru", "lstm", "transformer_lite", "transformer") else 1
    Xw, yw, gw = _windowize_by_group(X, y.reshape(-1), groups, seq_len=seq_len)
    if Xw.shape[0] == 0:
        raise ValueError(f"[temp] Not enough samples to build seq_len={seq_len} windows for kind={kind}")

    tr, va, te = _split_groups(gw)

    X_tr, y_tr = Xw[tr], yw[tr]
    X_va, y_va = Xw[va], yw[va]
    X_te, y_te = Xw[te], yw[te]

    model = _make_temporal_model(kind, in_dim=in_dim, meta=meta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(C.TEMP_LR), weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_path = out_dir / f"temporal_{kind}.pt"

    def _forward(batch_x: torch.Tensor):
        # batch_x: (B,T,F) for seq models, or (B,F) for mlp/physics
        if batch_x.ndim == 3:
            if isinstance(model, (TemporalMLP, PhysicsDeltaNet)):
                return model(batch_x[:, -1, :])     # (B,1)
            return model(batch_x)                   # (B,1)

        # 2D -> make safe
        if isinstance(model, (TemporalGRU, TemporalLSTM, TransformerLite)):
            return model(batch_x.unsqueeze(1))      # (B,1)
        return model(batch_x)                       # (B,1)

    bs = int(getattr(C, "TEMP_BATCH", 4096))
    for ep in range(1, int(C.TEMP_EPOCHS) + 1):
        model.train()
        perm = np.random.permutation(len(X_tr))
        total = 0.0

        for s in range(0, len(perm), bs):
            ix = perm[s:s + bs]
            xb = torch.tensor(X_tr[ix], dtype=torch.float32, device=device)
            y_ok = torch.tensor(y_tr[ix].reshape(-1, 1), dtype=torch.float32, device=device)

            pred = _forward(xb)
            loss = loss_fn(pred, y_ok)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(ix)

        tr_loss = total / max(len(perm), 1)

        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_va, dtype=torch.float32, device=device)
            y_ok = torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32, device=device)
            pred = _forward(xb)
            va_loss = float(loss_fn(pred, y_ok).item())

        if ep % 5 == 0 or ep == 1 or ep == int(C.TEMP_EPOCHS):
            print(f"[temp_{kind}] epoch={ep}/{int(C.TEMP_EPOCHS)} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "kind": kind,
                "seq_len": seq_len,
                "in_dim": in_dim,
                "hidden": int(getattr(C, "TEMP_HIDDEN", 64)),
                "layers": int(getattr(C, "TEMP_LAYERS", 3)),
                "dropout": float(getattr(C, "TEMP_DROPOUT", 0.10)),
                "state_dict": model.state_dict(),
                "meta": dict(meta, temp_in_dim=int(meta.get("temp_in_dim", in_dim)), seq_len=seq_len, kind=kind),
            }
            torch.save(ckpt, best_path)

    # eval
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_te, dtype=torch.float32, device=device)
        yp = _forward(xb).cpu().numpy().reshape(-1)
    yt = y_te.reshape(-1)

    mae, rmse, r2, adj = _reg_metrics(yp, yt, p_for_adj=in_dim)
    eval_csv = out_dir / f"temporal_{kind}_eval.csv"
    pd.DataFrame([{
        "kind": kind, "seq_len": seq_len, "MAE": mae, "RMSE": rmse, "R2": r2, "AdjR2": adj, "best_val_loss": best_val
    }]).to_csv(eval_csv, index=False)

    print(f"[temp_eval] saved -> {abs_path(eval_csv)}")
    print(f"[temp_eval] kind={kind} MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f} AdjR2={adj:.6f}")
    print(f"[temp_{kind}] best_val_loss={best_val:.6f} saved={abs_path(best_path)}")
    return str(best_path)


def train_temporal_all(X: np.ndarray, y: np.ndarray, groups: np.ndarray, meta: dict, out_dir: Path,
                       kinds=("mlp", "gru", "lstm", "transformer_lite", "physics_delta")) -> str:
    results = []
    best_kind = None
    best_mae = float("inf")

    for k in kinds:
        try:
            _ = train_temporal_model(X, y, groups, meta, out_dir, kind=k)
            ev = pd.read_csv(Path(out_dir) / f"temporal_{k}_eval.csv").iloc[0].to_dict()
            results.append(ev)
            if float(ev["MAE"]) < best_mae:
                best_mae = float(ev["MAE"])
                best_kind = k
        except Exception as e:
            print(f"[temp_all] skip {k}: {e}")

    summ = Path(out_dir) / "temporal_all_summary.csv"
    if results:
        pd.DataFrame(results).sort_values("MAE").to_csv(summ, index=False)
        print(f"[temp_all] summary -> {abs_path(summ)}")

    if best_kind is None:
        raise RuntimeError("No temporal model could be trained.")

    (Path(out_dir) / "best_temporal_kind.txt").write_text(str(best_kind), encoding="utf-8")
    print(f"[temp_all] BEST={best_kind} (MAE={best_mae:.6f})")
    return best_kind


def load_temporal_model(path: str):
    device = torch.device(C.DEVICE)
    ckpt = torch.load(path, map_location=device)
    meta = ckpt.get("meta", {})
    kind = str(ckpt.get("kind", meta.get("kind", "mlp"))).lower().strip()
    in_dim = int(ckpt.get("in_dim", meta.get("temp_in_dim", C.TEMP_IN_DIM)))
    dropout = float(ckpt.get("dropout", getattr(C, "TEMP_DROPOUT", 0.10)))

    if kind == "gru":
        model = TemporalGRU(in_dim=in_dim, hidden=int(ckpt.get("hidden", C.TEMP_HIDDEN)),
                            layers=max(1, int(ckpt.get("layers", C.TEMP_LAYERS))), dropout=dropout)
    elif kind == "lstm":
        model = TemporalLSTM(in_dim=in_dim, hidden=int(ckpt.get("hidden", C.TEMP_HIDDEN)),
                             layers=max(1, int(ckpt.get("layers", C.TEMP_LAYERS))), dropout=dropout)
    elif kind in ("transformer_lite", "transformer"):
        heads = int(getattr(C, "TEMP_HEADS", 2))
        layers = int(getattr(C, "TEMP_TF_LAYERS", 2))
        model = TransformerLite(in_dim=in_dim, hidden=int(ckpt.get("hidden", C.TEMP_HIDDEN)),
                                layers=layers, heads=heads, dropout=dropout,
                                max_len=int(meta.get("seq_len", getattr(C, "TEMP_SEQ_LEN", 5))))
    elif kind in ("physics", "physics_delta", "physics_nn"):
        time_scale = float(meta.get("time_scale", 10.0))
        t_idx = 4
        model = PhysicsDeltaNet(in_dim=in_dim, time_scale=time_scale, t_norm_index=t_idx,
                                hidden=int(ckpt.get("hidden", C.TEMP_HIDDEN)),
                                layers=max(1, int(ckpt.get("layers", C.TEMP_LAYERS))), dropout=dropout)
    else:
        model = TemporalMLP(in_dim=in_dim, hidden=int(ckpt.get("hidden", C.TEMP_HIDDEN)),
                            layers=max(1, int(ckpt.get("layers", C.TEMP_LAYERS))), dropout=dropout)

    model = model.to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, meta
