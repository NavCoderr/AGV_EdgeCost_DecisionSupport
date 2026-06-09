# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd

import config as C
from graph_data import GraphData

def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(' ', '').replace('_','').replace('-','')

def _pick_col(df: pd.DataFrame, candidates) -> str | None:
    m = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in m:
            return m[key]
    return None

def _ensure_standard_cols(d: pd.DataFrame) -> pd.DataFrame:
    # XY
    xcol = _pick_col(d, ['X-coordinate','x','x_coordinate','x-coordinate','X','pos_x'])
    ycol = _pick_col(d, ['Y-coordinate','y','y_coordinate','y-coordinate','Y','pos_y'])
    if xcol is not None and 'X-coordinate' not in d.columns:
        d['X-coordinate'] = pd.to_numeric(d[xcol], errors='coerce')
    if ycol is not None and 'Y-coordinate' not in d.columns:
        d['Y-coordinate'] = pd.to_numeric(d[ycol], errors='coerce')

    # speed
    spcol = _pick_col(d, ['Speed','speed','linear_speed','v'])
    if spcol is not None and 'Speed' not in d.columns:
        d['Speed'] = pd.to_numeric(d[spcol], errors='coerce')

    # power
    pwcol = _pick_col(d, ['power consumption','power_W','powerw','power','pw'])
    if pwcol is not None and 'power consumption' not in d.columns:
        d['power consumption'] = pd.to_numeric(d[pwcol], errors='coerce')

    # per-second energy (J in 1s): prefer energy_J_1s, else diff of cumulative, else power*1
    if 'Cumulative energy consumption' not in d.columns:
        e1col = _pick_col(d, ['energy_J_1s','energy_j_1s','energy1s','energy_step_j'])
        if e1col is not None:
            d['Cumulative energy consumption'] = pd.to_numeric(d[e1col], errors='coerce')
        else:
            cumcol = _pick_col(d, [
    "Cumulative energy consumption",
    "cumulative_energy",
    "energy_cum",
    "energy_cum_raw",
    "energycum"
])
            if cumcol is not None:
                cum = pd.to_numeric(d[cumcol], errors='coerce').to_numpy(dtype=float)
                # diff, keep non-negative
                diff = np.diff(np.concatenate([[cum[0]], cum]))
                diff = np.where(np.isfinite(diff) & (diff>=0), diff, 0.0)
                d['Cumulative energy consumption'] = diff
            else:
                # fallback: power_W * 1s
                if 'power consumption' in d.columns:
                    pw = pd.to_numeric(d['power consumption'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
                    d['Cumulative energy consumption'] = np.where(np.isfinite(pw) & (pw>0), pw, 0.0)
                else:
                    d['Cumulative energy consumption'] = 0.0

    return d



def load_timeseries_1hz_csv(path: Path) -> pd.DataFrame:

    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            from preprocess_nav2_to_1hz import run_preprocess_nav2
        except Exception as e:
            raise ImportError("preprocess_nav2_to_1hz.py is required to read NAV2 Excel input") from e

        out_1hz_dir = Path(C.OUT_DIR) / "_1hz_from_excel"
        out_1hz_dir.mkdir(parents=True, exist_ok=True)

        print("[1hz] NAV2 Excel detected -> generating 1Hz CSVs in:", out_1hz_dir.resolve())
        run_preprocess_nav2(path, out_dir=out_1hz_dir)

        path = out_1hz_dir / "nav_1hz_move_only.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected dejump output not found: {path}")
        print("[1hz] Using dejump 1Hz CSV:", path.resolve())

    if not path.exists():
        raise FileNotFoundError(f"Timeseries input not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # normalize required columns
    cols = {c.lower().strip(): c for c in df.columns}
    cx = cols.get("x", None)
    cy = cols.get("y", None)
    cs = cols.get("speed", None)

    if cx is None or cy is None or cs is None:
        raise ValueError(f"CSV missing required columns x,y,speed. Found={list(df.columns)}")

    # ensure numeric
    df[cx] = pd.to_numeric(df[cx], errors="coerce")
    df[cy] = pd.to_numeric(df[cy], errors="coerce")
    df[cs] = pd.to_numeric(df[cs], errors="coerce").fillna(0.0)
    if C.DROP_NEG_SPEED:
        df.loc[df[cs] < 0, cs] = 0.0

    # ensure t_sec exists
    if "t_sec" not in cols:
        df["t_sec"] = np.arange(len(df), dtype=int)

    # ensure ts exists (optional but nice)
    if "ts" in cols:
        df["ts"] = pd.to_datetime(df[cols["ts"]], errors="coerce")
    else:
        # create synthetic ts from t_sec
        t0 = pd.Timestamp("2020-01-01 00:00:00")
        df["ts"] = t0 + pd.to_timedelta(df["t_sec"].astype(int), unit="s")

    # rename stable
    out = df.copy()
    out = out.rename(columns={cx: "x", cy: "y", cs: "speed"})

    # power_W
    if "power_W" not in out.columns:
        # try other names
        for cand in ["power", "powerw", "power_watt", "powerconsumption", "powerconsumptionw"]:
            if cand in cols:
                out["power_W"] = pd.to_numeric(out[cols[cand]], errors="coerce")
                break
    if "power_W" in out.columns:
        out["power_W"] = pd.to_numeric(out["power_W"], errors="coerce").fillna(0.0)
        out.loc[out["power_W"] < 0, "power_W"] = 0.0
        out["energy_J_1s"] = out["power_W"].astype(float)  # W * 1s = J
    else:
        out["power_W"] = 0.0
        out["energy_J_1s"] = 0.0

    # keep only move (safety)
    out["speed"] = out["speed"].fillna(0.0)
    out.loc[out["speed"] < C.MIN_MOVE_SPEED, "speed"] = 0.0

    out = out.sort_values("t_sec").reset_index(drop=True)
    return out


def snap_xy_to_node_with_dist(x: float, y: float, G: GraphData) -> Tuple[int, float]:
    dx = G.node_x.astype(float) - float(x)
    dy = G.node_y.astype(float) - float(y)
    d2 = dx * dx + dy * dy
    j = int(np.argmin(d2))
    nid = int(G.node_ids[j])
    dist = float(np.sqrt(d2[j]))
    return nid, dist


def apply_snap(df_1hz: pd.DataFrame, G: GraphData) -> Tuple[pd.DataFrame, float]:
    """
    Adds node_id column. Keeps original x,y.
    fallback ffill allowed.
    """
    d = df_1hz.copy()
    d = _ensure_standard_cols(d)

    node_id = np.full(len(d), np.nan, dtype=float)
    dist_arr = np.full(len(d), np.nan, dtype=float)

    rad = float(C.SNAP_RADIUS_M)

    for i, (x, y) in enumerate(zip(d["X-coordinate"].to_numpy(), d["Y-coordinate"].to_numpy())):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        nid, dist = snap_xy_to_node_with_dist(float(x), float(y), G)
        if dist <= rad:
            node_id[i] = float(nid)
            dist_arr[i] = dist

    d["node_id_raw"] = node_id
    d["snap_dist_m"] = dist_arr

    if str(C.SNAP_FALLBACK).lower() == "ffill":
        # ffill only small gaps
        s = pd.Series(d["node_id_raw"])
        s_ff = s.ffill(limit=int(C.FFILL_MAX_GAP_S))
        d["node_id"] = s_ff
    else:
        d["node_id"] = d["node_id_raw"]

    return d, rad


from typing import Tuple
import numpy as np
import pandas as pd

def build_edge_observations(df_1s_snapped: pd.DataFrame, G: "GraphData") -> Tuple[pd.DataFrame, float]:
    d = df_1s_snapped.copy()
    d = _ensure_standard_cols(d)

    if "node_id" not in d.columns:
        raise ValueError("df_1s_snapped missing node_id. Run apply_snap first.")

    node_ids = pd.to_numeric(d["node_id"], errors="coerce").to_numpy(dtype=float)
    sp = pd.to_numeric(d.get("Speed", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pw = pd.to_numeric(d.get("power consumption", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    #per-second energy (J in 1s)
    if "step_energy" in d.columns:
        # best: already per-second J
        e1 = pd.to_numeric(d["step_energy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    elif "energy_J_1s" in d.columns:
        # also per-second J (some datasets use this name)
        e1 = pd.to_numeric(d["energy_J_1s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    elif "energy_cum_from_start" in d.columns:
        # cumulative -> diff to get per-second
        cum = pd.to_numeric(d["energy_cum_from_start"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        e1 = np.diff(np.concatenate([[cum[0]], cum]))
        e1 = np.where(np.isfinite(e1) & (e1 >= 0), e1, 0.0)

    elif "Cumulative energy consumption" in d.columns:
        # cumulative -> diff to get per-second
        cum = pd.to_numeric(d["Cumulative energy consumption"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        e1 = np.diff(np.concatenate([[cum[0]], cum]))
        e1 = np.where(np.isfinite(e1) & (e1 >= 0), e1, 0.0)

    else:
        # fallback: power(W) * 1s = Joule
        e1 = np.where(np.isfinite(pw) & (pw > 0), pw, 0.0)

    if getattr(C, "DROP_NEG_SPEED", False):
        sp = np.where(sp < 0, 0.0, sp)

    # speed reference (for slowdown_idx)
    move_sp = sp[(sp >= C.MIN_MOVE_SPEED) & np.isfinite(sp)]
    if getattr(C, "SPEED_REF", None) is not None:
        speed_ref = float(C.SPEED_REF)
    elif len(move_sp) > 10:
        mode = str(getattr(C, "SPEED_REF_MODE", "p95")).lower()
        if mode == "max":
            speed_ref = float(np.max(move_sp))
        elif mode == "median":
            speed_ref = float(np.median(move_sp))
        else:
            speed_ref = float(np.percentile(move_sp, 95))
    else:
        speed_ref = float(getattr(C, "DEFAULT_FALLBACK_SPEED_MPS", 0.2))

    speed_ref = max(speed_ref, 1e-3)

    dist_map = {
        (int(u), int(v)): float(dd)
        for u, v, dd in zip(G.edge_u_ids, G.edge_v_ids, G.edge_distance)
    }

    
    EDGE_TAIL_U_SEC = int(getattr(C, "EDGE_TAIL_U_SEC", 3))  # use last 3 seconds of u-run

    rows = []
    n = len(d)
    i = 0
    while i < n - 2:
        if not np.isfinite(node_ids[i]):
            i += 1
            continue

        u = int(node_ids[i])

        # run of u
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

        # small stabilization at v (up to 3s)
        k = j
        while k < n and np.isfinite(node_ids[k]) and int(node_ids[k]) == v and (k - j) < 3:
            k += 1

        # start edge at tail of u-run (prevents inflated time_s)
        i_edge = max(i, j - EDGE_TAIL_U_SEC)

        dt = k - i_edge
        if dt <= 2 or dt > int(C.MAX_SEG_SECONDS):
            i = j
            continue

        seg_sp = sp[i_edge:k]
        seg_pw = pw[i_edge:k]
        seg_e1 = e1[i_edge:k]

        move_mask = (seg_sp >= C.MIN_MOVE_SPEED) & np.isfinite(seg_sp)
        t_move = int(np.sum(move_mask))
        if t_move < max(2, int(C.MIN_MOVE_POINTS_PER_EDGE)):
            i = j
            continue

        e_move = float(np.sum(seg_e1[move_mask]))          # Joule
        ms = float(np.mean(seg_sp[move_mask]))             # m/s
        mean_power_W = float(e_move / max(t_move, 1e-6))   # J/s = W

        slowdown = float(np.clip(1.0 - (ms / speed_ref), 0.0, 1.0))
        dist = float(dist_map[(u, v)])
        jpm = float(e_move / max(dist, 1e-6))

        rows.append({
            "u_node_id": int(u),
            "v_node_id": int(v),
            "edge_distance": float(dist),
            "time_s": float(t_move),            # move-only seconds
            "energy_J": float(e_move),
            "mean_speed": float(ms),
            "mean_power_W": float(mean_power_W),
            "slowdown_idx": float(slowdown),
            "samples": int(t_move),
            "energy_J_per_m": float(jpm),
        })

        # advance to next segment
        i = j

    edge_samples = pd.DataFrame(rows)
    if len(edge_samples) == 0:
        raise ValueError("No edge traversal samples were built. Check snap radius / node coverage / data.")

    energy_per_m = float(np.median(edge_samples["energy_J_per_m"].to_numpy(dtype=float)))
    return edge_samples, energy_per_m
