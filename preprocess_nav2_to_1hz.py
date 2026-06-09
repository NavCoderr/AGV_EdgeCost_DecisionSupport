# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

# CONFIG
DT_SEC = 1.0

# move_only: keep only Speed 
MOVE_TH = 0.09

#XY de-jump (spike removal)
ENABLE_DEJUMP_XY = True
MAX_RAW_IMPLIED_SPEED_MPS = 0.6

# output clean
DROP_COLS_FINAL = ["t_sec_abs", "_t_raw_s"]

# which target columns may exist in NAV2 excel
TARGET_COL_CANDIDATES = ("Target reached", "Target reached2", "Target reached.1")


#IO
def read_any_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _rename_nav2_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Target reached.1" in df.columns and "Target reached2" not in df.columns:
        df = df.rename(columns={"Target reached.1": "Target reached2"})
    return df

# TIME
def parse_ts_to_seconds(ts) -> float:
    """
    Supports:
      - "MM:SS"
      - "MM:SS.s"
      - "HH:MM:SS.s"
      - numeric seconds
    """
    s = str(ts).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    parts = s.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except Exception:
        return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    s = series.astype(str)
    date_like = s.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True)
    return (date_like.mean() > 0.5)


def fmt_mmss(sec: float) -> str:
    sec = int(np.floor(float(sec)))
    mm = sec // 60
    ss = sec % 60
    return f"{mm:02d}:{ss:02d}"

# NUMERIC HELPERS
def _to_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _interp_linear(series: pd.Series) -> pd.Series:
    return _to_num(series).interpolate(method="linear", limit_direction="both").ffill().bfill()

# TARGET EVENT (0/1) from raw target state + goal-id
def pick_target_col(df: pd.DataFrame) -> str | None:
    for c in TARGET_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def make_target_event(out_1hz: pd.DataFrame, goal_id_col: str = "Going to ID") -> pd.Series:
    """
    Event rules:
      - rising edge 0->1 inside same goal-id
      - if goal-id changes and target is 1 on first row of new goal, mark event
      - ignore first row if already 1 (carry-over)
    """
    out = out_1hz.copy()
    target_src = pick_target_col(out)
    if target_src is None:
        return pd.Series(np.zeros(len(out), dtype=int), index=out.index)

    tr = _to_num(out[target_src]).fillna(0).astype(int)
    if len(tr) > 0:
        tr.iloc[0] = 0  # ignore carry-over

    prev = tr.shift(1, fill_value=0)

    if goal_id_col in out.columns:
        g = _to_num(out[goal_id_col]).fillna(-1)
        same_goal = g.eq(g.shift(1, fill_value=g.iloc[0]))
        event1 = (tr == 1) & (prev == 0) & same_goal
        event2 = (tr == 1) & (~same_goal)  # goal changed and already 1
        ev = (event1 | event2).astype(int)
    else:
        ev = ((tr == 1) & (prev == 0)).astype(int)

    return ev

# MOVE_ONLY: remove stops but keep target event by shifting upward
def build_move_only(
    out_1hz: pd.DataFrame,
    speed_col: str = "Speed",
    move_th: float = MOVE_TH,
    goal_id_col: str = "Going to ID",
) -> pd.DataFrame:
    """
    Output:
      - keep only rows where Speed >= move_th (so no stop rows)
      - Target reached is EVENT (single 1)
      - if Target reached happens on stop row => shift 1 upward to nearest previous row
        where Speed >= move_th (prefer same goal-id if possible)
      - keep ONLY ONE column: "Target reached" (0/1)
      - add t_sec_work and timestamp_work (perfect 1-sec work timeline)
    """
    out = out_1hz.copy()

    # moving mask
    sp = _to_num(out.get(speed_col, 0)).fillna(0.0)
    moving = sp >= float(move_th)

    # build event from raw target columns
    ev = make_target_event(out, goal_id_col=goal_id_col).astype(bool)

    # shift events that occur on non-moving rows
    if ev.any():
        idxs = np.where(ev.values)[0]
        for i in idxs:
            if moving.iloc[i]:
                continue

            ev.iloc[i] = False  # remove from stop row

            gid = None
            if goal_id_col in out.columns:
                gid = out[goal_id_col].iloc[i]

            # 1) try: go up to nearest MOVING row with SAME goal-id
            j = i - 1
            found = None
            while j >= 0:
                if moving.iloc[j]:
                    if gid is None or pd.isna(gid):
                        found = j
                        break
                    if goal_id_col in out.columns and out[goal_id_col].iloc[j] == gid:
                        found = j
                        break
                j -= 1

            # 2) fallback: nearest previous MOVING row (any goal)
            if found is None:
                j = i - 1
                while j >= 0 and (not moving.iloc[j]):
                    j -= 1
                if j >= 0:
                    found = j

            if found is not None:
                ev.iloc[found] = True

    # filter move_only
    out2 = out.loc[moving].copy().reset_index(drop=True)

    # rebuild Target reached on kept rows
    out2["Target reached"] = ev.loc[moving].reset_index(drop=True).astype(int).values

    # drop other target columns if present (only keep Target reached)
    drop_targets = [c for c in TARGET_COL_CANDIDATES if c in out2.columns and c != "Target reached"]
    out2 = out2.drop(columns=drop_targets, errors="ignore")

    # add work time
    out2["t_sec_work"] = np.arange(len(out2), dtype=int)
    out2["timestamp_work"] = out2["t_sec_work"].apply(fmt_mmss)

    return out2

# CORE: 1Hz builder
def to_1hz_full_features(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    x_col: str = "X-coordinate",
    y_col: str = "Y-coordinate",
    speed_col: str = "Speed",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = _rename_nav2_columns(df).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # required
    for c in [ts_col, x_col, y_col, speed_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # numeric + discrete
    numeric_cols = [
        x_col, y_col, speed_col,
        "power consumption",
        "current consuption",
        "Cumulative energy consumption",
        "Heading",
        "Position confidence",
        "RIGHT DRIVE SIGNALS.ActualSpeed_R",
        "LEFT DRIVE SIGNALS.ActualSpeed_L",
        "Battery value",
    ]
    discrete_cols = [
        "Going to ID",
        "Current segment",
    ]

    # keep target columns if present (for inspection + event creation)
    for tc in TARGET_COL_CANDIDATES:
        if tc in df.columns and tc not in discrete_cols:
            discrete_cols.append(tc)

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    discrete_cols = [c for c in discrete_cols if c in df.columns]

    # clean -99
    for c in ["RIGHT DRIVE SIGNALS.ActualSpeed_R", "LEFT DRIVE SIGNALS.ActualSpeed_L"]:
        if c in df.columns:
            df[c] = _to_num(df[c]).replace(-99, np.nan)

    # parse time
    have_dt = is_datetime_like(df[ts_col])
    if have_dt:
        ts_dt = pd.to_datetime(df[ts_col], errors="coerce")
        if ts_dt.notna().sum() < 3:
            raise ValueError(f"Timestamp looks datetime but cannot parse: {ts_col}")
        t0_dt = ts_dt.dropna().iloc[0]
        df["_t_raw_s"] = (ts_dt - t0_dt).dt.total_seconds()
    else:
        df["_t_raw_s"] = df[ts_col].map(parse_ts_to_seconds)

    df = df.dropna(subset=["_t_raw_s"]).sort_values("_t_raw_s").reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Not enough valid timestamps after parsing.")

    # dejump xy
    dejump_spikes = 0
    if ENABLE_DEJUMP_XY and len(df) >= 3:
        x = _to_num(df[x_col])
        y = _to_num(df[y_col])
        t = _to_num(df["_t_raw_s"])
        dx = x.diff()
        dy = y.diff()
        dt_raw = t.diff().replace(0, np.nan)
        implied_speed = np.sqrt(dx * dx + dy * dy) / dt_raw
        spike = (implied_speed > float(MAX_RAW_IMPLIED_SPEED_MPS)).fillna(False)
        if spike.any():
            df.loc[spike, [x_col, y_col]] = np.nan
            df[x_col] = _interp_linear(df[x_col])
            df[y_col] = _interp_linear(df[y_col])
            dejump_spikes = int(spike.sum())

    # bucket per sec
    df["t_sec_abs"] = np.floor(_to_num(df["_t_raw_s"]) / float(DT_SEC)) * float(DT_SEC)
    df["t_sec_abs"] = df["t_sec_abs"].astype(int)

    # aggregate per second
    agg = {c: "mean" for c in numeric_cols}
    agg.update({c: "last" for c in discrete_cols})

    d1 = (
        df.groupby("t_sec_abs", as_index=False)
          .agg(agg)
          .sort_values("t_sec_abs")
          .reset_index(drop=True)
    )

    # continuous 1Hz grid
    t_start = int(d1["t_sec_abs"].min())
    t_end = int(d1["t_sec_abs"].max())
    t_grid = np.arange(t_start, t_end + 1, int(DT_SEC))

    out = pd.DataFrame({"t_sec_abs": t_grid})
    out["t_sec"] = (out["t_sec_abs"] - out["t_sec_abs"].iloc[0]).astype(int)

    if have_dt:
        ts_out = t0_dt + pd.to_timedelta(out["t_sec_abs"], unit="s")
        out["timestamp_1hz"] = ts_out.dt.strftime("%H:%M:%S")
    else:
        out["timestamp_1hz"] = out["t_sec_abs"].apply(fmt_mmss)

    out = out.merge(d1, on="t_sec_abs", how="left")

    # fill numeric
    for c in numeric_cols:
        if c in out.columns:
            out[c] = _interp_linear(out[c])

    # fill discrete:
    #   - IDs/segments forward fill
    #   - target columns: fill NaN -> 0 
    for c in discrete_cols:
        if c in out.columns:
            if c in TARGET_COL_CANDIDATES:
                out[c] = _to_num(out[c]).fillna(0).astype(int)
            else:
                out[c] = out[c].ffill()

    # energy step
    if "Cumulative energy consumption" in out.columns:
        cum = _to_num(out["Cumulative energy consumption"]).ffill().bfill().cummax()
        out["Cumulative energy consumption"] = cum
        out["step_energy"] = cum.diff().fillna(0.0).clip(lower=0.0)
        out["energy_cum_from_start"] = out["step_energy"].cumsum()
    else:
        out["step_energy"] = np.nan
        out["energy_cum_from_start"] = np.nan

    # IMPORTANT:
    # Convert raw target columns into ONLY ONE final column "Target reached" = EVENT (0/1)
    out["Target reached"] = make_target_event(out, goal_id_col="Going to ID").astype(int)

    # drop other target columns from FULL too (keep clean, only one target column)
    drop_targets_full = [c for c in TARGET_COL_CANDIDATES if c in out.columns and c != "Target reached"]
    out = out.drop(columns=drop_targets_full, errors="ignore")

    # build move_only (no stop rows, but keep target event by shifting upward)
    out_move = build_move_only(out, speed_col=speed_col, move_th=MOVE_TH, goal_id_col="Going to ID")

    # clean outputs
    out_all_final = out.drop(columns=DROP_COLS_FINAL, errors="ignore")
    out_move_final = out_move.drop(columns=DROP_COLS_FINAL, errors="ignore")

    # summary
    sp_all = _to_num(out_all_final.get(speed_col, pd.Series([np.nan])))
    n_events_full = int((_to_num(out_all_final.get("Target reached", 0)).fillna(0).astype(int) == 1).sum())
    n_events_move = int((_to_num(out_move_final.get("Target reached", 0)).fillna(0).astype(int) == 1).sum())

    summary = {
        "input_rows": int(len(df)),
        "output_rows_1hz_full": int(len(out_all_final)),
        "output_rows_move_only": int(len(out_move_final)),
        "dt_sec": float(DT_SEC),
        "move_threshold_mps": float(MOVE_TH),
        "dejump_spikes_removed": int(dejump_spikes),
        "time_start_s": float(t_start),
        "time_end_s": float(t_end),
        "duration_s": float(t_end - t_start),
        "max_speed_mps": float(np.nanmax(sp_all.values)),
        "mean_speed_mps": float(np.nanmean(sp_all.values)),
        "n_target_events_full": n_events_full,
        "n_target_events_move_only": n_events_move,
    
    }

    return out_all_final, out_move_final, summary

# RUNNER
def run_preprocess_nav2(in_path: Path | str, out_dir: Path | str) -> dict:
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_any_table(in_path)

    out_full, out_move, summary = to_1hz_full_features(
        df,
        ts_col="timestamp",
        x_col="X-coordinate",
        y_col="Y-coordinate",
        speed_col="Speed",
    )

    full_csv = out_dir / "nav_1hz_full.csv"
    move_csv = out_dir / "nav_1hz_move_only.csv"
    sum_json = out_dir / "nav_1hz_summary.json"

    out_full.to_csv(full_csv, index=False)
    out_move.to_csv(move_csv, index=False)
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[saved]", full_csv.resolve())
    print("[saved]", move_csv.resolve())
    print("[saved]", sum_json.resolve())
    print("[summary]", summary)
    return summary


if __name__ == "__main__":
    # change paths if needed
    #in_path = Path  ("wholetesting_nav2_.xlsx") 
    in_path = Path  ("ground_data/plan1-4_.csv")  
    #out_dir = Path("out_1hz")
    out_dir = Path("out_2hz")
    run_preprocess_nav2(in_path, out_dir)
