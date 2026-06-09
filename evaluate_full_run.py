# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(".")
OUT_DIR  = BASE_DIR / "inductive_folder_new_data"

REAL_CANDIDATES = [
    BASE_DIR / "inductive_folder_new_data" / "data_1s_global.csv",
     # BASE_DIR / "out_1hz" / "nav_1hz_move_only.csv",
     #BASE_DIR / "out_2hz" / "nav_1hz_move_only.csv",
]

PLANNED_FILES = {
    "tau_warp": OUT_DIR / "planned_trajectory_1hz_CURVED_TAU_WARP.csv",

    "mlp_open": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_MLP.csv",
    "mlp_closed": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_MLP_CLOSED_LOOP.csv",

    "gru_open": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_GRU.csv",
    "gru_closed": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_GRU_CLOSED_LOOP.csv",

    "lstm_open": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_LSTM.csv",
    "lstm_closed": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_LSTM_CLOSED_LOOP.csv",

    "transformer_open": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_TRANSFORMER_LITE.csv",
    "transformer_closed": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_TRANSFORMER_LITE_CLOSED_LOOP.csv",

    "physics_open": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_PHYSICS_DELTA.csv",
    "physics_closed": OUT_DIR / "planned_trajectory_1hz_CURVED_TEMPORAL_PHYSICS_DELTA_CLOSED_LOOP.csv",
}


def _pick_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("No real 1Hz file found in REAL_CANDIDATES. Please edit REAL_CANDIDATES.")


def _load_real(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = df.columns

    # Case 1: snapped file (x,y)
    if {"x", "y"}.issubset(cols):
        tcol = "t_sec_work" if "t_sec_work" in cols else ("t_sec" if "t_sec" in cols else None)
        if tcol is None:
            df["t"] = np.arange(len(df), dtype=int)
        else:
            df["t"] = pd.to_numeric(df[tcol], errors="coerce").ffill().fillna(0).astype(int)

        out = pd.DataFrame({
            "t": df["t"].astype(int),
            "x_real": pd.to_numeric(df["x"], errors="coerce"),
            "y_real": pd.to_numeric(df["y"], errors="coerce"),
        })

        if "Speed" in cols:
            out["speed_real"] = pd.to_numeric(df["Speed"], errors="coerce")
        elif "speed" in cols:
            out["speed_real"] = pd.to_numeric(df["speed"], errors="coerce")
        else:
            out["speed_real"] = np.nan

        out["step_energy_real"] = pd.to_numeric(df["step_energy"], errors="coerce") if "step_energy" in cols else np.nan
        out["energy_cum_real"] = pd.to_numeric(df["energy_cum_from_start"], errors="coerce") if "energy_cum_from_start" in cols else np.nan

        out = out.dropna(subset=["t", "x_real", "y_real"]).sort_values("t").reset_index(drop=True)
        out["t"] = out["t"] - int(out["t"].iloc[0])
        out = out.groupby("t", as_index=False).first()
        return out

    # Case 2: raw nav_1hz_move_only (X-coordinate/Y-coordinate)
    if {"X-coordinate", "Y-coordinate"}.issubset(cols):
        if "t_sec_work" in cols:
            t = pd.to_numeric(df["t_sec_work"], errors="coerce")
        elif "t_sec" in cols:
            t = pd.to_numeric(df["t_sec"], errors="coerce")
        else:
            t = pd.Series(np.arange(len(df), dtype=int))

        t = t.ffill().fillna(0).astype(int)

        out = pd.DataFrame({
            "t": t.astype(int),
            "x_real": pd.to_numeric(df["X-coordinate"], errors="coerce"),
            "y_real": pd.to_numeric(df["Y-coordinate"], errors="coerce"),
            "speed_real": pd.to_numeric(df["Speed"], errors="coerce") if "Speed" in cols else np.nan,
            "step_energy_real": pd.to_numeric(df["step_energy"], errors="coerce") if "step_energy" in cols else np.nan,
            "energy_cum_real": pd.to_numeric(df["energy_cum_from_start"], errors="coerce") if "energy_cum_from_start" in cols else np.nan,
        })

        out = out.dropna(subset=["t", "x_real", "y_real"]).sort_values("t").reset_index(drop=True)
        out["t"] = out["t"] - int(out["t"].iloc[0])
        out = out.groupby("t", as_index=False).first()
        return out

    raise ValueError(f"Unrecognized real file format columns={list(cols)[:30]}")


def _load_plan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t_global_s" not in df.columns:
        raise ValueError(f"{path.name} missing t_global_s")

    # remove hold rows
    if "note" in df.columns:
        bad = df["note"].astype(str).str.contains("hold_after_finish", na=False)
        df = df.loc[~bad].copy()

    if "u" in df.columns and "v" in df.columns:
        uu = pd.to_numeric(df["u"], errors="coerce")
        vv = pd.to_numeric(df["v"], errors="coerce")
        df = df.loc[~((uu == -1) & (vv == -1))].copy()

    t = pd.to_numeric(df["t_global_s"], errors="coerce").fillna(0.0)
    t = np.round(t).astype(int)

    out = pd.DataFrame({
        "t": t,
        "x_pred": pd.to_numeric(df["x"], errors="coerce") if "x" in df.columns else np.nan,
        "y_pred": pd.to_numeric(df["y"], errors="coerce") if "y" in df.columns else np.nan,
        "speed_pred": pd.to_numeric(df["Speed"], errors="coerce") if "Speed" in df.columns else np.nan,
        "energy_cum_pred": pd.to_numeric(df["e_cum_J"], errors="coerce") if "e_cum_J" in df.columns else np.nan,
    })

    out = out.dropna(subset=["t", "x_pred", "y_pred"]).sort_values("t").reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"{path.name}: no valid rows after filtering hold rows")

    out["t"] = out["t"] - int(out["t"].iloc[0])
    out = out.groupby("t", as_index=False).first()
    return out


def _metrics(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, P95=np.nan, MAX=np.nan)
    return dict(
        MAE=float(np.mean(np.abs(arr))),
        RMSE=float(np.sqrt(np.mean(arr**2))),
        P95=float(np.quantile(np.abs(arr), 0.95)),
        MAX=float(np.max(np.abs(arr))),
    )


def _best_window_in_real_by_pos_rmse(real_df: pd.DataFrame, plan_df: pd.DataFrame, max_shift_s: int = 3):
    r = real_df[["t", "x_real", "y_real"]].copy().sort_values("t").reset_index(drop=True)
    p = plan_df[["t", "x_pred", "y_pred"]].copy().sort_values("t").reset_index(drop=True)

    T = int(p["t"].max())
    L = T + 1

    if len(r) < L:
        return {"real_start_t": int(r["t"].iloc[0]), "shift_s": 0, "rmse": float("inf")}

    rx = r["x_real"].to_numpy(dtype=float)
    ry = r["y_real"].to_numpy(dtype=float)
    px = p["x_pred"].to_numpy(dtype=float)
    py = p["y_pred"].to_numpy(dtype=float)
    rt = r["t"].to_numpy(dtype=int)

    best_rmse = float("inf")
    best_start_t = int(rt[0])
    best_shift = 0

    # inclusive end: len(r)-L
    for start_idx in range(0, len(r) - L + 1):
        wx = rx[start_idx:start_idx + L]
        wy = ry[start_idx:start_idx + L]

        for sh in range(-int(max_shift_s), int(max_shift_s) + 1):
            a0 = max(0, 0 + sh)
            a1 = min(L, L + sh)
            b0 = max(0, 0 - sh)
            b1 = min(L, L - sh)

            if (a1 - a0) < 5:
                continue

            dx = px[b0:b1] - wx[a0:a1]
            dy = py[b0:b1] - wy[a0:a1]
            rmse = float(np.sqrt(np.mean(dx * dx + dy * dy)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_start_t = int(rt[start_idx])
                best_shift = int(sh)

    return {"real_start_t": best_start_t, "shift_s": best_shift, "rmse": best_rmse}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_path = _pick_first_existing(REAL_CANDIDATES)
    real = _load_real(real_path)
    print(f"[real] {real_path} rows={len(real)} t=[{int(real.t.min())}..{int(real.t.max())}]")

    plans = {}
    for name, p in PLANNED_FILES.items():
        p = Path(p)
        if p.exists():
            plans[name] = _load_plan(p)
            print(f"[plan] {name}: {p} rows={len(plans[name])} t=[{int(plans[name].t.min())}..{int(plans[name].t.max())}]")
        else:
            print(f"[plan] skip missing: {name}: {p}")

    if not plans:
        raise FileNotFoundError("No planned trajectory files found. Edit PLANNED_FILES paths.")

    # FULL REAL grid (important)
    t_end_real = int(real["t"].max())
    t_grid_real = np.arange(0, t_end_real + 1, 1, dtype=int)

    real_i = (real.set_index("t")
                  .reindex(t_grid_real)
                  .interpolate(limit_direction="both")
                  .reset_index()
                  .rename(columns={"index": "t"}))
    real_i = real_i.dropna(subset=["x_real", "y_real"]).copy()
    real_i["t"] = real_i["t"].astype(int)

    # Mission length axis = max plan length
    T_max = int(max(pl["t"].max() for pl in plans.values()))
    t_mission = np.arange(0, T_max + 1, 1, dtype=int)

    per_second = pd.DataFrame({"t": t_mission})

    summary_rows = []

    for name, plan in plans.items():
        plan = plan.copy().sort_values("t").reset_index(drop=True)

        # ensure plan is complete on its own grid 0..T
        T = int(plan["t"].max())
        t_grid_plan = np.arange(0, T + 1, 1, dtype=int)
        plan_i = (plan.set_index("t")
                      .reindex(t_grid_plan)
                      .interpolate(limit_area="inside")
                      .reset_index()
                      .rename(columns={"index": "t"}))
        plan_i["t"] = plan_i["t"].astype(int)
        plan_i = plan_i.dropna(subset=["x_pred", "y_pred"]).copy()

        if len(plan_i) < 6:
            print(f"[warn] {name}: plan too short after cleaning: {len(plan_i)}")
            continue

        best = _best_window_in_real_by_pos_rmse(real_i, plan_i, max_shift_s=3)
        t0 = int(best["real_start_t"])
        sh = int(best["shift_s"])
        rmse0 = float(best["rmse"])
        print(f"[window] {name}: real_start_t={t0} shift_s={sh} rmse={rmse0:.3f}")

        # cut real window
        t1 = t0 + int(plan_i["t"].max())
        real_cut = real_i[(real_i["t"] >= t0) & (real_i["t"] <= t1)].copy()
        if len(real_cut) < 6:
            print(f"[warn] {name}: real_cut too short: {len(real_cut)}")
            continue

        real_cut["t"] = real_cut["t"] - int(real_cut["t"].iloc[0])
        real_cut = real_cut.groupby("t", as_index=False).first()

        # shift plan and keep overlap
        plan_cut = plan_i.copy()
        plan_cut["t"] = plan_cut["t"] + sh
        plan_cut = plan_cut[plan_cut["t"] >= 0].copy()
        plan_cut = plan_cut.groupby("t", as_index=False).first()

        m = pd.merge(real_cut, plan_cut, on="t", how="inner")
        if len(m) < 6:
            print(f"[warn] {name}: too few aligned seconds after window+shift: {len(m)}")
            continue

        pos_err = np.sqrt((m["x_pred"] - m["x_real"])**2 + (m["y_pred"] - m["y_real"])**2)

        if "speed_real" in m.columns and "speed_pred" in m.columns and m["speed_pred"].notna().any():
            speed_err = np.abs(m["speed_pred"].astype(float) - m["speed_real"].astype(float))
        else:
            speed_err = pd.Series([np.nan] * len(m))

        if ("energy_cum_pred" in m.columns and m["energy_cum_pred"].notna().any()
                and "step_energy_real" in m.columns and m["step_energy_real"].notna().any()):
            step_pred = m["energy_cum_pred"].astype(float).diff().fillna(0.0)
            energy_step_err = np.abs(step_pred - m["step_energy_real"].astype(float))
        else:
            energy_step_err = pd.Series([np.nan] * len(m))

        tmp = pd.DataFrame({
            "t": m["t"].astype(int),
            f"pos_err_{name}": pos_err.to_numpy(dtype=float),
            f"speed_err_{name}": speed_err.to_numpy(dtype=float),
            f"energy_step_err_{name}": energy_step_err.to_numpy(dtype=float),
        })

        per_second = per_second.merge(tmp, on="t", how="left")

        pm = _metrics(pos_err.to_numpy(dtype=float))
        sm = _metrics(speed_err.to_numpy(dtype=float))
        em = _metrics(energy_step_err.to_numpy(dtype=float))

        summary_rows.append({
            "model": name,
            "n_sec": int(len(m)),
            "best_real_start_t": int(t0),
            "best_shift_s": int(sh),
            "window_rmse_m": float(rmse0),
            "pos_MAE_m": pm["MAE"],
            "pos_RMSE_m": pm["RMSE"],
            "pos_P95_m": pm["P95"],
            "pos_max_m": pm["MAX"],
            "speed_MAE_mps": sm["MAE"],
            "speed_RMSE_mps": sm["RMSE"],
            "speed_P95_mps": sm["P95"],
            "speed_max_mps": sm["MAX"],
            "energy_step_MAE": em["MAE"],
            "energy_step_RMSE": em["RMSE"],
            "energy_step_P95": em["P95"],
            "energy_step_MAX": em["MAX"],
        })

    if not summary_rows:
        raise RuntimeError("No models produced valid evaluation rows. Check files/columns.")

    summary = pd.DataFrame(summary_rows).sort_values("pos_RMSE_m")
    summary_csv = OUT_DIR / "model_vs_real_summary_full1.csv"
    per_csv = OUT_DIR / "per_second_error_full1.csv"
    summary.to_csv(summary_csv, index=False)
    per_second.to_csv(per_csv, index=False)

    # plots
    plt.figure()
    for name in summary["model"].tolist():
        col = f"pos_err_{name}"
        if col in per_second.columns:
            plt.plot(per_second["t"], per_second[col], label=name)
    plt.xlabel("t (s)")
    plt.ylabel("Position error (m)")
    plt.title("Per-second XY position error (mission window)")
    plt.legend()
    pos_png = OUT_DIR / "pos_error_full.png"
    plt.savefig(pos_png, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    for name in summary["model"].tolist():
        col = f"speed_err_{name}"
        if col in per_second.columns:
            plt.plot(per_second["t"], per_second[col], label=name)
    plt.xlabel("t (s)")
    plt.ylabel("Speed abs error (m/s)")
    plt.title("Per-second speed error (mission window)")
    plt.legend()
    spd_png = OUT_DIR / "speed_error_full.png"
    plt.savefig(spd_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[saved] {summary_csv}")
    print(f"[saved] {per_csv}")
    print(f"[saved] {pos_png}")
    print(f"[saved] {spd_png}")
    print("\nTOP by position RMSE:")
    print(summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()