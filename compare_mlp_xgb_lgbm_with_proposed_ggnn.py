# -*- coding: utf-8 -*-
"""
safe_learning_baselines_eswa.py

FINAL ESWA REVISION BASELINE SCRIPT

Purpose:
- Compact planner-safe edge-cost baseline comparison.
- Models:
    1) MLP
    2) XGBoost
    3) LightGBM

Important:
- This script is for supporting baseline table only.
- Main proposed GGNN result comes separately from main_run.py.
- This script avoids target leakage by excluding:
    time_s, energy_J, mean_speed, mean_power_W, samples, energy_J_per_m
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 11

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "inductive_folder_new_data"

if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR

OUT_DIR = DATA_DIR / "eswa_learning_baselines_safe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_file(candidates):
    for name in candidates:
        p1 = DATA_DIR / name
        if p1.exists():
            return p1

        p2 = BASE_DIR / name
        if p2.exists():
            return p2

    raise FileNotFoundError(f"Could not find any of these files: {candidates}")


def load_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")

    df.columns = [c.strip() for c in df.columns]

    if "Node" not in df.columns:
        raise ValueError(f"Node file must contain 'Node'. Found: {df.columns.tolist()}")

    df["Node"] = pd.to_numeric(df["Node"], errors="coerce")
    df = df.dropna(subset=["Node"]).copy()
    df["Node"] = df["Node"].astype(int)

    for c in df.columns:
        if c != "Node":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_dataset(edge_samples: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    df = edge_samples.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["u_node_id", "v_node_id", "edge_distance", "time_s", "energy_J"]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column in edge_samples.csv: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    df["u_node_id"] = df["u_node_id"].astype(int)
    df["v_node_id"] = df["v_node_id"].astype(int)

    node_cols = [c for c in nodes.columns if c != "Node"]

    nodes_u = nodes[["Node"] + node_cols].copy()
    nodes_v = nodes[["Node"] + node_cols].copy()

    nodes_u = nodes_u.rename(
        columns={"Node": "u_node_id", **{c: f"u_{c}" for c in node_cols}}
    )

    nodes_v = nodes_v.rename(
        columns={"Node": "v_node_id", **{c: f"v_{c}" for c in node_cols}}
    )

    df = df.merge(nodes_u, on="u_node_id", how="left")
    df = df.merge(nodes_v, on="v_node_id", how="left")

    if "u_X-coordinate" in df.columns and "v_X-coordinate" in df.columns:
        df["dx"] = df["v_X-coordinate"] - df["u_X-coordinate"]
        df["dy"] = df["v_Y-coordinate"] - df["u_Y-coordinate"]
        df["abs_dx"] = df["dx"].abs()
        df["abs_dy"] = df["dy"].abs()

        angle = np.arctan2(df["dy"], df["dx"])
        df["angle_sin"] = np.sin(angle)
        df["angle_cos"] = np.cos(angle)

    return df


def get_planner_safe_features(df: pd.DataFrame):
    """
    Only features available before planning.

    Do NOT add:
    time_s, energy_J, mean_speed, mean_power_W, samples, energy_J_per_m.
    """

    candidate_features = [
        "edge_distance",

        "dx",
        "dy",
        "abs_dx",
        "abs_dy",
        "angle_sin",
        "angle_cos",

        "u_X-coordinate",
        "u_Y-coordinate",
        "u_Node_Degree",
        "u_charging_flag",
        "u_Type_Corridor",
        "u_Type_Intersection",
        "u_Type_Station",
        "u_dist_mean",
        "u_dist_min",
        "u_dist_max",

        "v_X-coordinate",
        "v_Y-coordinate",
        "v_Node_Degree",
        "v_charging_flag",
        "v_Type_Corridor",
        "v_Type_Intersection",
        "v_Type_Station",
        "v_dist_mean",
        "v_dist_min",
        "v_dist_max",
    ]

    features = [c for c in candidate_features if c in df.columns]

    if not features:
        raise ValueError("No planner-safe features found. Check Node_F3.csv columns.")

    return features


def split_by_edge_id(df: pd.DataFrame):
    """
    Edge-based split.
    Same directed edge ID is not allowed in both train and test.
    """

    data = df.copy()
    data["edge_id"] = data["u_node_id"].astype(str) + "_" + data["v_node_id"].astype(str)

    unique_edges = np.array(data["edge_id"].unique())

    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_edges)

    n = len(unique_edges)

    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n))

    train_edges = set(unique_edges[:n_train])
    val_edges = set(unique_edges[n_train:n_train + n_val])
    test_edges = set(unique_edges[n_train + n_val:])

    trainval = data[data["edge_id"].isin(train_edges | val_edges)].reset_index(drop=True)
    test = data[data["edge_id"].isin(test_edges)].reset_index(drop=True)

    return trainval, test


def metric_dict(y_true, y_pred):
    result = {}

    for idx, name in enumerate(["time", "energy"]):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        result[f"{name}_MAE"] = float(mean_absolute_error(yt, yp))
        result[f"{name}_RMSE"] = float(math.sqrt(mean_squared_error(yt, yp)))

        if len(np.unique(yt)) > 1:
            result[f"{name}_R2"] = float(r2_score(yt, yp))
        else:
            result[f"{name}_R2"] = np.nan

    return result


def train_mlp(X_train, y_train, X_test):
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
        ),
    )

    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_xgboost(X_train, y_train, X_test):
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        warnings.warn(f"XGBoost missing. Install: pip install xgboost. Error: {e}")
        return None

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=SEED,
            reg_lambda=1.0,
            n_jobs=1,
        )
    )

    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_lightgbm(X_train, y_train, X_test):
    try:
        from lightgbm import LGBMRegressor
    except Exception as e:
        warnings.warn(f"LightGBM missing. Install: pip install lightgbm. Error: {e}")
        return None

    model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
        )
    )

    model.fit(X_train, y_train)
    return model.predict(X_test)


def add_result(rows, detail_rows, model_name, pred, y_test, test, n_trainval, n_features):
    row = {
        "model": model_name,
        "feature_mode": "planner_safe",
        "n_trainval_samples": int(n_trainval),
        "n_test_samples": int(len(test)),
        "n_features": int(n_features),
    }

    row.update(metric_dict(y_test, pred))
    rows.append(row)

    detail = test[["u_node_id", "v_node_id", "edge_distance", "time_s", "energy_J"]].copy()
    detail["model"] = model_name
    detail["pred_time_s"] = pred[:, 0]
    detail["pred_energy_J"] = pred[:, 1]

    detail_rows.append(detail)


def main():
    edge_samples_path = find_file([
        "edge_samples.csv",
        "edge_samples(2).csv",
        "edge_samples(1).csv",
    ])

    node_path = find_file([
        "Node_F3.csv",
        "Node_F3(6).csv",
        "Node_F3(5).csv",
        "Node_F3(4).csv",
        "Node_F3(3).csv",
        "Node_F3(2).csv",
        "Node_F3(1).csv",
    ])

    print(f"[load] edge_samples={edge_samples_path}")
    print(f"[load] nodes={node_path}")

    edge_samples = pd.read_csv(edge_samples_path)
    nodes = load_nodes(node_path)

    df = build_dataset(edge_samples, nodes)
    features = get_planner_safe_features(df)

    for c in features + ["time_s", "energy_J"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=features + ["time_s", "energy_J"]).reset_index(drop=True)

    trainval, test = split_by_edge_id(df)

    print(f"[data] total_samples={len(df)}")
    print(f"[data] trainval_samples={len(trainval)}")
    print(f"[data] test_samples={len(test)}")
    print(f"[data] trainval_edges={trainval['edge_id'].nunique()}")
    print(f"[data] test_edges={test['edge_id'].nunique()}")
    print(f"[data] n_features={len(features)}")

    X_train = trainval[features].to_numpy(dtype=float)
    y_train = trainval[["time_s", "energy_J"]].to_numpy(dtype=float)

    X_test = test[features].to_numpy(dtype=float)
    y_test = test[["time_s", "energy_J"]].to_numpy(dtype=float)

    rows = []
    detail_rows = []

    print("[train] MLP")
    pred_mlp = train_mlp(X_train, y_train, X_test)
    add_result(rows, detail_rows, "MLP", pred_mlp, y_test, test, len(trainval), len(features))

    print("[train] XGBoost")
    pred_xgb = train_xgboost(X_train, y_train, X_test)
    if pred_xgb is not None:
        add_result(rows, detail_rows, "XGBoost", pred_xgb, y_test, test, len(trainval), len(features))

    print("[train] LightGBM")
    pred_lgbm = train_lightgbm(X_train, y_train, X_test)
    if pred_lgbm is not None:
        add_result(rows, detail_rows, "LightGBM", pred_lgbm, y_test, test, len(trainval), len(features))

    summary = pd.DataFrame(rows)

    if detail_rows:
        detail_all = pd.concat(detail_rows, ignore_index=True)
    else:
        detail_all = pd.DataFrame()

    summary_path = OUT_DIR / "eswa_learning_baselines_summary.csv"
    detail_path = OUT_DIR / "eswa_learning_baselines_detail.csv"
    feature_path = OUT_DIR / "eswa_learning_baselines_features.txt"

    summary.to_csv(summary_path, index=False)
    detail_all.to_csv(detail_path, index=False)

    feature_lines = [
        "FEATURE_MODE=planner_safe",
        "MODELS=MLP, XGBoost, LightGBM",
        "",
        "This is a compact supporting baseline check for ESWA revision.",
        "This is not a comprehensive estimator-family benchmark.",
        "Main proposed GGNN is evaluated separately using main_run.py.",
        "",
        "EXCLUDED_FROM_FEATURES:",
        "- observed traversal duration / time_s",
        "- observed energy / energy_J",
        "- observed mean speed",
        "- observed mean power",
        "- duration-derived power",
        "- energy-per-meter",
        "- traversal sample count",
        "",
        "FEATURES_USED:",
    ]

    feature_lines.extend(features)
    feature_path.write_text("\n".join(feature_lines), encoding="utf-8")

    print("\n[saved]", summary_path)
    print("[saved]", detail_path)
    print("[saved]", feature_path)

    print("\n========== ESWA PLANNER-SAFE BASELINE SUMMARY ==========")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()