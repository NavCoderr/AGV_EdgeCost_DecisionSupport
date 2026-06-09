# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

MODE = "train"  # first run train; after training change to "plan"

def _get_script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd().resolve()

SCRIPT_DIR = _get_script_dir()

NODE_FILE = SCRIPT_DIR / "Node_F3.csv"
EDGE_FILE = SCRIPT_DIR / "Edge_Distances3_.csv"

DATA_1HZ_CSV = SCRIPT_DIR / "out_1hz" / "nav_1hz_move_only.csv"
REAL_1HZ_CSV = DATA_1HZ_CSV

OUT_DIR = SCRIPT_DIR / "inductive_folder_new_data"

START_XY = None
START_NODE = 5
GOAL_NODE = 9

MAIN_COST = "combo"  # "time" | "energy" | "combo"
ALPHA = 0.7

BLOCKED_NODES = []
BLOCKED_EDGES = []

PLANNER = "astar"  # "astar" | "dijkstra"
ASTAR_W = 1.2

SNAP_RADIUS_M = 2.50
SNAP_FALLBACK = "ffill"  # "ffill" or "none"
FFILL_MAX_GAP_S = 3

DROP_NEG_SPEED = True
MIN_MOVE_SPEED = 0.09

V_MAX_MPS = 0.30
V_TURN_MPS = 0.12
SLOWDOWN_GAIN = 0.40

DELTA_FRAC_MIN_RATIO = 0.20
DELTA_FRAC_MAX_RATIO = 4.00

GOAL_KEEP_LIST = None

KEEP_IDLE_STATS = True
MAX_SEG_SECONDS = 300
MIN_MOVE_POINTS_PER_EDGE = 4

GEOM_K = 100
TAU_M = 101

GEOM_AUGMENT_FROM_RAW = True
GEOM_AUGMENT_OVERWRITE = True
GEOM_RAW_MIN_POINTS = 10
GEOM_FILL_STRAIGHT_MISSING = True

SPEED_REF = None
SPEED_REF_MODE = "p95"  # "max" | "median" | "p95"
DEFAULT_FALLBACK_SPEED_MPS = 0.249

EDGE_BACKBONE = "ggnn" 
EDGE_EPOCHS = 50
EDGE_LR = 5e-3
EDGE_HIDDEN = 128
EDGE_GNN_LAYERS = 2
EDGE_DROPOUT = 0.10

EDGE_SPLIT_SEED = 5
EDGE_SPLIT_TRAIN = 0.70
EDGE_SPLIT_VAL = 0.15
EDGE_SPLIT_TEST = 0.15

USE_TEMPORAL_MLP = True

TEMP_EPOCHS = 50
TEMP_LR = 2e-3
TEMP_HIDDEN = 128
TEMP_LAYERS = 3
TEMP_BATCH = 4096
TEMP_DROPOUT = 0.10

TEMP_MODEL = "all"  # "mlp" | "gru" | "lstm" | "transformer_lite" | "physics_delta" | "all"
TEMP_MODELS = ("mlp", "gru", "lstm", "transformer_lite", "physics_delta")

TEMP_SPLIT_SEED = 5
TEMP_SPLIT_TRAIN = 0.70
TEMP_SPLIT_VAL = 0.15
TEMP_SPLIT_TEST = 0.15

DEFAULT_FALLBACK_TIME_S_PER_M = 1.0 / 0.20
DEFAULT_FALLBACK_ENERGY_J_PER_M = 1282.32
PRED_TIME_MIN_S = 1.0
PRED_ENERGY_MIN_J = 1.0

SEED = 8
DEVICE = "cpu"  # "cpu" or "cuda"

EDGE_TAIL_U_SEC = 3
USE_CACHED_GLOBAL_1S = True

# Choose ONLY ONE: 9, 11, 13, 15, 17, 19
# 9  = base9  tau ON  lookahead OFF
# 11 = base9  tau ON  lookahead ON   (+2)
# 13 = base13 tau ON  lookahead OFF
# 15 = base13 tau ON  lookahead ON   (+2)
# 17 = base17 tau ON  lookahead OFF
# 19 = base17 tau ON  lookahead ON   (+2)
TEMP_FEATURE_SET = 15

if TEMP_FEATURE_SET == 9:
    USE_TAU_FEATURE = True
    USE_LOOKAHEAD = False
    TEMP_BASE_DIM = 9
    TEMP_IN_DIM = 9
elif TEMP_FEATURE_SET == 11:
    USE_TAU_FEATURE = True
    USE_LOOKAHEAD = True
    TEMP_BASE_DIM = 9
    TEMP_IN_DIM = 11
elif TEMP_FEATURE_SET == 13:
    USE_TAU_FEATURE = True
    USE_LOOKAHEAD = False
    TEMP_BASE_DIM = 13
    TEMP_IN_DIM = 13
elif TEMP_FEATURE_SET == 15:
    USE_TAU_FEATURE = True
    USE_LOOKAHEAD = True
    TEMP_BASE_DIM = 13
    TEMP_IN_DIM = 15
elif TEMP_FEATURE_SET == 17:
    USE_TAU_FEATURE = True
    USE_LOOKAHEAD = False
    TEMP_BASE_DIM = 17
    TEMP_IN_DIM = 17
elif TEMP_FEATURE_SET == 19:
    USE_TAU_FEATURE = True
    USE_LOOKAHEAD = True
    TEMP_BASE_DIM = 17
    TEMP_IN_DIM = 19
else:
    raise ValueError("TEMP_FEATURE_SET must be one of: 9,11,13,15,17,19")