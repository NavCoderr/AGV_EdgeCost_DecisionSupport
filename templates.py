# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import math 

from utils_io import abs_path, read_csv_auto
import config as C

# GraphData import (SAFE)
try:
    from graph_data import GraphData  # type: ignore
except Exception:
    GraphData = Any  # fallback so runtime does not crash

# Small helpers
def norm_colname(name: str) -> str:
    """Normalize column names for robust matching."""
    s = str(name).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s


# Geometry augmentation helpers
def _node_xy(G: GraphData, node_id: int):
    """Return (x,y) for node_id from GraphData."""
    i = getattr(G, "node_id_to_idx", {}).get(int(node_id), None)
    if i is None:
        return None
    return float(G.node_x[i]), float(G.node_y[i])


def _straight_polyline(u: int, v: int, G: GraphData, K: int = 100) -> np.ndarray:
    a = _node_xy(G, u)
    b = _node_xy(G, v)
    if a is None or b is None:
        return None
    x0, y0 = a
    x1, y1 = b
    xs = np.linspace(x0, x1, int(K))
    ys = np.linspace(y0, y1, int(K))
    return np.stack([xs, ys], axis=1)


def _resample_polyline(pts: np.ndarray, K: int = 100) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.repeat(pts[:1], int(K), axis=0) if pts.shape[0] else np.zeros((int(K), 2), dtype=float)
    seg = np.diff(pts, axis=0)
    d = np.sqrt((seg ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-9:
        return np.repeat(pts[:1], int(K), axis=0)
    t = np.linspace(0.0, total, int(K))
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def fill_geom_missing_with_straight(
    templates: Dict[Tuple[int, int], np.ndarray],
    G: GraphData,
    K: int = 100,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Fill any missing directed edges in G.edge_set with straight polylines."""
    K = int(K)
    added = 0
    for (u, v) in getattr(G, "edge_set", set()):
        if (u, v) in templates:
            continue
        pts = _straight_polyline(int(u), int(v), G, K=K)
        if pts is not None:
            templates[(int(u), int(v))] = pts
            added += 1
    if added:
        print(f"[geom_templates] filled_straight={added} (K={K})")
    return templates


def augment_geom_from_raw_excel(
    raw_xlsx_path,
    templates: Dict[Tuple[int, int], np.ndarray],
    G: GraphData,
    K: int = 100,
    min_points: int = 10,
) -> Dict[Tuple[int, int], np.ndarray]:
    raw_xlsx_path = str(raw_xlsx_path)
    try:
        df = pd.read_excel(raw_xlsx_path)
    except Exception as e:
        print(f"[geom_raw] could not read raw excel: {raw_xlsx_path} err={e}")
        return templates

    # detect XY columns
    cols = {norm_colname(c): c for c in df.columns}
    x_c = cols.get("x-coordinate") or cols.get("x") or cols.get("xcoord") or cols.get("x_coordinate")
    y_c = cols.get("y-coordinate") or cols.get("y") or cols.get("ycoord") or cols.get("y_coordinate")
    if x_c is None or y_c is None:
        print("[geom_raw] missing X/Y columns in raw excel -> skip")
        return templates

    xy = df[[x_c, y_c]].to_numpy(dtype=float)

    node_xy = np.stack([np.asarray(G.node_x, dtype=float), np.asarray(G.node_y, dtype=float)], axis=1)  # (N,2)
    node_ids = np.asarray(G.node_ids, dtype=int)

    # snap each raw point to nearest node (vectorized)
    d2 = ((xy[:, None, :] - node_xy[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    snapped = node_ids[idx]  # (M,)

    # compress consecutive duplicates
    keep = np.ones(len(snapped), dtype=bool)
    keep[1:] = snapped[1:] != snapped[:-1]
    comp_nodes = snapped[keep]
    comp_idx = np.nonzero(keep)[0]

    # collect segments (u->v with enough points)
    seg_map: Dict[Tuple[int, int], List[np.ndarray]] = {}
    min_points = int(min_points)

    for i in range(len(comp_nodes) - 1):
        u = int(comp_nodes[i])
        v = int(comp_nodes[i + 1])
        a = int(comp_idx[i])
        b = int(comp_idx[i + 1])
        if b - a < min_points:
            continue
        if (u, v) in templates:
            continue
        pts = xy[a:b]
        seg_map.setdefault((u, v), []).append(_resample_polyline(pts, K=int(K)))

    added = 0
    added_rev = 0

    # aggregate by median across traversals
    for (u, v), arrs in seg_map.items():
        if (u, v) in templates:
            continue
        stack = np.stack(arrs, axis=0)  # (T,K,2)
        med = np.median(stack, axis=0)
        templates[(u, v)] = med
        added += 1

        # reverse-fill if opposite edge exists and is missing
        if (v, u) in getattr(G, "edge_set", set()) and (v, u) not in templates:
            templates[(v, u)] = med[::-1].copy()
            added_rev += 1

    if added or added_rev:
        print(f"[geom_raw] added={added} reverse_added={added_rev} from {raw_xlsx_path} (K={int(K)} min_pts={min_points})")

    return templates

# Geometry template extraction


def save_geom_templates(
    templates: Dict[Tuple[int, int], np.ndarray],
    G: GraphData,
    out_csv,
    K: int,
    n_samples_used: Dict[Tuple[int, int], int] = None,
) -> None:
    """Save templates dict to CSV in the standard (u,v,k_idx,x,y,...) format.
    Any polyline not already length-K will be resampled to K points.
    """
    K = int(K)
    if n_samples_used is None:
        n_samples_used = {}
    rows = []
    for (u, v), pts in templates.items():
        u = int(u); v = int(v)
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            pts = _straight_polyline(u, v, G, K=K)
            if pts is None:
                continue
        if pts.shape[0] != K:
            pts = _resample_polyline(pts, K=K).astype(np.float32)
        ns = int(n_samples_used.get((u, v), 0))
        for kk in range(K):
            rows.append({
                "u_node_id": u,
                "v_node_id": v,
                "k_idx": int(kk),
                "x": float(pts[kk, 0]),
                "y": float(pts[kk, 1]),
                "n_samples_used": ns,
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] geom_templates -> {abs_path(out_csv)} edges={len(templates)} K={K}")

def resample_polyline_to_k(points_xy: np.ndarray, k: int) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.shape[0] < 2:
        return np.repeat(points_xy[:1], k, axis=0)

    dx = np.diff(points_xy[:, 0])
    dy = np.diff(points_xy[:, 1])
    seg = np.sqrt(dx * dx + dy * dy)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    if total < 1e-9:
        return np.repeat(points_xy[:1], k, axis=0)

    target = np.linspace(0.0, total, k)
    out = np.zeros((k, 2), dtype=np.float32)

    j = 0
    for i, t in enumerate(target):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        t0, t1 = cum[j], cum[j + 1]
        p0, p1 = points_xy[j], points_xy[j + 1]
        if t1 <= t0:
            out[i] = p0
        else:
            a = (t - t0) / (t1 - t0)
            out[i] = (1.0 - a) * p0 + a * p1
    return out

def extract_geom_templates(
    df_1s_snapped: pd.DataFrame,
    G: GraphData,
    out_csv,
    K: int
) -> Dict[Tuple[int, int], np.ndarray]:

    """Build geom_templates **for all edges in the graph**.

    Fixes:
    - If (1->7) exists but (7->1) missing => create BOTH using reverse curve.
    - If neither direction appears in 1Hz data => create straight template.
    - Saved CSV will contain templates for every directed edge in G.edge_set
      (e.g., 102 edges), not only observed edges (e.g., 58).
    """

    d = df_1s_snapped.copy()

    # ensure numeric types
    node_ids = pd.to_numeric(d["node_id"], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(d["X-coordinate"], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(d["Y-coordinate"], errors="coerce").to_numpy(dtype=float)

    sp = pd.to_numeric(d.get("Speed", np.nan), errors="coerce").to_numpy(dtype=float)
    sp = np.where(np.isfinite(sp), sp, 0.0)
    if getattr(C, "DROP_NEG_SPEED", True):
        sp = np.where(sp < 0, 0.0, sp)
    sp = np.where(sp < getattr(C, "MIN_MOVE_SPEED", 0.05), 0.0, sp)

    # 1) collect per-directed-edge samples from the 1Hz traces
    samples_by_dir: Dict[Tuple[int, int], List[np.ndarray]] = {}
    n = len(d)
    i = 0

    while i < n - 2:
        if not np.isfinite(node_ids[i]):
            i += 1
            continue

        u = int(node_ids[i])
        j = i
        while j < n and np.isfinite(node_ids[j]) and int(node_ids[j]) == u:
            j += 1

        if j >= n or (not np.isfinite(node_ids[j])):
            i = j
            continue

        v = int(node_ids[j])
        if v == u:
            i = j
            continue

        # accept only edges that exist in graph
        if (u, v) not in getattr(G, "edge_set", set()):
            i = j
            continue

        # include a few samples into the new node so we get enough points
        k = j
        while k < n and np.isfinite(node_ids[k]) and int(node_ids[k]) == v and (k - j) < 3:
            k += 1

        seg_len = k - i
        if seg_len > getattr(C, "MAX_SEG_SECONDS", 9999):
            i = j
            continue

        seg_x = xs[i:k]
        seg_y = ys[i:k]
        seg_sp = sp[i:k]

        good_xy = np.isfinite(seg_x) & np.isfinite(seg_y)
        good_move = np.isfinite(seg_sp) & (seg_sp >= getattr(C, "MIN_MOVE_SPEED", 0.05))
        good = good_xy & good_move
        if not good.any():
            i = j
            continue

        seg_xy = np.stack([seg_x[good], seg_y[good]], axis=1).astype(np.float32)
        if seg_xy.shape[0] >= max(4, int(getattr(C, "MIN_MOVE_POINTS_PER_EDGE", 6))):
            tpl = resample_polyline_to_k(seg_xy, int(K))
            samples_by_dir.setdefault((u, v), []).append(tpl)

        i = j

    # 2) merge into UNDIRECTED canonical edges (min,max) so (u,v) and (v,u) share a curve
    undirected_samples: Dict[Tuple[int, int], List[np.ndarray]] = {}
    undirected_counts: Dict[Tuple[int, int], int] = {}

    for (u, v), lst in samples_by_dir.items():
        a, b = (u, v) if u < v else (v, u)
        for tpl in lst:
            tpl_ab = tpl if (u == a and v == b) else tpl[::-1].copy()
            undirected_samples.setdefault((a, b), []).append(tpl_ab)
            undirected_counts[(a, b)] = undirected_counts.get((a, b), 0) + 1

    undirected_tpl: Dict[Tuple[int, int], np.ndarray] = {}
    for (a, b), arrs in undirected_samples.items():
        stack = np.stack(arrs, axis=0)  # (T,K,2)
        med = np.median(stack, axis=0).astype(np.float32)
        undirected_tpl[(a, b)] = med

    # 3) expand to ALL directed edges; if missing in data -> straight line
    templates: Dict[Tuple[int, int], np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    edge_set = getattr(G, "edge_set", set())
    K = int(K)

    edge_list = sorted([(int(u), int(v)) for (u, v) in edge_set], key=lambda t: (t[0], t[1]))

    straight_filled = 0
    copied_reversed = 0

    for (u, v) in edge_list:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in undirected_tpl:
            base = undirected_tpl[(a, b)]
            tpl_uv = base if (u == a and v == b) else base[::-1].copy()
            if (u != a or v != b):
                copied_reversed += 1
            templates[(u, v)] = tpl_uv
            n_used = int(undirected_counts.get((a, b), 1))
        else:
            pts = _straight_polyline(u, v, G, K=K)
            if pts is None:
                pts = np.zeros((K, 2), dtype=np.float32)
            templates[(u, v)] = np.asarray(pts, dtype=np.float32)
            n_used = 0
            straight_filled += 1

        tpl = templates[(u, v)]
        for kk in range(K):
            rows.append({
                "u_node_id": int(u),
                "v_node_id": int(v),
                "k_idx": int(kk),
                "x": float(tpl[kk, 0]),
                "y": float(tpl[kk, 1]),
                "n_samples_used": int(n_used),
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(
        f"[saved] geom_templates -> {abs_path(out_csv)} "
        f"edges={len(templates)} K={K} "
        f"(undirected_curves={len(undirected_tpl)} reversed_copies={copied_reversed} straight_filled={straight_filled})"
    )
    return templates



def load_geom_templates(path, G: GraphData) -> Dict[Tuple[int, int], np.ndarray]:
    path = str(path)
    try:
        df = read_csv_auto(path)
    except Exception:
        return {}

    need = {"u_node_id", "v_node_id", "k_idx", "x", "y"}
    if not need.issubset(set(df.columns)):
        print("[warn] geom_templates columns mismatch -> ignoring")
        return {}

    templates: Dict[Tuple[int, int], np.ndarray] = {}
    for (u, v), g in df.groupby(["u_node_id", "v_node_id"]):
        u = int(u); v = int(v)
        if (u, v) not in G.edge_set:
            continue
        g = g.sort_values("k_idx")
        pts = g[["x", "y"]].to_numpy(dtype=np.float32)
        if pts.shape[0] >= 2:
            templates[(u, v)] = pts

    # reverse-fill if opposite exists but missing
    for (u, v), pts in list(templates.items()):
        if (v, u) in G.edge_set and (v, u) not in templates:
            templates[(v, u)] = pts[::-1].copy()

    # optional raw augmentation
    if getattr(C, "GEOM_AUGMENT_FROM_RAW", False):
        raw_path = getattr(C, "DATA_1HZ_CSV", None)
        if raw_path is not None and str(raw_path).lower().endswith((".xlsx", ".xls")):
            templates = augment_geom_from_raw_excel(
                raw_path,
                templates,
                G,
                K=int(getattr(C, "GEOM_K", 100)),
                min_points=int(getattr(C, "GEOM_RAW_MIN_POINTS", 10)),
            )

    # optional straight-fill for remaining missing edges
    if getattr(C, "GEOM_FILL_STRAIGHT_MISSING", False):
        templates = fill_geom_missing_with_straight(templates, G, K=int(getattr(C, "GEOM_K", 100)))

    print(f"[geom_templates] loaded={len(templates)} from {path}")
    return templates


# Tau template helpers
def polyline_cumlen(pts: np.ndarray) -> np.ndarray:
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_point_to_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return 0.0, ax, ay
    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    return t, ax + t * abx, ay + t * aby


def progress_on_polyline(xy, pts: np.ndarray) -> float:
    px, py = xy
    cum = polyline_cumlen(pts)
    total = float(cum[-1]) if len(cum) else 0.0
    if total <= 1e-9:
        return 0.0

    best_d2 = 1e18
    best_s = 0.0

    for i in range(len(pts) - 1):
        ax, ay = float(pts[i, 0]), float(pts[i, 1])
        bx, by = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        t, xp, yp = project_point_to_segment(px, py, ax, ay, bx, by)
        d2 = (px - xp) ** 2 + (py - yp) ** 2
        if d2 < best_d2:
            seg_len = math.dist((ax, ay), (bx, by))
            best_d2 = d2
            best_s = float(cum[i] + t * seg_len)

    frac = max(0.0, min(1.0, best_s / total))
    return frac


def resample_curve_tau(frac_list: np.ndarray, M: int) -> np.ndarray:
    dt = len(frac_list)
    if dt <= 1:
        return np.linspace(0.0, 1.0, int(M)).astype(np.float32)

    tau_src = np.linspace(0.0, 1.0, dt)
    tau_tgt = np.linspace(0.0, 1.0, int(M))
    frac_res = np.interp(tau_tgt, tau_src, frac_list).astype(np.float32)

    frac_res = np.maximum.accumulate(frac_res)
    frac_res = np.clip(frac_res, 0.0, 1.0)
    frac_res[-1] = 1.0
    return frac_res


from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd

# assumes these exist in your project:
# - abs_path
# - read_csv_auto
# - progress_on_polyline
# - resample_curve_tau
# - GraphData
# - config as C


def build_tau_templates(
    df_1s_snapped: pd.DataFrame,
    G: GraphData,
    geom_templates: Dict[Tuple[int, int], np.ndarray],
    out_csv,
    M: int
) -> Dict[Tuple[int, int], np.ndarray]:

    d = df_1s_snapped.copy()

    node_ids = pd.to_numeric(d["node_id"], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(d["X-coordinate"], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(d["Y-coordinate"], errors="coerce").to_numpy(dtype=float)

    speed_arr = pd.to_numeric(d.get("Speed", np.nan), errors="coerce").to_numpy(dtype=float)
    speed_arr = np.where(np.isfinite(speed_arr), speed_arr, 0.0)
    if getattr(C, "DROP_NEG_SPEED", True):
        speed_arr = np.where(speed_arr < 0, 0.0, speed_arr)
    speed_arr = np.where(speed_arr < getattr(C, "MIN_MOVE_SPEED", 0.05), 0.0, speed_arr)

    edge_set = getattr(G, "edge_set", set())

    samples_by_dir: Dict[Tuple[int, int], List[np.ndarray]] = {}
    n = len(d)
    i = 0

    while i < n - 2:
        if not np.isfinite(node_ids[i]):
            i += 1
            continue

        u = int(node_ids[i])
        j = i
        while j < n and np.isfinite(node_ids[j]) and int(node_ids[j]) == u:
            j += 1

        if j >= n or (not np.isfinite(node_ids[j])):
            i = j
            continue

        v = int(node_ids[j])
        if v == u:
            i = j
            continue

        if (u, v) not in edge_set:
            i = j
            continue

        k = j
        while k < n and np.isfinite(node_ids[k]) and int(node_ids[k]) == v and (k - j) < 3:
            k += 1

        dt_total = k - i
        if dt_total <= 1 or dt_total > getattr(C, "MAX_SEG_SECONDS", 9999):
            i = j
            continue

        pts = geom_templates.get((u, v), None)
        if pts is None:
            if (u in G.node_id_to_idx) and (v in G.node_id_to_idx):
                ui = G.node_id_to_idx[u]
                vi = G.node_id_to_idx[v]
                pts = np.array([
                    [float(G.node_x[ui]), float(G.node_y[ui])],
                    [float(G.node_x[vi]), float(G.node_y[vi])],
                ], dtype=np.float32)
            else:
                i = j
                continue

        frac_seq = []
        for t in range(i, k):
            if (not np.isfinite(xs[t])) or (not np.isfinite(ys[t])) or (not np.isfinite(speed_arr[t])) or (speed_arr[t] < getattr(C, "MIN_MOVE_SPEED", 0.05)):
                frac_seq.append(np.nan)
            else:
                frac_seq.append(float(progress_on_polyline((float(xs[t]), float(ys[t])), pts)))

        frac_seq = np.array(frac_seq, dtype=float)
        if np.all(~np.isfinite(frac_seq)):
            i = j
            continue

        frac_seq = pd.Series(frac_seq).interpolate().ffill().bfill().to_numpy(dtype=float)
        frac_seq = pd.Series(frac_seq).rolling(window=3, center=True, min_periods=1).median().to_numpy()
        frac_seq = np.clip(frac_seq, 0.0, 1.0)
        frac_seq = np.maximum.accumulate(frac_seq)
        frac_seq[-1] = 1.0

        delta = np.diff(np.concatenate([[0.0], frac_seq]))
        delta = np.clip(delta, 0.0, 1.0)
        cap = float(np.percentile(delta, 95))
        cap = max(cap, 1e-6)
        delta = np.minimum(delta, cap)
        ssum = float(np.sum(delta))
        if ssum <= 1e-9:
            i = j
            continue
        delta = delta / ssum

        frac_seq = np.cumsum(delta)

        #force start=0 end=1 (reverse symmetry safe)
        f0 = float(frac_seq[0])
        den = (1.0 - f0)
        if np.isfinite(f0) and den > 1e-6:
            frac_seq = (frac_seq - f0) / den

        frac_seq = np.clip(frac_seq, 0.0, 1.0)
        frac_seq = np.maximum.accumulate(frac_seq)
        frac_seq[0] = 0.0
        frac_seq[-1] = 1.0

        frac_tau = resample_curve_tau(frac_seq, int(M))
        samples_by_dir.setdefault((u, v), []).append(frac_tau.astype(np.float32))

        i = j

    # linear fallback if nothing observed
    if not samples_by_dir:
        full_linear = np.linspace(0.0, 1.0, int(M)).astype(np.float32)
        rows = []
        full = {}
        for (u, v) in sorted([(int(a), int(b)) for (a, b) in edge_set], key=lambda t: (t[0], t[1])):
            if u == v:
                continue
            full[(u, v)] = full_linear.copy()
            for mi in range(int(M)):
                tau = mi / float(int(M) - 1) if int(M) > 1 else 1.0
                rows.append({
                    "u_node_id": int(u),
                    "v_node_id": int(v),
                    "tau_idx": int(mi),
                    "tau": float(tau),
                    "frac": float(full_linear[mi]),
                    "n_samples_used": 0,
                })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[saved] tau_templates -> {abs_path(out_csv)} edges={len(full)} M={int(M)} (linear fallback, no observed)")
        return full

 

    # 2) Merge to UNDIRECTED canonical curves
    #    so (u,v) and (v,u) share one tau
    undirected_samples: Dict[Tuple[int, int], List[np.ndarray]] = {}
    undirected_counts: Dict[Tuple[int, int], int] = {}

    for (u, v), lst in samples_by_dir.items():
        a, b = (u, v) if u < v else (v, u)
        for frac_tau in lst:
            # map direction to canonical (a->b)
            frac_ab = frac_tau if (u == a and v == b) else (1.0 - frac_tau[::-1]).copy()
            undirected_samples.setdefault((a, b), []).append(frac_ab.astype(np.float32))
            undirected_counts[(a, b)] = undirected_counts.get((a, b), 0) + 1

    undirected_med: Dict[Tuple[int, int], np.ndarray] = {}
    for (a, b), lst in undirected_samples.items():
        arr = np.stack(lst, axis=0)  # (T,M)
        med = np.median(arr, axis=0).astype(np.float32)
        med = np.maximum.accumulate(med)
        med = np.clip(med, 0.0, 1.0)
        med[-1] = 1.0
        undirected_med[(a, b)] = med

    # global fallback (for never-seen edges)
    global_fb = global_tau_fallback({k: v for k, v in undirected_med.items()}, int(M))

    # 3) Expand to ALL directed edges in G
    full: Dict[Tuple[int, int], np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    reversed_copies = 0
    fallback_filled = 0

    for (u, v) in sorted([(int(a), int(b)) for (a, b) in edge_set], key=lambda t: (t[0], t[1])):
        if u == v:
            continue

        a, b = (u, v) if u < v else (v, u)

        if (a, b) in undirected_med:
            base = undirected_med[(a, b)]
            n_used = int(undirected_counts.get((a, b), 1))
        else:
            base = global_fb
            n_used = 0
            fallback_filled += 1

        frac_uv = base if (u == a and v == b) else (1.0 - base[::-1]).copy()
        if not (u == a and v == b):
            reversed_copies += 1

        frac_uv = np.maximum.accumulate(frac_uv.astype(np.float32))
        frac_uv = np.clip(frac_uv, 0.0, 1.0)
        frac_uv[-1] = 1.0

        full[(u, v)] = frac_uv

        for mi in range(int(M)):
            tau = mi / float(int(M) - 1) if int(M) > 1 else 1.0
            rows.append({
                "u_node_id": int(u),
                "v_node_id": int(v),
                "tau_idx": int(mi),
                "tau": float(tau),
                "frac": float(frac_uv[mi]),
                "n_samples_used": int(n_used),
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(
        f"[saved] tau_templates -> {abs_path(out_csv)} "
        f"edges={len(full)} M={int(M)} "
        f"(undirected_curves={len(undirected_med)} reversed_copies={reversed_copies} fallback_filled={fallback_filled})"
    )
    return full


def load_tau_templates(path, G: GraphData) -> Dict[Tuple[int, int], np.ndarray]:
    path = str(path)
    try:
        df = read_csv_auto(path)
    except Exception:
        return {}

    need = {"u_node_id", "v_node_id", "tau_idx", "frac"}
    if not need.issubset(set(df.columns)):
        print("[warn] tau_templates columns mismatch -> ignoring")
        return {}

    templates: Dict[Tuple[int, int], np.ndarray] = {}
    for (u, v), g in df.groupby(["u_node_id", "v_node_id"]):
        u = int(u); v = int(v)
        if (u, v) not in G.edge_set:
            continue
        g = g.sort_values("tau_idx")
        frac = g["frac"].to_numpy(dtype=np.float32)
        if len(frac) >= 2:
            frac = np.maximum.accumulate(frac)
            frac = np.clip(frac, 0.0, 1.0)
            frac[-1] = 1.0
            templates[(u, v)] = frac

    # optional: ensure reverse exists if missing (safe)
    for (u, v), frac in list(templates.items()):
        if (v, u) in G.edge_set and (v, u) not in templates:
            templates[(v, u)] = (1.0 - frac[::-1]).copy()

    print(f"[tau_templates] loaded={len(templates)} from {path}")
    return templates


def global_tau_fallback(tau_templates: Dict[Tuple[int, int], np.ndarray], M: int) -> np.ndarray:
    if not tau_templates:
        return np.linspace(0.0, 1.0, int(M)).astype(np.float32)

    arr = np.stack(list(tau_templates.values()), axis=0)
    med = np.median(arr, axis=0).astype(np.float32)
    med = np.maximum.accumulate(med)
    med = np.clip(med, 0.0, 1.0)
    med[-1] = 1.0
    return med
