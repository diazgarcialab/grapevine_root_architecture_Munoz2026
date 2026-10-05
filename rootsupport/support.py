"""Tip-initiated paths and segment support (Supplementary Note S1).

Defaults are the paper's values.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupportParams:
    snap_dist: float = 15.0  # segment ends closer than this merge into one vertex
    max_turn_deg: float = 100.0  # theta_max
    upward_eps: float = 10.0  # a candidate may descend at most this many px (epsilon)
    confidence_gap_deg: Optional[float] = None  # minimum gap to the second-best turn; off in the paper
    length_key: str = "euclidean_len"
    max_steps: int = 10_000


def _is_terminal(rec) -> bool:
    t1, t2 = rec.get("type_1"), rec.get("type_2")
    return (t1 == "TIP" and t2 == "INNER") or (t1 == "INNER" and t2 == "TIP")


def _is_inner(rec) -> bool:
    return rec.get("type_1") == "INNER" and rec.get("type_2") == "INNER"


def _ends(rec):
    return (np.array([rec.get("x_1"), rec.get("y_1")], dtype=float),
            np.array([rec.get("x_2"), rec.get("y_2")], dtype=float))


def _valid(p0, p1) -> bool:
    return bool(np.isfinite(p0).all() and np.isfinite(p1).all())


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 and np.isfinite(n) else np.array([0.0, 0.0])


def _angle_deg(u, v) -> float:
    nu, nv = float(np.hypot(u[0], u[1])), float(np.hypot(v[0], v[1]))
    if nu == 0 or nv == 0 or not np.isfinite(nu) or not np.isfinite(nv):
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def build_graph(seg_info: Dict[int, Dict[str, Any]], snap_dist: float):
    """Snap segment ends into shared vertices (nearest vertex within snap_dist)."""
    grid = snap_dist
    valid_seg_ids, eps = [], []
    for sid, rec in seg_info.items():
        sid = int(sid)
        p0, p1 = _ends(rec)
        if not _valid(p0, p1):
            continue
        valid_seg_ids.append(sid)
        eps.append((float(p0[0]), float(p0[1]), sid, 0))
        eps.append((float(p1[0]), float(p1[1]), sid, 1))

    cell_to_nodes = defaultdict(list)
    node_centers: List[np.ndarray] = []
    endpoint_node: Dict[Tuple[int, int], int] = {}

    def cell_key(x, y):
        return (int(np.floor(x / grid)), int(np.floor(y / grid)))

    for x, y, sid, ei in eps:
        p = np.array([x, y], dtype=float)
        cx, cy = cell_key(x, y)
        best_nid, best_d = None, 1e18
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in cell_to_nodes.get((cx + dx, cy + dy), []):
                    d = float(np.hypot(*(p - node_centers[nid])))
                    if d < best_d:
                        best_d, best_nid = d, nid
        if best_nid is not None and best_d <= snap_dist:
            endpoint_node[(sid, ei)] = best_nid
            node_centers[best_nid] = 0.5 * (node_centers[best_nid] + p)
        else:
            nid = len(node_centers)
            node_centers.append(p)
            endpoint_node[(sid, ei)] = nid
            cell_to_nodes[(cx, cy)].append(nid)

    centers = np.array(node_centers, dtype=float) if node_centers else np.zeros((0, 2))
    seg_nodes, node_to_segs = {}, defaultdict(list)
    for sid in valid_seg_ids:
        n0, n1 = endpoint_node[(sid, 0)], endpoint_node[(sid, 1)]
        seg_nodes[sid] = (n0, n1)
        node_to_segs[n0].append(sid)
        node_to_segs[n1].append(sid)
    return valid_seg_ids, endpoint_node, centers, seg_nodes, node_to_segs


def trace_paths(seg_info: Dict[int, Dict[str, Any]], params: SupportParams = SupportParams()):
    """Trace one path per terminal segment. Returns (paths, seg_to_paths, path_info)."""
    valid_seg_ids, endpoint_node, centers, seg_nodes, node_to_segs = build_graph(seg_info, params.snap_dist)
    if not valid_seg_ids:
        return {}, defaultdict(set), {}

    def seg_length(rec) -> float:
        v = rec.get(params.length_key)
        try:
            v = float(v)
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
        p0, p1 = _ends(rec)
        return float(np.hypot(*(p1 - p0))) if _valid(p0, p1) else float("nan")

    def dir_from_node(sid: int, u: int):
        """Unit chord of segment `sid` leaving vertex `u`, plus the far vertex and the end flag."""
        p0, p1 = _ends(seg_info[sid])
        n0, n1 = seg_nodes[sid]
        if u == n0:
            return _unit(p1 - p0), n1, 0
        return _unit(p0 - p1), n0, 1

    paths: Dict[int, List[int]] = {}
    path_info: Dict[int, Dict[str, Any]] = {}
    pid = 0
    for start_sid in [s for s in valid_seg_ids if _is_terminal(seg_info[s])]:
        rec0 = seg_info[start_sid]
        tip_ei = 0 if rec0.get("type_1") == "TIP" else 1
        start_node = endpoint_node[(start_sid, tip_ei)]
        tip_xy = (float(rec0[f"x_{tip_ei + 1}"]), float(rec0[f"y_{tip_ei + 1}"]))

        cur_dir, cur_node, flag0 = dir_from_node(start_sid, start_node)
        path, flags, turns = [start_sid], [flag0], []
        on_path = {start_sid}
        prev_sid = start_sid
        for _ in range(params.max_steps):
            cand = [s for s in node_to_segs.get(cur_node, [])
                    if s != prev_sid and s not in on_path and _is_inner(seg_info[s])]
            if not cand:
                break
            y_cur = centers[cur_node][1]
            scored = []
            for sid in cand:
                dv, nxt, flag = dir_from_node(sid, cur_node)
                if centers[nxt][1] - y_cur > params.upward_eps:  # descends: skip
                    continue
                scored.append((_angle_deg(cur_dir, dv), sid, nxt, flag))
            if not scored:
                break
            scored.sort(key=lambda t: t[0])  # ties: lowest segment id
            best_turn, best_sid, best_nxt, best_flag = scored[0]
            if best_turn > params.max_turn_deg:
                break
            if params.confidence_gap_deg is not None and len(scored) >= 2 and (scored[1][0] - best_turn) < params.confidence_gap_deg:
                break
            turns.append(best_turn)
            path.append(best_sid)
            flags.append(best_flag)
            on_path.add(best_sid)
            prev_sid = best_sid
            cur_dir = dir_from_node(best_sid, cur_node)[0]
            cur_node = best_nxt

        # per-path metrics
        L = sum(l for l in (seg_length(seg_info[s]) for s in path) if np.isfinite(l))
        node = start_node
        for sid, f in zip(path, flags):
            n0, n1 = seg_nodes[sid]
            node = n1 if f == 0 else n0
        end_pt, tip_pt = centers[node], centers[start_node]
        straight = float(np.hypot(*(end_pt - tip_pt)))
        paths[pid] = path
        path_info[pid] = {
            "start_seg_id": int(start_sid), "start_tip_xy": tip_xy, "segment_ids": list(map(int, path)),
            "n_segments": len(path), "length_sum": float(L), "vertical_gain": float(tip_pt[1] - end_pt[1]),
            "turn_cost_sum": float(np.sum(turns)) if turns else 0.0, "max_turn": float(np.max(turns)) if turns else 0.0,
            "tortuosity": float(L / straight) if straight > 1e-6 else float("inf"),
            "end_xy": (float(end_pt[0]), float(end_pt[1])),
        }
        pid += 1

    # longest path first; cosmetic
    order = sorted(path_info, key=lambda p: path_info[p]["length_sum"], reverse=True)
    remap = {old: new for new, old in enumerate(order)}
    paths = {remap[o]: paths[o] for o in order}
    path_info = {remap[o]: path_info[o] for o in order}
    seg_to_paths: Dict[int, Set[int]] = defaultdict(set)
    for p, segs in paths.items():
        for s in segs:
            seg_to_paths[s].add(p)
    return paths, seg_to_paths, path_info


def support_counts(seg_info: Dict[int, Dict[str, Any]], seg_to_paths) -> Dict[int, int]:
    """Number of paths through each segment (0 if none)."""
    return {int(sid): len(seg_to_paths.get(int(sid), ())) for sid in seg_info}


def summarize_support(df: pd.DataFrame, length_col: str = "geodesic_len") -> Dict[str, float]:
    """Image-level, length-weighted summaries. df needs `support_count` and a length column."""
    L = df[length_col].fillna(0).astype(float)
    s = df["support_count"].fillna(0).astype(float)
    total = float(L.sum())
    if total <= 0:
        return {"total_length_px": 0.0}
    share = lambda m: float(L[m].sum() / total)
    return {
        "total_length_px": total,
        "n_segments": int(len(df)),
        "mean_support_length_weighted": float((L * s).sum() / total),
        "transport_root_fraction": share(s >= 2),
        "share_support_0": share(s == 0),
        "share_support_1": share(s == 1),
        "share_isolated": share((df.type_1 == "TIP") & (df.type_2 == "TIP")),
        "share_support_0_connected": share((s == 0) & (df.type_1 == "INNER") & (df.type_2 == "INNER")),
        "max_support": int(s.max()),
    }


def segments_with_support(df: pd.DataFrame, seg_info, params: SupportParams = SupportParams()) -> pd.DataFrame:
    """Segment table -> same table with a `support_count` column."""
    _, seg_to_paths, _ = trace_paths(seg_info, params)
    sc = support_counts(seg_info, seg_to_paths)
    out = df.copy()
    out["support_count"] = out["seg_id"].map(sc).fillna(0).astype(int)
    return out
