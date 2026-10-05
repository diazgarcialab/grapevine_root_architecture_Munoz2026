"""Binary mask -> skeleton -> segments with typed ends (PlantCV).

A segment end is TIP (free end of the skeleton) or INNER (cut at a branch point). Defaults are the paper's values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from plantcv import plantcv as pcv
from plantcv.plantcv._helpers import _find_tips, _iterative_prune


@dataclass(frozen=True)
class SkeletonParams:
    blur_kernel: Tuple[int, int] = (3, 3)
    close_kernel: int = 3
    prune_size: int = 50  # spurs shorter than this (contour points) are removed
    tip_pad: int = 5  # an end is TIP if a skeleton tip lies within this padding
    end_prune_size: int = 1  # pruning depth used to find the two ends of a segment


def _geodesic_length(seg_u8: np.ndarray) -> float:
    """Length of a 1-px raster: 1 per orthogonal step, sqrt(2) per diagonal."""
    s = (seg_u8 > 0).astype(np.uint8)
    ys, xs = np.nonzero(s)
    if len(xs) <= 1:
        return 0.0
    h, w = s.shape
    length, rt2 = 0.0, math.sqrt(2)
    for y, x in zip(ys, xs):
        if x + 1 < w and s[y, x + 1]:
            length += 1.0
        if y + 1 < h and s[y + 1, x]:
            length += 1.0
        if y + 1 < h and x + 1 < w and s[y + 1, x + 1]:
            length += rt2
        if y + 1 < h and x - 1 >= 0 and s[y + 1, x - 1]:
            length += rt2
    return float(length)


def _farthest_two(xy: np.ndarray):
    if xy.shape[0] <= 2:
        return list(range(xy.shape[0]))
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    a, b = np.unravel_index(np.argmax(d2), d2.shape)
    return [int(a), int(b)]


def preprocess_mask(mask: np.ndarray, params: SkeletonParams = SkeletonParams()) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, params.blur_kernel, 0)
    k = np.ones((params.close_kernel, params.close_kernel), np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def skeletonize(mask_closed: np.ndarray, params: SkeletonParams = SkeletonParams()):
    """Pruned skeleton plus one contour per segment between branch points."""
    pcv.params.debug = None
    skeleton_raw = pcv.morphology.skeletonize(mask=mask_closed)
    skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton_raw, size=params.prune_size, mask=mask_closed)
    _, segment_objects = pcv.morphology.segment_skeleton(skel_img=skeleton, mask=mask_closed)
    return skeleton, segment_objects


def segment_ends(skeleton: np.ndarray, segment_objects, params: SkeletonParams = SkeletonParams()) -> Dict[int, Dict[str, Any]]:
    """Per segment: both ends (x, y, TIP/INNER), geodesic and Euclidean length, orientation."""
    tips_img, _, _ = _find_tips(skeleton)
    H, W = skeleton.shape[:2]
    seg_info: Dict[int, Dict[str, Any]] = {}
    for i in range(len(segment_objects)):
        seg = np.zeros((H, W), np.uint8)
        cv2.drawContours(seg, segment_objects, i, 255, 1)
        geodesic_len = _geodesic_length(seg)

        pruned = _iterative_prune(seg, params.end_prune_size)
        ends = cv2.subtract(seg, pruned)
        nlab, lab = cv2.connectedComponents((ends > 0).astype(np.uint8), connectivity=8)
        endpoints = []
        for k in range(1, nlab):
            ys, xs = np.where(lab == k)
            if len(xs) == 0:
                continue
            x, y = int(round(xs.mean())), int(round(ys.mean()))
            x0, x1 = max(0, xs.min() - params.tip_pad), min(W, xs.max() + params.tip_pad + 1)
            y0, y1 = max(0, ys.min() - params.tip_pad), min(H, ys.max() + params.tip_pad + 1)
            etype = "TIP" if np.any(tips_img[y0:y1, x0:x1] > 0) else "INNER"
            endpoints.append({"x": x, "y": y, "type": etype})
        if len(endpoints) > 2:
            keep = _farthest_two(np.array([(e["x"], e["y"]) for e in endpoints], dtype=np.float32))
            endpoints = [endpoints[keep[0]], endpoints[keep[1]]]

        rec = {"geodesic_len": int(round(geodesic_len)), "euclidean_len": None,
               "orientation_deg": None, "orientation_0_90_deg": None,
               "x_1": None, "y_1": None, "type_1": None, "x_2": None, "y_2": None, "type_2": None}
        if len(endpoints) >= 1:
            rec.update(x_1=endpoints[0]["x"], y_1=endpoints[0]["y"], type_1=endpoints[0]["type"])
        if len(endpoints) == 2:
            rec.update(x_2=endpoints[1]["x"], y_2=endpoints[1]["y"], type_2=endpoints[1]["type"])
            dx, dy = rec["x_2"] - rec["x_1"], rec["y_2"] - rec["y_1"]
            rec["euclidean_len"] = int(round(math.hypot(dx, dy)))
            orientation = math.degrees(math.atan2(dy, dx)) % 180  # 0 = horizontal, 90 = vertical
            rec["orientation_deg"] = round(orientation, 1)
            rec["orientation_0_90_deg"] = round(min(orientation, 180 - orientation), 1)
        seg_info[i] = rec
    return seg_info


def mask_to_segments(mask: np.ndarray, params: SkeletonParams = SkeletonParams()):
    """Mask -> (skeleton, segment contours, seg_info dict, segment DataFrame)."""
    mask_closed = preprocess_mask(mask, params)
    skeleton, segment_objects = skeletonize(mask_closed, params)
    seg_info = segment_ends(skeleton, segment_objects, params)
    df = pd.DataFrame.from_dict(seg_info, orient="index")
    df.index.name = "seg_id"
    cols = ["x_1", "y_1", "type_1", "x_2", "y_2", "type_2", "geodesic_len", "euclidean_len", "orientation_deg", "orientation_0_90_deg"]
    return skeleton, segment_objects, seg_info, df[cols].reset_index()


def segment_pixels(skeleton: np.ndarray, segment_objects, seg_id: int) -> np.ndarray:
    """(N, 2) array of (x, y) pixels of one segment, for plotting."""
    canvas = np.zeros(skeleton.shape[:2], np.uint8)
    cv2.drawContours(canvas, segment_objects, seg_id, 255, 1)
    canvas = cv2.bitwise_and(canvas, skeleton)
    ys, xs = np.nonzero(canvas)
    return np.column_stack([xs, ys])
