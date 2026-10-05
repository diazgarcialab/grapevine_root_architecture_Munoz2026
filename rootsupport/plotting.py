"""Segments coloured by support, and overlays on the scan."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .skeleton import segment_pixels

MASK_TINT = (0.35, 0.95, 0.45, 0.28)  # translucent green


def _support_rgba(s: int, max_s: int, grey=(0.55, 0.55, 0.55)) -> Tuple[float, float, float, float]:
    # log-scaled inferno; 0 in grey (the paper draws 0 as 1)
    if s <= 0:
        return (*grey, 1.0)
    t = math.log(s) / math.log(max_s) if max_s > 1 else 0.0
    r, g, b, _ = colormaps["inferno"](0.2 + 0.7 * t)
    return (r, g, b, 1.0)


def root_window(skeleton, window=None, crop_to_roots=False, margin=100):
    """Region to draw as (x0, y0, w, h): the given window, the skeleton bbox plus margin, or the whole image."""
    H, W = skeleton.shape[:2]
    if window is not None:
        x0, y0, w = window[0], window[1], window[2]
        h = window[3] if len(window) > 3 else window[2]
        return x0, y0, min(w, W - x0), min(h, H - y0)
    if crop_to_roots:
        ys, xs = np.nonzero(skeleton)
        if len(xs):
            x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin)
            y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin)
            return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
    return 0, 0, W, H


def support_layer(skeleton, segment_objects, support: dict, window=None, width_scale: float = 1.0,
                  crop_to_roots: bool = False, margin: int = 100) -> np.ndarray:
    """RGBA layer with the segments coloured and widened by support."""
    import cv2
    x0, y0, w, h = root_window(skeleton, window, crop_to_roots, margin)
    scale = width_scale * max(1.0, max(h, w) / 600.0)
    width_fn = lambda s: max(1, int(round(scale * (1 if s == 0 else 1 + 1.2 * math.log1p(s)))))
    layer = np.zeros((h, w, 4), np.float32)
    max_s = max(support.values()) if support else 1
    for sid in sorted(support, key=lambda s: support[s]):  # high support on top
        pts = segment_pixels(skeleton, segment_objects, sid)
        if len(pts) == 0:
            continue
        sel = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + w) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + h)
        if not sel.any():
            continue
        canvas = np.zeros((h, w), np.uint8)
        canvas[pts[sel, 1] - y0, pts[sel, 0] - x0] = 255
        k = width_fn(support[sid])
        if k > 1:
            canvas = cv2.dilate(canvas, np.ones((k, k), np.uint8))
        layer[canvas > 0] = _support_rgba(support[sid], max_s)
    return layer


def plot_support_map(skeleton, segment_objects, support: dict, df=None, window=None, ax=None,
                     background="black", show_ends=None, title=None, crop_to_roots=False, margin=100, width_scale=1.0):
    """Segments coloured by support, with tips and vertices when a window is given."""
    layer = support_layer(skeleton, segment_objects, support, window, width_scale, crop_to_roots, margin)
    h, w = layer.shape[:2]
    if show_ends is None:
        show_ends = window is not None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(np.zeros((h, w, 3)) if background == "black" else np.ones((h, w, 3)))
    ax.imshow(layer)
    if show_ends and df is not None:
        x0, y0 = root_window(skeleton, window, crop_to_roots, margin)[:2]
        tips, verts = [], []
        for _, r in df.iterrows():
            for k in ("1", "2"):
                x, y, t = r[f"x_{k}"], r[f"y_{k}"], r[f"type_{k}"]
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    continue
                if x0 <= x < x0 + w and y0 <= y < y0 + h:
                    (tips if t == "TIP" else verts).append((x - x0, y - y0))
        fg = "white" if background == "black" else "black"
        if verts:
            v = np.array(verts); ax.scatter(v[:, 0], v[:, 1], s=10, c=fg, zorder=3)
        if tips:
            t = np.array(tips); ax.scatter(t[:, 0], t[:, 1], s=18, facecolors=background, edgecolors=fg, linewidths=1.2, zorder=3)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left")
    return ax


def plot_overlay(image, mask, skeleton, segment_objects, support: dict, window=None, ax=None, mask_tint=MASK_TINT, title=None,
                 crop_to_roots=False, margin=100, width_scale=1.0):
    """Scan, translucent mask and support-coloured segments (Figure S3e)."""
    layer = support_layer(skeleton, segment_objects, support, window, width_scale, crop_to_roots, margin)
    h, w = layer.shape[:2]
    x0, y0 = root_window(skeleton, window, crop_to_roots, margin)[:2]
    img = np.asarray(image)[y0:y0 + h, x0:x0 + w]
    m = np.asarray(mask)[y0:y0 + h, x0:x0 + w]
    tint = np.zeros((h, w, 4)); tint[m > 0] = mask_tint
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(img); ax.imshow(tint); ax.imshow(layer)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left")
    return ax
