"""Mask -> segments with support -> figure, for one example scan.

Default is ArUco 109 at 24 DAT (Figure S3); --date 2025-04-21 gives 38 DAT.
"""
import argparse, os, sys, time
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rootsupport import mask_to_segments, SkeletonParams, trace_paths, support_counts, summarize_support, plot_support_map, plot_overlay

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--aruco", default="109")
ap.add_argument("--date", default="2025-04-07")
ap.add_argument("--window", default="880,1520,450", help="x0,y0,size of the zoom window (Figure S3: 880,1520,450)")
ap.add_argument("--out", default=os.path.join(HERE, "output"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

scan_path = os.path.join(HERE, "data", f"aruco_{a.aruco}_date_{a.date}_aligned.jpg")
mask_path = os.path.join(HERE, "data", f"aruco_{a.aruco}_date_{a.date}_mask.png")
image = cv2.cvtColor(cv2.imread(scan_path), cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

t0 = time.time()
skeleton, segment_objects, seg_info, df = mask_to_segments(mask)
paths, seg_to_paths, path_info = trace_paths(seg_info)
support = support_counts(seg_info, seg_to_paths)
df["support_count"] = df["seg_id"].map(support).astype(int)
print(f"skeleton + support: {time.time() - t0:.1f} s; {len(df)} segments, {len(paths)} paths, max support {df.support_count.max()}")
for k, v in summarize_support(df).items():
    print(f"  {k:32s} {v:.3f}" if isinstance(v, float) else f"  {k:32s} {v}")
csv_path = os.path.join(a.out, f"aruco_{a.aruco}_date_{a.date}_segments_support.csv")
df.to_csv(csv_path, index=False); print("wrote", csv_path)

# whole-scan panel with a 5x5 closing kernel: with the paper's 3x3, two junctions of this image are cut by the
# vertex merge and the traffic stops short of the cutting. Table and window panels keep the paper's parameters.
skeleton5, segment_objects5, seg_info5, _ = mask_to_segments(mask, SkeletonParams(close_kernel=5))
_, seg_to_paths5, _ = trace_paths(seg_info5)
support5 = support_counts(seg_info5, seg_to_paths5)

x0, y0, size = (int(v) for v in a.window.split(","))
fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
axes[0].imshow(image); axes[0].add_patch(plt.Rectangle((x0, y0), size, size, fill=False, ec="#ffd700", lw=1.5)); axes[0].set_axis_off(); axes[0].set_title("a  aligned scan", loc="left")
plot_support_map(skeleton5, segment_objects5, support5, ax=axes[1], title="b  segments by support (whole scan, closing kernel 5)")
plot_support_map(skeleton, segment_objects, support, df, window=(x0, y0, size), ax=axes[2], title="c  window: segments by support")
plot_overlay(image, mask, skeleton, segment_objects, support, window=(x0, y0, size), ax=axes[3], title="d  window: scan + mask + segments")
fig.tight_layout()
png = os.path.join(a.out, f"aruco_{a.aruco}_date_{a.date}_support.png")
fig.savefig(png, dpi=150); print("wrote", png)
