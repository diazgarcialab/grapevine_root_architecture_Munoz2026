# Supplementary Note S1. Tip-initiated paths and segment support

Text of the note as published in the preprint. The parameter values the note leaves to this repository
are in the table at the end.

**Graph.** The skeleton of a root mask is cut at branch points into segments. A segment end is a tip if
it is a skeleton node of degree one, as root tips are defined in Trait extraction (this includes the end
of the system at the cutting and ends left by gaps in the mask), and a branch point otherwise; segment
ends meeting at a branch point form a vertex. A segment is terminal if it has one tip (set E_T), isolated
if it has two, and inner if it is bounded by two branch points (set E_I). Image coordinates have y
increasing downward, so the cutting lies at small y.

**Path.** Every terminal segment e₀ ∈ E_T starts one path P = (e₀, e₁, …), entered at its tip; isolated
segments start none and are never traversed. Segment directions are taken along the straight line
joining a segment's two ends. At the current vertex u, let d be the unit vector along the last segment,
oriented toward u. The candidates are the inner segments f ∈ E_I from u to a vertex w that is not lower
than u (within a tolerance ε) and not yet on the path; each has a turning angle θ(f) = cos⁻¹(d · d_f),
where d_f is the unit vector along f from u to w, so that θ = 0° is straight ahead and θ = 180° a
reversal. The candidate with the smallest θ is appended (ties by segment index), and the path ends when
no candidate exists or the smallest θ exceeds θ_max = 100°. A path is a "route" in the main text.

**Support.** s(e) is the number of paths containing segment e (0 if none). By construction s = 1 for
terminal segments, s = 0 for isolated segments, and s ≥ 0 for inner segments; s ≥ 2 marks a segment
shared by two or more paths (transport segments in the main text).

**Pseudocode.** Input: skeleton graph G = (V, E) with segment sets E_T, E_I; θ_max; ε. Output: support
s(e) for every e ∈ E.

```
s(e) ← 0 for all e ∈ E
for each e0 ∈ E_T do                                   ▷ one path per terminal segment
    P ← (e0); u ← branch-point vertex of e0; d ← unit vector of e0 from its tip to u
    loop
        C ← segments f ∈ E_I incident to u, f ∉ P, whose far vertex w has y(w) ≤ y(u) + ε
        if C is empty then exit loop
        for each f ∈ C: d_f ← unit vector of f from u to w; θ(f) ← cos⁻¹(d · d_f)
        f* ← the f ∈ C with the smallest θ(f)              ▷ ties: lowest segment index
        if θ(f*) > θ_max then exit loop
        append f* to P; d ← d_f*; u ← far vertex of f*
    end loop
    for each e ∈ P: s(e) ← s(e) + 1
end for
return s
```

Because every iteration appends a segment not yet in P, every path terminates.

## Implementation parameters (this repository)

Values are in pixels of the aligned scans (300 dpi, 0.085 mm per pixel).

| Step | Parameter | Value | Where |
|---|---|---|---|
| Mask smoothing | Gaussian blur kernel | 3x3 | `SkeletonParams.blur_kernel` |
| Mask smoothing | morphological closing kernel | 3x3 | `SkeletonParams.close_kernel` |
| Skeleton pruning | spur removal (`plantcv.morphology.prune`, size) | 50 | `SkeletonParams.prune_size` |
| Segment ends | tip detection padding around an end | 5 px | `SkeletonParams.tip_pad` |
| Graph | merge distance for segment ends into one vertex | 15 px (1.3 mm) | `SupportParams.snap_dist` |
| Path | θ_max | 100° | `SupportParams.max_turn_deg` |
| Path | ε (allowed descent per step) | 10 px (0.85 mm) | `SupportParams.upward_eps` |
| Path | ambiguity margin | not used | `SupportParams.confidence_gap_deg = None` |
| Path | tie rule | lowest segment index (stable sort) | `support.trace_paths` |

Vertices are built greedily: each segment end, in order, joins the nearest vertex within the merge
distance (whose centre moves to the midpoint) or starts a new one. Segments whose ends could not be
located (rare 1-2 px remnants) are left out of the graph and get support 0.
