# Grapevine root architecture (Munoz et al., 2026): segment support

Code for the skeleton and segment-support steps of

> Munoz JR, Sharma S, Lupo Y, Torres-Lomas E, McElrone A, Diaz-Garcia L. Low-cost rhizotron imaging and
> zero-shot deep-learning segmentation resolve temporal, spatial, and genetic variation in grapevine
> rootstock root systems. bioRxiv 2026. https://doi.org/10.64898/2026.09.10.750709

From a root mask it builds the skeleton with PlantCV, cuts it into segments at tips and branch points,
and counts for every segment how many tip-initiated routes pass through it. That count is the segment
support of the Methods and of Supplementary Note S1 ([docs/supplementary_note_S1.md](docs/supplementary_note_S1.md)).

- `rootsupport.skeleton`: `mask_to_segments`
- `rootsupport.support`: `trace_paths`, `support_counts`, `summarize_support`
- `rootsupport.plotting`: `plot_support_map`, `plot_overlay` (Figure 6a,b and Figure S3c-e)

The defaults of every function are the values used in the paper; they are listed in the supplementary note.

## Running the example

```bash
git clone https://github.com/diazgarcialab/grapevine_root_architecture_Munoz2026.git
cd grapevine_root_architecture_Munoz2026
make venv
make example
make test
```

`make venv` needs Python 3.10 or newer. `make example` takes about a minute and writes the per-segment
table and a figure to `examples/output/`. `make lab` opens the same steps as a notebook,
`examples/segment_support_example.ipynb`. `make help` lists the other targets.

`make test` recomputes the segment tables of the two example images and compares them with the
fingerprints in `tests/expected/`, computed from the tables used in the paper. Same hash, same table.

## Example data

`examples/data/`: ArUco 109 at 24 DAT (2025-04-07, the Figure S3 case) and at 38 DAT (2025-04-21),
aligned scans at 300 dpi (0.085 mm per pixel) with the BiRefNet masks at threshold 0.3.

Not included: image alignment, BiRefNet segmentation, mask cleaning, the other traits, soil moisture
mapping and the statistics.

## License

Code: MIT (see `LICENSE`). Example images: CC BY 4.0, (c) the authors. Please cite the paper above.
