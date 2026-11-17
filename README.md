# warpdrive: Warp Bubble Notebook

A Python/Jupyter notebook that visualizes and numerically tests an independent
researcher's warp-drive ideas: the Alcubierre metric, a cosine "thick wall" (the Vorwaller Metric), and a
dual-shell always-on design. Each chart comes with a plain-English caption.

The full spec is in [`docs/warp-bubble-notebook-spec.md`](docs/warp-bubble-notebook-spec.md).

> **Status (2026-11-17):** Sections 1 (wall shapes), 2 (energy vs wall thickness), 3 (quantum inequality) and 4 (dual-shell design) are done. The animation, time-dependent stresses and curvature, the WarpFactory comparison and `summary.pdf` are still to come.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```sh
.venv/bin/jupyter lab warp_bubble.ipynb                                              # interactive
.venv/bin/jupyter nbconvert --to notebook --execute --inplace warp_bubble.ipynb      # regenerate everything
.venv/bin/pytest                                                                     # shape-function tests
```

## Layout

- `warp_bubble.ipynb`: the notebook. Each section explains itself in plain English.
- `warp/shapes.py`: the bubble shape functions (Alcubierre, the author's cosine wall, dual shell).
- `warp/energy.py`: total negative energy of a steady bubble, in SI units.
- `warp/curvature.py`, `warp/quantum.py`: space-time curvature, and the quantum-inequality test.
- `warp/dual_shell.py`: the dual-shell design and its alternatives.
- `warp/plotting.py`: phone-readable figure style; draws each caption into its PNG.
- `tests/`: checks for every shape function (`.venv/bin/pytest`).

## Outputs

- `figures/`: PNGs sized for a phone (to share as images).
- `summary.pdf`: 3–5 pages, one chart per page with captions, ready to print.
