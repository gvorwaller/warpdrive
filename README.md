# warpdrive: Warp Bubble Notebook

A Python/Jupyter notebook that visualizes and numerically tests an independent
researcher's warp-drive ideas: the Alcubierre metric, a cosine "thick wall", and a
dual-shell always-on design. Each chart comes with a plain-English caption.

The full spec is in [`docs/warp-bubble-notebook-spec.md`](docs/warp-bubble-notebook-spec.md).

> **Status:** repo scaffolded 2026-11-17. The notebook, figures and summary PDF are not built yet.

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

## Outputs

- `figures/`: PNGs sized for a phone (to share as images).
- `summary.pdf`: 3–5 pages, one chart per page with captions, ready to print.
