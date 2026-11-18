# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## Project status: new (scaffolded 2026-11-17)

A Python/Jupyter notebook that visualizes and numerically tests an independent
researcher's warp-drive ideas from their handwritten notes: the Alcubierre metric, the cosine
"thick wall" (Vorwaller Metric), the superseded pulsed design, and the current
dual-shell always-on design. The audience is a non-physicist reader and the ideas'
author (who reads results as phone images and printed PDFs).

## Session startup (required)

1. **Read `cs.md`** (repo root). It holds the binding hard rules and wins over this file on any conflict.
2. Run `td usage --new-session` at conversation start (or after `/clear`), then
   `td usage -q` for later reads. `AGENTS.md` requires this.
3. Read `docs/warp-bubble-notebook-spec.md`. It was written by Claude (iOS) from the author's
   handwritten notes. Its tasks and framing may be improved wherever you see a better way (the owner
   has said so); record each change in the devlog. The author's *equations* stay exactly as written.
4. Read recent `docs/devlog/` entries.

## Stack

- Python 3 in `.venv/`, dependencies in `requirements.txt`: numpy, scipy,
  matplotlib, sympy, jupyterlab, pytest. Animation export uses matplotlib (GIF via
  pillow; MP4 needs ffmpeg).
- Phase 2 (Task 6): WarpFactory Python port JohnsterID/WarpFactory, pinned in
  `requirements-warpfactory.txt` and installed with `--no-deps` (its metadata pins numpy < 2; its
  239 tests pass on numpy 2). Choice recorded in the 2026-11-18 devlog.

## Code layout

- Physics lives in `warp/` (importable and tested); notebook cells stay thin: parameters, plot, caption.
- `warp/shapes.py`: f and df/dr_s for each shape. `½[1 ± tanh x]` is computed as `expit(±2x)`
  (identical math, no overflow, keeps the far tails accurate).
- `warp/energy.py`: SI energy density, total energy as mass E/c² (full 2-D cylindrical integral and
  the 1-D P&F eq. 26 reduction), closed form and optimum for the cosine wall, IAU mass constants.
  Steady bubble only.
- `warp/curvature.py`: Riemann and Einstein tensors for any shift S(t,x,y,z), derived once with sympy
  and evaluated numerically in the Eulerian orthonormal frame. Reuse it for the time-dependent designs (Task 5).
- `warp/quantum.py`: Ford–Roman quantum inequality applied to a wall (full-curvature r_min, Lorentzian
  sampling along the real geodesic). Choices deliberately favour the design.
- `warp/dual_shell.py`: dual-shell variants (as written + alternatives), ship motion vs distant space,
  energy vs time.
- `warp/time_dependent.py`: switching bubbles. S = Σ a_k(t) g_k(r) with a moving centre; exact (ct,x,y,z)
  derivatives (checked against brute force), frozen comparison, stresses, curvature and QI vs frequency.
- `warp/animate.py`: energy-density cross-section maps, GIF/MP4 animation and stills (fixed inch
  layout so frames match; caption drawn into every frame). Maps are checked against the 1-D totals.
- `warp/warpfactory_check.py`: Task 6. Hands WarpFactory only the metric (custom hyper-dual metric
  functions for the cosine wall and the dual shell; WarpFactory's own Alcubierre), then compares its
  Eulerian stress-energy, totals and energy-condition maps with ours. Nothing from warp.curvature or
  warp.energy is used on the WarpFactory side. Tests importorskip("warpfactory").
- `warp/summary.py`: builds `summary.pdf` (5 Letter pages, one figure each, bottom-line text from the
  notebook's computed numbers; built by the notebook's last cell).
- `warp/plotting.py`: `use_phone_style()`, `phone_figure()`, `save_figure(fig, name, caption)`, `sci_text()`.
  The caption is drawn into the PNG because figures are viewed without the notebook text.
- `tests/`: pytest with `filterwarnings = error`, so any numpy overflow fails the suite.
- Wall-thickness yardstick for tanh walls: Pfenning & Ford eq. 5, Δ ≈ 2/σ (1 m ↔ σ = 2/m).

## Deliverables (from the spec)

1. `warp_bubble.ipynb`. Each section opens with a 2–3 sentence plain-English explanation.
2. `figures/`: PNGs sized for a phone.
3. `summary.pdf`: 3–5 pages, one chart per page, each with a plain-English caption.
4. `README.md`: how to rerun everything.

## Reference papers

- Alcubierre (1994): https://arxiv.org/abs/gr-qc/0009013
- Pfenning & Ford (1997): https://arxiv.org/abs/gr-qc/9702026
- Helmerich et al. (2024), Warp Factory: https://arxiv.org/abs/2404.03095
- WarpFactory (MATLAB): https://github.com/NerdsWithAttitudes/WarpFactory
- WarpFactory Python port (used): https://github.com/JohnsterID/WarpFactory
- Fuchs et al. (2024), subluminal warp drive satisfying all energy conditions: https://arxiv.org/abs/2405.02709

## Conventions

- Always `cd` back to the repo root; use absolute paths.
- Only commit when explicitly asked. Record decisions in `docs/devlog/`.
- Repo setup (agent docs, `td`, devlog, `/review` and `/nn` commands) was copied from
  the owner's `trips` repo and adapted from SvelteKit/Postgres to Python.
  Nothing else from trips applies here.
