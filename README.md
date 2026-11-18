# warpdrive: Warp Bubble Notebook

A Python/Jupyter notebook that visualizes and numerically tests an independent
researcher's warp-drive ideas: the Alcubierre metric, a cosine "thick wall" (the Vorwaller
Metric), an earlier pulsed design, and a dual-shell always-on design. Each chart comes
with a plain-English caption.

**Start with [`summary.pdf`](summary.pdf)**: five pages, one chart each, and the version
to send or print. The notebook [`warp_bubble.ipynb`](warp_bubble.ipynb) has the full story
(GitHub renders it, figures included). The spec is in
[`docs/warp-bubble-notebook-spec.md`](docs/warp-bubble-notebook-spec.md).

> **Status (2026-11-17):** spec Tasks 1–5 are done, plus an added quantum-inequality task:
> notebook Sections 1–7, `summary.pdf`, and the animation. Task 6 (cross-checking against
> WarpFactory) is still to come.

## What's in the notebook

1. **Wall shapes:** Alcubierre's tanh wall vs. the cosine wall at 1–1000 m.
2. **Energy vs. wall thickness:** thicker walls help, up to a sweet spot near 1.9× the bubble radius.
3. **The quantum inequality:** the Ford–Roman limit on how long negative energy can last.
4. **The dual-shell design:** the formula as written, plus alternatives; motion and energy over a cycle.
5. **Animation:** the dual-shell bubble through two cycles (GIF and MP4).
6. **Switching:** what fast switching changes (stresses, curvature, the quantum test) and what it can't (the energy).
7. **The summary PDF.**

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The MP4 export needs `ffmpeg` on the PATH (e.g. `brew install ffmpeg`); everything else is pure Python.

## Rerun everything

```sh
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 warp_bubble.ipynb    # all numbers, figures, animation, summary.pdf (~1 min)
.venv/bin/pytest                                             # 131 tests; any numpy warning counts as a failure
```

For interactive exploration: `.venv/bin/jupyter lab warp_bubble.ipynb`. The settings at the
top of each section (bubble radius `R`, speed `BETA`, sampling fraction `ALPHA`, the
dual-shell parameters in `warp/dual_shell.py`) can be changed and the notebook rerun.

## How the results were checked

- Every formula was checked against the original papers: Alcubierre (1994) and Pfenning & Ford (1997).
- Key results were computed two or three independent ways, for example the full 2-D energy integral
  vs. the 1-D reduction vs. a closed form, and the curvature engine vs. brute-force finite differences.
- Three independent reviewers:
  - Claude subagents reviewed each section.
  - Gemini (AGY2/AGY3) independently re-derived the physics.
  - Codex (CODEX8) reviewed the code.
  - Their findings and fixes are recorded in [`docs/devlog/`](docs/devlog/).

## Layout

- `warp_bubble.ipynb`: the notebook. Each section explains itself in plain English.
- `summary.pdf`: the five-page summary, built by the notebook's last cell (`warp/summary.py`).
- `warp/shapes.py`: the bubble shape functions (Alcubierre, the cosine wall, the dual shell).
- `warp/energy.py`: energy density and total negative energy, in SI units.
- `warp/curvature.py`: Riemann and Einstein tensors for any warp-type flow (symbolic, then numeric).
- `warp/quantum.py`: the quantum-inequality test.
- `warp/dual_shell.py`: the dual-shell design, its alternatives, motion and energy over time.
- `warp/time_dependent.py`: stresses and curvature of switching bubbles.
- `warp/animate.py`: the dual-shell animation (GIF + MP4) and still frames.
- `warp/plotting.py`: phone-readable figure style; draws each caption into its PNG.
- `tests/`: 131 tests.
- `figures/`: PNGs sized for a phone (each with its caption drawn in), plus the animation as GIF
  and MP4. The MP4 is much smaller and plays well in messaging apps.
