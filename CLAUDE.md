# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## Project status: new (scaffolded 2026-11-17)

A Python/Jupyter notebook that visualizes and numerically tests an independent
researcher's warp-drive ideas from their handwritten notes: the Alcubierre metric, the cosine
"thick wall", the superseded pulsed design, and the current
dual-shell always-on design. The audience is a non-physicist reader and the ideas'
author (who reads results as phone images and printed PDFs).

## Session startup (required)

1. **Read `cs.md`** (repo root). It holds the binding hard rules and wins over this file on any conflict.
2. Run `td usage --new-session` at conversation start (or after `/clear`), then
   `td usage -q` for later reads. `AGENTS.md` requires this.
3. Read `docs/warp-bubble-notebook-spec.md`. It is the authoritative spec.
4. Read recent `docs/devlog/` entries.

## Stack

- Python 3 in `.venv/`, dependencies in `requirements.txt`: numpy, scipy,
  matplotlib, sympy, jupyterlab, pytest. Animation export uses matplotlib (GIF via
  pillow; MP4 needs ffmpeg).
- Phase 2 (Task 6) adds a WarpFactory Python port. Check which port is maintained
  before choosing one, and record the choice in the devlog.

## Deliverables (from the spec)

1. `warp_bubble.ipynb`. Each section opens with a 2–3 sentence plain-English explanation.
2. `figures/`: PNGs sized for a phone.
3. `summary.pdf`: 3–5 pages, one chart per page, each with a plain-English caption.
4. `README.md`: how to rerun everything.

## Reference papers

- Alcubierre (1994): https://arxiv.org/abs/gr-qc/0009013
- Pfenning & Ford (1997): https://arxiv.org/abs/gr-qc/9702026
- Helmerich et al. (2024), Warp Factory: https://arxiv.org/abs/2404.03095
- WarpFactory (MATLAB): https://github.com/pseudotensor/WarpFactory

## Conventions

- Always `cd` back to the repo root; use absolute paths.
- Only commit when explicitly asked. Record decisions in `docs/devlog/`.
- Repo setup (agent docs, `td`, devlog, `/review` and `/nn` commands) was copied from
  the owner's `trips` repo and adapted from SvelteKit/Postgres to Python.
  Nothing else from trips applies here.
