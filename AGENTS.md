# AGENTS — warpdrive

## Start here
1. Read **`cs.md`**: the hard rules (audience, physics, units, figures). They override defaults.
2. Read **`docs/warp-bubble-notebook-spec.md`**: the authoritative spec (written by Claude iOS). It lists the author's equations and Tasks 1–6.
3. Check recent `docs/devlog/` entries.

## MANDATORY: Use `td` for task management
Run `td usage --new-session` at conversation start (or after `/clear`) to see current work.
Use `td usage -q` for subsequent reads.

## Quick reference
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (or `uv venv && uv pip install -r requirements.txt`)
- Notebook: `.venv/bin/jupyter lab warp_bubble.ipynb`
- Tests: `.venv/bin/pytest`
- Before commit: `pytest` passes and the notebook runs top to bottom
  (`.venv/bin/jupyter nbconvert --to notebook --execute --inplace warp_bubble.ipynb`).
  Only commit when asked.
