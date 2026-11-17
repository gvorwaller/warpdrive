# Warp Bubble Notebook — Handoff Spec for Claude Code

## Who this is for

- **The reader**: an experienced software engineer, not a math person. Wants clear plain-English explanations of every result.
- **The author** of the ideas: self-taught in general relativity, working on their own warp drive concepts. Views results as **images and PDFs on a phone**, and may print them.

**Goal:** A Python/Jupyter notebook that visualizes and numerically tests the author's warp drive ideas from their handwritten notes. This is a learning project. Accuracy matters, but so does explanation. Every chart needs a short plain-English caption.

---

## Background

The author is working with the **Alcubierre warp drive metric** (1994). A "bubble" of spacetime contracts in front of a ship and expands behind it, so the ship rides inside it without moving locally. It requires **negative energy density** (exotic matter), and in the original design the amount is absurdly large.

The key obstacle is the **Ford–Roman quantum inequalities**: negative energy can exist only briefly, and the larger the magnitude, the shorter the allowed duration. Pfenning & Ford (1997) used this to show that the bubble wall must be only a few hundred Planck lengths thick, which drives the total energy to unphysical levels.

The author's ideas attack this problem:

1. **Thick wall ("Vorwaller Metric")**: a smooth cosine wall profile, widened from 1 m to 1000 m.
2. **Pulsed design** (earlier, now superseded): microsecond on/off bursts.
3. **Dual-shell always-on design** (current): two nested shells driven by cosine and sine so the bubble never switches off.

### Reference papers (please read before coding)

- Alcubierre (1994), "The warp drive: hyper-fast travel within general relativity": https://arxiv.org/abs/gr-qc/0009013
- Pfenning & Ford (1997), "The unphysical nature of 'Warp Drive'": https://arxiv.org/abs/gr-qc/9702026
- Helmerich et al. (2024), "Analyzing warp drive spacetimes with Warp Factory": https://arxiv.org/abs/2404.03095
- WarpFactory (MATLAB, MIT license): https://github.com/pseudotensor/WarpFactory
- WarpFactory Python ports exist (e.g. https://github.com/pseudotensor/warpfactorypython and https://github.com/JohnsterID/WarpFactory). Check which is maintained and best documented before choosing.

---

## The author's equations (transcribed from their notes)

Treat these as **the author's** formulas. Where something looks off, flag it and show both versions rather than silently "fixing" it.

### Metric

```
ds² = −c² dt² + (dx − v_s f(r_s) dt)² + dy² + dz²
r_s = sqrt( (x − x_s(t))² + y² + z² )
```

- `v_s` = bubble velocity, `x_s(t)` = bubble center, `R` = bubble radius, `σ` = wall-sharpness parameter.

### Alcubierre's shape function

```
f(r_s) = [ tanh(σ(r_s + R)) − tanh(σ(r_s − R)) ] / ( 2 tanh(σR) )
```

### The author's cosine wall ("Vorwaller Metric")

```
f = 1                                   for r_s < R
f = ½ [ 1 + cos( π (r_s − R) / Δ ) ]    for R ≤ r_s ≤ R + Δ
f = 0                                   for r_s > R + Δ

max |df/dr_s| = π / (2Δ)
  Δ = 1 m    → 1.57
  Δ = 1000 m → 0.00157
```

### Energy density (craft moving along x)

```
T⁰⁰ = −(c⁴ / 8πG) · (v_s² / 4) · ((y² + z²) / r_s²) · (df/dr_s)²
```

The author's note: "Energy goes as derivative², so (1000)² = 1,000,000× savings." **See Task 2.**

### Pulsed design (superseded, but worth one plot)

```
τ_on = 1 µs, τ_off = 9 µs, duty cycle = 10%
v_s(t) = 1.1c + 0.2c · sin(2π f_pulse t)      # 0.9c base, 1.3c peak
```

### Dual-shell always-on design (current)

```
Shell 1: σ = 2.0, R = 100 m
Shell 2: σ = 2.0, R = 90 m

f_total(r_s, t) = ½[1 + tanh(2.0(r_s − 100))] · cos(ωt)
               + ½[1 − tanh(2.0(r_s − 90))]  · sin(ωt)

ω = 2π × 10⁹ rad/s   (1 GHz)
Wall crossing time t_cross = 1.0 m / c ≈ 3.33 ns  →  f_min ≈ 300 MHz
```

The sketch says the shells are "180° out of phase."

Other figures in the notes: total negative energy `E ≈ −10⁴⁵ kg`; quantum inequality `|ρ| · τ⁴ ≤ constant`.

---

## Tasks

### Task 1 — Wall shape comparison

Plot Alcubierre's tanh wall (σ chosen to give about a 1 m wall) against the author's cosine wall at Δ = 1 m, 10 m, 100 m and 1000 m, with R = 100 m. Plot `f` and `df/dr_s` on separate panels.

### Task 2 — Total energy vs wall thickness

Numerically integrate T⁰⁰ over all space (the problem is axisymmetric, so cylindrical coordinates work well) for a range of Δ values. Report total energy in kg-equivalent (E/c²).

**Key question:** does total energy fall as 1/Δ² (the author's estimate) or roughly as 1/Δ? Pfenning & Ford give total energy ∝ R²/Δ for thin walls: energy *density* drops as 1/Δ², but wall *volume* grows as Δ. Show the log-log plot with the slope labeled, and explain the result in two plain sentences.

### Task 2b — Quantum-inequality check (added 2026-11-17)

*Added by Claude Code with the owner's approval; it was missing from the original task list.* The spec names the Ford–Roman quantum inequalities as the key obstacle, and the author's notes include `|ρ| · τ⁴ ≤ constant`, but no task tested them. Apply the QI to the cosine wall: the strongest negative energy density, the curvature-limited sampling time (P&F section 3), the Lorentzian-sampled energy along a real geodesic, and the violation factor vs wall thickness. Show how the result depends on the sampling fraction α and the speed. State the assumptions honestly: a free scalar field, flat-space QI over windows shorter than the curvature time.

### Task 3 — Dual-shell model

Implement `f_total(r_s, t)` exactly as written, and plot it versus r_s at several times across one cycle.

Check and report on these, without overruling the author, just showing what the math does:

1. **tanh direction.** As written, the shell 1 term `½[1 + tanh(...)]` is ~0 inside 100 m and ~1 outside, while the shell 2 term is ~1 inside 90 m and ~0 outside. The two terms face opposite directions. Plot both the as-written version and a version where both use `½[1 − tanh(...)]`.
2. **Phase.** Cosine and sine are **90°** apart, but the sketch says **180°**. Show both. At 90°, cos² + sin² = 1. At 180°, the shells partly cancel.
3. **Net motion.** Inside the bubble, f_total oscillates in sign (e.g. it follows sin(ωt) for r < 90 m). Does the ship make net forward progress over a cycle, or oscillate in place? Compute the time-averaged interior shift.
4. **Energy over time.** Does total negative energy actually stay constant over a cycle, or does it slosh between the two shells (they sit at different radii)?

### Task 4 — Animation

Animate the dual-shell bubble across one or two cycles, as a 2D cross-section with energy density as a color map. Export as GIF or MP4, plus a few still frames as PNG.

### Task 5 — Physics caveat to check

~~The T⁰⁰ formula above assumes a steady bubble … time-derivative terms of f will add to the stress-energy.~~

*Revised 2026-11-17 by Claude Code.* The original premise is only half right. For this metric family (flat spatial slices, unit lapse) the energy density the Eulerian observers see, G_0̂0̂/8π = −(S_y² + S_z²)/(32π) with S = v_s f, has **no** time-derivative terms, so the T⁰⁰ formula holds at every instant even for GHz designs (verified symbolically, `warp/curvature.py`). Time dependence does enter (a) the **stresses** (G_îĵ contains ∂_t∂_x S), (b) the **curvature**, which sets the quantum-inequality window, and (c) the **geodesics**, i.e. whether anything is actually carried along. Task 5 is therefore: compute the full Einstein tensor and curvature for the pulsed and dual-shell designs, show what time dependence changes and what it doesn't, and redo the quantum-inequality test (Task 2b) for them.

### Task 6 (phase 2) — WarpFactory

Install WarpFactory (Python port preferred). Reproduce a standard Alcubierre case as a sanity check, then run the Vorwaller Metric (cosine wall) as a custom metric. Report energy-condition violations and energy totals, and compare them with Tasks 2–3.

---

## Engineering notes

- Python 3, numpy, scipy, matplotlib; sympy is fine for derivatives. Use a virtual environment.
- Compute in geometric units if convenient, but **report results in SI** (meters, seconds, kg).
- Watch out for floating-point issues: tanh with σ·r around 200 saturates, and 1 GHz oscillations need adequate time resolution.
- Units slip to fix quietly in captions: "v_s = 10c → v_s² = 100c" should be **100c²**.
- Include a small test for each shape function: f ≈ 1 at the center, f ≈ 0 far away, and the derivative peaks where expected.

## Deliverables

1. `warp_bubble.ipynb`, with each section starting with a 2–3 sentence plain-English explanation.
2. A `figures/` folder of PNGs, sized to read well on a phone (they are viewed as images in a phone chat app).
3. `summary.pdf`: 3–5 pages, one chart per page, each with a plain-English caption. This is the version to share or print.
4. `README.md` explaining how to rerun everything.

## Tone of the write-up

Encouraging and honest. The author is doing real physics. Where the math has a slip, show it as "here's what the equation does as written, and here's the alternative," which is how a good professor would handle it. Celebrate the ideas that hold up: the thick-wall instinct is genuinely the right lever, and choosing the frequency against the wall-crossing time is real physics reasoning.
