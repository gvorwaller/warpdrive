"""Analysis of the dual-shell always-on design (spec Task 3).

The author's formula, exactly as written, plus the alternatives the spec asks for
(tanh direction, 180° phase) and one further alternative ("squared drive") that
keeps the bubble's interior moving steadily. See warp.shapes.dual_shell_f.

Bubble path: the spec's metric has r_s = |x − x_s(t)| but gives no x_s(t) for this design.
We use Alcubierre's own rule, which keeps the ship at the centre: the centre moves with
the flow there, x_s(t) = ∫ v_s f(0, t) dt (bubble_centre below). For the cos/sin drives
the whole bubble then rocks back and forth with the ship. The other reading,
x_s = v_s·t, would make the walls drift away; the ship picks up a few centimetres
while a wall sweeps past and then stays put. Either way it doesn't travel.

Motion is computed from exact antiderivatives of the drives (∫cos, ∫cos²), so the
result doesn't depend on how finely the caller samples t.

Two facts used here, both verified elsewhere:
- Alcubierre's normal (Eulerian) observers are free-fall paths for ANY shift S(t, x),
  because the lapse is 1. Their coordinate velocity is dx/dt = S = v_s f. So the ship
  at the centre moves at v_s f(0, t), and distant matter at v_s f(∞, t).
- The energy density formula holds at every instant even when f changes in time
  (warp.curvature: G_0̂0̂ has no time-derivative terms), so the total energy at time t
  is P&F eq. 26 applied to f(·, t).
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c
from scipy.integrate import quad

from warp import energy
from warp.shapes import DUAL_SHELL_AS_WRITTEN, dual_shell_df, dual_shell_f

VARIANTS = {
    "as written": dict(DUAL_SHELL_AS_WRITTEN),
    "both shells face inward": {**DUAL_SHELL_AS_WRITTEN, "shell1_sign": -1},
    "180° phase": {**DUAL_SHELL_AS_WRITTEN, "phase": np.pi},
    "inward + squared drive": {**DUAL_SHELL_AS_WRITTEN, "shell1_sign": -1, "drive": "squared"},
}

FAR = np.inf  # "far away": the shell profiles are logistic functions, exact at r = ∞


def period(params):
    return 2.0 * np.pi / params["omega"]


def relative_velocity(t, beta, params):
    """(Ship velocity − distant-space velocity) / c, i.e. β [f(0, t) − f(∞, t)]."""
    return beta * (dual_shell_f(0.0, t, **params) - dual_shell_f(FAR, t, **params))


def bubble_centre(t, beta, params):
    """Bubble centre x_s(t) (metres, from 0 at t[0]) under the rule dx_s/dt = β c f(0, t).

    The centre moves with the flow at the centre, so a ship there stays there. Measured
    against distant space, the bubble's motion equals relative_displacement.
    """
    p1c, p2c = _profile_values(0.0, params)
    i1, i2 = _drive_integrals(t, params)
    x = beta * c * (p1c * i1 + p2c * i2)
    return x - x[0]


def _drive_integrals(t, params):
    """Exact ∫₀ᵗ of the two shells' time factors (seconds)."""
    t = np.asarray(t, dtype=float)
    w, ph = params["omega"], params["phase"]
    if params.get("drive", "linear") == "linear":
        return np.sin(w * t) / w, (np.sin(w * t - ph) - np.sin(-ph)) / w
    return (t / 2 + np.sin(2 * w * t) / (4 * w),
            t / 2 + (np.sin(2 * (w * t - ph)) - np.sin(-2 * ph)) / (4 * w))


def _profile_values(r, params):
    from warp.shapes import shell_profile
    return (float(shell_profile(r, params["R1"], params["sigma"], params["shell1_sign"])),
            float(shell_profile(r, params["R2"], params["sigma"], params["shell2_sign"])))


def relative_displacement(t, beta, params):
    """Ship position relative to distant space (metres), starting from 0 at t[0]. Exact."""
    (p1c, p2c), (p1f, p2f) = _profile_values(0.0, params), _profile_values(FAR, params)
    i1, i2 = _drive_integrals(t, params)
    x = beta * c * ((p1c - p1f) * i1 + (p2c - p2f) * i2)
    return x - x[0]


def mean_relative_velocity(beta, params):
    """Average of relative_velocity over one full cycle, in units of c. Exact."""
    (p1c, p2c), (p1f, p2f) = _profile_values(0.0, params), _profile_values(FAR, params)
    m = 0.0 if params.get("drive", "linear") == "linear" else 0.5   # cycle means of cos, cos²
    return beta * ((p1c - p1f) * m + (p2c - p2f) * m)


def _shell_span(R, sigma):
    """Radial range holding a tanh shell's slope: |r - R| ≤ 40/σ (tails < e⁻¹⁶⁰), clipped at r = 0."""
    pad = 40.0 / sigma
    return max(0.0, R - pad), R + pad


def wall_integral_at(t, params):
    """∫ r² (∂f/∂r)² dr at time t over both shells (metres).

    Each shell's range is integrated separately and the pieces summed, so shells far
    apart (or of very different sharpness) are each resolved; the ranges are merged
    if they overlap, so overlapping shells keep their cross term.
    """
    spans = sorted([_shell_span(params["R1"], params["sigma"]), _shell_span(params["R2"], params["sigma"])])
    merged = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    centres = [params["R1"], params["R2"]]
    df = lambda r: dual_shell_df(r, t, **params)
    return sum(energy.wall_integral(df, lo, hi, points=[c for c in centres if lo < c < hi] or None)
               for lo, hi in merged)


def mass_vs_time(t, beta, params):
    """Negative energy (kg-equivalent, negative numbers) at each time in t."""
    return np.array([energy.mass_from_wall_integral(wall_integral_at(ti, params), beta) for ti in np.atleast_1d(t)])


def shell_overlap_integrals(params):
    """I₁₁, I₂₂, I₁₂ = ∫ r² g_i' g_j' dr for the two inward-facing shell profiles."""
    from warp.shapes import shell_profile_dr
    s = params["sigma"]
    g1 = lambda r: shell_profile_dr(r, params["R1"], s, -1)
    g2 = lambda r: shell_profile_dr(r, params["R2"], s, -1)
    lo = min(_shell_span(params["R1"], s)[0], _shell_span(params["R2"], s)[0])
    hi = max(_shell_span(params["R1"], s)[1], _shell_span(params["R2"], s)[1])
    pts = sorted({params["R1"], params["R2"]})
    i11 = single_shell_integral(params["R1"], s)
    i22 = single_shell_integral(params["R2"], s)
    i12, _ = quad(lambda r: r * r * float(g1(r)) * float(g2(r)), lo, hi, points=pts, limit=1000,
                  epsabs=1e-12 * max(i11, i22), epsrel=1e-10)
    return i11, i22, i12


def cheapest_steady_split(params, beta):
    """Steady bubble with interior f = 1 split over both (inward-facing) walls: drive a on the
    R1 shell, 1 − a on the R2 shell. The energy is the full quadratic
    a²I₁₁ + 2a(1−a)I₁₂ + (1−a)²I₂₂ (I₁₂ is the overlap of the two walls' slopes). It is
    smallest at a = (I₂₂ − I₁₂)/(I₁₁ − 2I₁₂ + I₂₂), clipped to [0, 1]. Coincident shells
    (denominator 0) act as one shell: every split costs the same.
    Returns (a, mass in kg, negative)."""
    i11, i22, i12 = shell_overlap_integrals(params)
    denom = i11 - 2.0 * i12 + i22
    if denom <= 1e-12 * max(i11, i22):
        return 0.5, energy.mass_from_wall_integral(i11, beta)
    a = min(max((i22 - i12) / denom, 0.0), 1.0)
    cost = a * a * i11 + 2.0 * a * (1.0 - a) * i12 + (1.0 - a) ** 2 * i22
    return a, energy.mass_from_wall_integral(cost, beta)


def single_shell_integral(R, sigma):
    """∫ r² g'² dr over r ≥ 0 for one shell ½[1 ± tanh(σ(r − R))] alone (direction doesn't matter)."""
    from warp.shapes import shell_profile_dr
    lo, hi = _shell_span(R, sigma)
    return energy.wall_integral(lambda r: shell_profile_dr(r, R, sigma, -1), lo, hi,
                                points=[R] if lo < R else None)
