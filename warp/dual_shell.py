"""Analysis of the dual-shell always-on design (spec Task 3).

The author's formula, exactly as written, plus the alternatives the spec asks for
(tanh direction, 180° phase) and one further alternative ("squared drive") that
keeps the bubble's interior moving steadily. See warp.shapes.dual_shell_f.

Bubble path: the spec's metric has r_s = |x − x_s(t)| but gives no x_s(t) for this design.
We use Alcubierre's own rule, which keeps the ship at the centre: the centre moves with
the flow there, x_s(t) = ∫ v_s f(0, t) dt (bubble_centre below). For the cos/sin drives
the whole bubble then rocks back and forth with the ship. The other reading,
x_s = v_s·t, would make the walls drift away while the ship stays put; it doesn't
change the conclusion that the ship goes nowhere.

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
from scipy.integrate import cumulative_trapezoid

from warp import energy
from warp.shapes import DUAL_SHELL_AS_WRITTEN, dual_shell_df, dual_shell_f

VARIANTS = {
    "as written": dict(DUAL_SHELL_AS_WRITTEN),
    "both shells face inward": {**DUAL_SHELL_AS_WRITTEN, "shell1_sign": -1},
    "180° phase": {**DUAL_SHELL_AS_WRITTEN, "phase": np.pi},
    "inward + squared drive": {**DUAL_SHELL_AS_WRITTEN, "shell1_sign": -1, "drive": "squared"},
}

FAR = 1.0e4  # metres: "far away" (≥ 4000 wall widths outside the outer shell)


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
    return cumulative_trapezoid(c * beta * dual_shell_f(0.0, t, **params), t, initial=0.0)


def relative_displacement(t, beta, params):
    """Ship position relative to distant space (metres), starting from 0 at t[0]."""
    return cumulative_trapezoid(c * relative_velocity(t, beta, params), t, initial=0.0)


def mean_relative_velocity(beta, params, n=20_001):
    """Average of relative_velocity over one full cycle, in units of c."""
    t = np.linspace(0.0, period(params), n)
    return np.trapezoid(relative_velocity(t, beta, params), t) / period(params)


def wall_integral_at(t, params, pad=20.0):
    """∫ r² (∂f/∂r)² dr at time t over both shells (metres)."""
    lo = min(params["R1"], params["R2"]) - pad
    hi = max(params["R1"], params["R2"]) + pad
    return energy.wall_integral(lambda r: dual_shell_df(r, t, **params), lo, hi,
                                points=sorted([params["R1"], params["R2"]]))


def mass_vs_time(t, beta, params):
    """Negative energy (kg-equivalent, negative numbers) at each time in t."""
    return np.array([energy.mass_from_wall_integral(wall_integral_at(ti, params), beta) for ti in np.atleast_1d(t)])


def cheapest_steady_split(params, beta):
    """Steady bubble with interior f = 1 split over both (inward-facing) walls: drive a on the
    R1 shell, 1 − a on the R2 shell. Energy a²E₁ + (1−a)²E₂ is smallest at a = E₂/(E₁+E₂),
    where it equals E₁E₂/(E₁+E₂). Returns (a, mass in kg, negative)."""
    e1 = single_shell_integral(params["R1"], params["sigma"])
    e2 = single_shell_integral(params["R2"], params["sigma"])
    a = e2 / (e1 + e2)
    return a, energy.mass_from_wall_integral(e1 * e2 / (e1 + e2), beta)


def single_shell_integral(R, sigma, pad=20.0):
    """∫ r² g'² dr for one shell ½[1 ± tanh(σ(r − R))] alone (direction doesn't matter)."""
    from warp.shapes import shell_profile_dr
    return energy.wall_integral(lambda r: shell_profile_dr(r, R, sigma, -1), R - pad, R + pad, points=[R])
