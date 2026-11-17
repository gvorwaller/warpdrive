"""Warp-bubble shape functions f(r_s) and their radial derivatives.

Every function here takes r_s (distance from the bubble centre, metres, r_s >= 0)
and returns a plain numpy array. The author's formulas are implemented exactly as
written in docs/warp-bubble-notebook-spec.md; alternatives are separate,
explicitly named options, never silent substitutions.

Numerical note: ½[1 ± tanh(x)] is computed as the logistic function expit(±2x),
which is mathematically identical but never overflows and keeps full relative
precision in the far tails (where 1 - tanh(x) would cancel to exactly 0).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

__all__ = [
    "alcubierre_f",
    "alcubierre_df",
    "alcubierre_ddf",
    "alcubierre_wall_thickness",
    "sigma_for_wall_thickness",
    "cosine_wall_f",
    "cosine_wall_df",
    "cosine_wall_ddf",
    "cosine_wall_max_slope",
    "cosine_wall_pf_thickness",
    "pf_ramp_f",
    "pf_ramp_df",
    "shell_profile",
    "shell_profile_dr",
    "dual_shell_f",
    "dual_shell_df",
    "DUAL_SHELL_AS_WRITTEN",
]


def _sech2(x):
    """sech²(x) = 4·expit(2x)·expit(-2x), stable for any x."""
    x = np.asarray(x, dtype=float)
    return 4.0 * expit(2.0 * x) * expit(-2.0 * x)


# ---------------------------------------------------------------------------
# Alcubierre (1994), eq. 6
#   f = [tanh(σ(r+R)) - tanh(σ(r-R))] / (2 tanh(σR))
# ---------------------------------------------------------------------------

def alcubierre_f(r, R, sigma):
    """Alcubierre's tanh shape function.

    Uses tanh(a) - tanh(b) = 2[expit(-2b) - expit(-2a)], which keeps relative
    precision for r >> R (both tanh terms -> 1 there).
    """
    r = np.asarray(r, dtype=float)
    a = sigma * (r + R)
    b = sigma * (r - R)
    return (expit(-2.0 * b) - expit(-2.0 * a)) / np.tanh(sigma * R)


def alcubierre_df(r, R, sigma):
    """d f / d r_s for Alcubierre's shape (negative for r_s > 0)."""
    r = np.asarray(r, dtype=float)
    return sigma * (_sech2(sigma * (r + R)) - _sech2(sigma * (r - R))) / (
        2.0 * np.tanh(sigma * R)
    )


def alcubierre_ddf(r, R, sigma):
    """d²f/dr_s² for Alcubierre's shape. Uses d/dx sech²(x) = -2 sech²(x) tanh(x)."""
    r = np.asarray(r, dtype=float)
    a = sigma * (r + R)
    b = sigma * (r - R)
    return sigma**2 * (_sech2(b) * np.tanh(b) - _sech2(a) * np.tanh(a)) / np.tanh(sigma * R)


def alcubierre_wall_thickness(R, sigma):
    """Pfenning & Ford (1997) eq. 5: Δ = [1 + tanh²(σR)]² / (2σ tanh(σR)).

    This is 1/|df/dr_s| at r_s = R, i.e. the width of a straight ramp with the
    same slope as the tanh wall. For σR >> 1 it reduces to Δ ≈ 2/σ.
    """
    t = np.tanh(sigma * R)
    return (1.0 + t * t) ** 2 / (2.0 * sigma * t)


def sigma_for_wall_thickness(delta, R):
    """Invert Pfenning & Ford eq. 5: the σ that gives wall thickness Δ.

    The thickness Δ(σ) falls monotonically as σ grows, so a bracketed root
    find is safe. Starts from the large-σR estimate 2/Δ.
    """
    if delta <= 0 or R <= 0:
        raise ValueError("delta and R must be positive")
    # Two limits of eq. 5: σR >> 1 gives Δ ≈ 2/σ; σR << 1 gives Δ ≈ 1/(2σ²R).
    # Bracket both estimates generously so any positive Δ is found.
    large, small = 2.0 / delta, 1.0 / np.sqrt(2.0 * delta * R)
    lo, hi = min(large, small) / 100.0, max(large, small) * 100.0
    return brentq(lambda s: alcubierre_wall_thickness(R, s) - delta, lo, hi, xtol=1e-14, rtol=1e-14)


# ---------------------------------------------------------------------------
# The author's cosine wall ("Vorwaller Metric")
#   f = 1                               r < R
#   f = ½[1 + cos(π (r - R)/Δ)]          R <= r <= R + Δ
#   f = 0                               r > R + Δ
# The wall starts at R and extends OUTWARD to R + Δ (Alcubierre's is centred on R).
# ---------------------------------------------------------------------------

def _check_delta(delta):
    if not delta > 0:
        raise ValueError("wall thickness delta must be positive")


def cosine_wall_f(r, R, delta):
    _check_delta(delta)
    r = np.asarray(r, dtype=float)
    u = (r - R) / delta
    inside_wall = 0.5 * (1.0 + np.cos(np.pi * np.clip(u, 0.0, 1.0)))
    return np.where(u < 0.0, 1.0, np.where(u > 1.0, 0.0, inside_wall))


def cosine_wall_df(r, R, delta):
    """d f / d r_s = -(π / 2Δ) sin(π (r - R)/Δ) inside the wall, 0 elsewhere.

    The slope is continuous (it is 0 at both wall edges), so f is C¹ but not C²:
    f'' jumps by π²/(2Δ²) at both edges. Harmless for the steady T⁰⁰ (uses only f'),
    but anything using second derivatives (full Einstein tensor) will see edge spikes.
    """
    _check_delta(delta)
    r = np.asarray(r, dtype=float)
    u = (r - R) / delta
    in_wall = (u >= 0.0) & (u <= 1.0)
    return np.where(in_wall, -(np.pi / (2.0 * delta)) * np.sin(np.pi * np.clip(u, 0.0, 1.0)), 0.0)


def cosine_wall_ddf(r, R, delta):
    """d²f/dr_s² = -(π²/2Δ²) cos(π (r - R)/Δ) inside the wall, 0 outside.

    Jumps from 0 to ∓π²/(2Δ²) at the wall edges (the cosine wall is C¹, not C²).
    At exactly r = R or R + Δ this returns the inside-wall value.
    """
    _check_delta(delta)
    r = np.asarray(r, dtype=float)
    u = (r - R) / delta
    in_wall = (u >= 0.0) & (u <= 1.0)
    return np.where(in_wall, -(np.pi**2 / (2.0 * delta**2)) * np.cos(np.pi * np.clip(u, 0.0, 1.0)), 0.0)


def cosine_wall_max_slope(delta):
    """The author's result: max |df/dr_s| = π / (2Δ), reached mid-wall at r = R + Δ/2."""
    return np.pi / (2.0 * delta)


def cosine_wall_pf_thickness(delta):
    """The cosine wall's thickness on Pfenning & Ford's yardstick (1 / max slope) = 2Δ/π.

    The author's Δ is the wall's full extent (R to R + Δ); P&F measure a wall by its
    steepest slope. Use this when comparing a cosine wall with a tanh wall or with
    P&F's thin-wall energy formula.
    """
    return 2.0 * delta / np.pi


# ---------------------------------------------------------------------------
# Pfenning & Ford (1997) eq. 4: straight-ramp wall of thickness Δ centred on R.
# Used only as a validation reference (their eq. 28 gives its energy in closed form).
# ---------------------------------------------------------------------------

def pf_ramp_f(r, R, delta):
    _check_delta(delta)
    r = np.asarray(r, dtype=float)
    return np.clip(0.5 - (r - R) / delta, 0.0, 1.0)


def pf_ramp_df(r, R, delta):
    _check_delta(delta)
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r - R) <= delta / 2.0, -1.0 / delta, 0.0)


# ---------------------------------------------------------------------------
# Dual-shell always-on design (as written in the author's notes)
#   f_total = ½[1 + tanh(2.0(r - 100))]·cos(ωt) + ½[1 - tanh(2.0(r - 90))]·sin(ωt)
# Generalised so the spec's alternatives can be shown alongside:
#   shell k profile  = ½[1 + s_k tanh(σ(r - R_k))]   (s_k = +1 rises outward, -1 falls outward)
#   shell 1 time     = cos(ωt)
#   shell 2 time     = cos(ωt - φ)                   (φ = 90° gives sin(ωt), as written)
#   drive            = "linear" (as written) or "squared": time factors cos²(ωt), cos²(ωt - φ),
#                      which never go negative and sum to 1 at φ = 90° (an "always on" alternative)
# ---------------------------------------------------------------------------

DUAL_SHELL_AS_WRITTEN = dict(
    R1=100.0,
    R2=90.0,
    sigma=2.0,
    omega=2.0 * np.pi * 1.0e9,  # 1 GHz, rad/s
    shell1_sign=+1,
    shell2_sign=-1,
    phase=np.pi / 2.0,
    drive="linear",
)


def shell_profile(r, R, sigma, sign):
    """½[1 + sign·tanh(σ(r - R))] = expit(2·sign·σ(r - R))."""
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    r = np.asarray(r, dtype=float)
    return expit(2.0 * sign * sigma * (r - R))


def shell_profile_dr(r, R, sigma, sign):
    """d/dr of shell_profile = ½·sign·σ·sech²(σ(r - R))."""
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    r = np.asarray(r, dtype=float)
    return 0.5 * sign * sigma * _sech2(sigma * (r - R))


def _time_factors(t, omega, phase, drive):
    t = np.asarray(t, dtype=float)
    wt = omega * t
    c1 = np.cos(wt)
    # Use sin directly for the as-written 90° case so it matches the author's formula bit-for-bit.
    c2 = np.sin(wt) if phase == np.pi / 2.0 else np.cos(wt - phase)
    if drive == "linear":
        return c1, c2
    if drive == "squared":
        return c1 * c1, c2 * c2
    raise ValueError("drive must be 'linear' or 'squared'")


def dual_shell_f(r, t, *, R1, R2, sigma, omega, shell1_sign, shell2_sign, phase, drive="linear"):
    """f_total(r_s, t). r and t broadcast against each other (numpy rules).

    Call with **DUAL_SHELL_AS_WRITTEN for the author's formula exactly as written.
    """
    c1, c2 = _time_factors(t, omega, phase, drive)
    return shell_profile(r, R1, sigma, shell1_sign) * c1 + shell_profile(r, R2, sigma, shell2_sign) * c2


def dual_shell_df(r, t, *, R1, R2, sigma, omega, shell1_sign, shell2_sign, phase, drive="linear"):
    """∂f_total/∂r_s at fixed t."""
    c1, c2 = _time_factors(t, omega, phase, drive)
    return (
        shell_profile_dr(r, R1, sigma, shell1_sign) * c1
        + shell_profile_dr(r, R2, sigma, shell2_sign) * c2
    )
