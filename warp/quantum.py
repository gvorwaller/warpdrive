"""Ford–Roman quantum inequality (QI) applied to a steady warp-bubble wall.

The QI (Ford & Roman 1995; Pfenning & Ford 1997 eq. 9) for a free massless
scalar field limits how negative the energy density can be when an inertial
observer averages it with a Lorentzian of width τ:

    (τ/π) ∫ ρ(t) dt / (t² + τ²)  ≥  -3ħ / (32π² c³ τ⁴)

i.e. a negative energy density of strength |ρ| may persist only for
τ ≲ (3ħ / (32π² c³ |ρ|))^¼. That is exactly the form |ρ|·τ⁴ ≤ constant.

In curved spacetime the flat-space QI is trusted only for τ shorter than the
local curvature time r_min/c, where r_min = 1/√(largest Riemann component in the
observer's frame). Following P&F we sample for τ = α·r_min/c with 0 < α ≪ 1.

Differences from P&F's estimate (eqs. 10–23):
- r_min uses the largest of *all* frame Riemann components, computed exactly
  (warp.curvature), not one component of a straight-ramp wall. By default the
  maximum is taken over the whole wall ("global"), which gives the shortest
  window and so the loosest limit: a deliberate worst case in the design's
  favour. curvature="worldline" uses only points the sampled observer passes.
- The sampled energy density is integrated along the real geodesic of the worst
  observer, not approximated by its peak value.

Scope: a free massless scalar field (the QI holds for every quantum state of it),
flat-space QI trusted over windows shorter than the curvature time, steady wall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import G, c, hbar
from scipy.integrate import quad, solve_ivp

from warp import curvature, energy

PLANCK_LENGTH = np.sqrt(hbar * G / c**3)
QI_CONSTANT = 3.0 * hbar / (32.0 * np.pi**2 * c**3)  # J·s⁴/m³ ; |ρ|·τ⁴ ≤ QI_CONSTANT


def qi_bound(tau):
    """Most negative allowed sampled energy density (J/m³) for sampling time τ (s)."""
    return -QI_CONSTANT / np.asarray(tau, dtype=float) ** 4


def allowed_duration(rho_abs):
    """Longest time (s) a negative energy density of size |ρ| (J/m³) may persist."""
    return (QI_CONSTANT / np.asarray(rho_abs, dtype=float)) ** 0.25


def max_frame_curvature(beta, f, df, ddf, r_lo, r_hi, n_r=801, n_theta=721):
    """Largest |R_âb̂ĉd̂| (1/m²) over the wall shell r_lo ≤ r ≤ r_hi, all directions.

    Scans a polar grid (r, θ) in a plane through the line of travel; the bubble is
    axially symmetric, so this covers every point of the wall.
    """
    r = np.linspace(r_lo, r_hi, n_r)
    th = np.linspace(0.0, np.pi, n_theta)
    rr, tt = np.meshgrid(r, th)
    S, dS, ddS = curvature.steady_bubble_derivatives(
        0.0, rr * np.cos(tt), rr * np.sin(tt), np.zeros_like(rr), beta, f, df, ddf)
    return float(np.max(curvature.max_riemann(S, dS, ddS)))


def worldline_max_curvature(beta, f, df, ddf, rho0, r_hi, n=20_001):
    """Largest |R_âb̂ĉd̂| (1/m²) along the sampled observer's path through the wall.

    In the bubble's frame the observer moves along the line y = rho0 (z = 0),
    from the equator (X = 0) out to where it leaves the wall; by symmetry the
    other half is the mirror image.
    """
    x_exit = np.sqrt(max(r_hi**2 - rho0**2, 0.0))
    X = np.linspace(0.0, x_exit, n)
    S, dS, ddS = curvature.steady_bubble_derivatives(
        0.0, X, np.full_like(X, rho0), np.zeros_like(X), beta, f, df, ddf)
    return float(np.max(curvature.max_riemann(S, dS, ddS)))


def sampled_energy_density(beta, f, df, rho0, tau_m, r_hi):
    """Lorentzian-sampled energy density (J/m³) along a geodesic Eulerian observer.

    The observer sits at distance rho0 from the line of travel and crosses the
    bubble's equator at t = 0. Times are in metres (t means ct); tau_m = c·τ.
    In the bubble's frame its offset X = x - x_s obeys dX/dt = -β(1 - f(r_s)).
    The motion is symmetric in t, so we integrate t ≥ 0 and double.
    """
    def rhs(t, X):
        r = np.hypot(X[0], rho0)
        return [-beta * (1.0 - float(f(r)))]

    def leaves_wall(t, X):
        return np.hypot(X[0], rho0) - r_hi
    leaves_wall.terminal = True

    rate = beta * max(1.0 - float(f(rho0)), 1e-12)
    t_max = 10.0 * (np.sqrt(max(r_hi**2 - rho0**2, 0.0)) / rate + 50.0 * tau_m)
    sol = solve_ivp(rhs, (0.0, t_max), [0.0], events=leaves_wall, dense_output=True,
                    rtol=1e-11, atol=1e-12 * r_hi, max_step=tau_m)
    t_end = sol.t[-1]

    def integrand(t):
        X = sol.sol(t)[0]
        return float(energy.energy_density(X, rho0, df, beta)) / (t * t + tau_m * tau_m)

    pts = [p for p in (tau_m, 10 * tau_m) if p < t_end]
    val, _ = quad(integrand, 0.0, t_end, points=pts or None, limit=2000, epsabs=0.0, epsrel=1e-9)
    return 2.0 * (tau_m / np.pi) * val


@dataclass
class QIResult:
    beta: float
    alpha: float
    peak_density: float      # J/m³ (negative): worst point, equator at max slope
    sampled_density: float   # J/m³ (negative): Lorentzian-averaged along the worst geodesic
    r_min: float             # m: smallest curvature radius anywhere in the wall
    tau: float               # s: sampling time α·r_min/c
    bound: float             # J/m³ (negative): most negative value the QI allows for τ
    allowed_duration: float  # s: how long |peak_density| may persist
    violation: float         # sampled/bound; > 1 means the QI is violated


def qi_check(beta, alpha, f, df, ddf, r_lo, r_hi, curvature="global"):
    """Apply the QI to a steady wall occupying r_lo ≤ r ≤ r_hi.

    curvature="global": r_min from the largest curvature anywhere in the wall (worst case,
    favours the design). curvature="worldline": only along the sampled observer's path.
    """
    r = np.linspace(r_lo, r_hi, 20_001)
    rho0 = float(r[np.argmax(np.abs(df(r)))])          # equatorial radius of steepest slope
    peak = float(energy.energy_density(0.0, rho0, df, beta))
    if curvature == "global":
        m = max_frame_curvature(beta, f, df, ddf, r_lo, r_hi)
    elif curvature == "worldline":
        m = worldline_max_curvature(beta, f, df, ddf, rho0, r_hi)
    else:
        raise ValueError("curvature must be 'global' or 'worldline'")
    r_min = 1.0 / np.sqrt(m)
    tau = alpha * r_min / c
    sampled = sampled_energy_density(beta, f, df, rho0, alpha * r_min, r_hi)
    bound = float(qi_bound(tau))
    return QIResult(beta, alpha, peak, sampled, r_min, tau, bound,
                    float(allowed_duration(abs(peak))), sampled / bound)


def pf_max_thickness(beta, alpha):
    """P&F eq. 22 (derived from eqs. 17–21): Δ ≤ (3/4)√(3/π)·β·L_P/α²  (metres)."""
    return 0.75 * np.sqrt(3.0 / np.pi) * beta * PLANCK_LENGTH / alpha**2
