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


def max_frame_curvature(beta, f, df, ddf, r_lo, r_hi, n_r=801, n_theta=721, n_fine=200_001, refine=True):
    """Largest |R_âb̂ĉd̂| (1/m²) over the wall shell r_lo ≤ r ≤ r_hi, all directions.

    The bubble is axially symmetric, so a polar grid (r, θ) in one plane through the
    line of travel covers every point of the wall. The radial grid is built to resolve
    the profile: a fine 1-D pass (n_fine points) finds the region where f', f'' are
    non-negligible and their length scale, and that region gets ≥ 40 points per scale.
    The best grid points are then refined with a bounded local optimiser. Walls with
    features finer than (r_hi - r_lo)/n_fine need a larger n_fine.
    """
    rf = np.linspace(r_lo, r_hi, n_fine)
    a = np.abs(df(rf))
    b = np.abs(ddf(rf))
    a_max, b_max = float(a.max()), float(b.max())
    scale = 1.0 / max(a_max, np.sqrt(b_max), 1.0 / (r_hi - r_lo))
    active = (a > 1e-10 * a_max) | (b > 1e-10 * b_max) if a_max > 0 or b_max > 0 else np.zeros_like(a, bool)
    pieces = [np.linspace(r_lo, r_hi, n_r), [rf[np.argmax(a)], rf[np.argmax(b)]]]
    if active.any():
        step = rf[1] - rf[0]
        lo_a = max(r_lo, rf[active].min() - step)
        hi_a = min(r_hi, rf[active].max() + step)
        n_a = int(min(max((hi_a - lo_a) / (scale / 40.0), 2), 40_001))
        pieces.append(np.linspace(lo_a, hi_a, n_a))
    r = np.unique(np.concatenate([np.atleast_1d(p_) for p_ in pieces]))
    th = np.linspace(0.0, np.pi, n_theta)

    def m_at(rr, tt):
        S, dS, ddS = curvature.steady_bubble_derivatives(
            0.0, rr * np.cos(tt), rr * np.sin(tt), np.zeros_like(rr), beta, f, df, ddf)
        return curvature.max_riemann(S, dS, ddS)

    best_val = np.full(len(th), -np.inf)
    best_r = np.zeros(len(th))
    chunk = max(1, 2_000_000 // len(r))              # bound memory: ~2M points per evaluation
    for i in range(0, len(th), chunk):
        tt, rr = np.meshgrid(th[i:i + chunk], r, indexing="ij")
        m = m_at(rr, tt)
        j = np.argmax(m, axis=1)
        best_val[i:i + chunk] = m[np.arange(m.shape[0]), j]
        best_r[i:i + chunk] = r[j]
    result = float(best_val.max())

    if refine:
        from scipy.optimize import minimize
        for k in np.argsort(best_val)[-6:]:
            x0 = [best_r[k], th[k]]
            res = minimize(lambda p_: -float(m_at(np.array(p_[0]), np.array(p_[1]))), x0,
                           method="Nelder-Mead", bounds=[(r_lo, r_hi), (0.0, np.pi)],
                           options={"xatol": scale * 1e-6, "fatol": 0.0, "maxiter": 2000})
            result = max(result, -float(res.fun))
    return result


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

    qi_check uses the observer whose path crosses the point of strongest energy
    density; that is not guaranteed to be the observer with the strongest *sampled*
    average (for a 1000 m wall a nearby path is ~0.08% stronger), which doesn't change
    any conclusion. The observer sits at distance rho0 from the line of travel and crosses the
    bubble's equator at t = 0. Times are in metres (t means ct); tau_m = c·τ.
    In the bubble's frame its offset X = x - x_s obeys dX/dt = -β(1 - f(r_s)).
    The motion is symmetric in t, so we integrate t ≥ 0 and double. The Lorentzian
    is resolved by substituting t = τ·tan(u), which turns the sampling integral into
    (2/π) ∫ ρ(t(u)) du over u ∈ [0, arctan(t_exit/τ)].
    """
    if not tau_m > 0:
        raise ValueError("tau_m must be positive")
    if beta == 0:
        return 0.0

    def rho_at(X):
        return float(energy.energy_density(X, rho0, df, beta))

    if rho0 >= r_hi:
        return rho_at(0.0) if rho0 == r_hi else 0.0
    drift = abs(beta) * (1.0 - float(f(rho0)))
    if drift <= 1e-14 * abs(beta):
        # co-moving with the bubble here: the observer stays at r_s = rho0 forever,
        # so the sample is the constant value (the Lorentzian weights integrate to 1)
        return rho_at(0.0)

    def rhs(t, X):
        return [-beta * (1.0 - float(f(np.hypot(X[0], rho0))))]

    def leaves_wall(t, X):
        return np.hypot(X[0], rho0) - r_hi
    leaves_wall.terminal = True

    x_exit = np.sqrt(r_hi**2 - rho0**2)
    t_max = 100.0 * x_exit / drift + 100.0 * tau_m
    sol = solve_ivp(rhs, (0.0, t_max), [0.0], method="DOP853", events=leaves_wall,
                    dense_output=True, rtol=1e-12, atol=1e-12 * r_hi)
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise RuntimeError(f"observer did not leave the wall within t = {t_max:.3g} m "
                           f"(solver status {sol.status}: {sol.message})")
    t_exit = float(sol.t_events[0][0])
    u_exit = np.arctan(t_exit / tau_m)
    val, _ = quad(lambda u: rho_at(sol.sol(tau_m * np.tan(u))[0]), 0.0, u_exit,
                  limit=500, epsabs=0.0, epsrel=1e-10)
    return 2.0 / np.pi * val


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
