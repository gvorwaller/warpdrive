"""Stresses and curvature of time-dependent (switching) bubbles: spec Task 5 (revised).

General shift built from shells with time-varying drives:

    S(t, x, y, z) = Σ_k a_k(t) g_k(r),   r = |(x − x_s(t), y, z)|

in geometric units (c = G = 1, lengths and ct in metres). a_k(t) is in units of c
(β times the drive), and g_k(r) is the shell's radial profile. The centre follows
the flow at the centre, dx_s/dt = u(t) = Σ a_k g_k(0) (see warp.dual_shell), so
positions here are given relative to the centre: X = x − x_s(t).

What switching can and cannot change (from the symbolic tensors in warp.curvature):
∂²S/∂t² never appears; time derivatives enter only as ∂_t∂_iS. So the energy
density G_0̂0̂ and momentum density G_0̂î are unchanged by switching, while the
stresses G_îĵ (ĵ = y, z) and the Riemann tensor gain terms proportional to the
drive rate. ``drive_terms=False`` gives the "frozen" comparison: the same
instantaneous shape carried along at the centre's speed, with no switching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c

from warp import curvature, energy, quantum
from warp.shapes import shell_profile, shell_profile_dr, shell_profile_drr, time_factor_derivatives


@dataclass
class Shell:
    """One shell: profile g(r) = ½[1 + sign·tanh(σ(r − R))] driven by a(t) (units of c)."""
    R: float
    sigma: float
    sign: int
    a: object      # a(t)  -> array, dimensionless
    a_t: object    # da/dt -> array, per second
    a_tt: object   # d²a/dt² -> array, per second²


def dual_shell_shells(params, beta):
    """The two shells of a warp.dual_shell parameter set, at speed scale β."""
    def factor(k, order):
        def fn(t):
            vals = time_factor_derivatives(t, params["omega"], params["phase"], params.get("drive", "linear"))
            return beta * vals[2 * order + k]
        return fn
    return [Shell(params["R1"], params["sigma"], params["shell1_sign"], factor(0, 0), factor(0, 1), factor(0, 2)),
            Shell(params["R2"], params["sigma"], params["shell2_sign"], factor(1, 0), factor(1, 1), factor(1, 2))]


def shift_derivatives(t, X, y, z, shells, drive_terms=True):
    """S, ∂S, ∂∂S (geometric units) at time t (seconds) and centre-relative positions (metres).

    Coordinates are (ct, x, y, z). With drive_terms=False the drives' time derivatives
    are set to zero (the frozen comparison); the centre still moves at u(t).
    """
    X, y, z = np.broadcast_arrays(np.asarray(X, float), np.asarray(y, float), np.asarray(z, float))
    t = float(t)
    r = np.sqrt(X * X + y * y + z * z)
    if np.any(r == 0):
        raise ValueError("evaluate away from the exact centre (the interior is flat there anyway)")
    n = np.stack([X / r, y / r, z / r])
    ddr_sp = np.empty((3, 3) + r.shape)
    for i in range(3):
        for j in range(3):
            ddr_sp[i, j] = ((1.0 if i == j else 0.0) - n[i] * n[j]) / r

    F = Fr = Frr = Ft = Frt = Ftt = 0.0
    u = udot = 0.0
    for sh in shells:
        a = float(sh.a(t))
        a_t = float(sh.a_t(t)) / c if drive_terms else 0.0          # per metre of ct
        a_tt = float(sh.a_tt(t)) / c**2 if drive_terms else 0.0
        g = shell_profile(r, sh.R, sh.sigma, sh.sign)
        g1 = shell_profile_dr(r, sh.R, sh.sigma, sh.sign)
        g2 = shell_profile_drr(r, sh.R, sh.sigma, sh.sign)
        F = F + a * g
        Fr = Fr + a * g1
        Frr = Frr + a * g2
        Ft = Ft + a_t * g
        Frt = Frt + a_t * g1
        Ftt = Ftt + a_tt * g
        g0 = float(shell_profile(0.0, sh.R, sh.sigma, sh.sign))
        u += a * g0
        udot += a_t * g0

    # derivatives of r in (ct, x, y, z), with ∂_t X = −u
    dr = np.concatenate([(-u * n[0])[None], n])
    ddr = np.empty((4, 4) + r.shape)
    ddr[1:, 1:] = ddr_sp
    ddr[0, 1:] = ddr[1:, 0] = -u * ddr_sp[0]
    ddr[0, 0] = -udot * n[0] + u * u * ddr_sp[0, 0]

    S = F
    dS = Fr * dr
    dS[0] = dS[0] + Ft
    ddS = Frr * np.einsum("a...,b...->ab...", dr, dr) + Fr * ddr
    for i in range(4):
        ddS[0, i] = ddS[0, i] + Frt * dr[i]
        ddS[i, 0] = ddS[i, 0] + Frt * dr[i]
    ddS[0, 0] = ddS[0, 0] + Ftt
    return S, dS, ddS


STRESS_KEYS = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]


def stresses_si(t, X, y, z, shells, drive_terms=True):
    """Eulerian-frame stress components T_îĵ = (c⁴/8πG) G_îĵ in pascals (J/m³)."""
    G = curvature.einstein_frame(*shift_derivatives(t, X, y, z, shells, drive_terms))
    return {k: energy.ENERGY_DENSITY_SCALE * G[k] for k in STRESS_KEYS}


def _wall_radii(shells, n_r=441, pad_widths=12.0):
    lo = min(sh.R - pad_widths / sh.sigma for sh in shells)
    hi = max(sh.R + pad_widths / sh.sigma for sh in shells)
    return max(lo, 1e-3), hi, np.linspace(max(lo, 1e-3), hi, n_r)


def max_over_cycle(shells, period, quantity, n_phase=64, n_r=441, n_theta=61, equator_only=False, refine=True):
    """Max of quantity(t, X, Y) over the walls and one cycle: a grid in (r, θ, t), then a bounded
    Nelder–Mead refinement from the best few grid points. equator_only restricts to θ = 90°
    (X = 0), where the energy density peaks."""
    from scipy.optimize import minimize
    r_lo, r_hi, r = _wall_radii(shells, n_r)
    th = np.array([np.pi / 2]) if equator_only else np.linspace(0.0, np.pi, n_theta)
    rr, tt = np.meshgrid(r, th)
    X, Y = rr * np.cos(tt), rr * np.sin(tt)
    cands = []
    for t in np.linspace(0.0, period, n_phase, endpoint=False):
        q = quantity(t, X, Y)
        k = np.unravel_index(np.argmax(q), q.shape)
        cands.append((float(q[k]), float(rr[k]), float(tt[k]), float(t)))
    cands.sort(reverse=True)
    best = cands[0][0]
    if refine:
        def neg(p):
            r_, th_, t_ = p if not equator_only else (p[0], np.pi / 2, p[1])
            return -float(quantity(t_, np.array(r_ * np.cos(th_)), np.array(r_ * np.sin(th_))))
        dr = (r_hi - r_lo) / max(n_r - 1, 1)
        for val, r0, th0, t0 in cands[:8]:
            x0 = [r0, t0] if equator_only else [r0, th0, t0]
            bounds = [(r_lo, r_hi), (-period, 2 * period)] if equator_only else \
                [(r_lo, r_hi), (0.0, np.pi), (-period, 2 * period)]
            # one grid step per unit, so the simplex starts at the grid's own resolution
            scale = np.array([dr, period / n_phase] if equator_only else [dr, np.pi / max(n_theta - 1, 1), period / n_phase])
            dim = len(x0)
            simplex = np.vstack([np.zeros(dim), np.eye(dim)])
            res = minimize(lambda z: neg(np.asarray(x0) + z * scale), np.zeros(dim), method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 0.0, "maxiter": 6000, "initial_simplex": simplex})
            z = np.asarray(x0) + res.x * scale
            if all(lo <= zi <= hi for zi, (lo, hi) in zip(z, bounds)):
                best = max(best, -float(res.fun))
    return best


def max_curvature(shells, period, drive_terms=True, **kw):
    """Largest |R_âb̂ĉd̂| (1/m²) anywhere in the walls at any moment of the cycle."""
    return max_over_cycle(shells, period, lambda t, X, Y: curvature.max_riemann(
        *shift_derivatives(t, X, Y, 0.0 * X, shells, drive_terms)), **kw)


def max_stress(shells, period, drive_terms=True, **kw):
    """Largest |stress component| (Pa) anywhere in the walls at any moment of the cycle."""
    def q(t, X, Y):
        s = stresses_si(t, X, Y, 0.0 * X, shells, drive_terms)
        return np.max(np.abs(np.stack(list(s.values()))), axis=0)
    return max_over_cycle(shells, period, q, **kw)


def switching_effect(shells, period, **kw):
    """Largest change switching makes to any frame Riemann component, anywhere in the walls at any
    moment, divided by the largest frozen component: how much the drive rate matters."""
    def diff(t, X, Y):
        full = curvature.riemann_frame(*shift_derivatives(t, X, Y, 0.0 * X, shells, True))
        frozen = curvature.riemann_frame(*shift_derivatives(t, X, Y, 0.0 * X, shells, False))
        return np.max(np.abs(np.stack([full[k] - frozen[k] for k in full])), axis=0)
    return max_over_cycle(shells, period, diff, **kw) / max_curvature(shells, period, False, **kw)


@dataclass
class SwitchingQI:
    frequency: float     # Hz
    r_min: float         # m
    tau: float           # s, sampling window α r_min / c
    sampled: float       # J/m³ (negative)
    violation: float     # sampled / bound; > 1 means the QI is violated


def dual_shell_qi(params, beta, alpha, drive_terms=True, equator_only=False, **kw):
    """Quantum-inequality test for a dual-shell design, same conventions as warp.quantum.

    Worst point: a shell's equator when its drive is at full strength, where the energy
    density is (c⁴/8πG)(β²/4)(σ/2)². The window τ = α r_min/c uses the largest curvature
    anywhere in the walls over the cycle (favouring the design, as in Section 3). At that
    point the energy density goes as the drive squared, cos²(ωt); its Lorentzian average
    over a window τ centred on the peak is ½(1 + e^(−2ωτ)) of the peak (exact for a
    Lorentzian, whose Fourier transform is e^(−|k|τ)). The bubble's rocking (centimetres)
    is neglected.

    equator_only=True takes r_min only from the equator, where that worst observer actually
    sits. Switching adds no curvature there (∂_t∂_xS vanishes when n_x = 0), so this
    stricter version is essentially the frozen result at any frequency.
    """
    omega = params["omega"]
    period = 2 * np.pi / omega
    shells = dual_shell_shells(params, beta)
    m = max_curvature(shells, period, drive_terms, equator_only=equator_only, **kw)
    r_min = 1.0 / np.sqrt(m)
    tau = alpha * r_min / c
    peak = -energy.ENERGY_DENSITY_SCALE * beta**2 / 4.0 * (params["sigma"] / 2.0) ** 2
    if params.get("drive", "linear") != "linear":
        raise NotImplementedError("sampling factor derived for linear (cos/sin) drives")
    factor = 0.5 * (1.0 + np.exp(-2.0 * omega * tau)) if drive_terms else 1.0
    sampled = peak * factor
    return SwitchingQI(omega / (2 * np.pi), r_min, tau, sampled, sampled / float(quantum.qi_bound(tau)))
