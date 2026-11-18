"""Independent cross-check with WarpFactory (spec Task 6, phase 2).

WarpFactory (Helmerich et al. 2024, Class. Quantum Grav. 41 095009) is the published
toolkit that warp-drive researchers use to analyse metrics. We use the maintained Python
port, JohnsterID/WarpFactory, pinned in requirements-warpfactory.txt (the only Python
port with tests, docs and recent activity; see the devlog for the choice).

Everything here hands WarpFactory a *metric* and lets it do all the rest: derivatives,
curvature, the Einstein equations, the Eulerian frame, energy conditions. Nothing from
warp.curvature or warp.energy is used on the WarpFactory side, so agreement is a real
second opinion. Two WarpFactory pipelines are used:

- GridSolver: finite differences on a sampled grid, the method of the MATLAB original
  (used for the paper's Alcubierre case).
- ExactGridSolver: the same curvature algebra, with derivatives of an analytic metric
  computed exactly by hyper-dual numbers (used for the custom metrics).

Coordinates are WarpFactory's: geometric units, (ct, x, y, z) all in metres. The metric
for every design here is ds² = −c²dt² + (dx − S c dt)² + dy² + dz², i.e. lapse 1, flat
space, shift βˣ = −S. Keep evaluation points off the exact bubble centre (r = 0 is a
coordinate singularity of √(x² + y² + z²) for the derivatives).
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c
from warpfactory.grid import (
    ExactGridSolver,
    GridSolver,
    HyperDual,
    alcubierre_metric,
    alcubierre_metric_fn,
    do_frame_transfer,
    eulerian_transformation_matrix,
    get_energy_conditions,
    si_energy_factor,
    stress_energy_to_si,
)

WARPFACTORY_PORT = "JohnsterID/WarpFactory @ 20ffc61 (MIT)"
CONDITIONS = ("Null", "Weak", "Dominant", "Strong")

# WarpFactory paper, Section 4.1 (arXiv 2404.03095): Alcubierre at v = 0.1c, R = 300 m,
# σ = 0.015 /m. The paper's Figure 2 shows a peak Eulerian energy density of about −6.8e35 J/m³.
PAPER_ALCUBIERRE = dict(v=0.1, R=300.0, sigma=0.015, peak_energy_density=-6.8e35)


def shift_metric(shift):
    """WarpFactory metric function for ds² = −dt² + (dx − S dt)² + dy² + dz², S = shift(t, x, y, z)."""
    def metric(t, x, y, z):
        bx = -shift(t, x, y, z)
        return [[-1.0 + bx * bx, bx, 0.0, 0.0],
                [bx, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
    return metric


def _radius(t, x, y, z, beta):
    return np.sqrt((x - beta * t) ** 2 + y ** 2 + z ** 2)


def cosine_wall_shape(r, R, delta):
    """The author's cosine wall: 1 inside R, ½[1 + cos(π(r − R)/Δ)] across the wall, 0 beyond R + Δ.

    Written afresh here (not imported from warp.shapes). On hyper-dual input, the derivatives
    across the wall come from WarpFactory's own hyper-dual cos; outside the wall f is constant.
    f is C¹ but not C²: f'' jumps by π²/(2Δ²) at both edges, and exactly at r = R or R + Δ the
    wall-side value is returned (no two-sided f'' exists there; cf. warp.shapes.cosine_wall_ddf).
    """
    inside = 0.5 * (1.0 + np.cos(np.pi * (r - R) / delta))
    if not isinstance(r, HyperDual):
        return np.where(r < R, 1.0, np.where(r > R + delta, 0.0, inside))
    below, above = r.f < R, r.f > R + delta
    flat = below | above
    return HyperDual(np.where(below, 1.0, np.where(above, 0.0, inside.f)),
                     np.where(flat, 0.0, inside.f1),
                     np.where(flat, 0.0, inside.f2),
                     np.where(flat, 0.0, inside.f12))


def cosine_wall_solver(R, delta, beta):
    """ExactGridSolver for the Vorwaller Metric (cosine wall) moving at β along x."""
    shift = lambda t, x, y, z: beta * cosine_wall_shape(_radius(t, x, y, z, beta), R, delta)
    return ExactGridSolver(shift_metric(shift), name="Vorwaller Metric (cosine wall)")


def alcubierre_solver(R, sigma, beta):
    """ExactGridSolver using WarpFactory's own analytic Alcubierre metric."""
    return ExactGridSolver(alcubierre_metric_fn(beta, R, sigma), name="Alcubierre")


def dual_shell_solver(params, beta):
    """ExactGridSolver for a dual-shell design (a warp.dual_shell parameter set), time-dependent.

    S = β[d₁(t) g₁(r) + d₂(t) g₂(r)], g_k = ½[1 + sign_k tanh(σ(r − R_k))], drives cos(ωt) and
    cos(ωt − phase) (or their squares), r measured from the centre x_s(t), which follows the flow
    at the centre: dx_s/dt = S(r = 0), with x_s(0) = 0. Written afresh from the formulas, not from
    warp.dual_shell. Needs ω > 0.

    Precision note: the pinned port's hyper-dual tanh takes its derivative as 1 − tanh², so far out
    in a shell's tail (|σ(r − R)| ≳ 18) the stress-energy rounds to exactly 0 where the true value is
    tiny but nonzero (≲ 1e-15 of the peak). Agreement is therefore quoted relative to the largest
    component, not pointwise in the tails.
    """
    if not params["omega"] > 0:
        raise ValueError("omega must be positive (the drive antiderivative divides by it)")
    k = params["omega"] / c                       # rad per metre of ct
    ph, s, drive = params["phase"], params["sigma"], params.get("drive", "linear")
    if drive not in ("linear", "squared"):
        raise ValueError("drive must be 'linear' or 'squared'")
    shells = [(params["R1"], params["shell1_sign"], 0.0), (params["R2"], params["shell2_sign"], ph)]
    g = lambda r, R, sign: 0.5 * (1.0 + sign * np.tanh(s * (r - R)))

    def drive_value(t, lag):
        d = np.cos(k * t - lag)
        return d * d if drive == "squared" else d

    def drive_integral(t, lag):                   # ∫₀ᵗ drive dt, exact
        if drive == "squared":
            return t / 2.0 + (np.sin(2.0 * (k * t - lag)) + np.sin(2.0 * lag)) / (4.0 * k)
        return (np.sin(k * t - lag) + np.sin(lag)) / k

    def shift(t, x, y, z):
        xs = beta * sum(g(0.0, R, sign) * drive_integral(t, lag) for R, sign, lag in shells)
        r = np.sqrt((x - xs) ** 2 + y ** 2 + z ** 2)
        return beta * sum(drive_value(t, lag) * g(r, R, sign) for R, sign, lag in shells)

    def centre(t):
        return beta * sum(g(0.0, R, sign) * drive_integral(t, lag) for R, sign, lag in shells)

    solver = ExactGridSolver(shift_metric(shift), name="dual shell")
    solver.centre = centre                        # x_s(ct), metres
    return solver


def eulerian_stress_si(solver, ct, x, y, z):
    """Stress-energy in the Eulerian orthonormal frame, covariant components T_âb̂ in J/m³.

    Shape (4, 4) + broadcast shape. [0, 0] is the energy density the Eulerian observer measures.
    """
    g = solver.metric_at(ct, x, y, z)
    T = solver.stress_energy_at(ct, x, y, z)
    M = eulerian_transformation_matrix(g)
    return np.einsum("am...,ab...,bn...->mn...", M, T, M) * si_energy_factor()


def total_mass(solver, r_lo, r_hi, splits=(), n_r=80, n_theta=80, ct=0.0, x_centre=0.0):
    """E/c² (kg) of the Eulerian energy density at time ct (metres), over the shell r_lo ≤ r ≤ r_hi
    around the bubble centre x_centre (metres along x).

    Gauss–Legendre in r (one block per interval between splits) and in polar angle θ about the
    direction of motion; the bubble is symmetric about that axis, so dV = 2π r² sin θ dr dθ.
    """
    edges = [r_lo, *splits, r_hi]
    gx, gw = np.polynomial.legendre.leggauss(n_r)
    r = np.concatenate([(b - a) / 2 * gx + (a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
    wr = np.concatenate([(b - a) / 2 * gw for a, b in zip(edges[:-1], edges[1:])])
    tx, tw = np.polynomial.legendre.leggauss(n_theta)
    th, wt = np.pi / 2 * (tx + 1), np.pi / 2 * tw
    rr, tt = np.meshgrid(r, th, indexing="ij")
    rho = eulerian_stress_si(solver, ct, x_centre + rr * np.cos(tt), rr * np.sin(tt), 0.0 * rr)[0, 0]
    return float(np.sum(rho * 2 * np.pi * rr**2 * np.sin(tt) * wr[:, None] * wt[None, :])) / c**2


def paper_alcubierre_check(spacing=5.0, half_length=600.0, half_height=450.0):
    """WarpFactory's finite-difference pipeline (as in the MATLAB original) on the paper's case.

    Returns the grid's x, y (z = 0 plane, y ≥ 0) and the Eulerian energy density there (J/m³),
    away from the grid edges where the 4th-order stencil is one-sided. The grid extends 4 rows
    below y = 0 so the axis is interior, not a grid edge. The peak is on the equator near r = R,
    so half_height must reach well past R.
    """
    p = PAPER_ALCUBIERRE
    pad = 4
    nx, ny, nz = int(2 * half_length / spacing) + 1, int(half_height / spacing) + 1 + pad, 9
    centre = (0.0, (nx - 1) * spacing / 2, pad * spacing, (nz - 1) * spacing / 2)
    metric = alcubierre_metric((1, nx, ny, nz), centre, v=p["v"], R=p["R"], sigma=p["sigma"],
                               grid_scale=(1, spacing, spacing, spacing))
    T = GridSolver(order=4).solve(metric)
    rho = stress_energy_to_si(do_frame_transfer(metric, T)).tensor[0, 0, 0, :, :, nz // 2]
    x = np.arange(nx) * spacing - centre[1]
    y = np.arange(ny) * spacing - centre[2]
    keep = (slice(pad, -pad), slice(pad, -pad))   # first kept row is y = 0, with 4 rows below it
    return x[keep[0]], y[keep[1]], rho[keep]


def energy_condition_maps(solver, half_width, n=160, num_angular_vec=100, num_time_vec=10):
    """WarpFactory's energy-condition maps on the z = 0 plane at ct = 0 (x along the motion).

    For each condition, WarpFactory's most-violating *score* over its sampled observer field at
    every point (scaled to J/m³); negative means a violation was found. These are diagnostic
    scores, not energies a physical observer measures: the port normalises observer vectors by
    Euclidean length and its "timelike" field includes speed c. Sampling can miss a violation,
    so a non-negative score means "none detected", not "the condition holds".

    Also returned: axial_null, the null-condition contraction T_âb̂ k^â k^b̂ for the six axial null
    directions k = (1, ±ê)/√2 (same normalisation), computed here from WarpFactory's own
    Eulerian-frame tensor. For a null k the strong condition's trace term drops out, so the same
    value also witnesses the strong condition. Note the pinned port's dominant-condition test
    checks only that the energy flux is not spacelike, not that it is future-directed: its
    negative scores are genuine violations, but it is a narrower test than the full condition.

    n must be even, which keeps the bubble centre (a coordinate singularity of r) off the grid.
    Returns (x, y, {condition: map}, axial_null); x, y are pixel centres, spacing 2·half_width/n.
    """
    if n % 2:
        raise ValueError("n must be even (an odd grid puts a point at the bubble centre)")
    h = 2.0 * half_width / n
    size, centre, scale = (1, n, n, 1), (0.0, half_width - h / 2, half_width - h / 2, 0.0), (1, h, h, h)
    metric = solver.metric_on_grid(size, centre, scale)
    T = solver.solve(size, centre, scale)
    maps = {cond: get_energy_conditions(T, metric, cond, num_angular_vec=num_angular_vec,
                                        num_time_vec=num_time_vec)[0, :, :, 0] * si_energy_factor()
            for cond in CONDITIONS}
    local = do_frame_transfer(metric, T).tensor[:, :, 0, :, :, 0]   # contravariant, Eulerian frame
    T_low = local * np.array([-1.0, 1, 1, 1])[:, None, None, None] * np.array([-1.0, 1, 1, 1])[None, :, None, None]
    axial = np.full(local.shape[2:], np.inf)
    for i in (1, 2, 3):
        for sgn in (1.0, -1.0):
            k = np.zeros(4)
            k[0], k[i] = 1.0 / np.sqrt(2.0), sgn / np.sqrt(2.0)
            axial = np.minimum(axial, np.einsum("mn...,m,n->...", T_low, k, k))
    x = np.arange(n) * h - centre[1]
    return x, x.copy(), maps, axial * si_energy_factor()
