"""Task 6: WarpFactory agrees with our own engine (skipped if WarpFactory isn't installed)."""

import numpy as np
import pytest
from scipy.constants import c

pytest.importorskip("warpfactory")

from warp import curvature, energy, shapes  # noqa: E402
from warp import time_dependent as td  # noqa: E402
from warp import warpfactory_check as wf  # noqa: E402
from warp.dual_shell import VARIANTS  # noqa: E402

R, BETA = 100.0, 1.0


def test_paper_alcubierre_matches_ours_and_paper():
    x, y, rho = wf.paper_alcubierre_check(spacing=10.0)
    p = wf.PAPER_ALCUBIERRE
    X, Y = np.meshgrid(x, y, indexing="ij")
    ours = energy.energy_density(X, Y, lambda r: shapes.alcubierre_df(r, p["R"], p["sigma"]), p["v"])
    assert np.max(np.abs(rho - ours)) < 1e-3 * np.max(np.abs(ours))      # 4th-order FD at 10 m
    # the peak sits on the equator (x = 0) where |f'| is largest; the paper reads "≈ −6.8e35"
    r = np.linspace(p["R"] - 100, p["R"] + 100, 200_001)
    peak = energy.energy_density(0.0, r, lambda s: shapes.alcubierre_df(s, p["R"], p["sigma"]), p["v"]).min()
    assert peak == pytest.approx(p["peak_energy_density"], rel=0.01)


@pytest.mark.parametrize("delta", [1.0, 10.0, 188.1, 1000.0])
def test_cosine_wall_total_matches_closed_form(delta):
    m_wf = wf.total_mass(wf.cosine_wall_solver(R, delta, BETA), R, R + delta)
    m_ours = energy.mass_from_wall_integral(energy.cosine_wall_integral_exact(R, delta), BETA)
    assert m_wf == pytest.approx(m_ours, rel=1e-10)


def test_alcubierre_total_matches_ours():
    sigma = 2.0                                     # 1 m wall
    pad = 40.0 / sigma
    m_wf = wf.total_mass(wf.alcubierre_solver(R, sigma, BETA), R - pad, R + pad, splits=(R,), n_r=200)
    integral = energy.wall_integral(lambda r: shapes.alcubierre_df(r, R, sigma), R - pad, R + pad, points=[R])
    assert m_wf == pytest.approx(energy.mass_from_wall_integral(integral, BETA), rel=1e-8)


def _random_wall_points(rng, n, r_lo, r_hi):
    rr = rng.uniform(r_lo, r_hi, n)
    th = rng.uniform(0.01, np.pi - 0.01, n)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    return rr * np.cos(th), rr * np.sin(th) * np.cos(ph), rr * np.sin(th) * np.sin(ph)


def test_moving_cosine_wall_full_tensor_matches_curvature_engine():
    rng = np.random.default_rng(1)
    delta = 188.1
    X, Y, Z = _random_wall_points(rng, 400, R + 1, R + delta - 1)
    t = rng.uniform(-50.0, 50.0, X.size)
    T_wf = wf.eulerian_stress_si(wf.cosine_wall_solver(R, delta, BETA), t, BETA * t + X, Y, Z)
    G = curvature.einstein_frame(*curvature.steady_bubble_derivatives(
        t, BETA * t + X, Y, Z, BETA, lambda r: shapes.cosine_wall_f(r, R, delta),
        lambda r: shapes.cosine_wall_df(r, R, delta), lambda r: shapes.cosine_wall_ddf(r, R, delta)))
    scale = max(np.abs(energy.ENERGY_DENSITY_SCALE * v).max() for v in G.values())
    for (a, b), v in G.items():
        np.testing.assert_allclose(T_wf[a, b], energy.ENERGY_DENSITY_SCALE * v, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_switching_dual_shell_matches_time_dependent(variant):
    params = VARIANTS[variant]
    rng = np.random.default_rng(2)
    X, Y, Z = _random_wall_points(rng, 60, 80.0, 110.0)
    shells = td.dual_shell_shells(params, BETA)
    solver = wf.dual_shell_solver(params, BETA)
    period = 2 * np.pi / params["omega"]
    worst = scale = 0.0
    for i, ts in enumerate(rng.uniform(0.0, period, X.size)):
        ct = c * ts
        T_wf = wf.eulerian_stress_si(solver, ct, solver.centre(ct) + X[i], Y[i], Z[i])
        G = curvature.einstein_frame(*td.shift_derivatives(ts, X[i:i + 1], Y[i:i + 1], Z[i:i + 1], shells))
        for (a, b), v in G.items():
            ours = energy.ENERGY_DENSITY_SCALE * float(v[0])
            worst = max(worst, abs(float(T_wf[a, b]) - ours))
            scale = max(scale, abs(ours))
    assert worst < 1e-12 * scale


def test_energy_conditions_fail_in_the_wall_only():
    delta = 188.1
    x, y, maps = wf.energy_condition_maps(wf.cosine_wall_solver(R, delta, BETA), R + delta + 20,
                                          n=60, num_angular_vec=40, num_time_vec=4)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.hypot(X, Y)
    deep = (r > R + 0.2 * delta) & (r < R + 0.8 * delta)
    outside = (r < R - 1) | (r > R + delta + 1)
    for cond, m in maps.items():
        assert np.all(m[deep & (np.abs(Y) > 5)] < 0), cond
        assert np.all(m[outside] > -1e-9 * np.abs(m).max()), cond
