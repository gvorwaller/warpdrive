"""Time-dependent (switching) bubble tests: derivative engine vs brute force, and what switching can't change."""

import numpy as np
import pytest
from scipy.constants import c

from warp import curvature as cv
from warp import dual_shell as ds
from warp import time_dependent as td
from warp.shapes import shell_profile, shell_profile_dr, shell_profile_drr

AS = ds.VARIANTS["as written"]
T = ds.period(AS)
SHELLS = td.dual_shell_shells(AS, 1.0)


def _S_direct(t, x, y, z, params=AS, beta=1.0):
    """The shift written out directly from its definition, with the centre's path x_s(t)."""
    x_s = ds.bubble_centre(np.array([0.0, t]), beta, params)[-1]
    r = np.sqrt((x - x_s) ** 2 + y**2 + z**2)
    return float(ds.dual_shell_f(r, t, **params)) * beta


def test_shell_second_derivative():
    r = np.linspace(95, 105, 2001)
    h = 1e-5
    fd = (shell_profile_dr(r + h, 100, 2.0, -1) - shell_profile_dr(r - h, 100, 2.0, -1)) / (2 * h)
    np.testing.assert_allclose(shell_profile_drr(r, 100, 2.0, -1), fd, atol=1e-8)


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.6])
@pytest.mark.parametrize("r0", [89.8, 100.3])
def test_derivatives_match_brute_force(theta, r0):
    """∂S and ∂∂S in (ct, x, y, z) against finite differences of the explicit shift."""
    t0 = 0.37 * T
    x_s0 = ds.bubble_centre(np.array([0.0, t0]), 1.0, AS)[-1]
    X0, y0, z0 = r0 * np.cos(theta), r0 * np.sin(theta) * 0.8, r0 * np.sin(theta) * 0.6
    S, dS, ddS = td.shift_derivatives(t0, X0, y0, z0, SHELLS)
    hx, ht = 1e-4, 1e-4 / c          # equal steps in metres of x and of ct; FD error ~ (ω h / c)² ≈ 1e-6
    p0 = np.array([t0, X0 + x_s0, y0, z0])
    steps = np.diag([ht, hx, hx, hx])
    scale = np.array([c, 1, 1, 1])     # d/d(ct) = (1/c) d/dt
    fn = lambda p: _S_direct(*p)
    assert float(S) == pytest.approx(fn(p0), rel=1e-12)
    for a in range(4):
        fd1 = (fn(p0 + steps[a]) - fn(p0 - steps[a])) / (2 * steps[a, a] * scale[a])
        assert float(dS[a]) == pytest.approx(fd1, rel=5e-6, abs=1e-9)
        for b in range(4):
            fd2 = (fn(p0 + steps[a] + steps[b]) - fn(p0 + steps[a] - steps[b])
                   - fn(p0 - steps[a] + steps[b]) + fn(p0 - steps[a] - steps[b])) / (
                4 * steps[a, a] * steps[b, b] * scale[a] * scale[b])
            assert float(ddS[a, b]) == pytest.approx(fd2, rel=2e-4, abs=2e-6), (a, b)


def test_steady_limit_matches_steady_engine():
    """One inward shell with a constant drive is an ordinary steady bubble."""
    beta = 0.7
    sh = [td.Shell(100.0, 2.0, -1, lambda t: beta, lambda t: 0.0, lambda t: 0.0)]
    pts = np.random.default_rng(3).uniform(-110, 110, (500, 3))
    a = td.shift_derivatives(0.0, pts[:, 0], pts[:, 1], pts[:, 2], sh)
    f = lambda r: shell_profile(r, 100.0, 2.0, -1)
    df = lambda r: shell_profile_dr(r, 100.0, 2.0, -1)
    ddf = lambda r: shell_profile_drr(r, 100.0, 2.0, -1)
    b = cv.steady_bubble_derivatives(0.0, pts[:, 0], pts[:, 1], pts[:, 2], beta, f, df, ddf)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, rtol=1e-12, atol=1e-15)


def test_switching_cannot_change_energy_or_momentum_density():
    th, r = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(85, 105, 201))
    X, Y = r * np.cos(th), r * np.sin(th)
    for t in [0.1 * T, 0.37 * T]:
        full = cv.einstein_frame(*td.shift_derivatives(t, X, Y, 0 * X, SHELLS, True))
        frozen = cv.einstein_frame(*td.shift_derivatives(t, X, Y, 0 * X, SHELLS, False))
        for k in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1)]:
            np.testing.assert_array_equal(full[k], frozen[k])
        assert np.max(np.abs(full[(2, 2)] - frozen[(2, 2)])) > 1.0   # but the stresses do change


def test_interior_stays_flat_while_switching():
    th, r = np.meshgrid(np.linspace(0, np.pi, 31), np.linspace(1, 70, 41))   # ≥ 20 m inside the 90 m wall
    X, Y = r * np.cos(th), r * np.sin(th)
    for t in np.linspace(0, T, 7):
        m = cv.max_riemann(*td.shift_derivatives(t, X, Y, 0 * X, SHELLS, True))
        assert np.max(m) < 1e-30          # tanh tails at 20 m: ~e^(-80)


def test_qi_reduces_to_frozen_at_low_frequency():
    slow = {**AS, "omega": 2 * np.pi * 1e3}
    full = td.dual_shell_qi(slow, 1.0, 0.1, True, n_phase=16)
    frozen = td.dual_shell_qi(slow, 1.0, 0.1, False, n_phase=16)
    assert full.r_min == pytest.approx(frozen.r_min, rel=1e-4)
    assert full.violation == pytest.approx(frozen.violation, rel=1e-3)


# ---------------------------------- analytic values, cross-checked by AGY3 (Gemini) independently

def test_curvature_maxima_match_analytic_values():
    w_c = AS["omega"] / c
    assert td.max_curvature(SHELLS, T, True) == pytest.approx(1.0 * w_c * 2.0 / 2, rel=1e-6)          # β(ω/c)(σ/2)
    assert td.max_curvature(SHELLS, T, True, equator_only=True) == pytest.approx(0.5 * w_c, rel=1e-6)  # ½β(ω/c)(σ/2)
    assert td.max_curvature(SHELLS, T, False) == pytest.approx(1.4328, abs=2e-4)                       # AGY3's value


def test_passing_frequency_matches_closed_form():
    """Asymptotically ½ρ_peak τ⁴ = α⁴c²/(64πG ω²): β and σ cancel, f_crit = α²/(2√(6π) t_P)."""
    from scipy.constants import G, hbar
    t_p = np.sqrt(hbar * G / c**5)
    f_crit = 0.1**2 / (2 * np.sqrt(6 * np.pi) * t_p)
    v = td.dual_shell_qi({**AS, "omega": 2 * np.pi * f_crit}, 1.0, 0.1, True, n_phase=16).violation
    assert v == pytest.approx(1.0, rel=1e-3)
    v2 = td.dual_shell_qi({**AS, "omega": 2 * np.pi * f_crit, "sigma": 5.0}, 2.0, 0.1, True, n_phase=16).violation
    assert v2 == pytest.approx(1.0, rel=1e-3)          # independent of β and σ
