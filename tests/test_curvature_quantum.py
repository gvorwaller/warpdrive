"""Curvature and quantum-inequality tests: checked against Alcubierre eq. 19, P&F eqs. 18 and 22, and analytic limits."""

import itertools

import numpy as np
import pytest
import sympy as sp
from scipy.constants import G, c, hbar

from warp import curvature as cv
from warp import quantum as q
from warp import shapes

R = 100.0


def tanh_funcs(sigma):
    return (lambda r: shapes.alcubierre_f(r, R, sigma), lambda r: shapes.alcubierre_df(r, R, sigma),
            lambda r: shapes.alcubierre_ddf(r, R, sigma))


def cos_funcs(delta):
    return (lambda r: shapes.cosine_wall_f(r, R, delta), lambda r: shapes.cosine_wall_df(r, R, delta),
            lambda r: shapes.cosine_wall_ddf(r, R, delta))


# ------------------------------------------------------------------ second derivatives

@pytest.mark.parametrize("funcs,lo,hi", [(tanh_funcs(0.5), 1.0, 150.0), (cos_funcs(10.0), 100.5, 109.5)])
def test_second_derivative_matches_finite_difference(funcs, lo, hi):
    _, df, ddf = funcs
    r = np.linspace(lo, hi, 999)
    h = 1e-5
    np.testing.assert_allclose(ddf(r), (df(r + h) - df(r - h)) / (2 * h), atol=1e-7)


# -------------------------------------------------------------------------- curvature

@pytest.mark.parametrize("funcs,beta,t", [(tanh_funcs(0.2), 1.3, 0.37), (cos_funcs(20.0), 0.7, -5.0)])
def test_einstein_energy_density_matches_alcubierre_eq19(funcs, beta, t):
    """G_n̂n̂ = -(β²/4)(ρ²/r²) f'²: the whole symbolic pipeline vs. the published formula."""
    f, df, ddf = funcs
    pts = np.random.default_rng(7).uniform(-130, 130, (3000, 3))
    S, dS, ddS = cv.steady_bubble_derivatives(t, pts[:, 0], pts[:, 1], pts[:, 2], beta, f, df, ddf)
    G00 = cv.einstein_frame(S, dS, ddS)[(0, 0)]
    X = pts[:, 0] - beta * t
    rho2 = pts[:, 1] ** 2 + pts[:, 2] ** 2
    r = np.sqrt(X**2 + rho2)
    np.testing.assert_allclose(G00, -(beta**2 / 4) * rho2 / r**2 * df(r) ** 2, atol=1e-15)


def test_constant_shift_is_flat():
    """S = constant is just a moving coordinate system: zero curvature."""
    S = np.full(5, 0.6)
    zero = np.zeros((4, 5))
    assert np.all(cv.max_riemann(S, zero, np.zeros((4, 4, 5))) == 0.0)


@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_pfenning_ford_eq18_at_equator(beta):
    """|R_t̂ŷt̂ŷ| = (3/4) β² f'² on the equator (y = ρ), P&F eq. 18."""
    f, df, ddf = tanh_funcs(2.0)
    rho = np.linspace(98.0, 102.0, 401)
    S, dS, ddS = cv.steady_bubble_derivatives(0.0, 0 * rho, rho, 0 * rho, beta, f, df, ddf)
    tyty = cv.riemann_frame(S, dS, ddS)[(0, 2, 0, 2)]
    np.testing.assert_allclose(np.abs(tyty), 0.75 * beta**2 * df(rho) ** 2, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("beta,delta", [(1.0, 1.0), (10.0, 1.0), (1.0, 188.0), (0.5, 1.0)])
def test_cosine_wall_max_curvature_is_beta2_fpp(beta, delta):
    """For β ≥ 0.5 the largest frame component is R_t̂x̂t̂x̂ = β² f'' at the outer edge on the axis:
    β²π²/(2Δ²) = (8/3)·P&F eq. 18."""
    f, df, ddf = cos_funcs(delta)
    m = q.max_frame_curvature(beta, f, df, ddf, R, R + delta)
    assert m == pytest.approx(beta**2 * np.pi**2 / (2 * delta**2), rel=1e-9)


def test_slow_bubble_max_curvature_is_not_beta2_fpp():
    """Below β ≈ 0.5 a component linear in β f'' wins, so the β² f'' shortcut must not be used there."""
    f, df, ddf = cos_funcs(1.0)
    beta = 0.3
    m = q.max_frame_curvature(beta, f, df, ddf, R, R + 1.0)
    assert m > 1.5 * beta**2 * np.pi**2 / 2


def test_all_frame_components_against_independent_explicit_metric():
    """Independent check of all 21 components: explicit Gaussian bubble, own Christoffel/Riemann code."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    b, w = sp.Rational(7, 10), sp.Integer(3)
    S = b * sp.exp(-((x - b * t) ** 2 + y**2 + z**2) / w**2)
    X = (t, x, y, z)
    g = sp.Matrix([[-1 + S**2, -S, 0, 0], [-S, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gi = g.inv()
    Gam = [[[sum(gi[a, d] * (sp.diff(g[d, i], X[j]) + sp.diff(g[d, j], X[i]) - sp.diff(g[i, j], X[d]))
                 for d in range(4)) / 2 for j in range(4)] for i in range(4)] for a in range(4)]
    pt = {t: sp.Rational(1, 3), x: sp.Rational(9, 10), y: sp.Rational(13, 10), z: sp.Rational(-1, 2)}

    def Rup(a, bb, c, d):
        e = sp.diff(Gam[a][d][bb], X[c]) - sp.diff(Gam[a][c][bb], X[d])
        e += sum(Gam[a][c][k] * Gam[k][d][bb] - Gam[a][d][k] * Gam[k][c][bb] for k in range(4))
        return e

    E = sp.Matrix([[1, S, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    Rup_cache = {}
    expected = {}
    for (A, B), (C, D) in itertools.combinations_with_replacement(pairs, 2):
        val = 0
        for a, bb, c, d in itertools.product(range(4), repeat=4):
            coeff = (E[A, a] * E[B, bb] * E[C, c] * E[D, d]).subs(pt)
            if coeff != 0:
                if (bb, c, d) not in Rup_cache:
                    Rup_cache[(bb, c, d)] = [Rup(e, bb, c, d).subs(pt) for e in range(4)]
                val += coeff * sum(g[a, e].subs(pt) * Rup_cache[(bb, c, d)][e] for e in range(4))
        expected[(A, B, C, D)] = float(sp.N(val, 30))

    bf, wf = 0.7, 3.0
    f = lambda r: np.exp(-(r**2) / wf**2)
    df = lambda r: -2 * r / wf**2 * f(r)
    ddf = lambda r: (4 * r**2 / wf**4 - 2 / wf**2) * f(r)
    Sn, dS, ddS = cv.steady_bubble_derivatives(1 / 3, 0.9, 1.3, -0.5, bf, f, df, ddf)
    got = cv.riemann_frame(Sn, dS, ddS)
    assert set(got) == set(expected) and len(got) == 21
    for k in expected:
        assert float(got[k]) == pytest.approx(expected[k], rel=1e-10, abs=1e-14), k


# ------------------------------------------------------------------ quantum inequality

def test_constants():
    assert q.PLANCK_LENGTH == pytest.approx(1.616255e-35, rel=1e-6)
    assert q.QI_CONSTANT == pytest.approx(3.7178e-62, rel=1e-4)
    tau = 1e-9
    assert q.allowed_duration(-q.qi_bound(tau)) == pytest.approx(tau, rel=1e-12)


@pytest.mark.parametrize("delta", [0.1, 1.0, 188.0])
def test_violation_matches_pfenning_ford_up_to_exact_factor(delta):
    """Consistency with P&F eq. 22: with the global (worst-case) curvature, our V = P&F's (Δ_eff/Δ_max)²
    ÷ (8/3)², times the sampled/peak ratio (the only places the two calculations differ)."""
    f, df, ddf = cos_funcs(delta)
    res = q.qi_check(1.0, 0.1, f, df, ddf, R, R + delta)
    v_pf = (2 * delta / np.pi / q.pf_max_thickness(1.0, 0.1)) ** 2
    ratio = res.sampled_density / res.peak_density
    assert 0.9 < ratio <= 1.0
    assert res.violation == pytest.approx(v_pf / (8 / 3) ** 2 * ratio, rel=1e-6)


def test_sampled_density_approaches_peak_for_short_sampling():
    f, df, _ = cos_funcs(1.0)
    rho0 = R + 0.5
    peak = -q.energy.ENERGY_DENSITY_SCALE / 4 * (np.pi / 2) ** 2
    assert q.sampled_energy_density(1.0, f, df, rho0, 1e-4, R + 1.0) == pytest.approx(peak, rel=1e-3)


def test_violation_scaling_with_alpha_and_beta():
    """V ∝ α⁴/β² (P&F): sampling time ∝ α/β, density ∝ β²."""
    f, df, ddf = cos_funcs(1.0)
    base = q.qi_check(1.0, 0.1, f, df, ddf, R, R + 1.0)
    a = q.qi_check(1.0, 0.2, f, df, ddf, R, R + 1.0)
    b = q.qi_check(2.0, 0.1, f, df, ddf, R, R + 1.0)
    assert a.violation / base.violation == pytest.approx(16.0, rel=0.02)   # sampled/peak shifts slightly
    assert b.violation / base.violation == pytest.approx(0.25, rel=0.02)


def test_qi_numbers_in_si():
    """Peak density for a 1 m cosine wall at v = c: (c⁴/8πG)(1/4)(π/2)² J/m³."""
    f, df, ddf = cos_funcs(1.0)
    res = q.qi_check(1.0, 0.1, f, df, ddf, R, R + 1.0)
    assert res.peak_density == pytest.approx(-(c**4 / (8 * np.pi * G)) * 0.25 * (np.pi / 2) ** 2, rel=1e-9)
    assert res.r_min == pytest.approx(np.sqrt(2) / np.pi, rel=1e-9)
    # closed form: V = π c³ α⁴ / (12 ħ G β² f'max²) × sampled/peak, f'max = π/2 per metre
    v_closed = np.pi * c**3 * 0.1**4 / (12 * hbar * G * (np.pi / 2) ** 2) * res.sampled_density / res.peak_density
    assert res.violation == pytest.approx(v_closed, rel=1e-9)


def test_worldline_curvature_is_smaller_than_global():
    """The observer's own path never reaches the axis edge, so its window is longer and the limit tighter."""
    f, df, ddf = cos_funcs(1.0)
    g = q.qi_check(1.0, 0.1, f, df, ddf, R, R + 1.0)
    w = q.qi_check(1.0, 0.1, f, df, ddf, R, R + 1.0, curvature="worldline")
    assert w.r_min > g.r_min
    assert w.violation > g.violation
    # largest component on the path vs P&F eq. 18's (3/4)(π/2)²: ≈ 1.32
    assert 1 / w.r_min**2 / (0.75 * (np.pi / 2) ** 2) == pytest.approx(1.32, abs=0.01)
