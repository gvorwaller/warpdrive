"""Shape-function tests (spec: f≈1 at centre, f≈0 far away, derivative peaks where expected)."""

import numpy as np
import pytest

from warp.shapes import (
    DUAL_SHELL_AS_WRITTEN,
    alcubierre_df,
    alcubierre_f,
    alcubierre_wall_thickness,
    cosine_wall_ddf,
    cosine_wall_df,
    cosine_wall_f,
    cosine_wall_max_slope,
    cosine_wall_pf_thickness,
    pf_ramp_df,
    pf_ramp_f,
    dual_shell_df,
    dual_shell_f,
    sigma_for_wall_thickness,
)

R = 100.0


def _finite_diff(fn, r, h):
    return (fn(r + h) - fn(r - h)) / (2.0 * h)


# --------------------------------------------------------------------- Alcubierre

@pytest.mark.parametrize("sigma", [0.02, 0.2, 2.0, 20.0])
def test_alcubierre_centre_and_far(sigma):
    assert alcubierre_f(0.0, R, sigma) == pytest.approx(1.0, abs=1e-12)
    far = R + 60.0 / sigma  # many wall-widths outside
    assert alcubierre_f(far, R, sigma) == pytest.approx(0.0, abs=1e-12)
    assert alcubierre_f(far, R, sigma) >= 0.0


def test_alcubierre_matches_literal_formula():
    """Our stable form equals the formula as written wherever the literal one is accurate."""
    sigma = 0.2
    r = np.linspace(0.0, 200.0, 2001)
    literal = (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / (2.0 * np.tanh(sigma * R))
    np.testing.assert_allclose(alcubierre_f(r, R, sigma), literal, atol=1e-14)


def test_alcubierre_far_tail_keeps_precision():
    """Literal tanh difference cancels to exactly 0 here; stable form keeps the true tiny value."""
    sigma, r = 2.0, 110.0  # 10 m outside, σΔr = 20
    expected = np.exp(-2.0 * sigma * (r - R)) * (1.0 - np.exp(-4.0 * sigma * R))  # leading terms
    assert alcubierre_f(r, R, sigma) == pytest.approx(expected, rel=1e-9, abs=0)


@pytest.mark.parametrize("sigma", [0.2, 2.0])
def test_alcubierre_derivative_matches_finite_difference(sigma):
    r = np.linspace(0.5, 150.0, 997)
    h = 1e-4 / sigma
    np.testing.assert_allclose(
        alcubierre_df(r, R, sigma), _finite_diff(lambda x: alcubierre_f(x, R, sigma), r, h), atol=1e-7
    )


def test_alcubierre_derivative_peaks_at_R():
    sigma = 2.0
    r = np.linspace(95.0, 105.0, 100_001)
    assert r[np.argmin(alcubierre_df(r, R, sigma))] == pytest.approx(R, abs=1e-3)
    # Peak steepness equals 1/Δ from Pfenning & Ford eq. 5
    assert -alcubierre_df(R, R, sigma) == pytest.approx(1.0 / alcubierre_wall_thickness(R, sigma), rel=1e-12)


def test_no_overflow_at_huge_radius():
    r = np.array([0.0, 1e3, 1e6, 1e12])
    with np.errstate(all="raise"):
        f = alcubierre_f(r, R, 2.0)
        df = alcubierre_df(r, R, 2.0)
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(df))


# ---------------------------------------------------- wall thickness ↔ σ (P&F eq. 5)

def test_one_metre_wall_is_sigma_two():
    assert sigma_for_wall_thickness(1.0, R) == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("delta", [0.001, 1.0, 10.0, 100.0, 1e4, 1e7, 1e10])
def test_sigma_round_trip(delta):
    s = sigma_for_wall_thickness(delta, R)
    assert alcubierre_wall_thickness(R, s) == pytest.approx(delta, rel=1e-10)


# ---------------------------------------------------------------------- cosine wall

DELTAS = [1.0, 10.0, 100.0, 1000.0]


@pytest.mark.parametrize("delta", DELTAS)
def test_cosine_centre_far_and_edges(delta):
    assert cosine_wall_f(0.0, R, delta) == 1.0
    assert cosine_wall_f(R, R, delta) == 1.0
    assert cosine_wall_f(R + delta / 2.0, R, delta) == pytest.approx(0.5, abs=1e-15)
    assert cosine_wall_f(R + delta, R, delta) == pytest.approx(0.0, abs=1e-15)
    assert cosine_wall_f(R + 2.0 * delta, R, delta) == 0.0


@pytest.mark.parametrize("delta", DELTAS)
def test_cosine_max_slope_is_authors_value(delta):
    """Independent of cosine_wall_df: slope measured by finite differences of f itself."""
    r = np.linspace(R, R + delta, 200_001)
    slope = -np.gradient(cosine_wall_f(r, R, delta), r)
    i = np.argmax(slope)
    assert slope[i] == pytest.approx(np.pi / (2.0 * delta), rel=1e-8)
    assert slope[i] == pytest.approx(cosine_wall_max_slope(delta), rel=1e-8)
    assert r[i] == pytest.approx(R + delta / 2.0, abs=delta * 1e-4)
    assert cosine_wall_pf_thickness(delta) == pytest.approx(1.0 / slope[i], rel=1e-8)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_cosine_rejects_nonpositive_delta(bad):
    with pytest.raises(ValueError):
        cosine_wall_f(R, R, bad)
    with pytest.raises(ValueError):
        cosine_wall_df(R, R, bad)


@pytest.mark.parametrize("delta", DELTAS)
def test_cosine_derivative_matches_finite_difference_and_is_continuous(delta):
    r = np.linspace(0.0, R + 2.0 * delta, 4001)
    h = delta * 1e-6
    np.testing.assert_allclose(
        cosine_wall_df(r, R, delta), _finite_diff(lambda x: cosine_wall_f(x, R, delta), r, h),
        atol=1e-6 / delta,
    )
    # slope is ~0 just inside both edges (C¹ wall)
    eps = delta * 1e-9
    assert abs(cosine_wall_df(R + eps, R, delta)) < 1e-7 / delta
    assert abs(cosine_wall_df(R + delta - eps, R, delta)) < 1e-7 / delta


# ----------------------------------------------------------------------- dual shell

def _literal_dual_shell(r, t):
    """The author's f_total transcribed character-for-character from the spec."""
    w = 2.0 * np.pi * 1.0e9
    return 0.5 * (1.0 + np.tanh(2.0 * (r - 100.0))) * np.cos(w * t) + 0.5 * (
        1.0 - np.tanh(2.0 * (r - 90.0))
    ) * np.sin(w * t)


def test_dual_shell_as_written_matches_literal():
    r = np.linspace(0.0, 200.0, 801)[:, None]
    t = np.linspace(0.0, 2e-9, 41)[None, :]
    np.testing.assert_allclose(dual_shell_f(r, t, **DUAL_SHELL_AS_WRITTEN), _literal_dual_shell(r, t), atol=1e-14)


def test_dual_shell_as_written_limits():
    """Documents what the formula does as written: interior follows sin(ωt), far field cos(ωt)."""
    t = np.linspace(0.0, 1e-9, 17)
    w = DUAL_SHELL_AS_WRITTEN["omega"]
    np.testing.assert_allclose(dual_shell_f(0.0, t, **DUAL_SHELL_AS_WRITTEN), np.sin(w * t), atol=1e-12)
    np.testing.assert_allclose(dual_shell_f(1e4, t, **DUAL_SHELL_AS_WRITTEN), np.cos(w * t), atol=1e-12)


def test_dual_shell_alternative_both_inward_vanishes_far_away():
    alt = {**DUAL_SHELL_AS_WRITTEN, "shell1_sign": -1}
    t = np.linspace(0.0, 1e-9, 17)
    np.testing.assert_allclose(dual_shell_f(1e4, t, **alt), 0.0, atol=1e-12)


def test_dual_shell_180_degree_phase():
    p180 = {**DUAL_SHELL_AS_WRITTEN, "phase": np.pi}
    t = np.linspace(0.0, 1e-9, 17)
    w = p180["omega"]
    np.testing.assert_allclose(dual_shell_f(0.0, t, **p180), -np.cos(w * t), atol=1e-12)


@pytest.mark.parametrize("variant", [{}, {"shell1_sign": -1}, {"phase": np.pi}])
def test_dual_shell_derivative_matches_finite_difference(variant):
    p = {**DUAL_SHELL_AS_WRITTEN, **variant}
    r = np.linspace(50.0, 150.0, 2001)[:, None]
    t = np.array([0.0, 0.1e-9, 0.37e-9])[None, :]
    np.testing.assert_allclose(
        dual_shell_df(r, t, **p), _finite_diff(lambda x: dual_shell_f(x, t, **p), r, 1e-5), atol=1e-7
    )

@pytest.mark.parametrize("variant", [{}, {"shell1_sign": -1}, {"phase": np.pi}])
def test_dual_shell_each_shell_peaks_at_its_own_radius(variant):
    p = {**DUAL_SHELL_AS_WRITTEN, **variant}
    period = 2.0 * np.pi / p["omega"]
    # Pick times when the shell's own time factor is ±1 (and, for 90°, the other's is 0).
    t_shell1 = 0.0
    t_shell2 = period / 4.0 if p["phase"] == np.pi / 2.0 else 0.0
    r1 = np.linspace(95.0, 105.0, 10_001)
    r2 = np.linspace(85.0, 94.0, 9_001)
    assert r1[np.argmax(np.abs(dual_shell_df(r1, t_shell1, **p)))] == pytest.approx(100.0, abs=1e-3)
    assert r2[np.argmax(np.abs(dual_shell_df(r2, t_shell2, **p)))] == pytest.approx(90.0, abs=1e-3)


# ------------------------------------------------- Pfenning & Ford straight ramp (eq. 4)

@pytest.mark.parametrize("delta", [0.5, 1.0, 10.0])
def test_pf_ramp_shape_and_slope(delta):
    assert pf_ramp_f(0.0, R, delta) == 1.0
    assert pf_ramp_f(R, R, delta) == pytest.approx(0.5)
    assert pf_ramp_f(R + delta, R, delta) == 0.0
    r = np.linspace(R - delta, R + delta, 2001)
    inner = np.abs(r - R) < 0.49 * delta
    fd = _finite_diff(lambda x: pf_ramp_f(x, R, delta), r[inner], delta * 1e-6)
    np.testing.assert_allclose(pf_ramp_df(r[inner], R, delta), fd, rtol=1e-6)
    assert np.all(pf_ramp_df(r[np.abs(r - R) > 0.51 * delta], R, delta) == 0.0)


# ------------------------------------------------- regressions from the Codex review (CODEX8)

def test_cosine_edges_exact_for_non_binary_width():
    """r = R+Δ must count as inside the wall even when (R+Δ-R)/Δ rounds above 1."""
    d = 0.01
    edge = R + d
    assert cosine_wall_ddf(edge, R, d) == pytest.approx(np.pi**2 / (2 * d**2), rel=1e-12)
    assert cosine_wall_ddf(np.nextafter(edge, np.inf), R, d) == 0.0
    assert cosine_wall_ddf(np.nextafter(R, -np.inf), R, d) == 0.0
    assert cosine_wall_df(edge, R, d) == 0.0 and cosine_wall_df(R, R, d) == 0.0
    assert cosine_wall_f(edge, R, d) == pytest.approx(0.0, abs=1e-15)


def test_alcubierre_broad_wall_is_stable():
    assert alcubierre_f(0.0, R, 1e-18) == pytest.approx(1.0, rel=1e-12)
    s = sigma_for_wall_thickness(1e29, R)
    assert alcubierre_wall_thickness(R, s) == pytest.approx(1e29, rel=1e-10)
