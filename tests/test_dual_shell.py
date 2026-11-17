"""Dual-shell analysis tests: analytic motion and energy results for each variant."""

import numpy as np
import pytest
from scipy.constants import G, c

from warp import dual_shell as ds

AS = ds.VARIANTS["as written"]
W = AS["omega"]
T = 2 * np.pi / W


def test_variant_shapes_at_key_moments():
    """Pin each variant's behaviour, not just its parameters: centre and far values at t = 0 and T/4."""
    f = lambda name, r, t: float(ds.dual_shell_f(r, t, **ds.VARIANTS[name]))
    # as written: centre = sin, far = cos
    assert (f("as written", 0, 0), f("as written", ds.FAR, 0)) == pytest.approx((0, 1), abs=1e-12)
    # 180°: shell 2 driven by −cos, so the centre starts at −1, and everything is 0 at T/4
    assert f("180° phase", 0, 0) == pytest.approx(-1, abs=1e-12)
    assert f("180° phase", 0, T / 4) == pytest.approx(0, abs=1e-12)
    assert f("180° phase", ds.FAR, T / 4) == pytest.approx(0, abs=1e-12)
    # both inward: far field always 0, centre = cos + sin (peaks at √2)
    assert f("both shells face inward", ds.FAR, T / 8) == pytest.approx(0, abs=1e-12)
    assert f("both shells face inward", 0, T / 8) == pytest.approx(np.sqrt(2), abs=1e-12)


def test_as_written_motion_is_pure_oscillation():
    """Ship moves at β sin ωt, far space at β cos ωt: relative displacement (βc/ω)[(1 − cos ωt) − sin ωt]."""
    t = np.linspace(0, 2 * T, 40_001)
    x = ds.relative_displacement(t, 1.0, AS)
    expected = c / W * ((1 - np.cos(W * t)) - np.sin(W * t))
    np.testing.assert_allclose(x, expected, atol=1e-6)
    assert ds.mean_relative_velocity(1.0, AS) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", ["both shells face inward", "180° phase"])
def test_other_linear_variants_have_no_net_motion(name):
    assert ds.mean_relative_velocity(1.0, ds.VARIANTS[name]) == pytest.approx(0.0, abs=1e-9)


def test_squared_drive_moves_steadily():
    p = ds.VARIANTS["inward + squared drive"]
    t = np.linspace(0, 2 * T, 2001)
    np.testing.assert_allclose(ds.relative_velocity(t, 0.8, p), 0.8, atol=1e-12)
    assert ds.relative_displacement(t, 0.8, p)[-1] == pytest.approx(0.8 * c * 2 * T, rel=1e-9)
    np.testing.assert_allclose(ds.bubble_centre(t, 0.8, p), 0.8 * c * t, rtol=1e-9, atol=1e-12)


def test_bubble_rocks_with_ship_as_written():
    """With dx_s/dt = βc f(0,t) the centre follows βc∫sin = (βc/ω)(1 − cos ωt): back to start each cycle."""
    t = np.linspace(0, 2 * T, 40_001)
    np.testing.assert_allclose(ds.bubble_centre(t, 1.0, AS), c / W * (1 - np.cos(W * t)), atol=1e-6)


def test_cheapest_steady_split_is_the_minimum():
    a_best, m_best = ds.cheapest_steady_split(ds.VARIANTS["both shells face inward"], 1.0)
    e1 = ds.single_shell_integral(100.0, 2.0)
    e2 = ds.single_shell_integral(90.0, 2.0)
    a = np.linspace(0, 1, 100_001)
    cost = a**2 * e1 + (1 - a) ** 2 * e2
    assert a_best == pytest.approx(a[np.argmin(cost)], abs=1e-4)
    assert -m_best == pytest.approx(c**2 / G / 12 * cost.min(), rel=1e-6)


def test_single_shell_energy_thin_wall_limit():
    """½[1 ± tanh(σ(r−R))] has ∫ g'² dr = σ/3, so ∫ r² g'² ≈ R²σ/3."""
    assert ds.single_shell_integral(100.0, 2.0) == pytest.approx(100.0**2 * 2.0 / 3.0, rel=1e-4)


@pytest.mark.parametrize("name,weights", [
    ("as written", lambda wt: (np.cos(wt) ** 2, np.sin(wt) ** 2)),
    ("both shells face inward", lambda wt: (np.cos(wt) ** 2, np.sin(wt) ** 2)),
    ("180° phase", lambda wt: (np.cos(wt) ** 2, np.cos(wt) ** 2)),
    ("inward + squared drive", lambda wt: (np.cos(wt) ** 4, np.sin(wt) ** 4)),
])
def test_energy_over_time_is_weighted_sum_of_shells(name, weights):
    """Shells 10 m apart with 1 m walls don't overlap, so E(t) = a(t)² E₁ + b(t)² E₂ (a, b = time factors)."""
    p = ds.VARIANTS[name]
    e1 = ds.single_shell_integral(p["R1"], p["sigma"])
    e2 = ds.single_shell_integral(p["R2"], p["sigma"])
    for t in np.linspace(0, T, 9):
        w1, w2 = weights(W * t)
        assert ds.wall_integral_at(t, p) == pytest.approx(w1 * e1 + w2 * e2, rel=1e-8, abs=1e-6)


def test_mass_vs_time_units():
    """At t = 0 only the 100 m shell is on: −(c²/G)(1/12)·R²σ/3 ≈ −7.48e29 kg (thin-wall value, independent)."""
    m = ds.mass_vs_time([0.0], 1.0, AS)[0]
    assert m == pytest.approx(-(c**2 / G) / 12 * 100.0**2 * 2.0 / 3.0, rel=2e-4)
    assert m == pytest.approx(-7.48e29, rel=1e-3)


# ------------------------------------------------- regressions from the Codex review (CODEX8)

def test_far_field_is_truly_far():
    p = {**AS, "R1": 20_000.0, "R2": 19_000.0}
    assert ds.relative_velocity(0.0, 1.0, p) == pytest.approx(-1.0, abs=1e-12)


def test_cheapest_split_includes_overlap():
    """Coincident shells act as one shell; separated shells match a direct integral of the combined slope."""
    from warp import energy
    from warp.shapes import shell_profile_dr
    same = {**ds.VARIANTS["both shells face inward"], "R2": 100.0}
    _, m_same = ds.cheapest_steady_split(same, 1.0)
    assert m_same == pytest.approx(energy.mass_from_wall_integral(ds.single_shell_integral(100.0, 2.0), 1.0), rel=1e-9)
    near = {**ds.VARIANTS["both shells face inward"], "R2": 99.5}     # walls overlap strongly
    a, m = ds.cheapest_steady_split(near, 1.0)
    direct = energy.wall_integral(lambda r: a * shell_profile_dr(r, 100.0, 2.0, -1)
                                  + (1 - a) * shell_profile_dr(r, 99.5, 2.0, -1), 70.0, 130.0, points=[99.5, 100.0])
    assert m == pytest.approx(energy.mass_from_wall_integral(direct, 1.0), rel=1e-8)


@pytest.mark.parametrize("sigma", [0.01, 1e4])
def test_single_shell_integral_any_sharpness(sigma):
    from warp import energy
    from warp.shapes import shell_profile_dr
    if sigma < 1:
        wide = energy.wall_integral(lambda r: shell_profile_dr(r, 100.0, sigma, -1), 0.0, 2500.0, points=[100.0])
        assert ds.single_shell_integral(100.0, sigma) == pytest.approx(wide, rel=1e-8)
    else:
        assert ds.single_shell_integral(100.0, sigma) == pytest.approx(100.0**2 * sigma / 3.0, rel=1e-6)


def test_motion_is_exact_on_sparse_grids():
    """One full cycle sampled at just its endpoints still returns exactly to the start (CODEX8)."""
    assert ds.relative_displacement(np.array([0.0, T]), 1.0, AS)[-1] == pytest.approx(0.0, abs=1e-15)
    sq = ds.VARIANTS["inward + squared drive"]
    assert ds.relative_displacement(np.array([0.0, T]), 1.0, sq)[-1] == pytest.approx(c * T, rel=1e-12)
    assert ds.mean_relative_velocity(1.0, sq) == pytest.approx(1.0, abs=1e-12)
