"""Energy tests: three independent routes must agree, and published results must be reproduced."""

import numpy as np
import pytest
from scipy.constants import G, c
from scipy.optimize import minimize_scalar

from warp import energy, shapes

R = 100.0


def cos_df(delta):
    return lambda r: shapes.cosine_wall_df(r, R, delta)


@pytest.mark.parametrize("delta", [1e-3, 1.0, 37.0, 188.0, 1e4])
def test_cosine_wall_1d_integral_matches_closed_form(delta):
    num = energy.wall_integral(cos_df(delta), R, R + delta)
    assert num == pytest.approx(energy.cosine_wall_integral_exact(R, delta), rel=1e-10)


@pytest.mark.parametrize("delta", [1e-3, 1e-2, 1.0, 188.0, 1e4, 1e5])
def test_cylindrical_2d_matches_closed_form(delta):
    """The full 2-D (x, ρ) integration agrees with the 1-D formula: checks the ρ²/r² angular factor."""
    m2d = energy.total_mass_cylindrical(cos_df(delta), R, R + delta, beta=1.0)
    m1d = energy.mass_from_wall_integral(energy.cosine_wall_integral_exact(R, delta), beta=1.0)
    assert m2d < 0
    assert m2d == pytest.approx(m1d, rel=1e-7)


def test_cylindrical_2d_tanh_wall():
    """Full integral of the 1 m tanh wall vs its 1-D reduction (no IntegrationWarning allowed)."""
    df = lambda r: shapes.alcubierre_df(r, R, 2.0)
    m2d = energy.total_mass_cylindrical(df, R - 30, R + 30, beta=1.0)
    m1d = energy.mass_from_wall_integral(energy.wall_integral(df, R - 30, R + 30, points=[R]), beta=1.0)
    assert m2d == pytest.approx(m1d, rel=1e-8)


def test_pfenning_ford_eq28_reproduced():
    """P&F straight-ramp wall: ∫ r² f'² dr = R²/Δ + Δ/12 (their eq. 28, times 12)."""
    for delta in [0.5, 1.0, 10.0]:
        num = energy.wall_integral(lambda r: shapes.pf_ramp_df(r, R, delta), R - delta / 2, R + delta / 2)
        assert num == pytest.approx(R * R / delta + delta / 12.0, rel=1e-10)


def test_tanh_wall_thin_limit():
    """Thin tanh wall: ∫ r² f'² dr ≈ R²·σ/3 (∫ sech⁴ = 4/3)."""
    sigma = 2.0
    num = energy.wall_integral(lambda r: shapes.alcubierre_df(r, R, sigma), R - 30, R + 30, points=[R])
    assert num == pytest.approx(R * R * sigma / 3.0, rel=2e-4)  # finite-thickness correction ~ 1/(σR)²


def test_unit_conversion():
    assert energy.KG_PER_METRE == pytest.approx(1.3466e27, rel=1e-4)
    assert energy.M_SUN == pytest.approx(1.9884e30, rel=1e-4)


def test_optimum_is_a_minimum():
    """Minimise the numerically integrated energy (not the closed form) and compare with R·√(A/B)."""
    res = minimize_scalar(lambda lg: energy.wall_integral(cos_df(10**lg), R, R + 10**lg), bounds=(1, 3.5),
                          method="bounded", options={"xatol": 1e-9})
    assert 10**res.x == pytest.approx(energy.cosine_wall_optimal_delta(R), rel=1e-6)
    assert energy.cosine_wall_optimal_delta(R) == pytest.approx(188.1, abs=0.1)


def test_energy_density_on_axis_and_centre_is_zero():
    rho_e = energy.energy_density(np.array([0.0, 50.0, 100.5]), np.array([0.0, 0.0, 0.0]), cos_df(1.0), 1.0)
    assert np.all(rho_e == 0.0)
    assert energy.energy_density(0.0, 100.5, cos_df(1.0), 1.0) < 0.0


def test_unresolvably_thin_wall_raises():
    """Δ = 1e-14 m at r = 100 m is below float resolution: refuse rather than return 0 (CODEX8)."""
    with pytest.raises(ValueError):
        energy.wall_integral(cos_df(1e-14), R, R + 1e-14)
