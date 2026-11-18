"""Animation-map tests: the colour maps must agree with the energy calculations they picture."""

import numpy as np
import pytest

from warp import animate as an
from warp import dual_shell as ds
from warp import energy
from warp.shapes import dual_shell_df

AS = ds.VARIANTS["as written"]
T = ds.period(AS)


@pytest.mark.parametrize("t", [0.0, T / 8, T / 4, 0.37 * T])
def test_map_total_matches_one_dimensional_energy(t):
    x, y, rho = an.energy_density_map(AS, t, 1.0, n=801)
    assert an.map_total_mass(x, y, rho) == pytest.approx(ds.mass_vs_time([t], 1.0, AS)[0], rel=1e-4)


def test_map_values_and_symmetry():
    x, y, rho = an.energy_density_map(AS, T / 8, 1.0, n=401)
    assert np.all(rho <= 0)
    np.testing.assert_allclose(rho, rho[::-1, :], atol=0)           # mirror across the line of travel
    np.testing.assert_allclose(rho, rho[:, ::-1], atol=0)           # front/back symmetric (no motion term)
    assert np.all(rho[len(y) // 2, :] == 0.0)                      # zero on the line of travel
    j, i = 123, 301
    expected = energy.energy_density(x[j], y[i], lambda r: dual_shell_df(r, T / 8, **AS), 1.0)
    assert rho[i, j] == pytest.approx(float(expected), rel=1e-14, abs=0)


def test_colour_scale_peak():
    """Peak is one fully driven shell at the equator: (c⁴/8πG)(β²/4)(σ/2)²; the map reaches it."""
    vmax = an.peak_density(AS, 1.0)
    assert vmax == pytest.approx(energy.ENERGY_DENSITY_SCALE / 4 * 1.0, rel=1e-12)
    x, y, rho = an.energy_density_map(AS, 0.0, 1.0, n=4001)
    assert -rho.min() == pytest.approx(vmax, rel=2e-3)
    assert -rho.min() <= vmax * (1 + 1e-12)
