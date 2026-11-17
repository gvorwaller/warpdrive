"""Negative energy of a steady (constant-velocity) warp bubble, in SI units.

Energy density seen by Alcubierre's Eulerian observers (Alcubierre 1994 eq. 19,
Pfenning & Ford 1997 eq. 8), converted from c = G = 1 units to SI:

    ρ_E(x, ρ) = -(c⁴ / 8πG) · (β² / 4) · (ρ² / r²) · (df/dr)²      [J/m³]

where β = v_s / c, ρ = √(y² + z²) is the distance from the line of travel and
r = √(x² + ρ²) is the distance from the bubble centre. (The spec writes v_s²/4,
which is the same thing in units where c = 1.)

Total energy E = ∫ ρ_E dV. The spatial slices of this metric are flat, so the
proper volume element is the ordinary one (P&F eq. 25). Doing the angle integral
analytically gives P&F eq. 26:

    E = -(c⁴/G) · (β² / 12) · ∫₀^∞ r² (df/dr)² dr

Both routes are implemented: total_mass_cylindrical() integrates the full 2-D
energy density as the spec asks; wall_integral() is the 1-D reduction. They are
independent cross-checks of each other.

For this metric family (flat slices, unit lapse) the energy density has no time
derivatives, so these formulas hold at every instant even when v_s or f change in
time (see warp.curvature). Time dependence affects the stresses and curvature, which
are not computed here.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import G, c
from scipy.integrate import quad

# --- Astronomical masses -------------------------------------------------------
# scipy.constants has none. IAU 2015 Resolution B3 defines exact nominal values
# of GM (m³/s²); dividing by G gives the mass in kg.
GM_SUN_NOMINAL = 1.3271244e20      # IAU 2015 B3, (GM)_sun^N
GM_JUPITER_NOMINAL = 1.2668653e17  # IAU 2015 B3, (GM)_J^N
GM_EARTH_NOMINAL = 3.986004e14     # IAU 2015 B3, (GM)_E^N
M_SUN = GM_SUN_NOMINAL / G
M_JUPITER = GM_JUPITER_NOMINAL / G
M_EARTH = GM_EARTH_NOMINAL / G

# c⁴/(8πG): converts curvature (1/m²) to energy density (J/m³).
ENERGY_DENSITY_SCALE = c**4 / (8.0 * np.pi * G)
# c²/G: converts a length in metres (geometric units) to kilograms (≈1.35e27 kg/m).
KG_PER_METRE = c**2 / G


def energy_density(x, rho, df, beta):
    """ρ_E in J/m³ at (x, ρ) relative to the bubble centre. `df` is a callable df/dr(r).

    On the axis (ρ = 0) the energy density is exactly zero, including at r = 0.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r = np.hypot(x, rho)
    rho2, r2 = np.broadcast_arrays(rho**2, r**2)
    ratio = np.divide(rho2, r2, out=np.zeros_like(r2), where=r2 > 0)
    return -ENERGY_DENSITY_SCALE * (beta**2 / 4.0) * ratio * df(r) ** 2


def _achievable_rtol(requested, r_lo, r_hi):
    """Tightest tolerance floating point allows for a wall spanning [r_lo, r_hi].

    Shape functions compute (r - R) from r ≈ R, so a wall of width w at radius r_hi
    only has about eps·r_hi/w relative precision. Asking quad for more than that
    makes it chase roundoff noise (IntegrationWarning) without improving the answer.
    """
    floor = 100.0 * np.finfo(float).eps * r_hi / (r_hi - r_lo)
    if floor > 1e-4:
        raise ValueError(
            f"wall [{r_lo!r}, {r_hi!r}] is too thin to integrate in absolute coordinates "
            f"(only ~{floor:.0e} relative precision); use a closed form or wall-local coordinates")
    return max(requested, floor)


def wall_integral(df, r_lo, r_hi, points=None, epsrel=1e-12):
    """∫ r² (df/dr)² dr over [r_lo, r_hi] (metres). Outside that range df must be ~0."""
    value, _ = quad(
        lambda r: r * r * float(df(r)) ** 2, r_lo, r_hi,
        points=points, limit=1000, epsabs=0.0, epsrel=_achievable_rtol(epsrel, r_lo, r_hi),
    )
    return value


def mass_from_wall_integral(integral, beta):
    """Mass-equivalent E/c² in kg (negative) from ∫ r² f'² dr in metres (P&F eq. 26)."""
    return -KG_PER_METRE * beta**2 * integral / 12.0


def total_mass_cylindrical(df, r_lo, r_hi, beta, epsrel=1e-10):
    """E/c² in kg from the full 2-D integral of ρ_E over (x, ρ), 2πρ dρ dx.

    The wall occupies the spherical shell r_lo ≤ r ≤ r_hi. For each ρ the inner
    integral runs over the x-range that lies inside that shell (x ≥ 0, doubled
    by symmetry), so thin walls are fully resolved without a huge grid.
    """
    epsrel = _achievable_rtol(epsrel, r_lo, r_hi)
    def over_x(rho):
        x_lo = np.sqrt(max(r_lo * r_lo - rho * rho, 0.0))
        x_hi = np.sqrt(max(r_hi * r_hi - rho * rho, 0.0))
        if x_hi <= x_lo:
            return 0.0
        val, _ = quad(
            lambda x: float(energy_density(x, rho, df, beta)), x_lo, x_hi,
            limit=500, epsabs=0.0, epsrel=epsrel,
        )
        return 2.0 * val * 2.0 * np.pi * rho

    energy, _ = quad(over_x, 0.0, r_hi, points=[r_lo], limit=1000, epsabs=0.0, epsrel=epsrel)
    return energy / c**2


# --- the author's cosine wall in closed form --------------------------------------

_A = np.pi**2 / 8.0                    # coefficient of R²/Δ and of R
_B = np.pi**2 / 24.0 - 1.0 / 16.0      # coefficient of Δ


def cosine_wall_integral_exact(R, delta):
    """∫ r² f'² dr for the author's cosine wall, done by hand:

        (π²/8)·R²/Δ + (π²/8)·R + (π²/24 − 1/16)·Δ

    Derivation: substitute r = R + Δu, use ∫₀¹ sin²(πu) du = 1/2,
    ∫₀¹ u sin²(πu) du = 1/4, ∫₀¹ u² sin²(πu) du = 1/6 − 1/(4π²).
    Thin wall (Δ ≪ R): ∝ 1/Δ.  Thick wall (Δ ≫ R): ∝ Δ.
    """
    return _A * R * R / delta + _A * R + _B * delta


def cosine_wall_optimal_delta(R):
    """Δ that minimises the cosine wall's energy: R·√(A/B) ≈ 1.881·R."""
    return R * np.sqrt(_A / _B)
