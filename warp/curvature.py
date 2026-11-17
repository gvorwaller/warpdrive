"""Curvature of a general warp metric, computed symbolically once and evaluated numerically.

Metric (geometric units, c = G = 1, lengths in metres, t means ct):

    ds² = -dt² + (dx - S dt)² + dy² + dz²

where S(t, x, y, z) is any function: v_s·f(r_s) for a steady bubble, or a
time-dependent profile for the pulsed / dual-shell designs. Every curvature
quantity depends only on S and its first and second partial derivatives, so the
symbolic Riemann and Einstein tensors are derived once (sympy) and turned into
fast numpy functions of (S, ∂S, ∂∂S).

Components are given in the orthonormal frame of the Eulerian observers
(Alcubierre's "normal" observers, at rest in the slicing):

    e_0 = n = ∂t + S ∂x,   e_1 = ∂x,   e_2 = ∂y,   e_3 = ∂z

so that G_00 / 8π is the energy density those observers measure. Units: 1/m².
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import sympy as sp

COORDS = sp.symbols("t x y z", real=True)
_IJ = [(i, j) for i in range(4) for j in range(i, 4)]  # 10 independent second derivatives


@lru_cache(maxsize=1)
def _symbolic():
    t, x, y, z = COORDS
    S = sp.Function("S")(t, x, y, z)
    g = sp.Matrix([[-1 + S**2, -S, 0, 0], [-S, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ginv = sp.simplify(g.inv())
    X = COORDS

    gamma = [[[sp.expand(sum(ginv[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
                             for d in range(4)) / 2)
               for c in range(4)] for b in range(4)] for a in range(4)]

    def riemann_up(a, b, c, d):  # R^a_{bcd}
        expr = sp.diff(gamma[a][d][b], X[c]) - sp.diff(gamma[a][c][b], X[d])
        expr += sum(gamma[a][c][e] * gamma[e][d][b] - gamma[a][d][e] * gamma[e][c][b] for e in range(4))
        return expr

    Rup = {}
    for a, b, c, d in itertools.product(range(4), repeat=4):
        if c < d:
            Rup[a, b, c, d] = sp.expand(riemann_up(a, b, c, d))
            Rup[a, b, d, c] = -Rup[a, b, c, d]
        elif c == d:
            Rup[a, b, c, d] = sp.Integer(0)

    def R_down(a, b, c, d):
        return sum(g[a, e] * Rup[e, b, c, d] for e in range(4))

    ricci = sp.Matrix(4, 4, lambda b, d: sum(Rup[a, b, a, d] for a in range(4)))
    scalar = sum(ginv[b, d] * ricci[b, d] for b in range(4) for d in range(4))
    einstein = ricci - g * scalar / 2

    # Orthonormal Eulerian frame (rows: e_0..e_3, contravariant components)
    E = sp.Matrix([[1, S, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Replace S and its derivatives by plain symbols so the result can be lambdified.
    s0 = sp.Symbol("S0")
    s1 = sp.symbols("S_t S_x S_y S_z")
    s2 = {ij: sp.Symbol(f"S_{'txyz'[ij[0]]}{'txyz'[ij[1]]}") for ij in _IJ}
    repl = {}
    for (i, j), sym in s2.items():
        repl[sp.diff(S, X[i], X[j])] = sym
    for i in range(4):
        repl[sp.diff(S, X[i])] = s1[i]

    def plain(expr):
        return sp.expand(sp.expand(expr).subs(repl).subs(S, s0))

    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]  # 6 antisymmetric index pairs
    R_frame = {}
    for (A, B), (C, D) in itertools.combinations_with_replacement(pairs, 2):
        val = 0
        for a, b, c, d in itertools.product(range(4), repeat=4):
            coeff = E[A, a] * E[B, b] * E[C, c] * E[D, d]
            if coeff != 0:
                val += coeff * R_down(a, b, c, d)
        R_frame[(A, B, C, D)] = plain(val)

    G_frame = {}
    for A, B in _IJ:
        val = sum(E[A, a] * E[B, b] * einstein[a, b] for a in range(4) for b in range(4))
        G_frame[(A, B)] = plain(val)

    args = [s0, *s1, *[s2[ij] for ij in _IJ]]
    r_keys = list(R_frame)
    g_keys = list(G_frame)
    r_fun = sp.lambdify(args, [R_frame[k] for k in r_keys], "numpy")
    g_fun = sp.lambdify(args, [G_frame[k] for k in g_keys], "numpy")
    return r_keys, r_fun, g_keys, g_fun, R_frame, G_frame


def _args(S, dS, ddS):
    """S: array; dS: (4, ...) first derivatives in (t,x,y,z); ddS: (4,4, ...) second derivatives."""
    return [S, *[dS[i] for i in range(4)], *[ddS[i][j] for i, j in _IJ]]


def riemann_frame(S, dS, ddS):
    """All independent Eulerian-frame Riemann components R_âb̂ĉd̂ (1/m²) as {(a,b,c,d): array}."""
    keys, fun, *_ = _symbolic()
    args = _args(S, dS, ddS)
    vals = fun(*args)
    shape = np.broadcast(*[np.asarray(a) for a in args]).shape
    return {k: np.broadcast_to(np.asarray(v, dtype=float), shape) for k, v in zip(keys, vals)}


def einstein_frame(S, dS, ddS):
    """Eulerian-frame Einstein tensor G_âb̂ (1/m²) as {(a,b): array}, a <= b."""
    _, _, keys, fun, *_ = _symbolic()
    args = _args(S, dS, ddS)
    vals = fun(*args)
    shape = np.broadcast(*[np.asarray(a) for a in args]).shape
    return {k: np.broadcast_to(np.asarray(v, dtype=float), shape) for k, v in zip(keys, vals)}


def max_riemann(S, dS, ddS):
    """Largest |R_âb̂ĉd̂| over all frame components at each point (1/m²)."""
    comps = riemann_frame(S, dS, ddS)
    return np.max(np.abs(np.stack(list(comps.values()))), axis=0)


# --- A steady bubble: S = β f(r_s), r_s = |(x - βt, y, z)| ------------------------

def steady_bubble_derivatives(t, x, y, z, beta, f, df, ddf):
    """S, ∂S, ∂∂S for a bubble moving at constant speed β (in units of c) along x.

    f, df, ddf are callables giving f(r), f'(r), f''(r). Time derivatives follow
    from the bubble's motion: ∂_t acts as -β ∂_x on anything that depends on x - βt.
    """
    X = np.asarray(x, dtype=float) - beta * np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    X, y, z = np.broadcast_arrays(X, y, z)
    r = np.sqrt(X * X + y * y + z * z)
    centre = r == 0
    if np.any(centre) and abs(float(df(0.0))) > 0:
        raise ValueError("f'(0) ≠ 0: the shape has a kink at the bubble centre, so curvature is undefined there")
    r_safe = np.where(centre, 1.0, r)
    n = np.stack([X / r_safe, y / r_safe, z / r_safe])       # unit vector, spatial (0 at the centre)
    f0, f1, f2 = f(r), df(r), ddf(r)
    r = r_safe

    # spatial first and second derivatives of r
    dr_sp = n
    ddr_sp = np.empty((3, 3) + r.shape)
    for i in range(3):
        for j in range(3):
            ddr_sp[i, j] = ((1.0 if i == j else 0.0) - n[i] * n[j]) / r

    # extend to (t, x, y, z) with ∂_t = -β ∂_x
    lift = np.array([[-beta, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)  # 4x3
    dr = np.einsum("ai,i...->a...", lift, dr_sp)
    ddr = np.einsum("ai,bj,ij...->ab...", lift, lift, ddr_sp)

    S = beta * f0
    dS = beta * f1 * dr
    ddS = beta * (f2 * np.einsum("a...,b...->ab...", dr, dr) + f1 * ddr)
    if np.any(centre):
        # smooth radial f with f'(0) = 0: ∇S = 0 and spatial Hessian β f''(0) I, then the same time lift
        dS = np.where(centre, 0.0, dS)
        hess = beta * float(ddf(0.0)) * (lift @ lift.T)
        ddS = np.where(centre, hess.reshape((4, 4) + (1,) * centre.ndim), ddS)
    return S, dS, ddS
