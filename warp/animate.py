"""Energy-density maps and animations of the dual-shell bubble (spec Task 4).

The map is a cross-section through the line of travel, in the bubble's own frame
(the bubble rocks along with the ship by centimetres, invisible at this scale).
The colour is the negative energy density seen by the Eulerian observers, which
holds at every instant even though the bubble changes in time (warp.curvature).
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from warp import energy
from warp.plotting import CAPTION_SIZE, CAPTION_WRAP, COLORS, FIG_WIDTH_IN
from warp.shapes import _time_factors, dual_shell_df


def energy_density_map(params, t, beta, half_width=130.0, n=1001):
    """Energy density (J/m³, ≤ 0) on an n×n grid over x, y ∈ [-half_width, half_width] m.

    Returns (x, y, rho) with rho[i, j] at (x[j], y[i]). r_s = √(x² + y²) in the z = 0 plane,
    so the distance from the line of travel is |y|.
    """
    x = np.linspace(-half_width, half_width, n)
    y = np.linspace(-half_width, half_width, n)
    X, Y = np.meshgrid(x, y)
    rho = energy.energy_density(X, Y, lambda r: dual_shell_df(r, t, **params), beta)
    return x, y, rho


def peak_density(params, beta):
    """Largest |energy density| over a cycle: a single shell at full drive, on the equator.

    A tanh shell's slope peaks at σ/2, so |ρ| = (c⁴/8πG)(β²/4)(σ/2)².
    """
    return energy.ENERGY_DENSITY_SCALE * beta**2 / 4.0 * (params["sigma"] / 2.0) ** 2


def map_total_mass(x, y, rho):
    """Total mass-equivalent (kg) of a map, treating it as axially symmetric about the x axis.

    Uses the upper half (y ≥ 0) as the (x, ρ) half-plane with volume element 2πρ dρ dx.
    A check that the map is consistent with the 1-D totals in warp.dual_shell.
    """
    from scipy.constants import c
    upper = y >= 0
    yy = y[upper]
    integrand = rho[upper, :] * 2.0 * np.pi * yy[:, None]
    return np.trapezoid(np.trapezoid(integrand, x, axis=1), yy) / c**2


ZOOM = (-60.0, 60.0, 78.0, 104.0)   # x0, x1, y0, y1 (m): the top of the bubble, where both walls show as bands


def make_frames(params, beta, times, title, caption, half_width=130.0, n=901, n_zoom=(1601, 351)):
    """Build the figure and an update function; returns (fig, update, period).

    Layout (portrait, phone width): full cross-section, a zoom on the top of the
    bubble where the 1 m walls are wide enough to see, one shared colour scale,
    the two drives over time with a moving marker, and the caption.
    """
    from matplotlib.patches import Rectangle
    from warp.plotting import sci_text
    vmax = peak_density(params, beta)
    period = 2 * np.pi / params["omega"]
    # Fixed layout in inches (top to bottom) so every animation frame has the same size.
    W, H = FIG_WIDTH_IN, 14.2
    fig = plt.figure(figsize=(W, H))
    left, width = 1.0, W - 1.0 - 0.45
    zoom_h = width * (ZOOM[3] - ZOOM[2]) / (ZOOM[1] - ZOOM[0])
    y = H - 0.55                    # below the title
    def place(gap, h):
        nonlocal y
        y -= gap + h
        return fig.add_axes((left / W, y / H, width / W, h / H))
    ax_map = place(0.0, width)
    ax_zoom = place(0.95, zoom_h)   # room for the map's x label and the zoom title
    ax_cb = place(0.55, 0.22)       # room for the zoom's tick labels
    ax_drv = place(1.05, 1.2)       # room for the colour-bar label and the drive legend
    caption_top = (y - 0.75) / H    # below the drive panel's x label

    def zoom_map(t):
        zx = np.linspace(ZOOM[0], ZOOM[1], n_zoom[0])
        zy = np.linspace(ZOOM[2], ZOOM[3], n_zoom[1])
        ZX, ZY = np.meshgrid(zx, zy)
        return energy.energy_density(ZX, ZY, lambda r: dual_shell_df(r, t, **params), beta)

    x, y, rho = energy_density_map(params, times[0], beta, half_width, n)
    common = dict(origin="lower", cmap="magma", vmin=0.0, vmax=1.0, interpolation="antialiased")
    img = ax_map.imshow(-rho / vmax, extent=(x[0], x[-1], y[0], y[-1]), **common)
    img_z = ax_zoom.imshow(-zoom_map(times[0]) / vmax, extent=ZOOM, **common)
    ax_map.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2],
                               fill=False, ec="white", lw=1.4, ls="--"))
    ax_map.plot(0, 0, marker="^", ms=11, color=COLORS["sky"], mec="white", mew=1.2)
    ax_map.annotate("ship", (0, 0), xytext=(0, -20), textcoords="offset points", ha="center",
                    color="white", fontsize=12)
    ax_map.annotate("", xy=(0.95, 0.05), xytext=(0.72, 0.05), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    ax_map.text(0.71, 0.05, "travel", transform=ax_map.transAxes, ha="right", va="center",
                color="white", fontsize=12)
    ax_map.set_xlabel("along the direction of travel  (m)")
    ax_map.set_ylabel("across  (m)")
    for ax in (ax_map, ax_zoom):
        ax.grid(False)
    title_txt = ax_map.set_title(title)
    ax_zoom.set_title("close-up of the dashed box", fontsize=13)
    ax_zoom.set_ylabel("across (m)", fontsize=12)
    for R_shell in (params["R1"], params["R2"]):
        ax_zoom.text(0.0, R_shell - 1.2, f"{R_shell:g} m shell", color="white", fontsize=11, ha="center", va="top")

    cb = fig.colorbar(img, cax=ax_cb, orientation="horizontal")
    cb.set_label(f"negative energy density, as a fraction of its peak ({sci_text(vmax)} J/m³)", fontsize=11)
    cb.ax.tick_params(labelsize=11)

    t_line = np.linspace(times[0], times[-1], 800)
    c1, c2 = _time_factors(t_line, params["omega"], params["phase"], params.get("drive", "linear"))
    ax_drv.plot(t_line * 1e9, c1, color=COLORS["orange"], lw=2.4, label=f"{params['R1']:g} m shell drive")
    ax_drv.plot(t_line * 1e9, c2, color=COLORS["green"], lw=2.4, ls="--", label=f"{params['R2']:g} m shell drive")
    ax_drv.axhline(0, color="0.5", lw=0.8)
    marker = ax_drv.axvline(times[0] * 1e9, color="black", lw=1.8)
    ax_drv.set_xlim(t_line[0] * 1e9, t_line[-1] * 1e9)
    ax_drv.set_ylim(-1.2, 1.2)
    ax_drv.set_xlabel("time  (ns)")
    ax_drv.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=10, ncol=2, frameon=False)

    wrapped = "\n\n".join(textwrap.fill(p.strip(), CAPTION_WRAP) for p in caption.strip().split("\n\n"))
    fig.text(0.03, caption_top, wrapped, ha="left", va="top", fontsize=CAPTION_SIZE)

    def update(t):
        _, _, rho_t = energy_density_map(params, t, beta, half_width, n)
        img.set_data(-rho_t / vmax)
        img_z.set_data(-zoom_map(t) / vmax)
        marker.set_xdata([t * 1e9, t * 1e9])
        title_txt.set_text(f"{title}   t = {t * 1e9:.2f} ns")
        return img, img_z, marker, title_txt

    update(times[0])
    return fig, update, period


def save_animation(params, beta, name, title, caption, cycles=2, frames_per_cycle=30, out_dir=None,
                   gif_dpi=110, mp4_dpi=160, fps=12):
    """Write figures/<name>.gif and figures/<name>.mp4. Returns the two paths."""
    from warp.plotting import FIGURES_DIR
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(exist_ok=True)
    period = 2 * np.pi / params["omega"]
    times = np.linspace(0.0, cycles * period, cycles * frames_per_cycle, endpoint=False)
    fig, update, _ = make_frames(params, beta, np.array([0.0, cycles * period]), title, caption)
    anim = animation.FuncAnimation(fig, update, frames=times, blit=False)
    gif = out_dir / f"{name}.gif"
    mp4 = out_dir / f"{name}.mp4"
    anim.save(gif, writer=animation.PillowWriter(fps=fps), dpi=gif_dpi)
    anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, codec="libx264",
                                                 extra_args=["-pix_fmt", "yuv420p", "-crf", "23"]),
              dpi=mp4_dpi)
    plt.close(fig)
    return gif, mp4


def save_still(params, beta, t, name, title, caption, cycles=2, dpi=160):
    """One frame as figures/<name>.png (same layout as the animation)."""
    from warp.plotting import FIGURES_DIR
    period = 2 * np.pi / params["omega"]
    fig, update, _ = make_frames(params, beta, np.array([0.0, cycles * period]), title, caption)
    update(t)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
