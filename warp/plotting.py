"""Phone-first figure style and saving.

Figures are viewed as images in a phone chat app, often without any of the
notebook text around them, so every saved PNG carries its own plain-English
caption drawn underneath the chart.
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# 6.75 in × 160 dpi ≈ 1080 px wide: a full-width phone image after a chat app's resize.
FIG_WIDTH_IN = 6.75
SAVE_DPI = 160
CAPTION_WRAP = 60  # characters per caption line at CAPTION_SIZE

BASE_SIZE = 14
CAPTION_SIZE = 13

# Okabe–Ito colour-blind-safe palette; every series also gets its own line style.
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
LINESTYLES = ["-", "--", "-.", ":", (0, (6, 2, 1, 2, 1, 2))]

PHONE_RC = {
    "font.size": BASE_SIZE,
    "axes.titlesize": BASE_SIZE + 2,
    "axes.labelsize": BASE_SIZE,
    "xtick.labelsize": BASE_SIZE - 1,
    "ytick.labelsize": BASE_SIZE - 1,
    "legend.fontsize": BASE_SIZE - 2,
    "lines.linewidth": 2.6,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def use_phone_style():
    """Apply the phone-readable style globally (call once at notebook start)."""
    mpl.rcParams.update(PHONE_RC)


def phone_figure(nrows=1, height_per_row=3.6, **subplot_kw):
    """A full-phone-width figure with `nrows` stacked panels sharing nothing by default."""
    fig, axes = plt.subplots(
        nrows, 1, figsize=(FIG_WIDTH_IN, height_per_row * nrows), layout="constrained", **subplot_kw
    )
    return fig, axes


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci_text(x, digits=1):
    """Readable scientific notation: 2.857e28 -> '2.9 × 10²⁸'."""
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    mant = x / 10**exp
    if round(abs(mant), digits) >= 10:  # e.g. 9.99 rounding up to 10.0
        mant, exp = mant / 10, exp + 1
    # non-breaking spaces so caption wrapping never splits the number
    return f"{mant:.{digits}f}\u00a0×\u00a010{str(exp).translate(_SUPERSCRIPT)}"


def add_caption(fig, caption):
    """Draw a wrapped plain-English caption under the figure.

    A number and its unit ("1 m", "188 m") are kept on the same line.
    """
    glued = re.sub(r"(\d) (m|mm|cm|km|kg|s|ns|GHz|MHz)\b", "\\1\u00a0\\2", caption.strip())
    paragraphs = [textwrap.fill(p.strip(), CAPTION_WRAP) for p in glued.split("\n\n")]
    fig.text(
        0.02, -0.01, "\n\n".join(paragraphs),
        ha="left", va="top", fontsize=CAPTION_SIZE, transform=fig.transFigure,
    )


def save_figure(fig, name, caption):
    """Add the caption and save to figures/<name>.png. Returns the saved path."""
    add_caption(fig, caption)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.25)
    return path
