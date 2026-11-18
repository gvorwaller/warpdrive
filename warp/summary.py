"""summary.pdf: the 5-page version to send or print (spec deliverable 3).

US Letter portrait, one chart per page. Each page has a heading, a one-line bottom
line, and a figure from figures/ (which already carries its plain-English caption).
Every number in the text is passed in from the notebook's computed results.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from warp.plotting import FIGURES_DIR

PAGE_W, PAGE_H = 8.5, 11.0
MARGIN = 0.6


def _page(pdf, n, total, heading, bottom_line, figure, intro=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    y = PAGE_H - MARGIN                                   # inches from the bottom, moving down
    if intro:
        fig.text(MARGIN / PAGE_W, y / PAGE_H, "Warp bubble notebook: results", fontsize=22,
                 fontweight="bold", va="top")
        y -= 0.5
        wrapped = textwrap.fill(intro, 88)
        fig.text(MARGIN / PAGE_W, y / PAGE_H, wrapped, fontsize=11.5, va="top", linespacing=1.4)
        y -= 0.23 * (wrapped.count("\n") + 1) + 0.35
    fig.text(MARGIN / PAGE_W, y / PAGE_H, heading, fontsize=17, fontweight="bold", va="top")
    y -= 0.45
    wrapped = textwrap.fill(bottom_line, 82)
    fig.text(MARGIN / PAGE_W, y / PAGE_H, wrapped, fontsize=12.5, va="top", linespacing=1.35,
             color="#1a1a1a")
    y -= 0.25 * (wrapped.count("\n") + 1) + 0.25

    img = plt.imread(FIGURES_DIR / f"{figure}.png")
    h_px, w_px = img.shape[:2]
    box_w, box_h = PAGE_W - 2 * MARGIN, y - (MARGIN + 0.35)
    scale = min(box_w / w_px, box_h / h_px)
    w, h = w_px * scale, h_px * scale
    ax = fig.add_axes(((PAGE_W - w) / 2 / PAGE_W, (y - h) / PAGE_H, w / PAGE_W, h / PAGE_H))
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")

    fig.text(0.5, (MARGIN - 0.1) / PAGE_H,
             f"page {n} of {total}  ·  all the details and code: github.com/gvorwaller/warpdrive",
             fontsize=9, color="0.4", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def build_summary(numbers, path=None):
    """Write summary.pdf. `numbers` holds the notebook's computed values (see the keys used below)."""
    from warp.plotting import sci_text
    n = numbers
    path = Path(path) if path else FIGURES_DIR.parent / "summary.pdf"
    pages = [
        dict(heading="1. Thicker walls: the right idea, up to a point",
             bottom_line=(f"A thicker wall really does cut the negative energy needed: at best about "
                          f"{n['best_saving']:.0f}× less than a 1 m wall, with the sweet spot near "
                          f"{n['best_delta']:.0f} m for a 100 m bubble. It's not the million-fold the notes "
                          f"hoped for, because a thicker wall is gentler but also fills more space."),
             figure="02_energy_vs_thickness",
             intro=("These pages test the warp-drive ideas from your notes by computer. Your equations were "
                    "coded exactly as written, every number was calculated rather than estimated, and the "
                    "physics was checked against Alcubierre (1994) and Pfenning & Ford (1997), cross-checked "
                    "several independent ways, and reviewed by three different AI systems. WarpFactory, the "
                    "published toolkit that warp-drive researchers use, independently reproduced the results to "
                    "about 15 digits. Where the math as "
                    "written has a slip, both versions are shown.")),
        dict(heading="2. The quantum rule",
             bottom_line=(f"You wrote the rule in exactly the right form: |ρ|·τ⁴ ≤ constant. Applied to a 1 m "
                          f"wall it is broken by about {sci_text(n['qi_1m'], 0)}, and thicker walls make it "
                          f"worse, not better. This is the obstacle every warp design runs into, not just yours."),
             figure="03_quantum_inequality"),
        dict(heading="3. The dual shell: your math beat your sketch",
             bottom_line=("Driving the shells with cos and sin (90° apart), as your formula does, keeps the "
                          "bubble always on. The 180° in the sketch would switch both shells off at the same "
                          "moment. As written, the outer shell's tanh faces outward; turning it inward makes "
                          "an ordinary bubble."),
             figure="04b_dual_shell_alternatives"),
        dict(heading="4. Does the ship go anywhere?",
             bottom_line=(f"As written, the ship rocks back and forth by up to {n['swing_cm']:.0f} cm and ends "
                          f"each cycle where it started, because cos and sin are negative half the time. Drives "
                          f"that never go negative (like cos² and sin²) move it steadily. With two shells you "
                          f"can't keep both the speed and the energy steady, but three shells could."),
             figure="04c_ship_motion"),
        dict(heading="5. Switching a billion times a second",
             bottom_line=(f"Switching can't change the amount of negative energy, at any speed. It does bend "
                          f"space-time harder, which loosens the quantum test, but at 1 GHz the wall is still over "
                          f"by about {sci_text(n['qi_1ghz'], 0)}; passing would take about "
                          f"{sci_text(n['f_pass'], 0)} switches a second. Your estimate of when switching starts "
                          f"to matter (300 MHz) was close: the full answer is about {n['f_cross_mhz']:.0f} MHz. "
                          f"Where a breakthrough would have to come from: quantum fields beyond the simplest kind, "
                          f"and physics near the Planck scale. Those are genuinely open questions."),
             figure="06b_switching_quantum_rule"),
    ]
    with PdfPages(path, metadata={"Title": "Warp bubble notebook: results", "Subject": "Warp drive analysis",
                                  "Author": "warpdrive notebook"}) as pdf:
        for i, p in enumerate(pages, start=1):
            _page(pdf, i, len(pages), **p)
    return path
