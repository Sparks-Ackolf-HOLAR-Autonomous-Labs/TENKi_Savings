"""Shared figure style for the three 2026 papers.

Every figure is created at its true print size, so a "7 pt" tick label
is 7 pt on the printed page; nothing is downscaled. The module defines
the print widths, the font sizes, the database-of-origin palette and
reader-facing source names, and a save helper.

Print geometry (from the LaTeX class files):
  RSC (Digital Discovery, 9pt two-column): text block 18.0 cm wide,
    column 8.6 cm; \textwidth figures ~17.1 cm after margins as used
    in the drafts. Body text 9 pt, captions ~8 pt.
  TMLR: 6.5 in text width, 10 pt body.

Figure fonts lie between caption size and body size at print:
  ticks 7 pt, axis labels 8 pt, panel titles 9 pt, legends 7 pt.

Palette: the drafts' CVD-validated database-of-origin colours.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print widths in inches
RSC_COL = 3.27       # 8.3 cm column
RSC_FULL = 6.73      # 17.1 cm full width
TMLR_FULL = 6.50
IOP_FULL = 6.02      # 153 mm text width of iopjournal.cls (MLST)

# Database-of-origin palette (CVD-validated)
MP_BLUE = "#2E5FA3"
JARVIS_GREEN = "#009E73"
SNUMAT_PURPLE = "#7A4DD8"
EXPT_GOLD = "#B8860B"
INK = "#333333"
GREY = "#8C8C8C"
ACCENT_RED = "#C4452C"

SOURCE_COLOURS = {
    "mp_gga": MP_BLUE,
    "mp_r2scan": MP_BLUE,
    "mp_gga_expsynth": MP_BLUE,
    "jarvis_optb88vdw": JARVIS_GREEN,
    "jarvis_mbj": JARVIS_GREEN,
    "snumat_gga": SNUMAT_PURPLE,
    "snumat_hse": SNUMAT_PURPLE,
    "expt_zhuo": EXPT_GOLD,
}

# Reader-facing names (plain vocabulary: these are "data sources")
SOURCE_NAMES = {
    "mp_gga": "MP GGA",
    "mp_r2scan": "MP r$^2$SCAN",
    "mp_gga_expsynth": "MP GGA (ICSD-matched)",
    "jarvis_optb88vdw": "JARVIS OptB88vdW",
    "jarvis_mbj": "JARVIS TB-mBJ",
    "snumat_gga": "SNUMAT GGA",
    "snumat_hse": "SNUMAT HSE06",
    "expt_zhuo": "Experiment (Zhuo)",
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
