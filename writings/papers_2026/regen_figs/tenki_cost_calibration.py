#!/usr/bin/env python3
"""Regenerate the TENKi cost-calibration figure at true print size.

Panels: one heatmap from
tenki/results/cost_calibration/cost_calibration.json,
cells.full.pair[donor][receiver].hours_saved_mid (verified orientation:
outer key donor, inner key receiver; the receiver's native unit is in
each cell's receiver_unit). Condition and arm: full reuse, pairwise
arm, whose values back the caption's claims (SNUMAT ->
HSE06 saving of order 2e5 CPU-core-hours; only the SNUMAT donors price
positive for the experimental receiver). Rows = receivers so each row
is one native cost unit; the experimental row is outlined. Symlog colour
scale, cell annotations from the artifact. Nothing recomputed.

Size and fonts: 0.9\textwidth figure (6.06 in), fonts per paper_fig_style.
RdBu_r centred at 0 (red = hours saved), labelled colourbar.
Labels use "data source".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    INK, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/cost_calibration/cost_calibration.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "cost_calibration.png")

SHORT = {
    "expt_zhuo": "Expt.\n(Zhuo)",
    "jarvis_mbj": "JARVIS\nTB-mBJ",
    "jarvis_optb88vdw": "JARVIS\nOptB88vdW",
    "mp_gga": "MP\nGGA",
    "mp_gga_expsynth": "MP GGA\n(ICSD)",
    "mp_r2scan": "MP\nr$^2$SCAN",
    "snumat_gga": "SNUMAT\nGGA",
    "snumat_hse": "SNUMAT\nHSE06",
}
# receivers as rows (each row one native cost unit), donors as columns
ORDER = ["expt_zhuo", "jarvis_mbj", "jarvis_optb88vdw", "mp_gga",
         "mp_gga_expsynth", "mp_r2scan", "snumat_gga", "snumat_hse"]


def main():
    apply_style()
    cells = json.load(open(ART))["cells"]["full"]["pair"]

    mat = []
    for recv in ORDER:  # row = receiver
        row = []
        for donor in ORDER:  # col = donor
            if donor == recv:
                row.append(float("nan"))
            else:
                row.append(cells[donor][recv]["hours_saved_mid"])
        mat.append(row)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    norm = mcolors.SymLogNorm(linthresh=100, vmin=-3e5, vmax=3e5, base=10)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([SHORT[s] for s in ORDER], fontsize=6)
    ylabels = [SHORT[s].replace("\n", " ") for s in ORDER]
    ylabels[0] += "\n(instrument-hours)"
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlabel("donor data source (archived screen)")
    ax.set_ylabel("receiver data source (row in its native cost unit)")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, recv in enumerate(ORDER):
        for j, donor in enumerate(ORDER):
            if donor == recv:
                continue
            v = mat[i][j]
            r, g, b, _ = im.cmap(im.norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, (f"{v / 1000:+.0f}k" if abs(v) >= 10000 else f"{v:+,.0f}"), ha="center", va="center",
                    fontsize=6.5,
                    color="white" if lum < 0.5 else INK)
    # outline the experimental receiver row (instrument-hours)
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 8, 1, fill=False,
                                    edgecolor=INK, linewidth=1.2))
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, aspect=24)
    cb.set_label("hours saved (receiver's own cost unit, log scale)",
                 fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, OUT)


if __name__ == "__main__":
    main()
