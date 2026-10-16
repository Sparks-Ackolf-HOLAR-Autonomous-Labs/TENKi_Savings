#!/usr/bin/env python3
"""Regenerate the TENKi cost-calibration figure at true print size.

Data: tenki/results/cost_calibration/cost_calibration.json,
cells.full.pair[donor][receiver] (outer key donor, inner key receiver;
the receiver's native unit is in each cell's receiver_unit). Condition
and arm: "full" = per-source full reuse at fixed trust (lambda = 8, from
the paired-campaign config), pairwise arm (k = 2), 50 seeds, budget 240,
saving measured at the end of the early window (120 evaluations).
Nothing recomputed.

Layout: rows = receivers (the cost unit belongs to the receiver), columns
= donors. The experimental receiver row (laboratory-hours) is drawn as a
separate one-row panel with its own colour scale; the seven DFT receiver
rows (CPU-core-hours) share a symlog scale. One number format throughout
(thousands, one decimal, "k"). Cells in which at least half of the seeds
are censored (the cold campaign never reaches the reuse arm's tau within
240 evaluations, so the saving is a lower bound) are hatched.

Size and fonts: IOP text width (6.02 in), fonts >= 7 pt.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    INK, IOP_FULL, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/cost_calibration/cost_calibration.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "cost_calibration.png")

SHORT = {
    "expt_zhuo": "Expt.\n(Zhuo)",
    "jarvis_mbj": "JARVIS\nTB-mBJ",
    "jarvis_optb88vdw": "JARVIS\nOptB88\nvdW",
    "mp_gga": "MP\nGGA",
    "mp_gga_expsynth": "MP GGA\n(ICSD-\nmatched)",
    "mp_r2scan": "MP\nr$^2$SCAN",
    "snumat_gga": "SNUMAT\nGGA",
    "snumat_hse": "SNUMAT\nHSE06",
}
ROWNAME = {
    "expt_zhuo": "Expt. (Zhuo)",
    "jarvis_mbj": "JARVIS TB-mBJ",
    "jarvis_optb88vdw": "JARVIS OptB88vdW",
    "mp_gga": "MP GGA",
    "mp_gga_expsynth": "MP GGA (ICSD-matched)",
    "mp_r2scan": "MP r$^2$SCAN",
    "snumat_gga": "SNUMAT GGA",
    "snumat_hse": "SNUMAT HSE06",
}
ORDER = ["expt_zhuo", "jarvis_mbj", "jarvis_optb88vdw", "mp_gga",
         "mp_gga_expsynth", "mp_r2scan", "snumat_gga", "snumat_hse"]
CENSOR_FRAC = 0.5
FS = 7
MINUS = "−"


def fmt(v):
    return "{:+.1f}k".format(v / 1000).replace("-", MINUS)


def draw(ax, rows, cells, cmap, norm):
    mat = np.full((len(rows), len(ORDER)), np.nan)
    cens = np.zeros_like(mat, dtype=bool)
    for i, recv in enumerate(rows):
        for j, donor in enumerate(ORDER):
            if donor == recv:
                continue
            c = cells[donor][recv]
            mat[i, j] = c["hours_saved_mid"]
            cens[i, j] = c["censored_frac"] >= CENSOR_FRAC
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cm, norm=norm,
                   aspect="auto")
    for i in range(len(rows)):
        for j in range(len(ORDER)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=FS, color=INK)
                continue
            v = mat[i, j]
            face = im.cmap(im.norm(v))
            lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            if cens[i, j]:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                    edgecolor=(1, 1, 1, 0.7) if lum < 0.5 else
                    (0.2, 0.2, 0.2, 0.45), linewidth=0))
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=FS,
                    color="white" if lum < 0.5 else INK,
                    bbox=dict(boxstyle="square,pad=0.08", lw=0, fc=face))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([ROWNAME[r] for r in rows], fontsize=FS)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im, int(cens.sum())


def main():
    apply_style()
    cells = json.load(open(ART, encoding="utf-8"))["cells"]["full"]["pair"]

    fig = plt.figure(figsize=(IOP_FULL, 4.5), layout="none")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 7],
                          width_ratios=[1, 0.03], hspace=0.16, wspace=0.03)
    ax_e = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[1, 0], sharex=ax_e)
    cax_e = fig.add_subplot(gs[0, 1])
    cax_d = fig.add_subplot(gs[1, 1])

    norm_e = mcolors.Normalize(vmin=-3000, vmax=3000)
    im_e, n_e = draw(ax_e, ["expt_zhuo"], cells, "PuOr_r", norm_e)
    norm_d = mcolors.SymLogNorm(linthresh=100, vmin=-3e5, vmax=3e5,
                                base=10)
    im_d, n_d = draw(ax_d, ORDER[1:], cells, "RdBu_r", norm_d)

    ax_e.tick_params(labelbottom=False, labeltop=False)
    ax_d.set_xticks(range(len(ORDER)))
    ax_d.set_xticklabels([SHORT[s] for s in ORDER], fontsize=FS)
    ax_d.set_xlabel("donor data source (archived ranking reused)",
                    fontsize=FS + 1)
    ax_e.set_title("receiver: experiment (laboratory-hours saved)",
                   fontsize=FS + 1, pad=3, loc="left")
    ax_d.set_title("receivers: DFT data sources (CPU-core-hours saved)",
                   fontsize=FS + 1, pad=3, loc="left")

    cb_e = fig.colorbar(im_e, cax=cax_e, ticks=[-3000, 0, 3000])
    cb_e.ax.set_yticklabels([MINUS + "3k", "0", "+3k"])
    cb_e.ax.tick_params(labelsize=FS)
    cb_d = fig.colorbar(im_d, cax=cax_d, ticks=[-1e5, -1e3, 0, 1e3, 1e5])
    cb_d.ax.set_yticklabels([MINUS + "100k", MINUS + "1k", "0", "+1k",
                             "+100k"])
    cb_d.ax.tick_params(labelsize=FS)
    cb_d.set_label("symmetric log scale", fontsize=FS)

    hatch = mpatches.Patch(facecolor="white", edgecolor=INK, hatch="////",
                           linewidth=0.5,
                           label="at least half of the seeds censored "
                                 "(saving is a lower bound)")
    fig.legend(handles=[hatch], loc="lower center", fontsize=FS,
               frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.235, right=0.9, top=0.94, bottom=0.19)
    save(fig, OUT)
    print("censored (lower-bound) cells: expt row", n_e, "DFT rows", n_d)


if __name__ == "__main__":
    main()
