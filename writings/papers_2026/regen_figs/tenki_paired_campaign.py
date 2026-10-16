#!/usr/bin/env python3
"""Regenerate the TENKi paired-campaign figure at true print size.

Panels: three, from
tenki/results/paired_campaign/paired_campaign.json.
Left and centre: directed early-window transfer-gain matrices for the
k-way arm, transfer_matrices.full.kway[donor][receiver].T_early and
transfer_matrices.selective.kway[...] (verified orientation: outer key
donor, inner key receiver; combined condition keys receivers under
"__combined__"). Right: summary["cond|arm"].mean_T_early bars for the
six condition/arm combinations. Every number is read from the artifact.

Size and fonts: full-width figure* (6.73 in), fonts per paper_fig_style.
Signed heatmaps RdBu_r centred at 0 (red = reuse beats matched-seed
cold start, matching the caption); shared labelled colourbar; bars use
GREY for the pairwise arm and MP_BLUE for the k-way arm. Labels use
"data source".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    MP_BLUE, GREY, INK, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/paired_campaign/paired_campaign.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "paired_campaign.png")

SHORT = {
    "expt_zhuo": "Expt. (Zhuo)",
    "jarvis_mbj": "JARVIS TB-mBJ",
    "jarvis_optb88vdw": "JARVIS OptB88vdW",
    "mp_gga": "MP GGA",
    "mp_gga_expsynth": "MP GGA (ICSD)",
    "mp_r2scan": "MP r$^2$SCAN",
    "snumat_gga": "SNUMAT GGA",
    "snumat_hse": "SNUMAT HSE06",
}
ORDER = ["expt_zhuo", "jarvis_mbj", "jarvis_optb88vdw", "mp_gga",
         "mp_gga_expsynth", "mp_r2scan", "snumat_gga", "snumat_hse"]
BAR_ORDER = ["full|pair", "full|kway", "selective|pair", "selective|kway",
             "combined|pair", "combined|kway"]
BAR_LABELS = {
    "full|pair": "full | pairwise",
    "full|kway": "full | $k$-way",
    "selective|pair": "selective | pairwise",
    "selective|kway": "selective | $k$-way",
    "combined|pair": "combined | pairwise",
    "combined|kway": "combined | $k$-way",
}


def matrix(d, cond):
    m = []
    for a in ORDER:
        row = []
        for b in ORDER:
            if a == b:
                row.append(float("nan"))
            else:
                row.append(d["transfer_matrices"][cond]["kway"][a][b]
                           ["T_early"])
        m.append(row)
    return m


def main():
    apply_style()
    d = json.load(open(ART))

    # Manual layout (fractions of the 6.02 x 2.85 in canvas) so the long
    # tick labels, the shared colour bar and the bar panel all fit.
    fig = plt.figure(figsize=(IOP_FULL, 2.85), layout="none")
    hb, hh = 0.405, 0.455
    axes = [fig.add_axes([0.178, hb, 0.215, hh]),
            fig.add_axes([0.401, hb, 0.215, hh]),
            fig.add_axes([0.815, 0.36, 0.165, 0.49])]
    cax = fig.add_axes([0.626, hb, 0.012, hh])
    vmax = 0.5
    ims = []
    for ax, cond, title in zip(
            axes[:2], ("full", "selective"),
            ("(a) Full reuse", "(b) Selective reuse")):
        m = matrix(d, cond)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal")
        ims.append(im)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([SHORT[s] for s in ORDER], fontsize=6.5,
                           rotation=90)
        ax.set_yticklabels([SHORT[s] for s in ORDER], fontsize=6.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8, pad=3)
    axes[0].set_ylabel("donor data source")
    axes[1].set_yticklabels([])
    fig.text(0.397, 0.005, "receiver data source", ha="center",
             va="bottom", fontsize=plt.rcParams["axes.labelsize"])
    cb = fig.colorbar(ims[1], cax=cax)
    cb.set_label("early transfer gain\n$T_{\\mathrm{early}}$ (Kendall $\\tau$)",
                 fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    axb = axes[2]
    # Bars grouped by reuse condition; colour gives the query type.
    vals = [d["summary"][k]["mean_T_early"] for k in BAR_ORDER]
    cols = [GREY if k.endswith("pair") else MP_BLUE for k in BAR_ORDER]
    ys = [0, 0.7, 1.8, 2.5, 3.6, 4.3]
    axb.barh(ys, vals, color=cols, height=0.62)
    axb.set_yticks([0.35, 2.15, 3.95])
    axb.set_yticklabels(["full", "selective", "combined"], fontsize=6.5)
    axb.invert_yaxis()
    for y, v in zip(ys, vals):
        axb.annotate(f"+{v:.3f}", xy=(v, y), xytext=(-2, 0),
                     textcoords="offset points", va="center", ha="right",
                     fontsize=6, color="white")
    axb.set_xlabel("early gain\n(Kendall $\\tau$)")
    axb.set_title("(c) Mean early gain by\nreuse and query type",
                  fontsize=8, pad=3, loc="right")
    axb.grid(axis="y", visible=False)
    axb.set_xlim(0, 0.26)
    axb.set_xticks([0, 0.1, 0.2])
    axb.tick_params(axis="y", length=0)
    axb.legend(handles=[Patch(color=GREY, label="pairwise"),
                        Patch(color=MP_BLUE, label="$k$-way")],
               loc="upper center", bbox_to_anchor=(0.35, -0.42), ncol=2,
               fontsize=6.5, handlelength=1.0, columnspacing=0.8,
               frameon=False)

    save(fig, OUT)


if __name__ == "__main__":
    main()
