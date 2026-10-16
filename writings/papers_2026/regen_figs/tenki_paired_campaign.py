#!/usr/bin/env python3
"""Regenerate the TENKi paired-campaign figure at true print size.

Panels: three, from
tenki/results/paired_campaign/paired_campaign.json.
Left and centre: directed early-window transfer-gain matrices for the
k-way arm, transfer_matrices.full.kway[donor][receiver].T_early and
transfer_matrices.selective.kway[...] (verified orientation: outer key
donor, inner key receiver; combined condition keys receivers under
"__combined__"). The diagonal (donor = receiver) is drawn grey and
hatched so it cannot be read as a zero gain. Right: mean early gain of the
six condition/arm combinations with an approximate 95% interval of the
mean, propagated from the per-cell percentile-bootstrap intervals over
seeds (T_early_ci; standard error per cell = CI width / 3.92, cells
treated as independent).

Size and fonts: IOP full width (6.02 in), fonts per paper_fig_style.
Signed heatmaps RdBu_r centred at 0 (red = reuse beats matched-seed
cold start); bars GREY for the pairwise arm and MP_BLUE for the k-way arm.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    MP_BLUE, GREY, INK, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/paired_campaign/paired_campaign.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "paired_campaign.png")

SHORT = {
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
BAR_ORDER = ["full|pair", "full|kway", "selective|pair", "selective|kway",
             "combined|pair", "combined|kway"]


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


def cells(d, cond, arm):
    T = d["transfer_matrices"][cond][arm]
    if cond == "combined":
        return list(T["__combined__"].values())
    return [T[a][b] for a in ORDER for b in ORDER if a != b]


def mean_ci(d, key):
    cond, arm = key.split("|")
    cs = cells(d, cond, arm)
    m = sum(c["T_early"] for c in cs) / len(cs)
    se = math.sqrt(sum(((c["T_early_ci"][1] - c["T_early_ci"][0]) / 3.92)
                       ** 2 for c in cs)) / len(cs)
    return m, 1.96 * se


def main():
    apply_style()
    d = json.load(open(ART))

    fig = plt.figure(figsize=(IOP_FULL, 3.0), layout="none")
    hb, hh = 0.47, 0.44
    axes = [fig.add_axes([0.235, hb, 0.2, hh]),
            fig.add_axes([0.445, hb, 0.2, hh]),
            fig.add_axes([0.855, hb, 0.13, hh])]
    cax = fig.add_axes([0.655, hb, 0.012, hh])
    vmax = 0.5
    ims = []
    for ax, cond, title in zip(
            axes[:2], ("full", "selective"),
            ("(a) Full reuse", "(b) Selective reuse")):
        m = matrix(d, cond)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal")
        ims.append(im)
        for i in range(len(ORDER)):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   facecolor="#BBBBBB", edgecolor="#777777",
                                   hatch="////", linewidth=0))
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([SHORT[s] for s in ORDER], fontsize=7,
                           rotation=90)
        ax.set_yticklabels([SHORT[s] for s in ORDER], fontsize=7)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8, pad=3)
    axes[0].set_ylabel("donor data source")
    axes[1].set_yticklabels([])
    fig.text(0.44, 0.005, "receiver data source", ha="center",
             va="bottom", fontsize=plt.rcParams["axes.labelsize"])
    cb = fig.colorbar(ims[1], cax=cax)
    cb.set_label("$T_{\\mathrm{early}}$ (Kendall $\\tau$)", fontsize=7,
                 labelpad=1)
    cb.ax.tick_params(labelsize=7)

    axb = axes[2]
    mc = [mean_ci(d, k) for k in BAR_ORDER]
    vals = [v for v, _ in mc]
    errs = [e for _, e in mc]
    for k, (v, e) in zip(BAR_ORDER, mc):
        print(f"{k}: {v:+.4f} +/- {e:.4f}")
    cols = [GREY if k.endswith("pair") else MP_BLUE for k in BAR_ORDER]
    ys = [0, 0.7, 1.8, 2.5, 3.6, 4.3]
    axb.barh(ys, vals, color=cols, height=0.62, xerr=errs,
             error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK})
    axb.set_yticks([0.35, 2.15, 3.95])
    axb.set_yticklabels(["full", "selective", "combined"], fontsize=7)
    axb.invert_yaxis()
    axb.set_xlabel("$T_{\\mathrm{early}}$\n(Kendall $\\tau$)")
    axb.set_title("(c) Mean gain", fontsize=8, pad=3, loc="right")
    axb.grid(axis="y", visible=False)
    axb.set_xlim(0, 0.3)
    axb.set_xticks([0, 0.1, 0.2])
    axb.tick_params(axis="y", length=0)
    axb.legend(handles=[Patch(color=GREY, label="pairwise"),
                        Patch(color=MP_BLUE, label="$k$-way")],
               loc="upper center", bbox_to_anchor=(0.4, -0.45), ncol=1,
               fontsize=7, handlelength=1.0, columnspacing=0.8,
               frameon=False)

    save(fig, OUT)


if __name__ == "__main__":
    main()
