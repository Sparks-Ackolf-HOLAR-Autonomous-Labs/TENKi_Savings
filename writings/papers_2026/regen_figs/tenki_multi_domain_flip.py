#!/usr/bin/env python3
"""Regenerate the TENKi multi-domain donor-flip figure at true print size.

Panels: two rows. Row 1: the full-record ceiling matrices of the three
single-score materials domains (polymer hardness, JARVIS leaderboard,
Materials Project fidelities), from domain_results[*].ceiling_matrix in
tenki/results/multi_domain_flip.json; their budget-one asymmetry is
identically zero (asserted). Row 2: the directed structure measured on the
record-level band-gap archive, from the 50-seed paired campaigns (exp 26,
CHPC, pairwise arm, fixed-trust full reuse): (a) the asymmetry matrix
A[i,j] = T_early(i->j) - T_early(j->i); (b) the net donor score of each
source (mean outgoing minus mean incoming gain) in the early window and at
the final budget, which shows the budget-dependent role changes; (c)
cross-source stability bars (mean off-diagonal ceiling tau) of the three
single-score domains.

Size and fonts: RSC full-width figure* (6.73 in), fonts per paper_fig_style.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    RSC_FULL, INK, GREY, ACCENT_RED, EXPT_GOLD, SOURCE_COLOURS, apply_style,
    save,
)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/multi_domain_flip.json"
PAIRED = ROOT / ("shared/output/bandgap_cluster_run2/paired_campaign_k4/"
                 "paired_campaign.json")
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "multi_domain_flip.png")

DOMAIN_TITLES = {
    "polymer_hardness": "Polymer hardness",
    "jarvis_leaderboard": "JARVIS leaderboard",
    "materials_project": "Materials Project",
}
STUDY_NAMES = {
    "shore_a": "Shore A", "shore_d": "Shore D", "rockwell_r": "Rockwell R",
    "band_gap_opt": "opt. gap", "form_energy": "form. E",
    "bulk_modulus": "bulk mod.", "shear_modulus": "shear mod.",
    "ehull": "E hull", "exfol_energy": "exfol. E", "band_gap_mbj": "TB-mBJ gap",
    "gga": "GGA", "hse06": "HSE06", "experimental": "expt.",
}
ORDER = ["polymer_hardness", "jarvis_leaderboard", "materials_project"]
SHORT = {
    "expt_zhuo": "Expt. (Zhuo)", "jarvis_mbj": "TB-mBJ",
    "jarvis_optb88vdw": "OptB88vdW", "mp_gga": "MP GGA",
    "mp_gga_expsynth": "MP ICSD", "mp_r2scan": "r$^2$SCAN",
    "snumat_gga": "SNUMAT GGA", "snumat_hse": "SNUMAT HSE06",
}

def heat(ax, mat, names, cmap, vmin, vmax, annotate):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6.5)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = mat[i][j]
                r, g, b, _ = im.cmap(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if lum < 0.45 else INK)
    return im


def main():
    apply_style()
    res = json.load(open(ART))["domain_results"]
    T = json.load(open(PAIRED))["transfer_matrices"]["full"]["pair"]
    src = sorted(T)

    fig = plt.figure(figsize=(IOP_FULL, 5.2), layout="none")
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
                  left=0.12, right=0.92, top=0.94, bottom=0.62, wspace=0.9)
    gb = GridSpec(1, 5, figure=fig, width_ratios=[1.0, 0.05, 0.62, 1.0, 0.85],
                  left=0.17, right=0.97, top=0.41, bottom=0.21, wspace=0.6)

    last_im = None
    for k, dom in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, k])
        r = res[dom]
        assert all(v == 0.0 for row in r["asymmetry_matrix"] for v in row)
        names = [STUDY_NAMES[s] for s in r["study_names"]]
        last_im = heat(ax, r["ceiling_matrix"], names, "viridis", 0.0, 1.0,
                       annotate=(len(names) <= 3))
        ax.set_title(f"({'abc'[k]}) {DOMAIN_TITLES[dom]}", fontsize=8, pad=3)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("full-record ceiling (Kendall $\\tau$)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # (d) directed asymmetry on the band-gap archive
    ax_a = fig.add_subplot(gb[0, 0])
    A = [[(T[a][b]["T_early"] - T[b][a]["T_early"]) if a != b else 0.0
          for b in src] for a in src]
    amax = max(abs(v) for row in A for v in row)
    im = heat(ax_a, A, [SHORT[s] for s in src], "RdBu_r", -amax, amax,
              annotate=False)
    ax_a.set_title("(d) Band-gap archive:\ndirected asymmetry", fontsize=8,
                   pad=4)
    cba = fig.colorbar(im, cax=fig.add_subplot(gb[0, 1]))
    cba.set_label("$T(i\\to j)-T(j\\to i)$", fontsize=7)
    cba.ax.tick_params(labelsize=6.5)

    # (e) net donor score, early window and final budget
    ax_n = fig.add_subplot(gb[0, 3])

    def net(key):
        return {a: sum(T[a][b][key] for b in src if b != a) / (len(src) - 1)
                - sum(T[b][a][key] for b in src if b != a) / (len(src) - 1)
                for a in src}
    ne, nf = net("T_early"), net("T_final")
    order = sorted(src, key=lambda a: -ne[a])
    y = range(len(order))
    ax_n.barh([i - 0.19 for i in y], [ne[a] for a in order], height=0.36,
              color=[SOURCE_COLOURS[a] for a in order], label="early window")
    ax_n.barh([i + 0.19 for i in y], [nf[a] for a in order], height=0.36,
              color="white", edgecolor=[SOURCE_COLOURS[a] for a in order],
              linewidth=0.9, label="final budget")
    ax_n.set_yticks(list(y))
    ax_n.set_yticklabels([SHORT[a] for a in order], fontsize=6.5)
    ax_n.invert_yaxis()
    ax_n.axvline(0, color=INK, linewidth=0.6)
    ax_n.grid(axis="y", visible=False)
    ax_n.set_xlabel("net donor score\n(Kendall $\\tau$)")
    ax_n.set_title("(e) Donor roles", fontsize=8, pad=4)
    ax_n.legend(loc="upper center", bbox_to_anchor=(0.5, -0.52), ncol=2, fontsize=6.5, handlelength=1.0,
                borderpad=0.25, frameon=False)

    # (f) cross-source stability of the single-score domains
    ax_b = fig.add_subplot(gb[0, 4])
    doms = sorted(ORDER, key=lambda d: res[d]["mean_ceiling"])
    vals = [res[d]["mean_ceiling"] for d in doms]
    cols = [ACCENT_RED if v < 0.6 else EXPT_GOLD for v in vals]
    ax_b.barh(range(len(doms)), vals, color=cols, height=0.6)
    ax_b.set_yticks(range(len(doms)))
    ax_b.set_yticklabels([{"Polymer hardness": "polymer", "JARVIS leaderboard": "JARVIS",
                           "Materials Project": "MP"}.get(DOMAIN_TITLES[d], DOMAIN_TITLES[d])
                          for d in doms], fontsize=6.5)
    ax_b.invert_yaxis()
    ax_b.axvline(0.6, color=ACCENT_RED, linestyle="--", linewidth=0.9)
    ax_b.axvline(0.85, color=GREY, linestyle="--", linewidth=0.9)
    for yy, v in enumerate(vals):
        ax_b.annotate(f"{v:.3f}", xy=(v, yy), xytext=(-3, 0),
                      textcoords="offset points", va="center", ha="right",
                      fontsize=6.5, color="white")
    ax_b.set_xlim(0, 1.0)
    ax_b.grid(axis="y", visible=False)
    ax_b.set_xlabel("mean ceiling\n(Kendall $\\tau$)")
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_title("(f) Stability", fontsize=8, pad=4)

    save(fig, OUT)


if __name__ == "__main__":
    main()
