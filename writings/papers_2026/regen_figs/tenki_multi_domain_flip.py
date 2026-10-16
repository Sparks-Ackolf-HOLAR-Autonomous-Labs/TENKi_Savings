#!/usr/bin/env python3
"""Regenerate the TENKi ceiling and donor-role figures at true print size.

Figure multi_domain_flip.png (ceilings): full-record ceiling matrices of
the three single-score materials domains (polymer hardness, JARVIS
leaderboard, Materials Project fidelities), with values printed, from
domain_results[*].ceiling_matrix in tenki/results/multi_domain_flip.json
(their budget-one asymmetry is identically zero, asserted), and the mean
off-diagonal ceiling of each domain with the 0.6 and 0.85 thresholds.

Figure donor_roles.png (roles): the directed structure measured on the
record-level band-gap archive, from the 50-seed paired campaigns (CHPC,
pairwise arm, fixed-trust full reuse): (a) the asymmetry matrix
A[i,j] = T_early(i->j) - T_early(j->i), diagonal hatched; (b) the net donor
score of each source (mean outgoing minus mean incoming gain) in the early
window, with an approximate 95% interval propagated from the per-cell
percentile-bootstrap intervals over seeds (T_early_ci; per-cell standard
error = CI width / 3.92, cells treated as independent), and at the final
budget (no per-cell interval stored for the final budget).

Size and fonts: IOP full width (6.02 in), fonts per paper_fig_style.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL, INK, GREY, SOURCE_COLOURS, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/multi_domain_flip.json"
PAIRED = ROOT / ("shared/output/bandgap_cluster_run2/paired_campaign_k4/"
                 "paired_campaign.json")
FIGS = ROOT / "writings/papers_2026/tenki_mlst/figures"

DOMAIN_TITLES = {
    "polymer_hardness": "(a) Polymer hardness",
    "jarvis_leaderboard": "(b) JARVIS leaderboard",
    "materials_project": "(c) MP band gaps",
}
STUDY_NAMES = {
    "shore_a": "Shore A", "shore_d": "Shore D", "rockwell_r": "Rockwell R",
    "band_gap_opt": "OptB88vdW gap", "form_energy": "form. energy",
    "bulk_modulus": "bulk modulus", "shear_modulus": "shear modulus",
    "ehull": "E above hull", "exfol_energy": "exfol. energy",
    "band_gap_mbj": "TB-mBJ gap",
    "gga": "GGA", "hse06": "HSE06", "experimental": "measured",
}
ORDER = ["polymer_hardness", "jarvis_leaderboard", "materials_project"]
SHORT = {
    "expt_zhuo": "Expt. (Zhuo)", "jarvis_mbj": "JARVIS TB-mBJ",
    "jarvis_optb88vdw": "JARVIS OptB88vdW", "mp_gga": "MP GGA",
    "mp_gga_expsynth": "MP GGA (ICSD-matched)", "mp_r2scan": "MP r$^2$SCAN",
    "snumat_gga": "SNUMAT GGA", "snumat_hse": "SNUMAT HSE06",
}


def heat(ax, mat, names, cmap, vmin, vmax, annotate, short=False,
         xrot=90):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if xrot == 90:
        ax.set_xticklabels(names, rotation=90, fontsize=7)
    else:
        ax.set_xticklabels(names, rotation=xrot, ha="right",
                           rotation_mode="anchor", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
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
                txt = f"{v:.2f}"
                if short:
                    txt = "1" if v > 0.995 else txt.replace("0.", ".")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if lum < 0.45 else INK)
    return im


def ceilings():
    res = json.load(open(ART))["domain_results"]
    fig = plt.figure(figsize=(IOP_FULL, 3.6), layout="none")
    pos = {"polymer_hardness": [0.15, 0.62, 0.15, 0.27],
           "jarvis_leaderboard": [0.47, 0.30, 0.28, 0.58],
           "materials_project": [0.15, 0.17, 0.15, 0.27]}
    last = None
    for dom in ORDER:
        ax = fig.add_axes(pos[dom])
        r = res[dom]
        assert all(v == 0.0 for row in r["asymmetry_matrix"] for v in row)
        names = [STUDY_NAMES[s] for s in r["study_names"]]
        big = len(names) > 3
        last = heat(ax, r["ceiling_matrix"], names, "viridis", 0.0, 1.0,
                    annotate=True, short=big, xrot=90 if big else 40)
        ax.set_title(DOMAIN_TITLES[dom], fontsize=8, pad=3,
                     loc="right" if big else "left")
    cax = fig.add_axes([0.80, 0.55, 0.012, 0.33])
    cb = fig.colorbar(last, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("full-record ceiling\n(Kendall $\\tau$)", fontsize=7)

    ax_b = fig.add_axes([0.875, 0.13, 0.11, 0.22])
    doms = ORDER
    vals = [res[d]["mean_ceiling"] for d in doms]
    ax_b.barh(range(3), vals, color=GREY, height=0.6)
    ax_b.set_yticks(range(3))
    ax_b.set_yticklabels(["polymer", "JARVIS", "MP"], fontsize=7)
    ax_b.invert_yaxis()
    ax_b.axvline(0.6, color=INK, linestyle="--", linewidth=0.8)
    ax_b.axvline(0.85, color=INK, linestyle=":", linewidth=0.8)
    for yy, v in enumerate(vals):
        ax_b.text(v - 0.03, yy, f"{v:.2f}", va="center", ha="right",
                  fontsize=7, color="white")
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_xticklabels(["0", "0.5", "1"])
    ax_b.grid(False)
    ax_b.set_xlabel("mean ceiling", fontsize=7, labelpad=1)
    ax_b.set_title("(d) Stability", fontsize=8, pad=3, loc="left")
    save(fig, FIGS / "multi_domain_flip.png")


def roles():
    d = json.load(open(PAIRED))
    T = d["transfer_matrices"]["full"]["pair"]
    src = sorted(T)
    n = len(src)
    fig = plt.figure(figsize=(IOP_FULL, 3.2), layout="none")
    ax_a = fig.add_axes([0.235, 0.42, 0.25, 0.51])
    cax = fig.add_axes([0.495, 0.42, 0.012, 0.51])
    ax_n = fig.add_axes([0.775, 0.42, 0.2, 0.51])

    A = [[(T[a][b]["T_early"] - T[b][a]["T_early"]) if a != b
          else float("nan") for b in src] for a in src]
    amax = max(abs(v) for row in A for v in row if v == v)
    im = heat(ax_a, A, [SHORT[s] for s in src], "RdBu_r", -amax, amax,
              annotate=False)
    for i in range(n):
        ax_a.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                                 facecolor="#BBBBBB", edgecolor="#777777",
                                 hatch="////", linewidth=0))
    ax_a.set_ylabel("source $i$ (donor)")
    ax_a.set_xlabel("source $j$ (receiver)")
    ax_a.set_title("(a) Directed asymmetry", fontsize=8, pad=3, loc="left")
    cba = fig.colorbar(im, cax=cax)
    cba.set_label("$T(i\\to j)-T(j\\to i)$", fontsize=7)
    cba.ax.tick_params(labelsize=7)

    def net(key):
        return {a: sum(T[a][b][key] for b in src if b != a) / (n - 1)
                - sum(T[b][a][key] for b in src if b != a) / (n - 1)
                for a in src}

    def net_err(a):
        se2 = 0.0
        for b in src:
            if b == a:
                continue
            for c in (T[a][b], T[b][a]):
                se = (c["T_early_ci"][1] - c["T_early_ci"][0]) / 3.92
                se2 += se ** 2
        return 1.96 * math.sqrt(se2) / (n - 1)

    ne, nf = net("T_early"), net("T_final")
    order = sorted(src, key=lambda a: -ne[a])
    for a in order:
        print(f"{a}: early {ne[a]:+.3f} +/- {net_err(a):.3f}, "
              f"final {nf[a]:+.3f}")
    y = range(len(order))
    ax_n.barh([i - 0.19 for i in y], [ne[a] for a in order], height=0.36,
              color=[SOURCE_COLOURS[a] for a in order],
              xerr=[net_err(a) for a in order],
              error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK},
              label="early window")
    ax_n.barh([i + 0.19 for i in y], [nf[a] for a in order], height=0.36,
              color="white", edgecolor=[SOURCE_COLOURS[a] for a in order],
              linewidth=0.9, label="final budget")
    ax_n.set_yticks(list(y))
    ax_n.set_yticklabels([SHORT[a] for a in order], fontsize=7)
    ax_n.invert_yaxis()
    ax_n.axvline(0, color=INK, linewidth=0.6)
    ax_n.grid(axis="y", visible=False)
    ax_n.set_xlabel("net donor score\n(Kendall $\\tau$)")
    ax_n.set_title("(b) Net donor score", fontsize=8, pad=3, loc="left")
    ax_n.legend(loc="upper center", bbox_to_anchor=(0.3, -0.36), ncol=1,
                fontsize=7, handlelength=1.0, columnspacing=0.8,
                frameon=False)
    save(fig, FIGS / "donor_roles.png")


def main():
    apply_style()
    ceilings()
    roles()


if __name__ == "__main__":
    main()
