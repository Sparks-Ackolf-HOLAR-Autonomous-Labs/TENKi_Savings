#!/usr/bin/env python3
"""Regenerate the TENKi cross-property transfer figure at true print size.

Panels: six (2x3) from
tenki/results/cross_property/cross_property.json, per_channel[*]:
checkpoints, curves.kway_learned / curves.kway_random and their
curves_std (20 evaluation seeds), plus final_tau for the in-panel
annotations. Each panel shows the fixed band-gap-trained policy against
random k-way selection (the hand-written/epsilon arms are in the artifact
but are not part of this figure's claim, so they are omitted).

AFLOW formation-energy panel: the 200-fresh-seed rerun from
tenki/results/cross_property/formation_energy_diagnosis.json
(fresh_200_seeds: final learned, random, paired difference and its 90%
bootstrap interval) is overlaid as open diamonds at the final budget and
quoted in the panel. Only final values were stored for the 200-seed
rerun, so no 200-seed curve is drawn.

MP formation-energy panel: four classes and k = 4, so every event holds
all classes and no acquisition rule can differ from random. The panel is
greyed out and says so.

Size and fonts: IOP text width (6.02 in), fonts >= 7 pt. Colours follow
the paper convention: trained policy JARVIS_GREEN, random 4-way MP_BLUE.
+-1 sd bands across seeds. One shared legend.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    MP_BLUE, JARVIS_GREEN, INK, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/cross_property/cross_property.json"
DIAG = ROOT / "tenki/results/cross_property/formation_energy_diagnosis.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "cross_property.png")

PANELS = [
    ("bulk_jarvis", "Bulk modulus, JARVIS"),
    ("bulk_aflow", "Bulk modulus, AFLOW"),
    ("bulk_mp", "Bulk modulus, MP"),
    ("hardness_chen_niu", "Vickers hardness, Chen--Niu"),
    ("fe_aflow_dHf", "Formation energy, AFLOW\n(Al--Mg--Si)"),
    ("fe_mp_dHf", "Formation energy, MP\n(Al--Mg--Si)"),
]
ARMS = [
    ("kway_random", "random 4-way", MP_BLUE, "-"),
    ("kway_learned", "trained policy", JARVIS_GREEN, "-"),
]
FS = 7


def main():
    apply_style()
    d = json.load(open(ART, encoding="utf-8"))["per_channel"]
    diag = json.load(open(DIAG, encoding="utf-8"))

    fig, axes = plt.subplots(2, 3, figsize=(IOP_FULL, 3.8),
                             sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ch = d[key]
        x = ch["checkpoints"]
        no_choice = key == "fe_mp_dHf"
        for arm, label, col, ls in ARMS:
            m = ch["curves"][arm]
            s = ch["curves_std"][arm]
            ax.plot(x, m, ls, color=col, label=label,
                    alpha=0.45 if no_choice else 1.0)
            ax.fill_between(x, [a - b for a, b in zip(m, s)],
                            [a + b for a, b in zip(m, s)],
                            color=col, alpha=0.06 if no_choice else 0.14,
                            linewidth=0)
        ax.set_title(title.replace("--", "–"), fontsize=8, pad=2)
        ft = ch["final_tau"]
        if key == "fe_aflow_dHf":
            f2 = diag["aflow_dHf"]["fresh_200_seeds"]
            lo, hi = f2["diff_ci90"]
            # just right of the axis end so the two marks do not overlap
            for val, col, mk, xm in ((f2["learned"], JARVIS_GREEN, "D", 249),
                                     (f2["random"], MP_BLUE, "o", 259)):
                ax.plot([xm], [val], mk, markersize=4.5,
                        markerfacecolor="white", markeredgecolor=col,
                        markeredgewidth=1.2, clip_on=False, zorder=5)
            txt = ("20 seeds: {:.2f} vs {:.2f}\n"
                   "200 seeds (open marks):\n{:.2f} vs {:.2f}, "
                   "difference {:+.2f}\n(90% CI {:+.2f} to {:+.2f})").format(
                       ft["kway_learned"], ft["kway_random"],
                       f2["learned"], f2["random"], f2["diff"], lo, hi)
            txt = txt.replace("-", "\u2212")
        elif no_choice:
            ax.set_facecolor("#EFEFEF")
            txt = ("every event holds all\n"
                   "four classes: no\n"
                   "selection possible")
        else:
            txt = "final: {:.2f} vs {:.2f}".format(
                ft["kway_learned"], ft["kway_random"])
        ax.annotate(txt, xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=FS, color=INK,
                    linespacing=1.15, annotation_clip=False
                    ).set_in_layout(False)
        ax.set_xlim(0, 240)
        ax.set_xticks([0, 80, 160, 240])
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[1, :]:
        ax.set_xlabel("cumulative evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Kendall $\\tau$ vs reference")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center",
               ncols=2, fontsize=FS, handlelength=1.8)
    save(fig, OUT)


if __name__ == "__main__":
    main()
