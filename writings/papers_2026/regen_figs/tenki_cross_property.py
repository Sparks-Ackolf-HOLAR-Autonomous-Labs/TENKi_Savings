#!/usr/bin/env python3
"""Regenerate the TENKi cross-property transfer figure at true print size.

Panels: six (2x3) from
tenki/results/cross_property/cross_property.json, per_channel[*]:
checkpoints, curves.kway_learned / curves.kway_random and their
curves_std, plus final_tau for the in-panel annotations. As stated in
the caption, each panel shows the fixed band-gap-trained policy against
random k-way selection (the three hand-written/epsilon arms are in the
artifact but are not part of this figure's claim, so they are omitted
for legibility).

Size and fonts: \textwidth figure (6.73 in), fonts per paper_fig_style.
k-way random solid MP_BLUE, fixed trained policy ACCENT_RED. +-1 sd
bands. One shared legend. Labels use "data source".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    MP_BLUE, ACCENT_RED, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/cross_property/cross_property.json"
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
    ("kway_random", "random $k$-way", MP_BLUE, "-"),
    ("kway_learned", "fixed trained policy", ACCENT_RED, "-"),
]


def main():
    apply_style()
    d = json.load(open(ART))["per_channel"]

    fig, axes = plt.subplots(2, 3, figsize=(IOP_FULL, 3.6),
                             sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ch = d[key]
        x = ch["checkpoints"]
        for arm, label, col, ls in ARMS:
            m = ch["curves"][arm]
            s = ch["curves_std"][arm]
            ax.plot(x, m, ls, color=col, label=label)
            ax.fill_between(x, [a - b for a, b in zip(m, s)],
                            [a + b for a, b in zip(m, s)],
                            color=col, alpha=0.14, linewidth=0)
        ax.set_title(title.replace("--", "\u2013"), fontsize=8, pad=2)
        ax.annotate(
            "final: policy {:.2f}, random {:.2f}".format(
                ch["final_tau"]["kway_learned"],
                ch["final_tau"]["kway_random"]),
            xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
            va="bottom", fontsize=6.5)
        ax.set_xlim(0, 240)
        ax.set_xticks([0, 80, 160, 240])
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[1, :]:
        ax.set_xlabel("cumulative evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Kendall $\\tau$ vs reference")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               fontsize=7, handlelength=1.8)
    save(fig, OUT)


if __name__ == "__main__":
    main()
