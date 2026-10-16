#!/usr/bin/env python3
"""TENKi annex figure: direct comparison with PRISMS on the computed Al-Mg-Si pool.

Panels: two (PBE receiver, HSE06 receiver). Horizontal bars: mean
share of receiver compute needed to find eight of the ten most stable
compounds, with 95% bootstrap intervals over 20 runs. PRISMS arms (fidelity
order supplied in advance) in grey, rank-prior arms in blue (MACE-MP-0 prior) and purple (three potentials
pooled); green is kept for the trained acquisition policy elsewhere. From
compute/almgsi_campaign/results/analysis/prisms_comparison/prisms_comparison_{qe_pbe,qe_hse06}.json.

Size and fonts: IOP text width, paper_fig_style fonts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL, MP_BLUE, SNUMAT_PURPLE, GREY, INK, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "compute/almgsi_campaign/results/analysis/prisms_comparison"
OUTS = [ROOT / "writings/papers_2026/tenki_mlst/figures/prisms_comparison.png"]

# Reader-facing row labels; potentials named as in Table 1 of the paper.
# "two potentials" row = PRISMS with MEAM MD as low fidelity and MACE-MP-0
# relaxation as high fidelity (NEP89 left out). The two MACE-MP-0 prior
# rows have zero-width intervals: the MACE-MP-0 relaxation energies carry
# no MD noise draw and the prior-ranked campaign has no random first batch,
# so all 20 runs are identical.
ARMS = [
    ("prisms_3fid:cost", "PRISMS, order by cost\n(MACE-MP-0, MEAM, NEP89)", GREY),
    ("prisms_3fid:class", "PRISMS, order by model class\n(MEAM, NEP89, MACE-MP-0)", GREY),
    ("prisms_3fid:measured", "PRISMS, order by measured $\\tau$\n(NEP89, MEAM, MACE-MP-0)", GREY),
    ("prisms_2fid:meam_to_mace", "PRISMS, two potentials\n(MEAM, MACE-MP-0)", GREY),
    ("prisms_single:mace_relax", "PRISMS, MACE-MP-0 only", GREY),
    ("surrogate", "no donor (surrogate only)", INK),
    ("fixed:mace_relax", "MACE-MP-0 prior, fixed trust", MP_BLUE),
    ("learned:mace_relax", "MACE-MP-0 prior, learned trust", MP_BLUE),
    ("pooled_learned", "three potentials pooled,\nlearned trust", SNUMAT_PURPLE),
]
PANELS = [("qe_pbe", "(a) receiver: PBE"), ("qe_hse06", "(b) receiver: HSE06")]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(IOP_FULL, 3.3), sharey=True)
    for ax, (rec, title) in zip(axes, PANELS):
        d = json.load(open(ART / f"prisms_comparison_{rec}.json", encoding="utf-8"))["receivers"][rec]["arms"]
        ys = range(len(ARMS))[::-1]
        for y, (key, _, col) in zip(ys, ARMS):
            a = d[key]
            m = 100 * a["cost_frac_to_8of10"]
            lo, hi = (100 * v for v in a["c8_ci"])
            ax.barh(y, m, color=col, height=0.65)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            ax.text(hi + 1.5, y, f"{m:.0f}", va="center", ha="left", fontsize=7, color=INK)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.supxlabel("receiver compute needed to find 8 of the 10 most stable compounds (%)", fontsize=8)
    axes[0].set_yticks(list(range(len(ARMS))[::-1]))
    axes[0].set_yticklabels([lab for _, lab, _ in ARMS])

    for o in OUTS:
        save(fig, o)


if __name__ == "__main__":
    main()
