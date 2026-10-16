#!/usr/bin/env python3
"""Regenerate the TENKi hyperedge-budget figure at true print size.

Panels: one. Data: checkpoints + aggregate.{pair_random,
kway_random,kway_active}.{mean,std} from
tenki/results/hyperedge_budget/hyperedge_budget.json (mean tau over the
eight data sources, +-1 sd band). Takeaway (caption): random 4-way
ranking events dominate random pairwise queries along the whole budget
axis; the epsilon-greedy active arm tracks random k-way selection.

Size and fonts: 0.62\textwidth figure (4.17 in), fonts per paper_fig_style.
Curve conventions: k-way random solid MP_BLUE, pairwise random dashed
GREY, active arm ACCENT_RED. Direct end-of-curve labels, no legend box.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    MP_BLUE, GREY, ACCENT_RED, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/hyperedge_budget/hyperedge_budget.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "hyperedge_budget.png")

ARMS = [
    ("pair_random", "pairwise random", GREY, "--"),
    ("kway_random", "$k$-way random", MP_BLUE, "-"),
    ("kway_active", "$k$-way active\n($\\epsilon$-greedy)", ACCENT_RED, "-"),
]


def main():
    apply_style()
    d = json.load(open(ART))
    x = d["checkpoints"]

    fig, ax = plt.subplots(figsize=(4.17, 2.9))
    for key, label, col, ls in ARMS:
        m = d["aggregate"][key]["mean"]
        s = d["aggregate"][key]["std"]
        ax.plot(x, m, ls, color=col, label=label)
        ax.fill_between(x, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)],
                        color=col, alpha=0.13, linewidth=0)
    # direct labels at the right edge, spread vertically to avoid overlap
    finals = {k: d["aggregate"][k]["mean"][-1] for k, _, _, _ in ARMS}
    offsets = {"pair_random": -0.012, "kway_random": 0.045,
               "kway_active": -0.045}
    for key, label, col, _ in ARMS:
        ax.annotate(label.replace("\n", " "),
                    xy=(x[-1], finals[key] + offsets[key]),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color=col)
    ax.set_xlim(0, 240)
    ax.set_xlabel("cumulative evaluations (one per queried entity)")
    ax.set_ylabel("Kendall $\\tau$ vs reference ranking")
    # extend x-limits a bit so the direct labels have room
    ax.set_xlim(0, 335)
    ax.set_xticks([0, 60, 120, 180, 240])
    save(fig, OUT)


if __name__ == "__main__":
    main()
