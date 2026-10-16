#!/usr/bin/env python3
"""Regenerate the TENKi physics-informed-budget figure at true print size.

Panels: two, from
tenki/results/physics_informed_budget/physics_informed_budget.json.
Left: tau vs budget for the five arms (checkpoints +
aggregate.{uninformed,physics_fixed,physics_adaptive,wrong_fixed,
wrong_adaptive}.mean), +-1 sd band only for uninformed and wrong_fixed
(the extreme arms) to keep the panel readable. Right:
lambda_trajectory_median for the two adaptive arms on a log axis, with
the lambda grid (lambda_grid) and the fixed lambda=4 marked. Caption
colours preserved: correct-prior learner green, reversed-prior learner
orange. Every number is read from the artifact.

Size and fonts: full-width figure* (6.73 in), fonts per paper_fig_style.
Labels use "data source".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    MP_BLUE, GREY, ACCENT_RED, JARVIS_GREEN, INK, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / ("tenki/results/physics_informed_budget/"
              "physics_informed_budget.json")
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "physics_informed_budget.png")

ORANGE = "#E69F00"  # Okabe-Ito orange, matches the caption's "orange"

ARMS = [
    ("uninformed", "uninformed", GREY, "-", True),
    ("physics_fixed", "donor prior, fixed trust", MP_BLUE, "-", False),
    ("physics_adaptive", "donor prior, learned trust", JARVIS_GREEN, "-",
     False),
    ("wrong_fixed", "reversed prior, fixed trust", ACCENT_RED, "--", True),
    ("wrong_adaptive", "reversed prior, learned trust", ORANGE, "--",
     False),
]


def main():
    apply_style()
    d = json.load(open(ART))
    x = d["checkpoints"]

    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(IOP_FULL, 2.7), width_ratios=[1.55, 1.0])

    for key, label, col, ls, band in ARMS:
        m = d["aggregate"][key]["mean"]
        ax.plot(x, m, ls, color=col, label=label)
        if band:
            s = d["aggregate"][key]["std"]
            ax.fill_between(x, [a - b for a, b in zip(m, s)],
                            [a + b for a, b in zip(m, s)],
                            color=col, alpha=0.12, linewidth=0)
    ax.set_xlim(0, 240)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.set_xlabel("cumulative evaluations")
    ax.set_ylabel("Kendall $\\tau$ vs reference ranking")
    ax.legend(loc="lower right", handlelength=1.8, borderpad=0.25)
    ax.set_title("Donor prior under budget (five arms)", fontsize=9, pad=3)

    grid = d["lambda_grid"]
    axl.set_yscale("log")
    for gv in grid:
        axl.axhline(gv, color="#DDDDDD", linewidth=0.5, zorder=0)
    axl.axhline(4.0, color=MP_BLUE, linewidth=0.9, linestyle=":", zorder=1)
    axl.annotate("fixed $\\lambda=4$", xy=(238, 4.0), xytext=(0, 3),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.5, color=MP_BLUE)
    axl.plot(x, d["lambda_trajectory_median"]["physics_adaptive"], "-",
             color=JARVIS_GREEN, label="correct prior")
    axl.plot(x, d["lambda_trajectory_median"]["wrong_adaptive"], "--",
             color=ORANGE, label="reversed prior")
    axl.set_xlim(0, 240)
    axl.set_xticks([0, 60, 120, 180, 240])
    axl.set_xlabel("cumulative evaluations")
    axl.set_ylabel("median learned trust $\\lambda$")
    axl.set_yticks(grid)
    axl.set_yticklabels([str(g) for g in grid])
    axl.minorticks_off()
    axl.legend(loc="lower right", handlelength=1.8, borderpad=0.25)
    axl.set_title("Learned trust in the prior", fontsize=9, pad=3)
    axl.grid(False)

    save(fig, OUT)


if __name__ == "__main__":
    main()
