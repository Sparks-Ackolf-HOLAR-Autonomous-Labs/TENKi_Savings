#!/usr/bin/env python3
"""TENKi worked-example figure: how to read the two core diagnostics.

Panels: two, both from committed artifacts.
(a) Reading a budget curve: mean Kendall-tau-versus-budget for random
    pairwise duels and random 4-way ranking events (experiment 24
    aggregate over the eight data sources;
    tenki/results/hyperedge_budget/hyperedge_budget.json). Annotated
    with the 90%-of-final threshold of the k-way arm and the budget at
    which each arm first reaches it: the same instrument time spent on
    four-sample ranked batches recovers the ordering sooner.
(b) Reading a ceiling: the polymer-hardness permanent gap
    (tenki/results/multi_domain_flip.json, ceiling and budget-one
    matrices): against the Rockwell R reference, Shore D tops out at
    0.867 while Shore A tops out at 0.467, and Shore A's budget-one
    value already equals its ceiling, so the very first Shore A lookup
    delivers everything Shore A will ever deliver; the rest is
    different indenter physics, not noise.

Size and fonts: full-width RSC figure (6.73 in), fonts per paper_fig_style.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    RSC_FULL, MP_BLUE, EXPT_GOLD, ACCENT_RED, GREY, INK,
    apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
HEB = ROOT / "tenki/results/hyperedge_budget/hyperedge_budget.json"
MDF = ROOT / "tenki/results/multi_domain_flip.json"
OUT = ROOT / "writings/papers_2026/tenki_mlst/figures/worked_example_tenki.png"


def first_crossing(budgets, means, level):
    for b, m in zip(budgets, means):
        if m >= level:
            return b
    return None


def main():
    apply_style()
    heb = json.load(open(HEB))
    budgets = heb["checkpoints"]
    agg = heb["aggregate"]
    pair = agg["pair_random"]["mean"]
    kway = agg["kway_random"]["mean"]

    mdf = json.load(open(MDF))["domain_results"]["polymer_hardness"]
    studies = mdf["study_names"]  # [shore_a, shore_d, rockwell_r]
    i_a, i_d, i_r = (studies.index(s) for s in
                     ("shore_a", "shore_d", "rockwell_r"))
    ceil_a = mdf["ceiling_matrix"][i_a][i_r]
    ceil_d = mdf["ceiling_matrix"][i_d][i_r]
    tau1_a = mdf["tau_n1_matrix"][i_a][i_r]

    fig, axes = plt.subplots(1, 2, figsize=(IOP_FULL, 2.6),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) reading a budget curve
    ax = axes[0]
    ax.plot(budgets, pair, "--", color=GREY, label="pairwise duels (cost 2 each)")
    ax.plot(budgets, kway, "-", color=MP_BLUE,
            label="4-way ranked batches (cost 4 each)")
    level = 0.9 * kway[-1]
    b_kway = first_crossing(budgets, kway, level)
    b_pair = first_crossing(budgets, pair, level)
    ax.axhline(level, color=INK, lw=0.7, ls=":")
    ax.annotate("90% of final batch value",
                xy=(budgets[0] + 2, level + 0.012), fontsize=6.5, color=INK)
    if b_kway is not None:
        ax.axvline(b_kway, color=MP_BLUE, lw=0.7, ls=":")
        ax.annotate(f"batches: {b_kway}", xy=(b_kway + 6, 0.445),
                    fontsize=6.8, color=MP_BLUE,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=MP_BLUE, lw=0.6))
    if b_pair is None:
        ax.annotate("duels: not reached\nby budget 240",
                    xy=(budgets[-1] - 2, pair[-1] - 0.13), ha="right",
                    fontsize=6.8, color=GREY)
    ax.set_xlabel("cumulative evaluations (cost)")
    ax.set_ylabel("agreement with the reference\nordering (Kendall $\\tau$)")
    ax.set_title("(a) same budget, better queries:\nranked batches beat duels",
                 fontsize=8)
    ax.legend(loc="lower right")

    # (b) reading a ceiling
    ax = axes[1]
    bars = [("Shore D", ceil_d, MP_BLUE), ("Shore A", ceil_a, EXPT_GOLD)]
    ys = [0, 1]
    for y, (name, v, col) in zip(ys, bars):
        ax.barh(y, v, color=col, height=0.55)
        ax.annotate(f"{v:.3f}", xy=(v + (0.055 if name == "Shore A" else 0.015), y),
                    va="center", fontsize=7.5)
    ax.scatter([tau1_a], [1], marker="D", s=30, color=ACCENT_RED, zorder=3)
    ax.annotate("first lookup equals the ceiling",
                xy=(tau1_a, 1.12), xytext=(0.02, 1.85), fontsize=6.8,
                color=ACCENT_RED,
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=0.8))
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(2.5, -0.6)
    ax.set_xlabel("full-record ceiling vs the Rockwell R\n"
                  "ordering (ten common polymers)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) when no budget helps:\nthe hardness-scale ceiling",
                 fontsize=8)
    save(fig, OUT)


if __name__ == "__main__":
    main()
