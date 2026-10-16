#!/usr/bin/env python3
"""Acquisition rules over a long campaign, with uncertainty (TENKi Annex B.6).

Data: exp 27 stopped at 240 evaluations and
stored seed means; exp 27b reran the five acquisition rules for 1,920
evaluations with 100 seeds per held-out source (800 campaigns per rule).
One column-width panel: the advantage of the trained policy over each
other rule (Kendall tau of the trained policy minus that of the rule, on
the same held-out source and seed index) against evaluations spent.
Line: mean over the 800 campaign pairs; dark band: 90% bootstrap
interval of the mean; light band (4-way random only, for legibility): middle half of the
campaign pairs.
The zero line is where the trained policy has no advantage.

Size and fonts: 0.62 of the IOP text width at print size, fonts per
paper_fig_style, log budget axis.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import INK, IOP_FULL, apply_style, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/learned_acquisition_long"
OUT = ROOT / "writings/papers_2026/tenki_mlst/figures/acquisition_long.png"
RIVALS = [("kway_random", "4-way random", "#0072B2"),
          ("kway_fisher", "Fisher-greedy", "#009E73"),
          ("pair_random", "pairwise random", "#8C8C8C")]


def main():
    apply_style()
    rng = np.random.default_rng(0)
    curves, cps = {}, None
    for f in sorted(ART.glob("fold_*.json")):
        d = json.load(open(f))
        cps = np.array(d["checkpoints"])
        for arm, c in d["curves"].items():
            curves.setdefault(arm, []).extend(c)
    L = np.array(curves["kway_learned"])
    fig, ax = plt.subplots(figsize=(0.62 * IOP_FULL, 2.9))
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvline(240, color=INK, lw=0.7, ls="--")
    for arm, lab, col in RIVALS:
        D = L - np.array(curves[arm])
        m = D.mean(0)
        boots = np.array([D[rng.integers(0, len(D), len(D))].mean(0) for _ in range(400)])
        if arm == "kway_random":
            ax.fill_between(cps, np.percentile(D, 25, 0), np.percentile(D, 75, 0),
                            color=col, alpha=0.12, lw=0,
                            label="middle half of campaigns")
        ax.fill_between(cps, np.percentile(boots, 5, 0), np.percentile(boots, 95, 0),
                        color=col, alpha=0.45, lw=0)
        ax.plot(cps, m, color=col, lw=1.4, label=f"vs {lab}")
    ax.set_xscale("log")
    ax.set_xlim(16, 1920)
    ax.set_xlabel("Evaluations spent (log scale)")
    ax.set_ylabel(r"Trained policy advantage ($\tau$)")
    ax.text(240 * 1.08, -0.2, "training\nhorizon", fontsize=7, color=INK, va="bottom")
    fig.legend(loc="outside lower center", ncol=2, handlelength=1.4, columnspacing=1.0)
    save(fig, OUT)


if __name__ == "__main__":
    main()
