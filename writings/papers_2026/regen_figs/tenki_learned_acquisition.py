#!/usr/bin/env python3
"""Regenerate the TENKi learned-acquisition figure at true print size.

Panels: two, from
tenki/results/learned_acquisition/learned_acquisition.json.
Left: tau vs budget for the five arms (checkpoints +
aggregate.{arm}.mean); +-1 sd band shown only for the two arms the
caption compares (kway_random, kway_learned) to keep the panel legible.
Right: learned_weights_per_fold (8 held-out folds x 4 features,
feature_names) as per-fold dots plus mean +- sd (learned_weights_mean,
learned_weights_std); the closeness weight is negative on average.
Every number is read from the artifact.

Size and fonts: \textwidth figure (6.73 in), fonts per paper_fig_style.
Curves: k-way random solid MP_BLUE, pairwise random dashed GREY,
trained policy JARVIS_GREEN (matches its dot colour on the right),
epsilon-greedy ACCENT_RED dotted, Fisher-greedy EXPT_GOLD dash-dot.
Labels use "data source" (leave-one-data-source-out).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL,
    MP_BLUE, GREY, ACCENT_RED, EXPT_GOLD, JARVIS_GREEN, INK,
    apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "tenki/results/learned_acquisition/learned_acquisition.json"
OUT = ROOT / ("writings/papers_2026/tenki_mlst/figures/"
              "learned_acquisition.png")

ARMS = [
    ("pair_random", "pairwise random", GREY, "--", False),
    ("kway_random", "$k$-way random", MP_BLUE, "-", True),
    ("kway_active", "$\\epsilon$-greedy", ACCENT_RED, ":", False),
    ("kway_fisher", "Fisher-greedy", EXPT_GOLD, "-.", False),
    ("kway_learned", "trained policy", JARVIS_GREEN, "-", True),
]
FEATURE_LABELS = {
    "coverage": "coverage", "uncertainty": "uncertainty",
    "closeness": "closeness", "noise": "noise",
}


def main():
    apply_style()
    d = json.load(open(ART))
    x = d["checkpoints"]

    fig, (ax, axw) = plt.subplots(
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
    ax.legend(loc="lower right", ncols=1, handlelength=1.8)
    ax.set_title("Acquisition arms (leave-one-data-source-out)",
                 fontsize=9, pad=3)

    feats = d["feature_names"]
    mean = d["learned_weights_mean"]
    std = d["learned_weights_std"]
    per_fold = d["learned_weights_per_fold"]
    xs = range(len(feats))
    for fi in xs:
        vals = [per_fold[fold][fi] for fold in per_fold]
        axw.plot([fi - 0.16] * len(vals), vals, "o", color=GREY,
                 markersize=3, alpha=0.75, markeredgewidth=0)
    axw.errorbar([fi + 0.14 for fi in xs], mean, yerr=std, fmt="s",
                 color=JARVIS_GREEN, ecolor=JARVIS_GREEN, capsize=2.5,
                 markersize=4, linewidth=1.2)
    axw.axhline(0, color=INK, linewidth=0.8)
    axw.set_xticks(list(xs))
    axw.set_xticklabels([FEATURE_LABELS[f] for f in feats], rotation=25, ha="right")
    axw.set_ylabel("learned feature weight")
    axw.set_title("Trained-policy weights per fold", fontsize=9, pad=3)
    axw.grid(axis="x", visible=False)
    # tiny in-panel key
    axw.plot([], [], "o", color=GREY, markersize=3, label="held-out fold")
    axw.plot([], [], "s", color=JARVIS_GREEN, markersize=4,
             label="mean $\\pm$ sd")
    axw.legend(loc="lower right", handlelength=1.0, borderpad=0.2)

    save(fig, OUT)


if __name__ == "__main__":
    main()
