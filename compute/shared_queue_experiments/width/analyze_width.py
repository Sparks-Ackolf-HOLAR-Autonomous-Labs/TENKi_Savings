"""Summarise TENKi exp 31 (parallel width) on both budget axes.

Threshold per receiver: 0.9 x the mean final tau of the single-agent
random campaign at the full budget. For every campaign: evaluations and
rounds until tau first reaches the threshold (censored at the budget if it
never does). Writes width_summary.json, width_summary.md, fig_width.png.
ASCII only.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
raw = json.load(open(HERE / "parallel_width_raw.json", encoding="utf-8"))
E = raw["config"]["evals"]
widths = raw["config"]["widths"]
rules = ["random", "active", "active_claim"]
by_rcv = {}
for r in raw["results"]:
    by_rcv.setdefault(r["receiver"], []).append(r)

rows = {(rule, P): {"evals": [], "rounds": [], "reached": [], "final": [],
                    "half": [], "dup": []} for rule in rules for P in widths}
thresholds = {}
for rcv, runs in by_rcv.items():
    base = np.mean([run["curves"]["random|1"][-1][2] for run in runs])
    thr = 0.9 * base
    thresholds[rcv] = thr
    for run in runs:
        for rule in rules:
            for P in widths:
                c = run["curves"][f"{rule}|{P}"]
                hit = next((pt for pt in c if pt[2] >= thr), None)
                d = rows[(rule, P)]
                d["reached"].append(hit is not None)
                d["evals"].append(hit[1] if hit else E)
                d["rounds"].append(hit[0] if hit else c[-1][0])
                d["final"].append(c[-1][2])
                mid = min(c, key=lambda pt: abs(pt[1] - E / 2))
                d["half"].append(mid[2])
                d["dup"].append(sum(pt[3] for pt in c)
                                / max(1, len(c) * P))

summary = []
for rule in rules:
    for P in widths:
        d = rows[(rule, P)]
        ev, rd = np.asarray(d["evals"]), np.asarray(d["rounds"])
        summary.append({
            "rule": rule, "P": P, "n": len(ev),
            "evals_median": float(np.median(ev)),
            "evals_q25": float(np.percentile(ev, 25)),
            "evals_q75": float(np.percentile(ev, 75)),
            "rounds_median": float(np.median(rd)),
            "rounds_q25": float(np.percentile(rd, 25)),
            "rounds_q75": float(np.percentile(rd, 75)),
            "share_reached": float(np.mean(d["reached"])),
            "tau_half_budget": float(np.mean(d["half"])),
            "tau_final": float(np.mean(d["final"])),
            "duplicate_share": float(np.mean(d["dup"])),
            "duplicate_median": float(np.median(d["dup"])),
            "duplicate_q25": float(np.percentile(d["dup"], 25)),
            "duplicate_q75": float(np.percentile(d["dup"], 75)),
        })
json.dump({"thresholds": thresholds, "budget": E, "summary": summary},
          open(HERE / "width_summary.json", "w", encoding="utf-8"), indent=1)

lines = ["# Parallel-width sweep: summary", "",
         f"Budget {E} evaluations, k = 4, receivers {len(by_rcv)}, "
         f"seeds per receiver {len(next(iter(by_rcv.values())))}. "
         "Threshold = 0.9 x mean final tau of the one-agent random campaign "
         "(per receiver). Medians with interquartile range; campaigns that "
         "never reach the threshold count at the budget.", "",
         "| rule | P | evaluations to threshold | rounds to threshold | "
         "reached | tau at half budget | final tau | duplicate queries |",
         "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
for s in summary:
    lines.append(
        f"| {s['rule']} | {s['P']} | {s['evals_median']:.0f} "
        f"({s['evals_q25']:.0f}-{s['evals_q75']:.0f}) | "
        f"{s['rounds_median']:.0f} ({s['rounds_q25']:.0f}-"
        f"{s['rounds_q75']:.0f}) | {s['share_reached']:.2f} | "
        f"{s['tau_half_budget']:.3f} | {s['tau_final']:.3f} | "
        f"{s['duplicate_share']:.2f} |")
open(HERE / "width_summary.md", "w", encoding="utf-8").write(
    "\n".join(lines) + "\n")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "writings" / "papers_2026"))
from paper_fig_style import IOP_FULL, MP_BLUE, JARVIS_GREEN, ACCENT_RED, GREY, INK, apply_style, save  # noqa: E402
PAPER_FIG = Path(__file__).resolve().parents[3] / "writings" / "papers_2026" / "tenki_mlst" / "figures"
apply_style()
from matplotlib.patches import Patch  # noqa: E402
fig, axes = plt.subplots(1, 3, figsize=(IOP_FULL, 2.55))
colors = {"random": GREY, "active": MP_BLUE, "active_claim": ACCENT_RED}
labels = {"random": "random trays", "active": "least-resolved tray",
          "active_claim": "least-resolved tray, with claims"}
offs = {"random": 0.94, "active": 1.0, "active_claim": 1.06}
for rule in rules:
    S = [s for s in summary if s["rule"] == rule]
    x = [s["P"] * offs[rule] for s in S]
    for ax, key in zip(axes, ["evals", "rounds", "duplicate"]):
        if key == "duplicate":
            # line: mean share over campaigns (the value quoted in the
            # text); band: interquartile range over campaigns
            med = [100 * s["duplicate_share"] for s in S]
            lo = [100 * s["duplicate_q25"] for s in S]
            hi = [100 * s["duplicate_q75"] for s in S]
        else:
            med = [s[f"{key}_median"] for s in S]
            lo = [s[f"{key}_q25"] for s in S]
            hi = [s[f"{key}_q75"] for s in S]
        ax.plot(x, med, "-o", ms=3, lw=1.2, color=colors[rule],
                label=labels[rule])
        ax.fill_between(x, lo, hi, color=colors[rule], alpha=0.15, lw=0)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.minorticks_off()
    ax.set_xlabel("agents per round, P")
axes[1].set_yscale("log")
axes[1].set_yticks([1, 2, 5, 10, 20, 50])
axes[1].set_yticklabels(["1", "2", "5", "10", "20", "50"])
axes[2].set_ylim(-3, 103)
axes[0].set_ylabel("evaluations to threshold")
axes[1].set_ylabel("rounds to threshold")
axes[2].set_ylabel("duplicate trays (%)")
axes[0].set_title("(a) evaluations", loc="left")
axes[1].set_title("(b) rounds", loc="left")
axes[2].set_title("(c) repeated trays", loc="left")
handles = [plt.Line2D([], [], color=colors[r], marker="o", ms=3, lw=1.2)
           for r in rules] + [Patch(color=GREY, alpha=0.3, lw=0)]
fig.legend(handles, [labels[r] for r in rules]
           + ["band: interquartile range"], loc="outside upper center",
           ncol=4, frameon=False, fontsize=7, handlelength=1.6,
           columnspacing=1.2)
fig.savefig(HERE / "fig_width.png", dpi=200)
save(fig, PAPER_FIG / "parallel_width.png")
print("\n".join(lines))
