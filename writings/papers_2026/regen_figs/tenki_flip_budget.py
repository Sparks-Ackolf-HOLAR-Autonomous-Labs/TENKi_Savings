#!/usr/bin/env python3
"""TENKi flip diagnostic on the band-gap archive: tau(B) with ceilings.

For each data source i and budget B (archived records drawn from source i,
spread evenly over the anion families that both source i and the measured
compilation hold with >= 30 records), the class ranking by median sampled
gap is compared with the full-record ranking of the measured gaps
(Kendall tau). The full-record ceiling is the tau of source i's full-record
ranking. The measured compilation itself is included: its curve is what the
group's own measurements reach after B measurements, and the flip budget
B* is the first budget at which its median curve exceeds a donor's ceiling.

Stage 1 (needs the rebuilt archive, see publish/reproduce/40_bandgap_archive.md)
writes tenki/results/flip_budget/flip_budget.json; stage 2 draws the figure
from that JSON alone.

  python tenki_flip_budget.py [--universe PATH] [--recompute]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    IOP_FULL, INK, GREY, MP_BLUE, SNUMAT_PURPLE, EXPT_GOLD, JARVIS_GREEN,
    SOURCE_NAMES, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
UNIVERSE = ROOT / "shared/output/bandgap_archive_build/bandgap_universe.jsonl"
RES = ROOT / "tenki/results/flip_budget/flip_budget.json"
OUT = ROOT / "writings/papers_2026/tenki_mlst/figures/flip_budget.png"

REF = "expt_zhuo"
SEEDS = 200
BUDGETS = list(range(8, 241, 8))
MIN_CLASS = 30
ANION_PRIORITY = [
    ("halide", {"F", "Cl", "Br", "I"}), ("oxide", {"O"}), ("nitride", {"N"}),
    ("chalcogenide", {"S", "Se", "Te"}), ("carbide", {"C"}),
    ("boride", {"B"}), ("pnictide", {"P", "As", "Sb", "Bi"}),
]


def classify(formula):
    sp = set(re.findall(r"[A-Z][a-z]?", formula))
    for name, el in ANION_PRIORITY:
        if sp & el:
            return name
    return "other"


def load(path):
    per = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            f = (r.get("x") or {}).get("formula")
            g = (r.get("y") or {}).get("bandgap_eV")
            s = r.get("s")
            if not f or g is None or s is None or float(g) <= 0.01:
                continue
            per.setdefault(s, {}).setdefault(classify(f), []).append(float(g))
    return {s: {c: np.asarray(v) for c, v in cl.items() if len(v) >= MIN_CLASS}
            for s, cl in per.items()}


def kendall(order_a, order_b):
    pos = {c: i for i, c in enumerate(order_b)}
    x = [pos[c] for c in order_a]
    n = len(x)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if x[j] > x[i] else -1
    return s / (n * (n - 1) / 2)


def rank(meds):
    return sorted(meds, key=lambda c: (-meds[c], c))


def compute(universe):
    data = load(universe)
    out = {"reference": REF, "seeds": SEEDS, "budgets": BUDGETS,
           "sources": {}}
    for s in sorted(data):
        common = sorted(set(data[s]) & set(data[REF]))
        ref_order = rank({c: float(np.median(data[REF][c])) for c in common})
        ceiling = kendall(rank({c: float(np.median(data[s][c]))
                                for c in common}), ref_order)
        rng = np.random.default_rng(42)
        curves = np.zeros((SEEDS, len(BUDGETS)))
        for seed in range(SEEDS):
            for bi, B in enumerate(BUDGETS):
                per = np.full(len(common), B // len(common))
                per[: B % len(common)] += 1
                rng.shuffle(per)
                meds = {}
                for c, m in zip(common, per):
                    m = max(int(m), 1)
                    meds[c] = float(np.median(rng.choice(data[s][c], m)))
                curves[seed, bi] = kendall(rank(meds), ref_order)
        out["sources"][s] = {
            "classes": common, "ceiling": ceiling,
            "median": np.median(curves, 0).tolist(),
            "mean": curves.mean(0).tolist(),
            "p05": np.percentile(curves, 5, 0).tolist(),
            "p95": np.percentile(curves, 95, 0).tolist(),
            "p25": np.percentile(curves, 25, 0).tolist(),
            "p75": np.percentile(curves, 75, 0).tolist(),
        }
        print(f"{s}: classes={len(common)} ceiling={ceiling:+.3f} "
              f"tau(8)={out['sources'][s]['median'][0]:+.3f} "
              f"tau(240)={out['sources'][s]['median'][-1]:+.3f}")
    RES.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES, "w", encoding="utf-8"), indent=1)
    return out


SHOW = [("expt_zhuo", EXPT_GOLD, "-"), ("snumat_hse", SNUMAT_PURPLE, "-"),
        ("mp_gga", MP_BLUE, "-")]


def first_cross(budgets, curve, level):
    for b, v in zip(budgets, curve):
        if v > level:
            return b
    return None


def figure(res):
    apply_style()
    B = res["budgets"]
    S = res["sources"]
    fig, axes = plt.subplots(1, 2, figsize=(IOP_FULL, 2.8),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    for s, col, ls in SHOW:
        d = S[s]
        lab = ("Measured gaps (own measurements)" if s == REF
               else SOURCE_NAMES[s])
        ax.plot(B, d["mean"], color=col, ls=ls, label=lab)
        ax.fill_between(B, d["p25"], d["p75"], color=col, alpha=0.15, lw=0)
        if s != REF:
            ax.axhline(d["ceiling"], color=col, ls=":", lw=1.0)
            ax.text(B[-1] + 3, d["ceiling"], f"{d['ceiling']:.2f}",
                    color=col, fontsize=7, va="center")
    exp = S[REF]["mean"]
    for s in ("mp_gga", "snumat_hse"):
        bstar = first_cross(B, exp, S[s]["ceiling"])
        if bstar is not None:
            ax.plot([bstar], [S[s]["ceiling"]], "o", ms=4, mfc="white",
                    mec=INK, zorder=5)
            ax.annotate(f"$B^*={bstar}$", xy=(bstar, S[s]["ceiling"]),
                        xytext=(6, -12), textcoords="offset points",
                        fontsize=7, color=INK)
    ax.set_xlim(0, B[-1])
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("records drawn from the source (budget $B$)")
    ax.set_ylabel("Kendall $\\tau$ with measured-gap order")
    ax.set_title("(a) Agreement against budget",
                 loc="left", fontsize=8)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.6)

    ax = axes[1]
    order = sorted([s for s in S if s != REF], key=lambda s: -S[s]["ceiling"])
    y = np.arange(len(order))
    ceil = [S[s]["ceiling"] for s in order]
    t8 = [S[s]["mean"][0] for s in order]
    from paper_fig_style import SOURCE_COLOURS
    cols = [SOURCE_COLOURS[s] for s in order]
    ax.barh(y, ceil, color=cols, height=0.6, label="full-record ceiling")
    ax.scatter(t8, y, marker="|", s=60, color=INK, zorder=4,
               label="mean at $B=8$")
    ax.set_yticks(y)
    ax.set_yticklabels([SOURCE_NAMES[s] for s in order], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlim(-0.2, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Kendall $\\tau$\n(bar: ceiling; tick: $B=8$)")
    ax.set_title("(b) Computed sources",
                 loc="left", fontsize=8)
    save(fig, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=UNIVERSE)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    if a.recompute or not RES.exists():
        res = compute(a.universe)
    else:
        res = json.load(open(RES, encoding="utf-8"))
    figure(res)


if __name__ == "__main__":
    main()
