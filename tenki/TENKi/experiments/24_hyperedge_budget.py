"""
Experiment 24 -- Hyperedge query budgets: k-way ranking events vs pairs.

The hypergraph companion experiment on the TENKi side (see
shared/docs/HYPERGRAPH_RATINGS.md section 4). PEGKi's hypergraph
extension shows rounds ARE k-way events; this experiment asks the
budgeted-acquisition question TENKi owns:

    For a fixed evaluation budget (each entity queried in one event costs
    1 evaluation: a pair costs 2, a k-way event costs k), which query
    design recovers the reference ranking fastest?

Arms
----
  pair_random   : random 2-subsets (the classic comparison-graph baseline)
  kway_random   : random k-subsets, k = --kway (one k-way event = k/2
                  pairs' worth of budget but C(k,2) implied comparisons
                  sharing one noise realisation)
  kway_active   : k-subsets chosen where the current fit is least resolved:
                  the window of k consecutive entities (in fitted-strength
                  order) with the smallest strength spread, epsilon-greedy
                  (eps = 0.2) random otherwise. Prototype acquisition rule;
                  experiment 27 (27_learned_acquisition.py) replaces it
                  with a greedy Fisher-information rule and a trained
                  policy, and the trained policy is the arm that
                  accelerates learning over random k-way selection.

Data (no invented observations)
-------------------------------
Entities are material classes within one channel of the frozen band-gap archive
band-gap universe (o=(x,s,m,y,u,p) records; build with
compute/bandgap_archive/build_inventory.py). A query event samples ONE real archived
record value per participating class (uniform over that class's records)
and ranks participants by value -- retrospective querying of the archive,
identical in spirit to the stride-indexed events of
shared/scripts/hypergraph_trial_bandgap.py but randomised per replicate.
The reference ranking per channel orders classes by their median gap over
ALL records (wide-gap-first guidance, higher_is_better=True throughout).

Metric
------
Kendall tau between the fitted Plackett-Luce ranking (MM fit; exact
Bradley-Terry at k=2, so the pairwise arm is the classic model, not a
strawman) and the reference ranking, as a function of cumulative
evaluation budget; mean +/- std over --n-seeds replicates, averaged over
channels.

Success criteria
----------------
  Primary  : at the final matched budget, kway_random tau >= pair_random
             tau (k-way events are at least as budget-efficient as pairs).
  Secondary: kway_active reaches 90% of its final tau at a smaller budget
             than kway_random (active selection helps).

Outputs (default results/hyperedge_budget/, override --output-dir)
------------------------------------------------------------------
  hyperedge_budget.json / .md / .png

Run (from tenki/TENKi, needs PEGKI_ROOT when shared/ is not the sibling):
  uv run python experiments/24_hyperedge_budget.py
Self-check of both score orientations: --self-test (runs, then exits).
ASCII only; seeded and reproducible (--seed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pegki_root() -> Path:
    env = os.environ.get("PEGKI_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(_HERE) / ".." / ".." / ".." / "shared").resolve()


# PEGKi's hypergraph module is the model under test; import it from the
# shared root so both projects fit the SAME Plackett-Luce implementation.
sys.path.insert(0, str(_pegki_root()))
from src.rating.hypergraph_rating import (  # noqa: E402
    PlackettLuceModel,
    hyperedge_from_scores,
    _kendall_tau,
)

# Mirrors classify_by_anion (aflow_mp_jarvis_phase2.py) and
# hypergraph_trial_bandgap.py exactly.
ANION_PRIORITY = [
    ("halide", {"F", "Cl", "Br", "I"}),
    ("oxide", {"O"}),
    ("nitride", {"N"}),
    ("chalcogenide", {"S", "Se", "Te"}),
    ("carbide", {"C"}),
    ("boride", {"B"}),
    ("pnictide", {"P", "As", "Sb", "Bi"}),
]
MIN_GAP_EV = 0.01


def classify_formula(formula: str) -> str:
    species = set(re.findall(r"[A-Z][a-z]?", formula))
    for name, elems in ANION_PRIORITY:
        if species & elems:
            return name
    return "other"


def load_universe(path: Path, min_class_size: int):
    """-> {channel: {class: np.array of gaps}} with classes >= min_class_size."""
    per_channel: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            formula = (rec.get("x") or {}).get("formula")
            gap = (rec.get("y") or {}).get("bandgap_eV")
            channel = rec.get("s")
            if not formula or gap is None or channel is None:
                continue
            gap = float(gap)
            if gap <= MIN_GAP_EV:
                continue
            cls = classify_formula(formula)
            per_channel.setdefault(channel, {}).setdefault(cls, []).append(gap)
    out = {}
    for channel, classes in per_channel.items():
        kept = {c: np.asarray(sorted(v), dtype=float)
                for c, v in classes.items() if len(v) >= min_class_size}
        if len(kept) >= 4:
            out[channel] = kept
    return out


def reference_ranking(classes: dict) -> list:
    """Classes ordered by median gap, best (widest) first; (value, name)
    tie-break -- the package's canonical rule."""
    med = {c: float(np.median(v)) for c, v in classes.items()}
    return sorted(med, key=lambda c: (-med[c], c))


def sample_event(classes: dict, subset: list, rng: np.random.Generator):
    """One real archived value per participating class -> Hyperedge or None."""
    scores = {c: float(classes[c][rng.integers(0, len(classes[c]))])
              for c in subset}
    return hyperedge_from_scores(scores, higher_is_better=True)


def pick_active_subset(model: PlackettLuceModel, names: list, k: int,
                       rng: np.random.Generator, eps: float = 0.2) -> list:
    """k consecutive entities (fitted-strength order) with minimal spread."""
    if rng.random() < eps or not model.log_strengths:
        return list(rng.choice(names, size=k, replace=False))
    theta = {n: float(model.log_strengths.get(n, 0.0)) for n in names}
    ordered = sorted(names, key=lambda n: (-theta[n], n))
    best_i, best_spread = 0, float("inf")
    for i in range(len(ordered) - k + 1):
        window = ordered[i:i + k]
        spread = theta[window[0]] - theta[window[-1]]
        if spread < best_spread:
            best_i, best_spread = i, spread
    return ordered[best_i:best_i + k]


def run_arm(classes: dict, arm: str, kway: int, budget: int,
            checkpoints: list, seed: int):
    """-> list of tau values, one per checkpoint budget."""
    rng = np.random.default_rng(seed)
    names = sorted(classes)
    ref = reference_ranking(classes)
    k = 2 if arm == "pair_random" else kway
    edges, taus, spent, ci = [], [], 0, 0
    model = PlackettLuceModel()
    while spent < budget and ci < len(checkpoints):
        if arm == "kway_active":
            subset = pick_active_subset(model, names, k, rng)
        else:
            subset = list(rng.choice(names, size=k, replace=False))
        e = sample_event(classes, subset, rng)
        spent += k  # a queried evaluation costs 1 whether or not it ties out
        if e is not None:
            edges.append(e)
        while ci < len(checkpoints) and spent >= checkpoints[ci]:
            if edges:
                model = PlackettLuceModel()
                model.fit(edges)
                taus.append(_kendall_tau(model.ranking(), ref))
            else:
                taus.append(0.0)
            ci += 1
    while len(taus) < len(checkpoints):
        taus.append(taus[-1] if taus else 0.0)
    return taus


def orientation_self_test() -> int:
    """Both score orientations on a toy input (sign-convention rule)."""
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    hi = hyperedge_from_scores(scores, higher_is_better=True)
    lo = hyperedge_from_scores(scores, higher_is_better=False)
    ok = hi.ranking == ("a", "b", "c") and lo.ranking == ("c", "b", "a")
    print(f"[{'PASS' if ok else 'FAIL'}] orientation: "
          f"higher_is_better {hi.ranking}, lower_is_better {lo.ranking}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="TENKi exp 24: hyperedge budgets")
    ap.add_argument("--universe",
                    default="output/bandgap_archive_build/bandgap_universe.jsonl",
                    help="band-gap archive universe jsonl, relative to PEGKI_ROOT")
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240,
                    help="max evaluations per replicate")
    ap.add_argument("--budget-step", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return orientation_self_test()

    universe = _pegki_root() / args.universe
    if not universe.exists():
        print(f"[error] universe not found: {universe}; "
              "run compute/bandgap_archive/build_inventory.py first")
        return 1
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(_HERE).parent / "results" / "hyperedge_budget")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_channel = load_universe(universe, args.min_class_size)
    print(f"[load] {len(per_channel)} channels with >= 4 classes of "
          f">= {args.min_class_size} records")
    if not per_channel:
        return 1

    checkpoints = list(range(args.budget_step, args.budget + 1,
                             args.budget_step))
    arms = ["pair_random", "kway_random", "kway_active"]
    # curves[arm][channel] = (n_seeds, n_checkpoints)
    curves = {a: {} for a in arms}
    for channel in sorted(per_channel):
        classes = per_channel[channel]
        for arm in arms:
            rows = [run_arm(classes, arm, args.kway, args.budget,
                            checkpoints, seed=args.seed + 1000 * s)
                    for s in range(args.n_seeds)]
            curves[arm][channel] = np.asarray(rows)
        print(f"[done] {channel}: {len(classes)} classes")

    # Aggregate: mean over seeds, then over channels
    agg = {}
    for arm in arms:
        stack = np.stack([curves[arm][ch].mean(axis=0)
                          for ch in sorted(per_channel)])
        agg[arm] = {"mean": stack.mean(axis=0).tolist(),
                    "std": stack.std(axis=0).tolist()}

    final = {a: agg[a]["mean"][-1] for a in arms}
    primary_pass = final["kway_random"] >= final["pair_random"] - 1e-9

    def budget_to_reach(arm, frac=0.9):
        target = frac * agg[arm]["mean"][-1]
        for i, v in enumerate(agg[arm]["mean"]):
            if v >= target:
                return checkpoints[i]
        return checkpoints[-1]

    b_active = budget_to_reach("kway_active")
    b_random = budget_to_reach("kway_random")
    secondary_pass = b_active < b_random

    results = {
        "config": {k: v for k, v in vars(args).items() if k != "self_test"},
        "checkpoints": checkpoints,
        "channels": sorted(per_channel),
        "aggregate": agg,
        "final_tau": final,
        "per_channel_final": {
            arm: {ch: float(curves[arm][ch].mean(axis=0)[-1])
                  for ch in sorted(per_channel)} for arm in arms},
        "criteria": {
            "primary_kway_ge_pair_at_final_budget": bool(primary_pass),
            "secondary_active_faster_to_90pct": bool(secondary_pass),
            "budget_to_90pct": {"kway_active": b_active,
                                "kway_random": b_random},
        },
    }
    with open(out_dir / "hyperedge_budget.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"pair_random": ("tab:gray", "--"),
              "kway_random": ("tab:blue", "-"),
              "kway_active": ("tab:red", "-")}
    for arm in arms:
        m = np.asarray(agg[arm]["mean"])
        s = np.asarray(agg[arm]["std"])
        c, ls = styles[arm]
        ax.plot(checkpoints, m, ls, color=c, label=arm)
        ax.fill_between(checkpoints, m - s, m + s, color=c, alpha=0.15)
    ax.set_xlabel("cumulative evaluations (1 per queried entity)")
    ax.set_ylabel("Kendall tau vs reference ranking")
    ax.set_title(f"Hyperedge budget curves (k={args.kway}, "
                 f"{len(per_channel)} band-gap archive channels, "
                 f"{args.n_seeds} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hyperedge_budget.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Experiment 24: hyperedge query budgets",
        "",
        f"Channels: {', '.join(sorted(per_channel))}",
        f"k = {args.kway}; budget = {args.budget} evaluations; "
        f"{args.n_seeds} seeds; seed base {args.seed}.",
        "",
        "| arm | final tau (mean over channels) |",
        "|---|---|",
    ]
    for arm in arms:
        lines.append(f"| {arm} | {final[arm]:+.4f} |")
    lines += [
        "",
        f"Primary (kway_random >= pair_random at final budget): "
        f"**{'PASS' if primary_pass else 'FAIL'}**",
        f"Secondary (active reaches 90% of its final tau earlier than "
        f"random: {b_active} vs {b_random} evaluations): "
        f"**{'PASS' if secondary_pass else 'FAIL'}**",
        "",
        "Each evaluation = one real archived record sampled from the",
        "queried class (retrospective querying; no invented observations).",
        "Plackett-Luce MM fit; k=2 arm is exact Bradley-Terry.",
    ]
    with open(out_dir / "hyperedge_budget.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[save] {out_dir}")
    print(f"final tau: " + ", ".join(f"{a}={final[a]:+.4f}" for a in arms))
    print(f"primary {'PASS' if primary_pass else 'FAIL'}; "
          f"secondary {'PASS' if secondary_pass else 'FAIL'} "
          f"(90% budget: active {b_active} vs random {b_random})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
