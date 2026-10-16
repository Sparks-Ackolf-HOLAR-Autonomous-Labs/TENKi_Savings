"""
Experiment 25 -- Physics-informed relative priors under a query budget.

The PINN question asked on rank data: PINNs add a
physics-residual term to the loss so the network needs less data and trains
faster. Our observations are RELATIVE (orderings inside k-way query events),
so the physics term must be relative too: a prior over strength DIFFERENCES,
taken from a different environment/system, injected into the Plackett-Luce
fit as an anchored MAP term with trust weight lambda
(src/rating/hypergraph_rating.py), and ADJUSTED/LEARNED from the event
stream (AdaptivePriorSelector, prequential scoring).

The physics prior here is real cross-system information, not synthetic: for
each receiver channel of the frozen band-gap archive band-gap universe, the prior is
the material-class ranking of the DONOR channel (the largest OTHER channel)
-- e.g. MP GGA's class ordering used as physics knowledge when querying the
experimental channel. Donor and receiver are different fidelity systems, so
the prior is genuinely imperfect ("prior quality" tau is reported per
channel), which is exactly what the learned trust weight is for. A
deliberately WRONG (reversed) prior tests robustness.

Arms (all use random k-way events, k = --kway, same costs as exp 24)
--------------------------------------------------------------------
  uninformed       : flat prior (exp 24's kway_random baseline)
  physics_fixed    : donor-channel prior, fixed lambda = --lam
  physics_adaptive : donor-channel prior, lambda learned prequentially
  wrong_fixed      : REVERSED donor prior, fixed lambda = --lam
  wrong_adaptive   : REVERSED donor prior, lambda learned prequentially

Metrics and success criteria
----------------------------
  tau-vs-budget curves (performance / information at matched cost);
  evaluations saved: budget the uninformed arm needs to match the
  physics arm's tau at the FIRST checkpoint (head start);
  mean MM iterations per fit (training time; anchored fits warm-start
  at the prior);
  lambda trajectory of the adaptive arms.

  Primary   : physics_fixed mean tau over the first half of checkpoints
              > uninformed's, AND physics_fixed final tau >= uninformed
              final tau - 0.02 (early gain, no long-run harm).
  Secondary : wrong_adaptive final tau >= wrong_fixed final tau AND
              wrong_adaptive final tau >= uninformed final tau - 0.05
              (the learned trust recovers from a misspecified prior).
  Tertiary  : mean MM iterations per fit, physics_fixed < uninformed
              (the prior is also a warm start).

Data (no invented observations)
-------------------------------
Identical retrospective querying to exp 24: one real archived record per
participating class per event; classes and reference rankings mirror
classify_by_anion / exp 24 exactly.

Outputs (default results/physics_informed_budget/, override --output-dir)
-------------------------------------------------------------------------
  physics_informed_budget.json / .md / .png

Run (from tenki/TENKi, needs PEGKI_ROOT when shared/ is not the sibling):
  uv run python experiments/25_physics_informed_budget.py
Self-check (orientation + prior regimes): --self-test (runs, then exits).
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


sys.path.insert(0, str(_pegki_root()))
from src.rating.hypergraph_rating import (  # noqa: E402
    AdaptivePriorSelector,
    Hyperedge,
    PlackettLuceModel,
    hyperedge_from_scores,
    _kendall_tau,
)

# Mirrors classify_by_anion (aflow_mp_jarvis_phase2.py) and experiment 24
# exactly.
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

LAMBDA_GRID = (0.01, 0.5, 2.0, 8.0, 32.0)


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
    med = {c: float(np.median(v)) for c, v in classes.items()}
    return sorted(med, key=lambda c: (-med[c], c))


def donor_prior(receiver_classes: dict, donor_classes: dict) -> dict:
    """Relative prior over the RECEIVER's classes from the DONOR's ordering.

    Donor classes are ranked by their donor-channel median; receiver classes
    present in the donor get evenly spaced log-strengths from +1 (donor-best)
    to -1 (donor-worst). Receiver classes the donor never saw anchor at 0
    (no relative information about them). Ordinal information only: the
    donor's actual eV values never cross systems, mirroring "relative
    information from a different environment".
    """
    donor_rank = reference_ranking(donor_classes)
    shared = [c for c in donor_rank if c in receiver_classes]
    prior = {c: 0.0 for c in receiver_classes}
    if len(shared) >= 2:
        for pos, c in enumerate(shared):
            prior[c] = float(1.0 - 2.0 * pos / (len(shared) - 1))
    return prior


def pick_donor(channel: str, per_channel: dict) -> str:
    """Largest other channel by record count; name tie-break."""
    sizes = {ch: sum(len(v) for v in cls.values())
             for ch, cls in per_channel.items() if ch != channel}
    return max(sorted(sizes), key=lambda ch: sizes[ch])


def sample_event(classes: dict, subset: list, rng: np.random.Generator):
    scores = {c: float(classes[c][rng.integers(0, len(classes[c]))])
              for c in subset}
    return hyperedge_from_scores(scores, higher_is_better=True)


def run_arm(classes: dict, arm: str, prior: dict, lam: float, kway: int,
            budget: int, checkpoints: list, seed: int):
    """-> (taus per checkpoint, lambdas per checkpoint, mean MM iterations)."""
    rng = np.random.default_rng(seed)
    names = sorted(classes)
    ref = reference_ranking(classes)
    anchored = arm != "uninformed"
    adaptive = arm.endswith("_adaptive")
    # Batched prequential scoring (refresh_every=4): each event is scored by
    # a model lagging at most 3 events -- still strictly predictive, ~4x
    # cheaper than the fully online scheme.
    selector = (AdaptivePriorSelector(prior, LAMBDA_GRID, refresh_every=4)
                if adaptive else None)
    prior_rank = (sorted(prior, key=lambda c: (-prior[c], c))
                  if anchored else None)

    edges, taus, lams, iters = [], [], [], []
    spent, ci = 0, 0
    while spent < budget and ci < len(checkpoints):
        subset = list(rng.choice(names, size=kway, replace=False))
        e = sample_event(classes, subset, rng)
        spent += kway
        if e is not None:
            if adaptive:
                selector.observe(e)
            edges.append(e)
        while ci < len(checkpoints) and spent >= checkpoints[ci]:
            if adaptive:
                model = selector.fit_current()
                lams.append(selector.best_lambda())
                if selector.edges:
                    iters.append(model.n_iterations)
                    taus.append(_kendall_tau(model.ranking(), ref))
                else:
                    taus.append(_kendall_tau(prior_rank, ref))
            elif edges:
                model = PlackettLuceModel(prior_weight=lam if anchored else 0.01)
                model.fit(edges,
                          prior_log_strengths=prior if anchored else None)
                iters.append(model.n_iterations)
                taus.append(_kendall_tau(model.ranking(), ref))
                lams.append(lam if anchored else 0.0)
            else:
                taus.append(_kendall_tau(prior_rank, ref) if anchored else 0.0)
                lams.append(lam if anchored else 0.0)
            ci += 1
    while len(taus) < len(checkpoints):
        taus.append(taus[-1] if taus else 0.0)
        lams.append(lams[-1] if lams else 0.0)
    mean_iters = float(np.mean(iters)) if iters else 0.0
    return taus, lams, mean_iters


ARMS = ["uninformed", "physics_fixed", "physics_adaptive",
        "wrong_fixed", "wrong_adaptive"]


def self_test() -> int:
    """Orientation both ways + prior dominance regimes (sign-convention rule)."""
    failures = 0
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    hi = hyperedge_from_scores(scores, higher_is_better=True)
    lo = hyperedge_from_scores(scores, higher_is_better=False)
    ok = hi.ranking == ("a", "b", "c") and lo.ranking == ("c", "b", "a")
    print(f"[{'PASS' if ok else 'FAIL'}] orientation: {hi.ranking} / {lo.ranking}")
    failures += 0 if ok else 1

    # Prior regimes: with 1 contradicting event, high lambda keeps the prior
    # ordering; with 100, low lambda follows the data.
    prior = {"a": 1.0, "b": 0.0, "c": -1.0}
    contra = [Hyperedge(ranking=("c", "b", "a"))]
    keep = PlackettLuceModel(prior_weight=8.0).fit(
        contra, prior_log_strengths=prior).ranking()
    flip = PlackettLuceModel(prior_weight=0.5).fit(
        contra * 100, prior_log_strengths=prior).ranking()
    ok = keep == ["a", "b", "c"] and flip == ["c", "b", "a"]
    print(f"[{'PASS' if ok else 'FAIL'}] prior regimes: sparse {keep}, plenty {flip}")
    failures += 0 if ok else 1
    return failures


def main():
    ap = argparse.ArgumentParser(
        description="TENKi exp 25: physics-informed budgets")
    ap.add_argument("--universe",
                    default="output/bandgap_archive_build/bandgap_universe.jsonl",
                    help="band-gap archive universe jsonl, relative to PEGKI_ROOT")
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--budget-step", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--lam", type=float, default=4.0,
                    help="fixed trust weight for the *_fixed arms")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    universe = _pegki_root() / args.universe
    if not universe.exists():
        print(f"[error] universe not found: {universe}; "
              "run compute/bandgap_archive/build_inventory.py first")
        return 1
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(_HERE).parent / "results" / "physics_informed_budget")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_channel = load_universe(universe, args.min_class_size)
    print(f"[load] {len(per_channel)} channels with >= 4 classes of "
          f">= {args.min_class_size} records")
    if not per_channel:
        return 1

    checkpoints = list(range(args.budget_step, args.budget + 1,
                             args.budget_step))
    curves = {a: {} for a in ARMS}       # arm -> channel -> (seeds, ckpts)
    lam_tracks = {a: {} for a in ARMS}
    iter_means = {a: {} for a in ARMS}
    donors, prior_quality = {}, {}
    for channel in sorted(per_channel):
        classes = per_channel[channel]
        donor = pick_donor(channel, per_channel)
        donors[channel] = donor
        prior = donor_prior(classes, per_channel[donor])
        wrong = {c: -v for c, v in prior.items()}
        prior_rank = sorted(prior, key=lambda c: (-prior[c], c))
        prior_quality[channel] = _kendall_tau(prior_rank,
                                              reference_ranking(classes))
        for arm in ARMS:
            p = wrong if arm.startswith("wrong") else prior
            rows, lrows, irows = [], [], []
            for s in range(args.n_seeds):
                taus, lams, mi = run_arm(
                    classes, arm, p, args.lam, args.kway, args.budget,
                    checkpoints, seed=args.seed + 1000 * s)
                rows.append(taus)
                lrows.append(lams)
                irows.append(mi)
            curves[arm][channel] = np.asarray(rows)
            lam_tracks[arm][channel] = np.asarray(lrows)
            iter_means[arm][channel] = float(np.mean(irows))
        print(f"[done] {channel}: {len(classes)} classes, donor={donor}, "
              f"prior quality tau={prior_quality[channel]:+.3f}")

    agg, lam_agg, iters_agg = {}, {}, {}
    for arm in ARMS:
        stack = np.stack([curves[arm][ch].mean(axis=0)
                          for ch in sorted(per_channel)])
        agg[arm] = {"mean": stack.mean(axis=0).tolist(),
                    "std": stack.std(axis=0).tolist()}
        lstack = np.stack([np.median(lam_tracks[arm][ch], axis=0)
                           for ch in sorted(per_channel)])
        lam_agg[arm] = np.median(lstack, axis=0).tolist()
        iters_agg[arm] = float(np.mean([iter_means[arm][ch]
                                        for ch in sorted(per_channel)]))

    half = len(checkpoints) // 2
    early = {a: float(np.mean(agg[a]["mean"][:half])) for a in ARMS}
    final = {a: agg[a]["mean"][-1] for a in ARMS}

    primary = (early["physics_fixed"] > early["uninformed"]
               and final["physics_fixed"] >= final["uninformed"] - 0.02)
    secondary = (final["wrong_adaptive"] >= final["wrong_fixed"] - 1e-9
                 and final["wrong_adaptive"] >= final["uninformed"] - 0.05)
    tertiary = iters_agg["physics_fixed"] < iters_agg["uninformed"]

    # Head start: budget the uninformed arm needs to reach the physics arm's
    # tau at the FIRST checkpoint
    target = agg["physics_fixed"]["mean"][0]
    saved = checkpoints[-1] - checkpoints[0]
    for i, v in enumerate(agg["uninformed"]["mean"]):
        if v >= target:
            saved = checkpoints[i] - checkpoints[0]
            break

    results = {
        "config": {k: v for k, v in vars(args).items() if k != "self_test"},
        "lambda_grid": list(LAMBDA_GRID),
        "checkpoints": checkpoints,
        "channels": sorted(per_channel),
        "donors": donors,
        "prior_quality_tau": prior_quality,
        "aggregate": agg,
        "lambda_trajectory_median": lam_agg,
        "mean_mm_iterations": iters_agg,
        "early_mean_tau": early,
        "final_tau": final,
        "evaluations_saved_vs_uninformed": saved,
        "criteria": {
            "primary_early_gain_no_final_harm": bool(primary),
            "secondary_adaptive_recovers_from_wrong_prior": bool(secondary),
            "tertiary_warm_start_fewer_iterations": bool(tertiary),
        },
    }
    with open(out_dir / "physics_informed_budget.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5),
                             gridspec_kw={"width_ratios": [3, 2]})
    styles = {"uninformed": ("tab:gray", "-"),
              "physics_fixed": ("tab:blue", "-"),
              "physics_adaptive": ("tab:green", "-"),
              "wrong_fixed": ("tab:red", "--"),
              "wrong_adaptive": ("tab:orange", "--")}
    for arm in ARMS:
        m = np.asarray(agg[arm]["mean"])
        s = np.asarray(agg[arm]["std"])
        c, ls = styles[arm]
        axes[0].plot(checkpoints, m, ls, color=c, label=arm)
        axes[0].fill_between(checkpoints, m - s, m + s, color=c, alpha=0.12)
    axes[0].set_xlabel("cumulative evaluations (1 per queried entity)")
    axes[0].set_ylabel("Kendall tau vs reference ranking")
    axes[0].set_title(f"Physics-informed budget curves (k={args.kway}, "
                      f"{len(per_channel)} channels, {args.n_seeds} seeds)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for arm in ("physics_adaptive", "wrong_adaptive"):
        c, _ = styles[arm]
        axes[1].plot(checkpoints, lam_agg[arm], color=c, label=arm)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cumulative evaluations")
    axes[1].set_ylabel("median learned lambda (prior trust)")
    axes[1].set_title("Learned trust in the physics prior")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "physics_informed_budget.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Experiment 25: physics-informed relative priors under budget",
        "",
        f"Channels: {', '.join(sorted(per_channel))}",
        f"k = {args.kway}; budget = {args.budget}; {args.n_seeds} seeds; "
        f"fixed lambda = {args.lam}; grid = {list(LAMBDA_GRID)}.",
        "",
        "| channel | donor (prior source) | prior quality tau |",
        "|---|---|---|",
    ]
    for ch in sorted(per_channel):
        lines.append(f"| {ch} | {donors[ch]} | {prior_quality[ch]:+.3f} |")
    lines += [
        "",
        "| arm | early mean tau | final tau | mean MM iters/fit |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {early[arm]:+.4f} | {final[arm]:+.4f} | "
                     f"{iters_agg[arm]:.1f} |")
    lines += [
        "",
        f"Evaluations saved by the physics prior (budget for uninformed to "
        f"match physics_fixed's first-checkpoint tau): **{saved}** of "
        f"{checkpoints[-1]}.",
        "",
        f"Primary (early gain, no final harm): "
        f"**{'PASS' if primary else 'FAIL'}**",
        f"Secondary (adaptive recovers from wrong prior): "
        f"**{'PASS' if secondary else 'FAIL'}**",
        f"Tertiary (warm start, fewer MM iterations): "
        f"**{'PASS' if tertiary else 'FAIL'}**",
        "",
        "Priors are ordinal-only cross-channel information (donor medians",
        "never cross systems as values). Retrospective querying of the",
        "frozen universe; no invented observations.",
    ]
    with open(out_dir / "physics_informed_budget.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[save] {out_dir}")
    print("final tau: " + ", ".join(f"{a}={final[a]:+.4f}" for a in ARMS))
    print(f"primary {'PASS' if primary else 'FAIL'}; "
          f"secondary {'PASS' if secondary else 'FAIL'}; "
          f"tertiary {'PASS' if tertiary else 'FAIL'}; "
          f"evaluations saved = {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
