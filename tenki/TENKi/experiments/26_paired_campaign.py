"""
Experiment 26 -- Paired cold/full/selective campaign protocol (band-gap archive).

The last open evidence item of compute/bandgap_archive/DESIGN.md: the paired
protocol of update.md Section 7, run retrospectively on the frozen
band-gap universe, with the hypergraph layer as a first-class arm.
Also instantiates the "reference policies" item: each protocol condition
IS one of the reference decision policies (see mapping below).

Protocol
--------
For every ordered channel pair A -> B (the database-of-origin shift, the
severe axis of DESIGN.md), a campaign learns the material-class ranking
of receiver B under an evaluation budget, with MATCHED SEEDS across
conditions so every comparison is paired:

  cold       : no donor information (reference policy: random/uninformed)
  full       : donor A's class ordering as a relative prior with FIXED
               high trust lambda = --lam (reference policy: per-channel
               full reuse -- "trust the donor's model as-is")
  selective  : same prior, trust lambda LEARNED prequentially
               (AdaptivePriorSelector; reference policy: transfer-aware
               selective reuse)
  combined   : rank-averaged prior over ALL other channels, learned
               trust (reference policy: combined/pooled reuse);
               donor id "__combined__" in outputs

Each condition runs both query arms at equal cumulative cost
(exp 24 accounting: a pair costs 2 evaluations, a k-way event costs k):

  pair : random 2-subsets  (classic pairwise campaign; exact Bradley-Terry)
  kway : random k-subsets  (hypergraph campaign, k = --kway)

Transfer scores and matrices
----------------------------
T^{A->B} per (condition, arm) = mean over the FIRST HALF of budget
checkpoints of (tau_condition - tau_cold) at matched seed, averaged over
seeds -- the early-budget advantage donor reuse buys, in Kendall tau
units. Also reported at the final budget (no-long-run-harm check).
Percentile bootstrap CIs over seeds. Directed transfer matrices per
(condition, arm) over all ordered channel pairs.

Diagnostics (hypergraph layer item 3)
-------------------------------------
Per receiver, on the SAME cold k-way event stream: mean hyperedge
residual (hyperedge_residuals) vs pairwise cyclic-energy fraction of the
clique expansion (clique_expansion_hodge) -- the shredding-artifact vs
genuine-intransitivity split.

Data (no invented observations)
-------------------------------
Identical retrospective querying to exps 24/25: one real archived record
per participating class per event, from the frozen universe built by
compute/bandgap_archive/build_inventory.py. By default donor priors carry ORDINAL
donor information only (class ranks, never eV values across systems).
--prior-shape margin keeps the donor's relative class spacing as well
(standardised median margins; see donor_prior), still never an absolute
eV value.

Preemption safety (CHPC guest partitions)
-----------------------------------------
One JSON checkpoint per (receiver, seed) under
<output-dir>/checkpoints/; existing checkpoints are skipped on restart,
so a preempted and resubmitted guest job resumes where it died.
--workers N parallelises over (receiver, seed) jobs.

Outputs (default results/paired_campaign/, override --output-dir)
-----------------------------------------------------------------
  paired_campaign.json / .md / .png  (+ checkpoints/*.json)

Run (from tenki/TENKi, needs PEGKI_ROOT when shared/ is not the sibling):
  uv run python experiments/26_paired_campaign.py
Self-check (orientation + paired-transfer sanity): --self-test.
ASCII only; seeded and reproducible (--seed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
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
    clique_expansion_hodge,
    hyperedge_from_scores,
    hyperedge_residuals,
    _kendall_tau,
)

# Mirrors classify_by_anion (aflow_mp_jarvis_phase2.py) and exps 24/25
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
CONDITIONS = ["cold", "full", "selective", "combined"]
ARMS = ["pair", "kway"]
COMBINED = "__combined__"


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


PRIOR_SHAPES = ("even", "margin")


def donor_prior(receiver_classes: dict, donor_classes: dict,
                shape: str = "even") -> dict:
    """Relative prior over the receiver's classes from the donor's classes.

    shape="even" (default; the committed exp 25/26 construction): shared
    classes are ordered by donor-channel median and receive evenly spaced
    log-strengths from +1 (donor-best) to -1 (donor-worst). Only the donor's
    class ORDER is used; its margins are discarded.

    shape="margin": the donor's relative spacing is kept. For the shared
    classes S (n = |S| >= 2), take the donor-channel class medians m_c,
    standardise them over S (z_c = (m_c - mean_S m) / std_S m, population
    std), then rescale to the spread of the even prior on the same n
    classes: prior_c = z_c * std(e), with e_pos = 1 - 2 pos / (n - 1).
    Both shapes therefore have mean 0 and the same variance over S; only
    the SHAPE differs (how far apart the donor places the classes, relative
    to its own spread). Standardisation removes the donor's eV scale and
    offset, so no absolute eV value crosses systems; the prior stays a
    relative (interval-scale) statement. When the donor medians are evenly
    spaced the two shapes coincide (self-test). If all shared medians tie
    (std = 0) the margin prior carries no relative information and is 0.

    In both shapes receiver classes the donor never saw anchor at 0.
    """
    if shape not in PRIOR_SHAPES:
        raise ValueError(f"unknown prior shape: {shape}")
    donor_rank = reference_ranking(donor_classes)
    shared = [c for c in donor_rank if c in receiver_classes]
    prior = {c: 0.0 for c in receiver_classes}
    if len(shared) >= 2:
        even = [float(1.0 - 2.0 * pos / (len(shared) - 1))
                for pos in range(len(shared))]
        if shape == "even":
            for c, v in zip(shared, even):
                prior[c] = v
        else:
            med = np.asarray([float(np.median(donor_classes[c]))
                              for c in shared], dtype=float)
            sd = float(med.std())
            if sd > 0:
                z = (med - med.mean()) / sd
                scale = float(np.asarray(even).std())
                for c, v in zip(shared, z):
                    prior[c] = float(v * scale)
    return prior


def combined_prior(receiver: str, per_channel: dict,
                   shape: str = "even") -> dict:
    """Mean prior over all donors except the receiver.

    Each donor contributes its donor_prior of the given shape (evenly
    spaced ordinal for "even", standardised median margins for "margin");
    the combined prior is the mean over donors that saw the class (donors
    that never saw a class contribute nothing for it).
    """
    receiver_classes = per_channel[receiver]
    sums = {c: 0.0 for c in receiver_classes}
    counts = {c: 0 for c in receiver_classes}
    for donor, donor_classes in per_channel.items():
        if donor == receiver:
            continue
        p = donor_prior(receiver_classes, donor_classes, shape)
        donor_saw = set(reference_ranking(donor_classes))
        for c, v in p.items():
            if c in donor_saw:
                sums[c] += v
                counts[c] += 1
    return {c: (sums[c] / counts[c] if counts[c] else 0.0)
            for c in receiver_classes}


def sample_event(classes: dict, subset: list, rng: np.random.Generator):
    scores = {c: float(classes[c][rng.integers(0, len(classes[c]))])
              for c in subset}
    return hyperedge_from_scores(scores, higher_is_better=True)


def run_campaign(classes: dict, mode: str, prior: dict | None, lam: float,
                 k: int, budget: int, checkpoints: list, seed: int,
                 collect_edges: bool = False):
    """One campaign replicate -> (taus per checkpoint, edges or None).

    mode: "cold" (flat prior), "fixed" (anchored, lambda=lam),
    "adaptive" (anchored, prequential lambda).
    """
    rng = np.random.default_rng(seed)
    names = sorted(classes)
    ref = reference_ranking(classes)
    k = min(k, len(names))
    adaptive = mode == "adaptive"
    anchored = mode in ("fixed", "adaptive")
    selector = (AdaptivePriorSelector(prior, LAMBDA_GRID, refresh_every=4)
                if adaptive else None)
    prior_rank = (sorted(prior, key=lambda c: (-prior[c], c))
                  if anchored else None)

    edges, taus = [], []
    spent, ci = 0, 0
    while spent < budget and ci < len(checkpoints):
        subset = list(rng.choice(names, size=k, replace=False))
        e = sample_event(classes, subset, rng)
        spent += k
        if e is not None:
            if adaptive:
                selector.observe(e)
            edges.append(e)
        while ci < len(checkpoints) and spent >= checkpoints[ci]:
            if adaptive:
                if selector.edges:
                    taus.append(_kendall_tau(selector.fit_current().ranking(),
                                             ref))
                else:
                    taus.append(_kendall_tau(prior_rank, ref))
            elif edges:
                model = PlackettLuceModel(
                    prior_weight=lam if anchored else 0.01)
                model.fit(edges,
                          prior_log_strengths=prior if anchored else None)
                taus.append(_kendall_tau(model.ranking(), ref))
            else:
                taus.append(_kendall_tau(prior_rank, ref) if anchored else 0.0)
            ci += 1
    while len(taus) < len(checkpoints):
        taus.append(taus[-1] if taus else 0.0)
    return taus, (edges if collect_edges else None)


def _job(args):
    """All conditions x arms for one (receiver, seed) -- one checkpoint unit.

    Matched seeds: every condition/arm reuses the same seed, so transfer
    deltas are paired per replicate (DESIGN.md protocol requirement).
    """
    (receiver, seed, per_channel, kway, budget, checkpoints, lam) = args[:7]
    shape = args[7] if len(args) > 7 else "even"
    classes = per_channel[receiver]
    out: dict = {"receiver": receiver, "seed": seed, "curves": {}}
    ks = {"pair": 2, "kway": kway}

    diag_edges = None
    for arm, k in ks.items():
        taus, edges = run_campaign(classes, "cold", None, lam, k, budget,
                                   checkpoints, seed,
                                   collect_edges=(arm == "kway"))
        out["curves"][f"cold|{arm}"] = taus
        if arm == "kway":
            diag_edges = edges

    for donor in sorted(per_channel):
        if donor == receiver:
            continue
        prior = donor_prior(classes, per_channel[donor], shape)
        for arm, k in ks.items():
            taus, _ = run_campaign(classes, "fixed", prior, lam, k, budget,
                                   checkpoints, seed)
            out["curves"][f"full|{arm}|{donor}"] = taus
            taus, _ = run_campaign(classes, "adaptive", prior, lam, k,
                                   budget, checkpoints, seed)
            out["curves"][f"selective|{arm}|{donor}"] = taus

    prior = combined_prior(receiver, per_channel, shape)
    for arm, k in ks.items():
        taus, _ = run_campaign(classes, "adaptive", prior, lam, k, budget,
                               checkpoints, seed)
        out["curves"][f"combined|{arm}|{COMBINED}"] = taus

    # Hyperedge-vs-clique diagnostic on the cold k-way event stream
    if diag_edges:
        model = PlackettLuceModel()
        model.fit(diag_edges)
        res = hyperedge_residuals(model, diag_edges)
        hodge = clique_expansion_hodge(diag_edges)
        out["diagnostic"] = {
            "mean_hyperedge_residual": res["mean_edge_residual"],
            "clique_cyclic_energy_fraction": hodge["cyclic_energy_fraction"],
        }
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 n_boot: int = 2000):
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    means = [float(values[rng.integers(0, len(values), len(values))].mean())
             for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def self_test() -> int:
    """Orientation + paired-transfer sanity on synthetic classes."""
    failures = 0
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    hi = hyperedge_from_scores(scores, higher_is_better=True)
    lo = hyperedge_from_scores(scores, higher_is_better=False)
    ok = hi.ranking == ("a", "b", "c") and lo.ranking == ("c", "b", "a")
    print(f"[{'PASS' if ok else 'FAIL'}] orientation: "
          f"{hi.ranking} / {lo.ranking}")
    failures += 0 if ok else 1

    # Synthetic receiver: 6 well-separated classes (self-test only; the
    # experiment itself queries archived records exclusively)
    rng = np.random.default_rng(7)
    classes = {f"c{i}": np.sort(rng.normal(6 - i, 0.8, 60))
               for i in range(6)}
    good = donor_prior(classes, classes)              # aligned donor
    bad = {c: -v for c, v in good.items()}            # reversed donor
    checkpoints = [12, 24, 36, 48]
    d_good, d_bad, d_bad_sel = [], [], []
    for seed in range(6):
        cold, _ = run_campaign(classes, "cold", None, 8.0, 4, 48,
                               checkpoints, seed)
        full, _ = run_campaign(classes, "fixed", good, 8.0, 4, 48,
                               checkpoints, seed)
        wrong, _ = run_campaign(classes, "fixed", bad, 8.0, 4, 48,
                                checkpoints, seed)
        wsel, _ = run_campaign(classes, "adaptive", bad, 8.0, 4, 48,
                               checkpoints, seed)
        half = len(checkpoints) // 2
        d_good.append(np.mean(full[:half]) - np.mean(cold[:half]))
        d_bad.append(np.mean(wrong) - np.mean(cold))
        d_bad_sel.append(np.mean(wsel) - np.mean(wrong))
    ok = float(np.mean(d_good)) > 0
    print(f"[{'PASS' if ok else 'FAIL'}] aligned donor early gain "
          f"{np.mean(d_good):+.3f} > 0")
    failures += 0 if ok else 1
    ok = float(np.mean(d_bad)) < 0
    print(f"[{'PASS' if ok else 'FAIL'}] reversed donor at fixed high trust "
          f"harms: {np.mean(d_bad):+.3f} < 0")
    failures += 0 if ok else 1
    ok = float(np.mean(d_bad_sel)) > 0
    print(f"[{'PASS' if ok else 'FAIL'}] selective (learned trust) beats "
          f"full reuse on a reversed donor: {np.mean(d_bad_sel):+.3f} > 0")
    failures += 0 if ok else 1

    # Margin shape reduces to the even shape on evenly spaced donor medians
    even_donor = {f"c{i}": np.asarray([10.0 - 1.5 * i - 0.3,
                                       10.0 - 1.5 * i,
                                       10.0 - 1.5 * i + 0.3])
                  for i in range(6)}
    recv = {f"c{i}": np.zeros(1) for i in range(7)}   # c6 unseen by donor
    pe = donor_prior(recv, even_donor, "even")
    pm = donor_prior(recv, even_donor, "margin")
    ok = all(abs(pe[c] - pm[c]) < 1e-12 for c in recv) and pm["c6"] == 0.0
    print(f"[{'PASS' if ok else 'FAIL'}] margin prior == even prior on "
          f"evenly spaced donor medians (max diff "
          f"{max(abs(pe[c] - pm[c]) for c in recv):.1e})")
    failures += 0 if ok else 1
    # ... and keeps the donor's margins (same variance) when they are uneven
    uneven = {c: v + (5.0 if c == "c0" else 0.0)
              for c, v in even_donor.items()}
    pu = donor_prior(recv, uneven, "margin")
    shared = [f"c{i}" for i in range(6)]
    ok = (sorted(shared, key=lambda c: -pu[c]) == shared
          and abs(np.std([pu[c] for c in shared])
                  - np.std([pe[c] for c in shared])) < 1e-12
          and (pu["c0"] - pu["c1"]) > (pu["c1"] - pu["c2"]))
    print(f"[{'PASS' if ok else 'FAIL'}] margin prior keeps donor order and "
          f"spread, widens the c0-c1 margin: "
          f"{pu['c0'] - pu['c1']:.3f} > {pu['c1'] - pu['c2']:.3f}")
    failures += 0 if ok else 1
    print(f"self-test: {'PASS' if failures == 0 else 'FAIL'}")
    return failures


def main():
    ap = argparse.ArgumentParser(
        description="TENKi exp 26: paired cold/full/selective campaigns")
    ap.add_argument("--universe",
                    default="output/bandgap_archive_build/bandgap_universe.jsonl",
                    help="band-gap archive universe jsonl, relative to PEGKI_ROOT")
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--budget-step", type=int, default=24)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--lam", type=float, default=8.0,
                    help="fixed trust for the full-reuse condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prior-shape", choices=PRIOR_SHAPES, default="even",
                    help="donor prior shape: 'even' = evenly spaced ordinal "
                         "(default, committed runs); 'margin' = donor "
                         "median margins, standardised to the even spread")
    ap.add_argument("--receivers", nargs="*", default=None,
                    help="restrict receiver channels (default: all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a checkpoint exists")
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
        Path(_HERE).parent / "results" / "paired_campaign")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_channel = load_universe(universe, args.min_class_size)
    receivers = sorted(args.receivers or per_channel)
    missing = [r for r in receivers if r not in per_channel]
    if missing:
        print(f"[error] unknown receivers: {missing}")
        return 1
    print(f"[load] {len(per_channel)} channels; receivers: "
          f"{', '.join(receivers)}")

    checkpoints = list(range(args.budget_step, args.budget + 1,
                             args.budget_step))
    jobs = []
    for receiver in receivers:
        for s in range(args.n_seeds):
            seed = args.seed + 1000 * s
            ck = ckpt_dir / f"{receiver}_s{seed}.json"
            if ck.exists() and not args.force:
                continue
            job = (receiver, seed, per_channel, args.kway,
                   args.budget, checkpoints, args.lam)
            if args.prior_shape != "even":
                job = job + (args.prior_shape,)
            jobs.append(job)
    print(f"[plan] {len(jobs)} (receiver, seed) jobs to run; "
          f"{len(receivers) * args.n_seeds - len(jobs)} already "
          f"checkpointed")

    def save(res):
        ck = ckpt_dir / f"{res['receiver']}_s{res['seed']}.json"
        tmp = ck.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f)
        os.replace(tmp, ck)

    if args.workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_job, jobs):
                save(res)
                print(f"[done] {res['receiver']} seed {res['seed']}")
    else:
        for job in jobs:
            res = _job(job)
            save(res)
            print(f"[done] {res['receiver']} seed {res['seed']}")

    # ---- aggregate from checkpoints ------------------------------------
    rows = []
    for receiver in receivers:
        for s in range(args.n_seeds):
            ck = ckpt_dir / f"{receiver}_s{args.seed + 1000 * s}.json"
            if ck.exists():
                rows.append(json.loads(ck.read_text(encoding="utf-8")))
    if not rows:
        print("[error] no checkpoints to aggregate")
        return 1

    half = max(1, len(checkpoints) // 2)
    boot_rng = np.random.default_rng(args.seed)
    # matrices[cond][arm][donor][receiver] = {mean, ci, final_mean}
    matrices: dict = {c: {a: {} for a in ARMS}
                      for c in ("full", "selective", "combined")}
    deltas: dict = {}
    for row in rows:
        rec = row["receiver"]
        for key, taus in row["curves"].items():
            parts = key.split("|")
            if parts[0] == "cold":
                continue
            cond, arm, donor = parts
            cold = row["curves"][f"cold|{arm}"]
            d_early = float(np.mean(taus[:half]) - np.mean(cold[:half]))
            d_final = float(taus[-1] - cold[-1])
            deltas.setdefault((cond, arm, donor, rec),
                              []).append((d_early, d_final))
    for (cond, arm, donor, rec), vals in deltas.items():
        early = np.asarray([v[0] for v in vals])
        final = np.asarray([v[1] for v in vals])
        matrices[cond][arm].setdefault(donor, {})[rec] = {
            "T_early": float(early.mean()),
            "T_early_ci": bootstrap_ci(early, boot_rng),
            "T_final": float(final.mean()),
            "n_seeds": len(vals),
        }

    diag: dict = {}
    for row in rows:
        if "diagnostic" in row:
            diag.setdefault(row["receiver"], []).append(row["diagnostic"])
    diagnostics = {
        rec: {
            "mean_hyperedge_residual": float(np.mean(
                [d["mean_hyperedge_residual"] for d in ds])),
            "clique_cyclic_energy_fraction": float(np.mean(
                [d["clique_cyclic_energy_fraction"] for d in ds])),
        } for rec, ds in diag.items()}

    # Condition-level summary (mean T over all donor->receiver cells)
    summary = {}
    for cond in matrices:
        for arm in ARMS:
            cells = [c for d in matrices[cond][arm].values()
                     for c in d.values()]
            if cells:
                summary[f"{cond}|{arm}"] = {
                    "mean_T_early": float(np.mean(
                        [c["T_early"] for c in cells])),
                    "mean_T_final": float(np.mean(
                        [c["T_final"] for c in cells])),
                    "n_cells": len(cells),
                    "frac_cells_positive_early": float(np.mean(
                        [c["T_early"] > 0 for c in cells])),
                }

    results = {
        # prior_shape is recorded only when non-default so that default
        # (even) runs stay byte-identical to the committed outputs
        "config": {k: v for k, v in vars(args).items()
                   if k != "self_test"
                   and not (k == "prior_shape" and v == "even")},
        "checkpoints": checkpoints,
        "early_window_checkpoints": checkpoints[:half],
        "receivers": receivers,
        "policy_mapping": {
            "cold": "random/uninformed baseline",
            "full": "per-channel full reuse (fixed lambda)",
            "selective": "transfer-aware selective reuse (learned lambda)",
            "combined": "pooled reuse over all other channels",
        },
        "transfer_matrices": matrices,
        "summary": summary,
        "diagnostics_hyperedge_vs_clique": diagnostics,
    }
    with open(out_dir / "paired_campaign.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- plot: heatmaps (kway arm) + condition summary ------------------
    channels = sorted(per_channel)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, cond in zip(axes[:2], ("full", "selective")):
        M = np.full((len(channels), len(receivers)), np.nan)
        for i, d in enumerate(channels):
            for j, r in enumerate(receivers):
                cell = matrices[cond]["kway"].get(d, {}).get(r)
                if cell:
                    M[i, j] = cell["T_early"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(receivers)),
                      [r.replace("_", "\n") for r in receivers], fontsize=7)
        ax.set_yticks(range(len(channels)),
                      [c.replace("_", " ") for c in channels], fontsize=7)
        ax.set_xlabel("receiver B")
        ax.set_ylabel("donor A")
        ax.set_title(f"T_early^(A->B), {cond} reuse (k-way arm)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    labels = sorted(summary)
    vals = [summary[k]["mean_T_early"] for k in labels]
    colors = ["tab:blue" if k.endswith("kway") else "tab:gray"
              for k in labels]
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean early transfer gain T_early (tau units)")
    ax.set_title("condition x arm summary")
    fig.tight_layout()
    fig.savefig(out_dir / "paired_campaign.png", dpi=150)
    plt.close(fig)

    # ---- markdown report ------------------------------------------------
    lines = [
        "# Experiment 26: paired cold/full/selective campaign protocol",
        "",
        f"Receivers: {', '.join(receivers)}; k={args.kway}; "
        f"budget={args.budget}; {args.n_seeds} seeds (matched across "
        f"conditions); lambda_full={args.lam}; early window = first "
        f"{half} of {len(checkpoints)} checkpoints."
        + ("" if args.prior_shape == "even" else
           f" Donor prior shape: {args.prior_shape} (donor median margins,"
           " standardised to the even-prior spread)."),
        "",
        "Reference-policy mapping: cold=random, full=per-channel, "
        "selective=transfer-aware, combined=pooled.",
        "",
        "## Condition x arm summary (mean over all donor->receiver cells)",
        "",
        "| condition | arm | mean T_early | mean T_final | cells>0 (early) |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(summary):
        s = summary[key]
        cond, arm = key.split("|")
        lines.append(f"| {cond} | {arm} | {s['mean_T_early']:+.4f} | "
                     f"{s['mean_T_final']:+.4f} | "
                     f"{100 * s['frac_cells_positive_early']:.0f}% |")
    lines += [
        "",
        "## Hyperedge residual vs clique-expansion cyclic energy "
        "(cold k-way stream)",
        "",
        "| receiver | mean hyperedge residual | clique cyclic fraction |",
        "|---|---|---|",
    ]
    for rec in sorted(diagnostics):
        d = diagnostics[rec]
        lines.append(f"| {rec} | {d['mean_hyperedge_residual']:.4f} | "
                     f"{d['clique_cyclic_energy_fraction']:.4f} |")
    lines += [
        "",
        "Full directed matrices with bootstrap CIs: paired_campaign.json.",
        "Each evaluation = one real archived record (no invented "
        "observations); donor priors are "
        + ("ordinal-only." if args.prior_shape == "even" else
           "relative (standardised donor margins, no absolute eV)."),
    ]
    with open(out_dir / "paired_campaign.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[save] {out_dir}")
    for key in sorted(summary):
        print(f"  {key}: T_early {summary[key]['mean_T_early']:+.4f}, "
              f"T_final {summary[key]['mean_T_final']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
