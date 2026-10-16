"""
Experiment 30 -- Failure records as transferred knowledge (exclusion lists).

Motivated by Agensh (Zhan et al. 2026, arXiv:2609.26781), which
treats the record of a failed attempt as the most useful shared entry. The band-gap analogue: the donor data
source sends only the list of its bottom-m material classes ("these
families did not give wide gaps"), with no ordering of the rest and no
values. The receiver campaign then uses that list.

Same protocol as exp 26 (matched seeds, one archived record per class per
event, cost = number of participants, budget checkpoints, Kendall tau to
the receiver's own median ranking). Donor-information conditions:

  cold          : no donor information
  full_fixed    : donor class ordering as an anchored prior, trust 8
  full_learned  : same prior, trust learned prequentially
  excl_fixed    : prior -1 on the donor's bottom-m classes, 0 elsewhere,
                  trust 8 (only the exclusion list is used)
  excl_learned  : same exclusion prior, trust learned prequentially
  excl_acq      : exclusion list applied to ACQUISITION: excluded classes
                  are never queried; they are placed below all queried
                  classes in the final ranking, in a fixed alphabetical
                  order (the list carries no order among them)
  top_fixed     : prior +1 on the donor's top-m classes, 0 elsewhere,
                  trust 8 (a "best families" list of the same size)
  top_learned   : same top list, trust learned prequentially

Query arms: pair (k = 2) and kway (k = --kway), as in exp 26.

Metrics per (receiver, donor, condition, arm), paired against cold at the
same seed: early-budget tau advantage (mean over the first half of the
checkpoints), final tau advantage, and top-3 recovery (share of the
receiver's three widest-gap classes found in the fitted top three) early
and final. Percentile bootstrap CIs over seeds.

Data rule: no invented observations; every event value is an archived
record. Only class names (and, for full_*, the donor's class order) cross
between data sources.

Run (from tenki/TENKi):
  python experiments/30_exclusion_list_transfer.py \
      --universe /abs/path/bandgap_universe.jsonl --workers 18 \
      --output-dir ../../compute/shared_queue_experiments/exclusion
ASCII only; seeded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "exp26", os.path.join(_HERE, "26_paired_campaign.py"))
exp26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp26)

from src.rating.hypergraph_rating import (  # noqa: E402  (path set by exp26)
    AdaptivePriorSelector,
    PlackettLuceModel,
    _kendall_tau,
)

LAMBDA_GRID = exp26.LAMBDA_GRID
CONDITIONS = ["full_fixed", "full_learned", "excl_fixed", "excl_learned",
              "excl_acq", "top_fixed", "top_learned"]
ARMS = ["pair", "kway"]


def list_prior(receiver_classes: dict, donor_classes: dict, m: int,
               which: str) -> dict:
    """+1 on the donor's top-m or -1 on its bottom-m shared classes."""
    donor_rank = [c for c in exp26.reference_ranking(donor_classes)
                  if c in receiver_classes]
    prior = {c: 0.0 for c in receiver_classes}
    if which == "bottom":
        for c in donor_rank[-m:]:
            prior[c] = -1.0
    else:
        for c in donor_rank[:m]:
            prior[c] = 1.0
    return prior


def top3(ranking: list, ref: list) -> float:
    return len(set(ranking[:3]) & set(ref[:3])) / 3.0


def run(classes: dict, mode: str, prior: dict | None, lam: float, k: int,
        budget: int, checkpoints: list, seed: int,
        excluded: list | None = None):
    """-> (taus, top3s) per checkpoint.

    mode: cold | fixed | adaptive (as exp 26). If `excluded` is given,
    those classes are never queried and are appended below the fitted
    ranking in alphabetical order.
    """
    rng = np.random.default_rng(seed)
    ref = exp26.reference_ranking(classes)
    excluded = sorted(excluded or [])
    names = sorted(c for c in classes if c not in excluded)
    k = min(k, len(names))
    adaptive = mode == "adaptive"
    anchored = mode in ("fixed", "adaptive")
    selector = (AdaptivePriorSelector(prior, LAMBDA_GRID, refresh_every=4)
                if adaptive else None)
    prior_rank = (sorted(prior, key=lambda c: (-prior[c], c))
                  if anchored else None)

    def finish(rank):
        rank = [c for c in rank if c not in excluded] + excluded
        return _kendall_tau(rank, ref), top3(rank, ref)

    edges, taus, t3s = [], [], []
    spent, ci = 0, 0
    while spent < budget and ci < len(checkpoints):
        subset = list(rng.choice(names, size=k, replace=False))
        e = exp26.sample_event(classes, subset, rng)
        spent += k
        if e is not None:
            if adaptive:
                selector.observe(e)
            edges.append(e)
        while ci < len(checkpoints) and spent >= checkpoints[ci]:
            if adaptive:
                rank = (selector.fit_current().ranking() if selector.edges
                        else prior_rank)
            elif edges:
                model = PlackettLuceModel(
                    prior_weight=lam if anchored else 0.01)
                model.fit(edges,
                          prior_log_strengths=prior if anchored else None)
                rank = model.ranking()
            else:
                rank = prior_rank if anchored else names
            # classes never seen by a cold fit keep their sorted position
            rank = list(rank) + [c for c in names if c not in rank]
            t, s = finish(rank)
            taus.append(t)
            t3s.append(s)
            ci += 1
    while len(taus) < len(checkpoints):
        taus.append(taus[-1] if taus else 0.0)
        t3s.append(t3s[-1] if t3s else 0.0)
    return taus, t3s


def _job(args):
    receiver, seed, per_channel, kway, budget, checkpoints, lam, m = args
    classes = per_channel[receiver]
    ks = {"pair": 2, "kway": kway}
    out = {"receiver": receiver, "seed": seed, "tau": {}, "top3": {}}
    for arm, k in ks.items():
        t, s = run(classes, "cold", None, lam, k, budget, checkpoints, seed)
        out["tau"][f"cold|{arm}"] = t
        out["top3"][f"cold|{arm}"] = s
    for donor in sorted(per_channel):
        if donor == receiver:
            continue
        dcl = per_channel[donor]
        full = exp26.donor_prior(classes, dcl, "even")
        excl = list_prior(classes, dcl, m, "bottom")
        top = list_prior(classes, dcl, m, "top")
        excluded = [c for c, v in excl.items() if v < 0]
        specs = {
            "full_fixed": ("fixed", full, None),
            "full_learned": ("adaptive", full, None),
            "excl_fixed": ("fixed", excl, None),
            "excl_learned": ("adaptive", excl, None),
            "excl_acq": ("cold", None, excluded),
            "top_fixed": ("fixed", top, None),
            "top_learned": ("adaptive", top, None),
        }
        for cond, (mode, prior, exc) in specs.items():
            for arm, k in ks.items():
                t, s = run(classes, mode, prior, lam, k, budget,
                           checkpoints, seed, excluded=exc)
                out["tau"][f"{cond}|{arm}|{donor}"] = t
                out["top3"][f"{cond}|{arm}|{donor}"] = s
    return out


def summarise(results: list, receivers: list, donors_of: dict,
              n_check: int, rng: np.random.Generator) -> dict:
    half = n_check // 2
    by = {}
    for res in results:
        by.setdefault(res["receiver"], []).append(res)
    rows = []
    for rcv in receivers:
        runs = sorted(by[rcv], key=lambda r: r["seed"])
        for donor in donors_of[rcv]:
            for cond in CONDITIONS:
                for arm in ARMS:
                    key = f"{cond}|{arm}|{donor}"
                    row = {"receiver": rcv, "donor": donor,
                           "condition": cond, "arm": arm}
                    for metric in ("tau", "top3"):
                        cold = np.asarray([r[metric][f"cold|{arm}"]
                                           for r in runs])
                        arr = np.asarray([r[metric][key] for r in runs])
                        early = (arr[:, :half].mean(1)
                                 - cold[:, :half].mean(1))
                        final = arr[:, -1] - cold[:, -1]
                        row[f"{metric}_early_delta"] = float(early.mean())
                        row[f"{metric}_early_ci"] = exp26.bootstrap_ci(
                            early, rng)
                        row[f"{metric}_final_delta"] = float(final.mean())
                        row[f"{metric}_final_ci"] = exp26.bootstrap_ci(
                            final, rng)
                        row[f"{metric}_early_abs"] = float(
                            arr[:, :half].mean())
                        row[f"{metric}_final_abs"] = float(arr[:, -1].mean())
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--budget-step", type=int, default=24)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--m", type=int, default=2,
                    help="size of the exclusion (and top) list")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--receivers", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    per_channel = exp26.load_universe(Path(args.universe),
                                      args.min_class_size)
    receivers = sorted(args.receivers or per_channel)
    checkpoints = list(range(args.budget_step, args.budget + 1,
                             args.budget_step))
    jobs, results = [], []
    for r in receivers:
        for s in range(args.n_seeds):
            seed = args.seed + 1000 * s
            ck = ck_dir / f"{r}_s{seed}.json"
            if ck.exists():
                with open(ck, "r", encoding="utf-8") as fh:
                    results.append(json.load(fh))
                continue
            jobs.append((r, seed, per_channel, args.kway, args.budget,
                         checkpoints, args.lam, args.m))
    print(f"[plan] {len(jobs)} jobs, {len(results)} checkpointed")

    def save(res):
        ck = ck_dir / f"{res['receiver']}_s{res['seed']}.json"
        with open(ck, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        results.append(res)
        print(f"[done] {res['receiver']} seed {res['seed']}", flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_job, jobs):
                save(res)
    else:
        for j in jobs:
            save(_job(j))

    donors_of = {r: [d for d in sorted(per_channel) if d != r]
                 for r in receivers}
    rows = summarise(results, receivers, donors_of, len(checkpoints),
                     np.random.default_rng(args.seed))
    lists = {r: {d: {"bottom": [c for c, v in list_prior(
        per_channel[r], per_channel[d], args.m, "bottom").items() if v < 0],
        "top": [c for c, v in list_prior(
            per_channel[r], per_channel[d], args.m, "top").items() if v > 0]}
        for d in donors_of[r]} for r in receivers}
    payload = {"config": {k: v for k, v in vars(args).items()},
               "checkpoints": checkpoints,
               "reference_rankings": {r: exp26.reference_ranking(
                   per_channel[r]) for r in receivers},
               "donor_lists": lists, "rows": rows}
    with open(out_dir / "exclusion_list_transfer.json", "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[write] {out_dir / 'exclusion_list_transfer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
