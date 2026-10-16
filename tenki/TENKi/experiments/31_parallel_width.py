"""
Experiment 31 -- Parallel width: cost axis versus wall-clock axis.

Motivated by Agensh (Zhan et al. 2026, arXiv:2609.26781), which
compares agent teams at equal wall clock; TENKi compares at equal evaluation cost. This experiment puts both
axes on the same campaigns.

P agents (instruments, furnaces, workers) query one receiver's band-gap
archive concurrently. Each round every agent submits one k-way event
(cost k), so a round costs P*k evaluations and one time step. All events go
to one shared log; the Plackett--Luce fit is refreshed once per round, so
agents in the same round choose from the same (stale) fit.

Acquisition rules:
  random       : each agent draws a random k-subset
  active       : each agent applies the exp 24 rule independently (the k
                 consecutive classes in fitted order with the smallest
                 strength spread; epsilon-greedy, eps = 0.2). Agents in a
                 round see the same fit, so they tend to pick the same
                 window (duplicated work).
  active_claim : as active, but agents choose in turn and a window already
                 claimed in this round is unavailable (the Agensh
                 "claim on the whiteboard"); if every window is claimed the
                 agent draws a random subset.

Every campaign stops at the same total evaluation budget E. Output per
(receiver, P, rule, seed): tau after every round, with the evaluations
spent and the round index, so tau can be read against either axis.

Data rule: one archived record per participating class per event; no
invented observations. No donor information (cold campaigns).

Run (from tenki/TENKi):
  python experiments/31_parallel_width.py --universe /abs/universe.jsonl \
      --workers 18 --output-dir ../../compute/shared_queue_experiments/width
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

from src.rating.hypergraph_rating import (  # noqa: E402
    PlackettLuceModel,
    _kendall_tau,
)

RULES = ["random", "active", "active_claim"]
EPS = 0.2


def windows(model, names, k):
    """All k-windows in fitted order, sorted by strength spread (ascending)."""
    theta = {n: float(model.log_strengths.get(n, 0.0)) for n in names}
    ordered = sorted(names, key=lambda n: (-theta[n], n))
    wins = []
    for i in range(len(ordered) - k + 1):
        w = ordered[i:i + k]
        wins.append((theta[w[0]] - theta[w[-1]], i, tuple(w)))
    wins.sort()
    return [w for _, _, w in wins]


def campaign(classes, P, rule, k, E, seed):
    rng = np.random.default_rng(seed)
    names = sorted(classes)
    ref = exp26.reference_ranking(classes)
    k = min(k, len(names))
    model = PlackettLuceModel()
    fitted = False
    edges, curve = [], []
    spent, rnd = 0, 0
    while spent + P * k <= E:
        rnd += 1
        claimed = set()
        ranked_wins = windows(model, names, k) if fitted else []
        n_dup = 0
        subsets = []
        for _ in range(P):
            if rule == "random" or not fitted or rng.random() < EPS:
                sub = tuple(sorted(rng.choice(names, size=k, replace=False)))
            elif rule == "active":
                sub = ranked_wins[0]
            else:
                free = [w for w in ranked_wins
                        if tuple(sorted(w)) not in claimed]
                sub = (free[0] if free else
                       tuple(sorted(rng.choice(names, size=k,
                                               replace=False))))
            key = tuple(sorted(sub))
            if key in claimed:
                n_dup += 1
            claimed.add(key)
            subsets.append(list(sub))
        for sub in subsets:
            e = exp26.sample_event(classes, sub, rng)
            if e is not None:
                edges.append(e)
        spent += P * k
        if edges:
            model = PlackettLuceModel()
            model.fit(edges)
            fitted = True
            tau = _kendall_tau(model.ranking(), ref)
        else:
            tau = 0.0
        curve.append([rnd, spent, float(tau), n_dup])
    return curve


def _job(args):
    receiver, seed, per_channel, k, E, widths = args
    classes = per_channel[receiver]
    out = {"receiver": receiver, "seed": seed, "curves": {}}
    for P in widths:
        for rule in RULES:
            out["curves"][f"{rule}|{P}"] = campaign(classes, P, rule, k, E,
                                                    seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--evals", type=int, default=512)
    ap.add_argument("--widths", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--min-class-size", type=int, default=30)
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
    jobs, results = [], []
    for r in receivers:
        for s in range(args.n_seeds):
            seed = args.seed + 1000 * s
            ck = ck_dir / f"{r}_s{seed}.json"
            if ck.exists():
                with open(ck, "r", encoding="utf-8") as fh:
                    results.append(json.load(fh))
                continue
            jobs.append((r, seed, per_channel, args.kway, args.evals,
                         args.widths))
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

    payload = {"config": vars(args), "receivers": receivers,
               "reference_rankings": {r: exp26.reference_ranking(
                   per_channel[r]) for r in receivers},
               "results": sorted(results,
                                 key=lambda r: (r["receiver"], r["seed"]))}
    with open(out_dir / "parallel_width_raw.json", "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[write] {out_dir / 'parallel_width_raw.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
