"""
Experiment 27b -- Long-horizon, per-seed evaluation of the acquisition rules.

Why. The decision-body loop analysis in PEGKi
(shared/scripts/decision_body_cycles.py) could only use exp 27's
seed-MEAN curves, because exp 27 stored one averaged curve per held-out
source. That left eight rounds (one per held-out source) for the
acquisition-rule tournament and a leave-one-out range instead of a
bootstrap. Active-learning campaigns run for far more than eight rounds,
so this script re-evaluates the same five arms

    pair_random, kway_random, kway_active, kway_fisher, kway_learned

on every held-out source with
  * many more evaluation seeds (default 100, disjoint from the CEM
    training seeds of exp 27, identical seed formula 42 + 1000 s), and
  * a budget eight times longer (default 1920 evaluations, i.e. 960
    pairwise rounds or 480 four-way rounds; exp 27 stopped at 240),
and stores EVERY per-seed curve, so each (source, seed, checkpoint) is a
separate tournament round.

Nothing is retrained. The kway_learned arm uses the committed per-fold
weights from results/learned_acquisition_salted_seed/fold_<source>.json (originally results/learned_acquisition/) (each fold's
policy was trained with that source held out), so the evaluation stays
leave-one-source-out. Budgets beyond 240 are an extrapolation of a policy
trained on the 240-evaluation area under the curve; this is reported, not
hidden.

Data: the version-locked band-gap archive (sha256[:16] checked and
recorded); one archived record per participating class per event; no
invented observations (identical sampling to exp 24/27).

Outputs (default tenki/results/learned_acquisition_long/):
  fold_<source>.json   per-seed curves for all arms (written per fold)
  summary.json / summary.md   seed-level means and 5/25/50/75/95 percentiles

Run (from tenki/TENKi):
  uv run python experiments/27b_acquisition_long_eval.py [--n-seeds 100]
ASCII only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "exp27", _HERE / "27_learned_acquisition.py")
exp27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp27)

ARMS = ["pair_random", "kway_random", "kway_active", "kway_fisher",
        "kway_learned"]
_U = None


def checkpoint_grid(budget: int) -> list:
    """Dense up to 240 (the exp 27 horizon), coarser beyond it."""
    cps = list(range(16, min(budget, 240) + 1, 16))
    cps += list(range(288, budget + 1, 48))
    return cps


def _init(path: str, min_class: int):
    global _U
    _U = exp27.load_universe(Path(path), min_class)


def _run(task):
    source, arm, seed, kway, budget, weights = task
    cps = checkpoint_grid(budget)
    w = np.asarray(weights) if weights is not None else None
    return exp27.run_arm(_U[source], arm, kway, budget, cps, seed, weights=w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe",
                    default="output/bandgap_archive_build/bandgap_universe.jsonl")
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=1920)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    universe = exp27._pegki_root() / args.universe
    sha = hashlib.sha256(universe.read_bytes()).hexdigest()[:16]
    print(f"[archive] sha256[:16] = {sha}")
    wdir = _HERE.parent.parent / "results" / "learned_acquisition_salted_seed"
    out = Path(args.output_dir) if args.output_dir else (
        _HERE.parent.parent / "results" / "learned_acquisition_long")
    out.mkdir(parents=True, exist_ok=True)
    per = exp27.load_universe(universe, args.min_class_size)
    sources = sorted(per)
    cps = checkpoint_grid(args.budget)
    seeds = [args.seed + 1000 * s for s in range(args.n_seeds)]
    nw = args.workers or min(16, os.cpu_count() or 1)

    with Pool(nw, initializer=_init,
              initargs=(str(universe), args.min_class_size)) as pool:
        for src in sources:
            f = out / f"fold_{src}.json"
            if f.exists():
                print(f"[skip] {src}")
                continue
            w = json.load(open(wdir / f"fold_{src}.json",
                               encoding="utf-8"))["weights"]
            tasks = [(src, a, sd, args.kway, args.budget,
                      w if a == "kway_learned" else None)
                     for a in ARMS for sd in seeds]
            res = pool.map(_run, tasks, chunksize=4)
            curves = {a: [] for a in ARMS}
            for t, r in zip(tasks, res):
                curves[t[1]].append([round(float(x), 6) for x in r])
            rec = {"held_out": src, "weights": w, "seeds": seeds,
                   "checkpoints": cps, "curves": curves,
                   "archive_sha256_16": sha}
            f.write_text(json.dumps(rec), encoding="utf-8")
            fin = {a: float(np.mean([c[-1] for c in curves[a]]))
                   for a in ARMS}
            print(f"[done] {src} final:",
                  ", ".join(f"{a}={v:+.3f}" for a, v in fin.items()),
                  flush=True)

    # summary over all (source, seed) curves
    summ = {"archive_sha256_16": sha, "checkpoints": cps,
            "config": vars(args), "arms": {}}
    for a in ARMS:
        stack = np.array([c for src in sources for c in json.load(
            open(out / f"fold_{src}.json", encoding="utf-8"))["curves"][a]])
        q = np.percentile(stack, [5, 25, 50, 75, 95], axis=0)
        summ["arms"][a] = {"n_curves": int(stack.shape[0]),
                           "mean": stack.mean(0).round(4).tolist(),
                           "p05": q[0].round(4).tolist(),
                           "p25": q[1].round(4).tolist(),
                           "p50": q[2].round(4).tolist(),
                           "p75": q[3].round(4).tolist(),
                           "p95": q[4].round(4).tolist()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1),
                                      encoding="utf-8")
    L = ["# Exp 27b: acquisition rules, long horizon, per seed", "",
         f"Archive sha256[:16] `{sha}`; {len(sources)} held-out sources x "
         f"{args.n_seeds} seeds; k = {args.kway}; budget {args.budget} "
         "evaluations. kway_learned uses the committed exp 27 fold weights "
         "(not retrained).", "",
         "| arm | tau at 240 (mean) | tau at final (mean) | final p05-p95 |",
         "|---|---|---|---|"]
    i240 = cps.index(240)
    for a in ARMS:
        s = summ["arms"][a]
        L.append(f"| {a} | {s['mean'][i240]:+.3f} | {s['mean'][-1]:+.3f} | "
                 f"{s['p05'][-1]:+.3f} to {s['p95'][-1]:+.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
