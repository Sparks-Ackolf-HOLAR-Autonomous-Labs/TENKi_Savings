"""
Experiment 28b -- Why the transferred acquisition policy failed on the
AFLOW Al-Mg-Si formation-energy source in experiment 28.

Experiment 28 reported learned final tau +0.683 against +0.790 for
random 4-way selection on the AFLOW formation-energy source (5 classes:
elemental, Al-Mg, Al-Si, Mg-Si, Al-Mg-Si; 20 evaluation seeds). This
script tests three explanations on the same records, the same frozen
policy weights, and exp 28's own run_arm:

  1. Evaluation noise: rerun the exp 28 seeds (to reproduce the
     number), then 200 fresh matched seeds, and report the paired
     learned-minus-random difference with a 90% bootstrap interval.
  2. Ceiling: fit Plackett-Luce to 10,000 random 4-way events and
     compare its order with the median-based reference ordering.
  3. Class structure: per-class median, quartiles, and the pairwise
     probability that a record of one class is more negative than a
     record of another (near-ties make tau coarse with 5 classes).

The MP formation-energy source has 4 classes, so with k = 4 every
event contains every class and no selection rule can differ from
random except through the random record draw; it is included as a
control.

Outputs: tenki/results/cross_property/formation_energy_diagnosis.{json,md}
Run from tenki/TENKi:  python experiments/28b_formation_energy_diagnosis.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "exp28", _HERE / "28_cross_property_policy.py")
e28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e28)

ROOT = _HERE.parents[2]
OUT = ROOT / "tenki" / "results" / "cross_property"
K, BUDGET, STEP, BASE_SEED = 4, 240, 8, 42


def paired(cl, seeds, w):
    cps = list(range(STEP, BUDGET + 1, STEP))
    arms = {}
    for arm in ("kway_random", "kway_learned"):
        arms[arm] = np.array([
            e28.run_arm(cl, arm, K, BUDGET, cps, sd, weights=w,
                        higher_is_better=False)[-1] for sd in seeds])
    d = arms["kway_learned"] - arms["kway_random"]
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return {"n_seeds": len(seeds),
            "learned": float(arms["kway_learned"].mean()),
            "random": float(arms["kway_random"].mean()),
            "learned_sd": float(arms["kway_learned"].std()),
            "diff": float(d.mean()),
            "diff_ci90": [float(x) for x in np.percentile(boot, [5, 95])]}


def ceiling(cl, n_events=10000):
    rng = np.random.default_rng(1)
    names = sorted(cl)
    edges = []
    for _ in range(n_events):
        sub = list(rng.choice(names, size=K, replace=False))
        e = e28.sample_event(cl, sub, rng, False)
        if e is not None:
            edges.append(e)
    m = e28.PlackettLuceModel()
    m.fit(edges)
    ref = e28.reference_ranking(cl, False)
    fit = [str(x) for x in m.ranking()]
    return {"reference_by_median": ref, "fit_order_10000_events": fit,
            "ceiling_tau": float(e28._kendall_tau(fit, ref))}


def structure(cl):
    names = sorted(cl)
    per = {c: {"n": int(len(v)), "median": float(np.median(v)),
               "q25": float(np.percentile(v, 25)),
               "q75": float(np.percentile(v, 75))} for c, v in cl.items()}
    pmat = {a: {b: float(np.mean(cl[a][:, None] < cl[b][None, :]))
                for b in names if b != a} for a in names}
    return {"classes": per, "p_more_negative": pmat}


def main():
    w = json.load(open(ROOT / "tenki" / "results" / "cross_property"
                       / "policy_weights.json"))["weights"]
    fe = e28.load_fe_channels(
        ROOT / "compute" / "almgsi_archive" / "results"
        / "tracka_channels.jsonl", 10)
    exp28_seeds = [BASE_SEED + 1000 * s for s in range(20)]
    fresh = [777777 + 1000 * s for s in range(200)]
    out = {}
    for ch, (cl, _counts) in fe.items():
        out[ch] = {"exp28_seeds": paired(cl, exp28_seeds, w),
                   "fresh_200_seeds": paired(cl, fresh, w),
                   "ceiling": ceiling(cl), "structure": structure(cl)}
        print(ch, json.dumps(out[ch]["exp28_seeds"]),
              json.dumps(out[ch]["fresh_200_seeds"]),
              out[ch]["ceiling"]["ceiling_tau"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "formation_energy_diagnosis.json", "w") as f:
        json.dump(out, f, indent=2)
    L = ["# Experiment 28b: formation-energy transfer diagnosis", "",
         "Frozen exp 28 policy weights; k = 4; budget 240; exp 28 run_arm.",
         "", "| source | seeds | learned tau | random tau | learned minus "
         "random (90% CI) |", "|---|---|---|---|---|"]
    for ch, r in out.items():
        for key, lab in (("exp28_seeds", "exp 28 seeds (20)"),
                         ("fresh_200_seeds", "fresh seeds (200)")):
            p = r[key]
            L.append(f"| {ch} | {lab} | {p['learned']:+.3f} | "
                     f"{p['random']:+.3f} | {p['diff']:+.3f} "
                     f"[{p['diff_ci90'][0]:+.3f}, {p['diff_ci90'][1]:+.3f}] |")
    L += ["", "| source | reference order (median) | fit order, 10,000 "
          "random events | ceiling tau |", "|---|---|---|---|"]
    for ch, r in out.items():
        c = r["ceiling"]
        L.append(f"| {ch} | {', '.join(c['reference_by_median'])} | "
                 f"{', '.join(c['fit_order_10000_events'])} | "
                 f"{c['ceiling_tau']:+.2f} |")
    L += ["", "Class medians and quartiles (eV/atom):", ""]
    for ch, r in out.items():
        for c, s in r["structure"]["classes"].items():
            L.append(f"- {ch} {c}: n = {s['n']}, median {s['median']:+.3f}, "
                     f"quartiles {s['q25']:+.3f} to {s['q75']:+.3f}")
    with open(OUT / "formation_energy_diagnosis.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
