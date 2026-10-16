"""
Experiment 27c -- spread of the experiment 27 result over training seeds.

Experiment 27 seeded its cross-entropy training with Python's salted
str hash(), so its committed numbers could not be reproduced. After the
fix (zlib.crc32), the experiment was rerun with five values of --seed
(42, 1, 2, 3, 7). Each value changes both the training draws and the
evaluation seeds, so every run is a complete, independent replicate of
experiment 27. This script collects the per-run summaries.

Usage (from tenki/TENKi):
  python experiments/27c_training_seed_spread.py RUN_DIR [RUN_DIR ...]
Each RUN_DIR holds learned_acquisition.json and fold_<source>.json files
written by 27_learned_acquisition.py.
Outputs: tenki/results/learned_acquisition_seeds/seed_spread.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = (Path(__file__).resolve().parents[2] / "results"
       / "learned_acquisition_seeds")


def per_source(run: Path):
    wins = {}
    for f in sorted(run.glob("fold_*.json")):
        d = json.load(open(f))
        c = d["curves"]  # mean tau per checkpoint over eval seeds
        lr, rr = c["kway_learned"][-1], c["kway_random"][-1]
        wins[d["held_out"]] = float(lr - rr)
    return wins


def main(dirs):
    runs = []
    for d in map(Path, dirs):
        r = json.load(open(d / "learned_acquisition.json"))
        ps = per_source(d)
        runs.append({
            "seed": r["config"]["seed"],
            "final_learned": r["final_tau"]["kway_learned"],
            "final_random_kway": r["final_tau"]["kway_random"],
            "final_pair": r["final_tau"]["pair_random"],
            "final_minspread": r["final_tau"]["kway_active"],
            "final_fisher": r["final_tau"]["kway_fisher"],
            "b90_learned": r["budget_to_90pct_of_random_final"]["kway_learned"],
            "b90_random": r["budget_to_90pct_of_random_final"]["kway_random"],
            "sources_learned_ge_random": int(sum(v >= 0 for v in ps.values())),
            "per_source_learned_minus_random": ps,
            "weights_mean": r["learned_weights_mean"],
        })
    diff = np.array([x["final_learned"] - x["final_random_kway"] for x in runs])
    summary = {
        "n_runs": len(runs),
        "final_learned_minus_random_mean": float(diff.mean()),
        "final_learned_minus_random_min": float(diff.min()),
        "final_learned_minus_random_max": float(diff.max()),
        "runs_learned_ge_random_final": int((diff >= 0).sum()),
        "runs_learned_faster_to_90pct": int(sum(
            x["b90_learned"] < x["b90_random"] for x in runs)),
        "closeness_weight_mean_per_run": [x["weights_mean"][2] for x in runs],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"runs": runs, "summary": summary},
              open(OUT / "seed_spread.json", "w"), indent=2)
    L = ["# Experiment 27c: experiment 27 over five training seeds", "",
         "k = 4; budget 240; 20 evaluation seeds per run; leave-one-source-out.",
         "Each --seed value changes training draws and evaluation seeds.", "",
         "| seed | learned final tau | random 4-way | min-spread | Fisher-greedy"
         " | pairwise | budget to 90%: learned / random | sources learned >= random |",
         "|---|---|---|---|---|---|---|---|"]
    for x in runs:
        L.append(f"| {x['seed']} | {x['final_learned']:+.3f} | "
                 f"{x['final_random_kway']:+.3f} | {x['final_minspread']:+.3f} | "
                 f"{x['final_fisher']:+.3f} | {x['final_pair']:+.3f} | "
                 f"{x['b90_learned']} / {x['b90_random']} | "
                 f"{x['sources_learned_ge_random']} of 8 |")
    s = summary
    L += ["", f"Learned minus random 4-way at the final budget: mean "
          f"{s['final_learned_minus_random_mean']:+.3f} (range "
          f"{s['final_learned_minus_random_min']:+.3f} to "
          f"{s['final_learned_minus_random_max']:+.3f}); learned at or above "
          f"random in {s['runs_learned_ge_random_final']} of {s['n_runs']} runs; "
          f"faster to 90% of the random arm's final tau in "
          f"{s['runs_learned_faster_to_90pct']} of {s['n_runs']} runs.",
          "", "Mean closeness weight per run: " + ", ".join(
              f"{w:+.2f}" for w in s["closeness_weight_mean_per_run"])]
    open(OUT / "seed_spread.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
