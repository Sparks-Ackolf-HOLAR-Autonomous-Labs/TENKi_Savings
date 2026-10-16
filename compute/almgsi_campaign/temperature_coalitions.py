#!/usr/bin/env python
"""Coalition files for the higher-temperature MD donor games.

temperature_study.py (analysis D) runs every donor coalition of the six hot
receivers (MEAM, NEP89 and MACE-MP-0 MD at 453 K and 803 K; six donors each,
64 coalitions) but keeps only the seed-averaged value. The data-valuation
check (ATEKi experiment 20, --almgsi-hot) also needs each run's value, so this
script reruns the same campaigns with the same seeds and writes, per receiver,
a coalitions.json in the format campaign.py --shapley writes:

    {mode: {players, runs, value: {key: mean}, per_run: {key: [run values]}}}

with key = '+'.join(sorted coalition members in player order) and mode
pooled_learned (the published game) and pooled_fixed (single donors run with
fixed trust, pools with pooled fixed trust).

The campaign seed only breaks exact ties (campaign.py adds a 1e-9 jitter), so
every coalition with at least one donor makes the same choices in all 20
runs; only the cold start differs. With --resample-sem each run instead draws
every MD value (receiver and donors) from a normal distribution centred on the
time-averaged value with the measured standard error of that average as its
width (0 K sources have no sampling error and are not perturbed). The draw is
fixed per (source, run), so all coalitions of one run see the same values.
This gives run-to-run noise from MD sampling, which the consistency check of
experiment 20 needs.

Usage (from singular_obj/shared, shared env):
    python ../compute/almgsi_campaign/temperature_coalitions.py [--resample-sem]
Outputs: results_T/analysis/coalitions/<receiver>/coalitions.json
         (coalitions_sem.json with --resample-sem)
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("temperature_study", HERE / "temperature_study.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
camp = ts.camp


def perturbed(table, source, s):
    # same draw as campaign.md_draw, which temperature_study.py uses for its campaigns
    return camp.md_draw(table, source, s)


def game(src, rec_name, mode, k, seeds, resample=False):
    rec0 = ts.formation(src[rec_name])
    pool = sorted(rec0)
    X = {c: ts.fractions_noase(c) for c in pool}
    total = sum(rec0[c]["cost"] for c in pool)
    names = ts.donors_for(rec_name, src)
    if resample:
        recs = [perturbed(rec0, rec_name, s) for s in range(seeds)]
        dons = [{d: perturbed(ts.formation(src[d]), d, s) for d in names} for s in range(seeds)]
    else:
        recs = [rec0] * seeds
        dons = [{d: ts.formation(src[d]) for d in names}] * seeds
    single = mode.split("_")[1] + ":"
    value, per_run = {}, {}
    for r_ in range(len(names) + 1):
        for S in itertools.combinations(names, r_):
            vals = []
            for s in range(seeds):
                rec = recs[s]
                if not S:
                    res = camp.run_campaign("surrogate", rec, {}, pool, X, k, camp.crc(f"T|surrogate|{k}|{s}"))
                else:
                    sub = {d: dons[s][d] for d in S}
                    arm = mode if len(S) > 1 else single + S[0]
                    res = camp.run_campaign(arm, rec, sub, pool, X, k,
                                            camp.crc(f"T|{mode}|{'+'.join(S)}|{k}|{s}"))
                vals.append(float(camp.summarise(res["curve"], total)["early_recall_area"]))
            key = "+".join(S)
            per_run[key] = vals
            value[key] = float(np.mean(vals))
    return {"players": names, "runs": [f"k{k}|s{s}" for s in range(seeds)],
            "value": value, "per_run": per_run}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-t", default=str(HERE / "results_T"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--resample-sem", action="store_true")
    args = ap.parse_args()
    rt = Path(args.results_t)
    src, _ = ts.load_sources(rt, reaggregate=False)
    receivers = [f"{e}@{T}K" for e in ts.ENGINES for T in ts.HOT if f"{e}@{T}K" in src]
    for rec_name in receivers:
        out = {m: game(src, rec_name, m, args.k, args.seeds, args.resample_sem) for m in ("pooled_learned", "pooled_fixed")}
        d = rt / "analysis" / "coalitions" / rec_name.replace("@", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / ("coalitions_sem.json" if args.resample_sem else "coalitions.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[coalitions] {rec_name}: v(all) learned {out['pooled_learned']['value']['+'.join(out['pooled_learned']['players'])]:.3f}"
              f" fixed {out['pooled_fixed']['value']['+'.join(out['pooled_fixed']['players'])]:.3f}", flush=True)


if __name__ == "__main__":
    main()
