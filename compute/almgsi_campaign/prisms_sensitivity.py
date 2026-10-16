"""Sensitivity of the PRISMS comparison to its settings.

``prisms_comparison.py`` runs PRISMS with one setting of its weights and
budgets (lambda1 = 0.1 for the lowest ranker and 10 for the higher rankers,
lambda2 = 0.1, kU = 5, b2 = 100, b3 = 50), because the PRISMS paper reports
ranges, not defaults. This script varies one setting at a time around that
point and reruns the two orders a group could choose before any receiver
value exists (cost order and model-class order, three rankers each) on both
DFT receivers, with the same pool, MD draws and metric as the comparison.

Grid (one at a time; the default is included once):
  lambda1 of the lowest ranker  0.01, 0.1, 1
  lambda1 of the higher rankers 1, 10, 100
  lambda2                       0.01, 0.1, 1
  kU                            3, 5, 10
  (b2, b3)                      (50, 25), (100, 50), (200, 100)

Output: results/analysis/prisms_comparison/prisms_sensitivity.{json,md}.
Run from this folder: python prisms_sensitivity.py --seeds 20 --procs 12
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

import prisms_comparison as P
from campaign import RES, crc, fractions, load_index, load_tables, md_draw, summarise

DEFAULT = {"lam_low": 0.1, "lam_high": 10.0, "lambda2": 0.1, "kU": 5, "budgets": (100, 50)}
GRID = {"lam_low": [0.01, 1.0], "lam_high": [1.0, 100.0], "lambda2": [0.01, 1.0],
        "kU": [3, 10], "budgets": [(50, 25), (200, 100)]}
ORDERS = {"cost": ["mace_relax", "lammps_meam_md", "gpumd_nep89_md"],
          "class": ["lammps_meam_md", "gpumd_nep89_md", "mace_relax"]}


def configs():
    out = [("default", dict(DEFAULT))]
    for key, vals in GRID.items():
        for v in vals:
            c = dict(DEFAULT)
            c[key] = v
            out.append((f"{key}={v}", c))
    return out


_CTX = {}


def _init(receivers, k):
    idx = load_index()
    tables = load_tables()
    for recname in receivers:
        rec = tables[recname]
        pool = sorted(c for c in rec if all(c in tables[d] for d in P.RANKERS))
        X = {c: fractions(c, idx) for c in pool}
        pool = [c for c in pool if (X[c] > 0).sum() >= 2]
        _CTX[recname] = (rec, pool, X, sum(rec[c]["cost"] for c in pool))
    _CTX["tables"] = tables
    _CTX["k"] = k


def _run(job):
    recname, cname, cfg, oname, s = job
    rec, pool, X, total = _CTX[recname]
    tables = _CTX["tables"]
    P.LAMBDA1 = [cfg["lam_low"], cfg["lam_high"], cfg["lam_high"]]
    P.LAMBDA2 = cfg["lambda2"]
    P.KU = cfg["kU"]
    P.BUDGETS = [None, cfg["budgets"][0], cfg["budgets"][1]]
    r_tab = md_draw(rec, recname, s)
    rk = [(d, md_draw(tables[d], d, s)) for d in ORDERS[oname]]
    # same seed string as prisms_comparison.py, so the default row reproduces it
    r = P.prisms_campaign(r_tab, rk, pool, X, _CTX["k"], crc(f"prisms|prisms_3fid:{oname}|{s}"))
    return recname, cname, oname, s, summarise(r["curve"], total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receivers", default="qe_pbe,qe_hse06")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    receivers = args.receivers.split(",")
    cfgs = configs()
    jobs = [(r, cn, c, o, s) for r in receivers for cn, c in cfgs for o in ORDERS for s in range(args.seeds)]
    res = {}
    with Pool(args.procs, initializer=_init, initargs=(receivers, args.k)) as pool:
        for n, (rn, cn, on, s, sm) in enumerate(pool.imap_unordered(_run, jobs, chunksize=2)):
            res.setdefault(rn, {}).setdefault(cn, {}).setdefault(on, []).append(sm)
            if n % 50 == 0:
                print(n, len(jobs), flush=True)
    summ = {}
    for rn, byc in res.items():
        for cn, byo in byc.items():
            for on, runs in byo.items():
                c8 = np.array([r["cost_frac_to_8of10"] for r in runs])
                e = np.array([r["early_recall_area"] for r in runs])
                boot = np.random.default_rng(0).integers(0, len(c8), (2000, len(c8)))
                summ.setdefault(rn, {}).setdefault(cn, {})[on] = {
                    "cost_frac_to_8of10": float(c8.mean()),
                    "c8_ci": [float(np.percentile(c8[boot].mean(1), 2.5)),
                              float(np.percentile(c8[boot].mean(1), 97.5))],
                    "early_recall_area": float(e.mean())}
    out = RES / "analysis" / "prisms_comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prisms_sensitivity.json").write_text(json.dumps(
        {"k": args.k, "seeds": args.seeds, "default": {**DEFAULT, "budgets": list(DEFAULT["budgets"])},
         "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in GRID.items()},
         "orders": ORDERS, "receivers": summ}, indent=1), encoding="utf-8")
    lines = ["# PRISMS settings sensitivity on the computed Al-Mg-Si pool", "",
             f"k = {args.k}, {args.seeds} runs with MD draws; one setting varied at a time. "
             "Cells: receiver-cost fraction to find 8 of the 10 lowest [95% CI].", ""]
    for rn in receivers:
        lines += [f"## Receiver {rn}", "", "| setting | cost order | model-class order |", "|---|---|---|"]
        for cn, _ in cfgs:
            row = summ[rn][cn]
            lines.append(f"| {cn} | " + " | ".join(
                f"{row[o]['cost_frac_to_8of10']:.3f} [{row[o]['c8_ci'][0]:.3f}, {row[o]['c8_ci'][1]:.3f}]"
                for o in ORDERS) + " |")
        lines.append("")
    (out / "prisms_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
