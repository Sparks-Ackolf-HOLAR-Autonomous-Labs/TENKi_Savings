#!/usr/bin/env python
"""PEGKi transfer scores into the two DFT receivers (QE PBE and QE HSE06),
compared with the measured campaigns of campaign.py.

For each receiver R in {qe_pbe, qe_hse06} and each donor D computed on the
same pool, fit Plackett-Luce class ratings per source (same fit as
temperature_study.py: the reduced formula of an ordering is its class) and
compute Delta(D -> R) = sigma + rho - omega. The measured outcome is the
early share of R's ten most stable compounds found by the learned-trust
campaign with donor D (mean over k = 1 and 4, 20 seeds), read from
results/campaigns/<R>_all/summary.json.

Also reported: the donor-inclusive cost to find 8 of 10 (receiver compute
fraction plus the donor's own compute on the pool, both in wall seconds of
core-seconds: QE PBE ran on 16 cores and QE HSE06 on 32, so wall seconds are
multiplied by the core count; the GPU and one-core donors are counted at their
wall time, which is below 0.5% of either receiver pool), which is the relevant
number when the donor has not already been computed.

Output: results/analysis/dft_receiver_study.json and .md
With --r2scan the receiver is GPAW r2SCAN (gpaw_r2scan, 16 cores) and QE PBE and
QE HSE06 join the donors; output dft_receiver_study_r2scan.{json,md}.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from scipy.stats import spearmanr

import temperature_study as ts

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RECEIVERS = ("qe_pbe", "qe_hse06")
DONORS = ("mace_relax", "gpumd_nep89_md", "lammps_meam_md", "qe_pbe", "qe_hse06")
LABEL = {"mace_relax": "MACE-MP-0, 0 K", "gpumd_nep89_md": "NEP89 MD, 300 K",
         "lammps_meam_md": "MEAM MD, 300 K", "qe_pbe": "QE PBE", "qe_hse06": "QE HSE06",
         "archive_mp": "MP archive"}
CORES = {"qe_pbe": 16, "qe_hse06": 32, "gpaw_r2scan": 16}   # per-query device / ncores field
LABEL["gpaw_r2scan"] = "GPAW r2SCAN"
SUFFIX = ""


def load_src():
    idx = ts.camp.load_index()
    lv = json.loads((RES / "live_values.json").read_text(encoding="utf-8"))
    src = {}
    for s in DONORS:
        t = {}
        for cid, r in lv.get(s, {}).items():
            if cid.startswith("_") or cid not in idx:
                continue
            if r.get("status") != "ok" or r.get("value") is None or not math.isfinite(r["value"]):
                continue
            t[cid] = {"value": float(r["value"]), "sem": float(r.get("sem") or 0.0),
                      "cost": float(r.get("wall_s") or 0.0), "property": r["property"]}
        src[s] = t
    return src, idx


def main():
    global RECEIVERS, DONORS, SUFFIX
    if "--r2scan" in sys.argv:
        RECEIVERS, DONORS, SUFFIX = ("gpaw_r2scan",), DONORS + ("gpaw_r2scan",), "_r2scan"
    src, idx = load_src()
    # transfer scores on the time-averaged values and on MD-noise draws of the MD donors
    # (campaign.md_draw, the draw of run j of the campaigns), fitted in parallel
    draws = 0 if "--no-md-noise" in sys.argv else 20
    from multiprocessing import Pool
    jobs = [(src, idx, list(RECEIVERS), None)] + [(src, idx, list(RECEIVERS), j) for j in range(draws)]
    with Pool(min(10, len(jobs))) as pool:
        fits = pool.map(ts._pegki_draw, jobs, chunksize=1)
    D = fits[0]["deltas"]
    noise = ts.collect_noise(fits[1:]) if draws else {}
    out = {"receivers": {}, "md_noise_draws": draws}
    lines = ["# PEGKi transfer scores into the DFT receivers", "",
             "| Receiver | Donor | Delta | sigma | rho | omega | learned-trust early share | "
             "fixed-trust cost to 8 of 10 | donor pool cost / receiver pool cost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for R in RECEIVERS:
        summ = json.loads((RES / "campaigns" / f"{R}_all" / "summary.json").read_text(encoding="utf-8"))
        arms = summ["arms"]
        total = summ["receiver_total_cost"]
        rows = {}
        for d in DONORS:
            key = f"{d}->{R}"
            if key not in D or f"learned:{d}|k1" not in arms:
                continue
            early = 0.5 * (arms[f"learned:{d}|k1"]["early_recall_area"]["mean"]
                           + arms[f"learned:{d}|k4"]["early_recall_area"]["mean"])
            c8 = arms[f"fixed:{d}|k4"]["cost_frac_to_8of10"]["mean"]
            dcost = (summ["agreement"][d]["donor_total_cost"] * CORES.get(d, 1)) / (total * CORES[R])
            rows[d] = {**D[key], "learned_early_share": early, "fixed_cost_frac_to_8of10_k4": c8,
                       "donor_cost_frac": dcost, "donor_inclusive_cost_frac": c8 + dcost,
                       "tau_full": summ["agreement"][d]["tau_full"]}
            lines.append(f"| {LABEL[R]} | {LABEL[d]} | {D[key]['delta']:+.3f} | {D[key]['sigma']:+.3f} | "
                         f"{D[key]['rho']:+.3f} | {D[key]['omega']:+.3f} | {early:.3f} | {c8:.3f} | {dcost:.4f} |")
        names = list(rows)
        rs = spearmanr([rows[d]["delta"] for d in names], [rows[d]["learned_early_share"] for d in names]).statistic
        top_delta = max(names, key=lambda d: rows[d]["delta"])
        top_meas = max(names, key=lambda d: rows[d]["learned_early_share"])
        # how often each donor is the top-Delta pick, and Spearman per draw, over the MD-noise draws
        pick_share, rs_draws = {}, []
        if draws:
            for j in range(draws):
                dj = {d: noise[f"{d}->{R}"]["draws"][j] for d in names}
                top = max(dj, key=dj.get)
                pick_share[top] = pick_share.get(top, 0) + 1 / draws
                rs_draws.append(float(spearmanr([dj[d] for d in names],
                                                [rows[d]["learned_early_share"] for d in names]).statistic))
            for d in names:
                rows[d]["delta_md_noise"] = {k: noise[f"{d}->{R}"][k] for k in ("mean", "p05", "p95")}
        out["receivers"][R] = {"donors": rows, "spearman_delta_vs_early": float(rs),
                               "top_pick_share_over_md_draws": pick_share,
                               "spearman_over_md_draws": {"mean": float(np.mean(rs_draws)) if rs_draws else None,
                                                          "min": min(rs_draws) if rs_draws else None,
                                                          "max": max(rs_draws) if rs_draws else None},
                               "top_by_delta": top_delta, "top_measured": top_meas,
                               "no_donor_cost_frac_to_8of10_k4": arms["surrogate|k4"]["cost_frac_to_8of10"]["mean"],
                               "receiver_total_wall_s": total}
        lines += ["", f"{LABEL[R]}: Spearman(Delta, measured early share) = {rs:+.2f}; "
                      f"top by Delta = {LABEL[top_delta]}; top measured = {LABEL[top_meas]}", ""]
        if draws:
            sd = out["receivers"][R]["spearman_over_md_draws"]
            lines += [f"Over {draws} MD-noise draws: top-Delta pick " + ", ".join(
                f"{LABEL[d]} {v:.0%}" for d, v in sorted(pick_share.items(), key=lambda kv: -kv[1]))
                + f"; Spearman mean {sd['mean']:+.2f} (range {sd['min']:+.2f} to {sd['max']:+.2f}); Delta 5-95%: "
                + ", ".join(f"{LABEL[d]} [{rows[d]['delta_md_noise']['p05']:+.3f}, {rows[d]['delta_md_noise']['p95']:+.3f}]"
                            for d in names), ""]
    (RES / "analysis" / f"dft_receiver_study{SUFFIX}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RES / "analysis" / f"dft_receiver_study{SUFFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
