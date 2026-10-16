"""Discovery campaigns, PEGKi transfer scores and ATEKi credit on the computed
semiconductor pool.

Question: a group screens the pool for the widest band gaps (UV emitters,
power electronics, transparent conductors) at a stated receiver level: QE
HSE06 computed on demand, or the measured gap. Each receiver evaluation is
charged its measured cost (HSE06: core-seconds; measured gaps: one unit per
compound, since the laboratory cost is not known here). Donors are the
on-demand QE PBE gaps and the archived gaps (MP GGA, MP r2SCAN, JARVIS
OptB88vdW and TB-mBJ, SNUMAT GGA and HSE06). A donor that does not hold a
candidate (every archive, for the ordered alloys no database holds) gives it
the neutral score.

The campaign machinery (acquisition, fixed and learned trust, pooled donors,
recall of the receiver's 10 widest gaps against receiver cost, exact Shapley
over donor coalitions) is the one used for the Al-Mg-Si campaign
(``../almgsi_campaign/campaign.py``), applied to -gap so that "lowest" means
"widest gap". The no-donor surrogate works on a composition descriptor
(mean and spread of Pauling electronegativity, mean atomic number, anion
family), scaled to [0, 1].

PEGKi: per-source Plackett-Luce ratings of the anion families (oxides,
nitrides, chalcogenides, pnictides, halides, other) from each source's gaps,
and Delta(D -> R) = sigma + rho - omega, compared with the measured early
recall of the learned-trust campaign with that donor.

Pools: ``all`` and ``nohalide`` (no F, Cl, Br, I: the oxide, nitride and
chalcogenide semiconductors that power and UV devices use).

Usage: python campaign_gap.py [--receiver qe_hse06|expt_zhuo] [--pool all|nohalide] [--shapley]
       [--cost receiver|combined]   (combined: HSE06 plus PBE core-seconds, see --help)
Output: results/campaigns/<receiver>_<pool>/summary.json, campaign_runs.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Element
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
SO = HERE.parents[1]
RES = HERE / "results"
sys.path.insert(0, str(SO / "shared"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


camp = _load("almgsi_campaign_core", HERE.parent / "almgsi_campaign" / "campaign.py")
HALOGENS = {"F", "Cl", "Br", "I"}
ARCHIVES = ("mp_gga", "mp_r2scan", "jarvis_optb88vdw", "jarvis_mbj", "snumat_gga", "snumat_hse")
LABEL = {"qe_pbe": "QE PBE (computed)", "qe_hse06": "QE HSE06 (computed)", "expt_zhuo": "Measured (Zhuo)",
         "mp_gga": "MP GGA", "mp_gga_expsynth": "MP GGA, ICSD-matched", "mp_r2scan": "MP r2SCAN",
         "jarvis_optb88vdw": "JARVIS OptB88vdW", "jarvis_mbj": "JARVIS TB-mBJ",
         "snumat_gga": "SNUMAT GGA", "snumat_hse": "SNUMAT HSE06"}


def family(formula: str) -> str:
    els = {str(e) for e in Composition(formula).elements}
    if els & HALOGENS:
        return "halide"
    if "O" in els:
        return "oxide"
    if "N" in els:
        return "nitride"
    if els & {"S", "Se", "Te"}:
        return "chalcogenide"
    if els & {"P", "As", "Sb"}:
        return "pnictide"
    return "other"


def descriptor(formula: str) -> np.ndarray:
    c = Composition(formula)
    fr = np.array([c.get_atomic_fraction(e) for e in c.elements])
    chi = np.array([e.X for e in c.elements])
    z = np.array([e.Z for e in c.elements])
    fam = ["oxide", "nitride", "chalcogenide", "pnictide", "halide", "other"].index(family(formula))
    return np.array([fr @ chi, chi.max() - chi.min(), fr @ z, fam / 5.0])


def load_tables() -> tuple[dict, dict]:
    idx = {r["cid"]: r for r in json.loads((HERE / "candidates_index.json").read_text())}
    tabs: dict[str, dict] = {}
    for s in ("qe_pbe", "qe_hse06"):
        t = {}
        for p in sorted((RES / s).glob("*.json")) if (RES / s).exists() else []:
            if p.name.endswith(".failed.json"):
                continue
            r = json.loads(p.read_text())
            if r.get("status") == "ok" and r.get("gap_eV") is not None and r["cid"] in idx:
                t[r["cid"]] = {"value": -float(r["gap_eV"]), "gap": float(r["gap_eV"]), "sem": 0.0,
                               "cost": float(r.get("core_s") or 0.0)}
        tabs[s] = t
    arch = json.loads((RES / "archive_donors.json").read_text())
    for s, d in arch.items():
        tabs[s] = {c: {"value": -float(v["value"]), "gap": float(v["value"]), "sem": 0.0,
                       "cost": 1.0 if s == "expt_zhuo" else 0.0} for c, v in d.items() if c in idx}
    return tabs, idx


def pegki(tabs, idx, receiver, donors, pool):
    setop = _load("set_operation_study", SO / "shared" / "scripts" / "set_operation_study.py")
    spaces = {}
    for s in [receiver] + donors:
        classes: dict[str, list[float]] = {}
        for c in pool:
            if c in tabs[s]:
                classes.setdefault(family(idx[c]["formula"]), []).append(tabs[s][c]["gap"])
        classes = {k: sorted(v) for k, v in classes.items() if len(v) >= 2}
        if len(classes) < 3:
            continue
        ks, _ = setop.fit_space(s, classes, 400, 20, 11)
        if ks is not None:
            spaces[s] = ks
    ltw = setop.ltw
    out = {}
    if receiver not in spaces:
        return out
    for d in donors:
        if d not in spaces:
            continue
        a, b = spaces[d], spaces[receiver]
        comp = [ltw.compute_structural_compatibility(a, b), ltw.compute_mapping_efficiency(a, b),
                ltw.compute_transfer_overhead(a, b)]
        out[d] = {"sigma": float(comp[0]), "rho": float(comp[1]), "omega": float(comp[2]),
                  "delta": float(comp[0] + comp[1] - comp[2])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receiver", default="qe_hse06")
    ap.add_argument("--pool", default="all", choices=["all", "nohalide"])
    ap.add_argument("--k", default="1,4")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shapley", action="store_true")
    ap.add_argument("--cost", default="receiver", choices=["receiver", "combined"],
                    help="combined (HSE06 receiver only): charge each query its HSE06 plus PBE core-seconds "
                         "(the PBE relaxation gives the HSE06 geometry); arms whose donors include the PBE "
                         "ordering pay PBE for the whole pool before the first query")
    args = ap.parse_args()
    tabs, idx = load_tables()
    rec = tabs[args.receiver]
    pool = sorted(rec)
    if args.pool == "nohalide":
        pool = [c for c in pool if family(idx[c]["formula"]) != "halide"]
    donor_names = ["qe_pbe", "qe_hse06", *ARCHIVES, "expt_zhuo"]
    donors = {d: tabs[d] for d in donor_names if d != args.receiver and d in tabs
              and sum(c in tabs[d] for c in pool) >= 10}
    X = {c: descriptor(idx[c]["formula"]) for c in pool}
    lo = np.min([X[c] for c in pool], 0)
    hi = np.max([X[c] for c in pool], 0)
    X = {c: (X[c] - lo) / np.where(hi > lo, hi - lo, 1) for c in pool}
    pbe_upfront = 0.0
    rec_query = rec
    if args.cost == "combined":
        if args.receiver != "qe_hse06":
            raise SystemExit("--cost combined is defined for the HSE06 receiver only")
        missing = [c for c in pool if c not in tabs["qe_pbe"]]
        if missing:
            raise SystemExit(f"no PBE record for {len(missing)} pool members, e.g. {missing[:3]}")
        rec_query = {c: {**rec[c], "cost": rec[c]["cost"] + tabs["qe_pbe"][c]["cost"]} for c in rec
                     if c in tabs["qe_pbe"]}
        pbe_upfront = sum(tabs["qe_pbe"][c]["cost"] for c in pool)
    total = sum(rec_query[c]["cost"] for c in pool)
    ks = [int(x) for x in args.k.split(",")]

    def run(arm, k, seed):
        """One campaign; with combined cost, PBE-donor arms pay the pool PBE first, then HSE06 per query."""
        uses_pbe = args.cost == "combined" and (arm.endswith(":qe_pbe") or arm.startswith("pooled"))
        r = camp.run_campaign(arm, rec if uses_pbe else rec_query, donors, pool, X, k, seed)
        if uses_pbe:
            r["curve"] = [{**p, "cost": p["cost"] + pbe_upfront} for p in r["curve"]]
        return r
    arms = ["random", "surrogate"] + [f"{m}:{d}" for d in donors for m in ("fixed", "learned")] \
        + ["pooled_fixed", "pooled_learned"]
    tag = f"{args.receiver}_{args.pool}" + ("_combined" if args.cost == "combined" else "")
    out = RES / "campaigns" / tag
    out.mkdir(parents=True, exist_ok=True)
    rows, bands = [], {}
    grid = np.linspace(0, 1, 201)
    for k in ks:
        for arm in arms:
            g = []
            for s in range(args.seeds):
                r = run(arm, k, camp.crc(f"{arm}|{k}|{s}"))
                sm = camp.summarise(r["curve"], total)
                cx = np.array([0.0] + [p["cost"] / total for p in r["curve"]])
                cy = np.array([0.0] + [p["recall"] for p in r["curve"]])
                g.append([float(cy[np.searchsorted(cx, x, side="right") - 1]) for x in grid])
                rows.append({"arm": arm, "k": k, "seed": s, **sm})
            g = np.array(g)
            bands[f"{arm}|k{k}"] = {"mean": g.mean(0).tolist(), "p05": np.percentile(g, 5, 0).tolist(),
                                   "p95": np.percentile(g, 95, 0).tolist()}
    with open(out / "campaign_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    truth = {c: rec[c]["gap"] for c in pool}
    agree = {}
    for d, t in donors.items():
        cs = [c for c in pool if c in t]
        tt = kendalltau([t[c]["gap"] for c in cs], [truth[c] for c in cs]).statistic if len(cs) >= 3 else float("nan")
        diff = [t[c]["gap"] - truth[c] for c in cs]
        agree[d] = {"tau": float(tt), "coverage": len(cs) / len(pool), "n": len(cs),
                    "mean_shift_eV": float(np.mean(diff)), "mae_eV": float(np.mean(np.abs(diff))),
                    "donor_cost_core_s": float(sum(t[c]["cost"] for c in cs))}
    summary = {"receiver": args.receiver, "pool": args.pool, "cost": args.cost, "pbe_upfront_core_s": pbe_upfront,
               "n_pool": len(pool), "receiver_total_cost": total,
               "top10": sorted(pool, key=lambda c: -truth[c])[:10],
               "top10_gaps": sorted([truth[c] for c in pool], reverse=True)[:10],
               "top10_suggested": [c for c in sorted(pool, key=lambda c: -truth[c])[:10] if idx[c]["kind"] == "suggested"],
               "agreement": agree, "recall_bands": bands, "arms": {}}
    for k in ks:
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and r["k"] == k]
            summary["arms"][f"{arm}|k{k}"] = {m: {"mean": float(np.mean([r[m] for r in sel])),
                                                  "p05": float(np.percentile([r[m] for r in sel], 5)),
                                                  "p95": float(np.percentile([r[m] for r in sel], 95))}
                                              for m in ("early_recall_area", "cost_frac_to_8of10", "n_to_8of10")}
    pk = pegki(tabs, idx, args.receiver, list(donors), pool)
    gain = {d: summary["arms"][f"learned:{d}|k4"]["early_recall_area"]["mean"] for d in pk}
    if len(pk) >= 3:
        rho_s = spearmanr([pk[d]["delta"] for d in pk], [gain[d] for d in pk]).statistic
    else:
        rho_s = float("nan")
    summary["pegki"] = {"delta": pk, "learned_k4_early": gain, "spearman_delta_vs_early": float(rho_s),
                        "pick": max(pk, key=lambda d: pk[d]["delta"]) if pk else None,
                        "best_measured": max(gain, key=gain.get) if gain else None}
    if args.shapley:
        names = [d for d in donors if d in ("qe_pbe", "qe_hse06", "mp_gga", "jarvis_mbj", "snumat_hse", "expt_zhuo")]
        shap = {}
        for mode in ("pooled_fixed", "pooled_learned"):
            v = {}
            for r_ in range(len(names) + 1):
                for S in itertools.combinations(names, r_):
                    vals = []
                    for k in ks:
                        for s in range(args.seeds):
                            # combined cost: a coalition holding the PBE ordering pays the pool PBE
                            # upfront, then HSE06 per query; any other coalition pays HSE06 + PBE per query
                            pays_pbe = args.cost == "combined" and "qe_pbe" in S
                            r_tab = rec if (args.cost == "receiver" or pays_pbe) else rec_query
                            if not S:
                                res = camp.run_campaign("surrogate", r_tab, {}, pool, X, k, camp.crc(f"surrogate|{k}|{s}"))
                            else:
                                arm = mode if len(S) > 1 else ("fixed:" if mode == "pooled_fixed" else "learned:") + S[0]
                                res = camp.run_campaign(arm, r_tab, {d: donors[d] for d in S}, pool, X, k,
                                                        camp.crc(f"{mode}|{'+'.join(S)}|{k}|{s}"))
                            if pays_pbe:
                                res["curve"] = [{**p, "cost": p["cost"] + pbe_upfront} for p in res["curve"]]
                            vals.append(camp.summarise(res["curve"], total)["early_recall_area"])
                    v[frozenset(S)] = float(np.mean(vals))
            n = len(names)
            phi = {}
            for p in names:
                others = [q for q in names if q != p]
                phi[p] = sum(math.factorial(r_) * math.factorial(n - r_ - 1) / math.factorial(n)
                             * (v[frozenset(S) | {p}] - v[frozenset(S)])
                             for r_ in range(n) for S in itertools.combinations(others, r_))
            best = max(v, key=v.get)
            shap[mode] = {"shapley": phi, "v_empty": v[frozenset()], "v_all": v[frozenset(names)],
                          "best_coalition": sorted(best), "v_best": v[best]}
        summary["shapley"] = shap
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"n_pool": len(pool), "top10": summary["top10"],
                      "early": {a: round(v["early_recall_area"]["mean"], 3) for a, v in summary["arms"].items()},
                      "pegki": summary["pegki"], "agreement": agree,
                      "shapley": {m: {k: round(x, 3) for k, x in d["shapley"].items()}
                                  for m, d in summary.get("shapley", {}).items()}}, indent=1))


if __name__ == "__main__":
    main()
