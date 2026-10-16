#!/usr/bin/env python
"""Temperature study: a context shift inside one source, and what PEGKi,
TENKi and ATEKi do with it.

Setting. The live Al-Mg-Si pool (193 candidates: archived MP structures,
suggested fcc orderings, Mg/Si solute clusters in Al) was computed with three
MD engines (LAMMPS MEAM, GPUMD NEP89, MACE-MP-0) at three temperatures taken
from 6xxx heat treatment:

    300 K  natural ageing
    453 K  artificial ageing (180 C, T6)
    803 K  solution treatment (530 C)

plus the 0 K sources (MACE-MP-0 relaxation, QE PBE, QE HSE06, MP archive).
A source is "engine@T"; the same engine at another temperature is the same
instrument in a different processing environment, which is the context shift
the retrospective band-gap archive cannot provide.

Property: formation energy per atom from time-averaged potential energies at
T (references at the same T and from the same engine), so it contains thermal
expansion and anharmonic relaxation but no vibrational or configurational
entropy. Cluster binding energies are reported alongside.

Analyses (all read the committed per-query results; nothing is invented):

  A  size of the shift: Kendall tau and top-10 overlap between engine@300 K
     and engine@T, coverage lost at T, and the candidates whose rank moves most
  B  PEGKi: directed transfer scores Delta(A -> engine@T) = sigma + rho - omega
     from Plackett-Luce ratings of composition classes (the formula of an
     ordering is its class; its orderings are the class's records), computed
     before any hot MD is used by a campaign
  C  TENKi: discovery campaigns at 453 K and 803 K (receiver engine@T, 4-way
     batches, cost = measured MD seconds) that borrow a donor ordering with
     fixed or learned trust, pooled or single; recall of the 10 most stable
     candidates in the early quarter of the receiver budget, and budget to
     find 8 of them
  D  ATEKi: exact Shapley credit of the donors in the pooled learned-trust
     campaign (every coalition is a campaign that was actually run)

Usage (shared uv env; no ase needed):
    cd ../../shared && uv run python ../compute/almgsi_campaign/temperature_study.py
Outputs: results_T/analysis/temperature_study.{json,md} and
results_T/analysis/temperature_study.png
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
SO = HERE.parents[1]
sys.path.insert(0, str(SO / "shared"))

TEMPS = (300, 453, 803)
HOT = (453, 803)
ENGINES = {"lammps_meam_md": "MEAM (LAMMPS)", "gpumd_nep89_md": "NEP89 (GPUMD)",
           "mace_md": "MACE-MP-0 MD"}
ZERO_K = {"mace_relax": "MACE-MP-0 relax", "qe_pbe": "QE PBE", "qe_hse06": "QE HSE06"}
T_LABEL = {300: "300 K (natural ageing)", 453: "453 K (T6 ageing)", 803: "803 K (solution treatment)"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


camp = _load("live_campaign_core", HERE / "campaign.py")
agg = _load("live_aggregate", HERE / "aggregate_sources.py")


def label(src: str) -> str:
    eng, _, t = src.partition("@")
    return f"{ENGINES.get(eng) or ZERO_K.get(eng) or eng} @ {t}"


# ----------------------------------------------------------------- data

def aggregate(tdir: Path):
    subprocess.run([sys.executable, str(HERE / "aggregate_sources.py"), "--results-dir", str(tdir)],
                   check=True, capture_output=True)


def load_sources(results_t: Path, reaggregate: bool = True):
    """{source: {cid: {value, sem, cost, property}}} for all engine@T and 0 K sources."""
    idx = camp.load_index()
    out = {}
    for T in TEMPS:
        tdir = results_t / f"T{T}"
        if not tdir.exists():
            continue
        if reaggregate:
            aggregate(tdir)
        lv = json.loads((tdir / "live_values.json").read_text(encoding="utf-8"))
        for eng, rows in lv.items():
            if eng not in ENGINES:
                continue
            out[f"{eng}@{T}K"] = rows
    lv0 = json.loads((HERE / "results" / "live_values.json").read_text(encoding="utf-8"))
    for eng in ZERO_K:
        if eng in lv0:
            out[f"{eng}@0K"] = lv0[eng]
    clean = {}
    for src, rows in out.items():
        t = {}
        for cid, r in rows.items():
            if cid.startswith("_") or cid not in idx:
                continue
            if r.get("status") != "ok" or r.get("value") is None or not math.isfinite(r["value"]):
                continue
            t[cid] = {"value": float(r["value"]), "sem": float(r.get("sem") or 0.0),
                      "cost": float(r.get("wall_s") or 0.0), "property": r["property"]}
        clean[src] = t
    return clean, idx


def formation(tab):
    return {c: v for c, v in tab.items() if v["property"] == "formation_energy"}


def fractions_noase(cid):
    comp, n = agg.composition(cid)
    return np.array([comp.get("Al", 0) / n, comp.get("Mg", 0) / n, comp.get("Si", 0) / n])


def formula_of(cid, idx):
    return idx[cid]["formula"]


# ----------------------------------------------------------------- A: shift size

def shift_stats(src, idx):
    out = {}
    for eng in ENGINES:
        base = src.get(f"{eng}@300K")
        if not base:
            continue
        b = formation(base)
        n300 = len(b)
        for T in HOT:
            hot = src.get(f"{eng}@{T}K")
            if not hot:
                continue
            h = formation(hot)
            common = sorted(set(b) & set(h))
            if len(common) < 5:
                continue
            x = [b[c]["value"] for c in common]
            y = [h[c]["value"] for c in common]
            top_b = set(sorted(common, key=lambda c: b[c]["value"])[:10])
            top_h = set(sorted(common, key=lambda c: h[c]["value"])[:10])
            rb = {c: i for i, c in enumerate(sorted(common, key=lambda c: b[c]["value"]))}
            rh = {c: i for i, c in enumerate(sorted(common, key=lambda c: h[c]["value"]))}
            movers = sorted(common, key=lambda c: -abs(rb[c] - rh[c]))[:6]
            lost = sorted(set(b) - set(h))
            out[f"{eng}@{T}K"] = {
                "n_common": len(common), "n_at_300K": n300, "n_at_T": len(h),
                "lost_at_T": len(lost), "lost_formulas": dict(Counter(formula_of(c, idx) for c in lost)),
                "tau_vs_300K": float(kendalltau(x, y).statistic),
                "spearman_vs_300K": float(spearmanr(x, y).statistic),
                "top10_overlap": len(top_b & top_h),
                "spread_ratio": float(np.std(y) / np.std(x)),
                "mean_shift_eV_atom": float(np.mean(np.array(y) - np.array(x))),
                "median_abs_shift_eV_atom": float(np.median(np.abs(np.array(y) - np.array(x)))),
                "largest_rank_moves": [{"cid": c, "formula": formula_of(c, idx), "rank_300K": rb[c] + 1,
                                        "rank_T": rh[c] + 1} for c in movers],
            }
    return out


def cluster_table(src):
    rows = {}
    for s, tab in src.items():
        for cid, v in tab.items():
            if v["property"] == "cluster_binding":
                rows.setdefault(cid, {})[s] = {"value": v["value"], "sem": v["sem"]}
    return rows


# ----------------------------------------------------------------- B: PEGKi

def pegki_spaces(src, idx, n_events=400, n_boot=20, seed=11, min_size=2):
    setop = _load("set_operation_study", SO / "shared" / "scripts" / "set_operation_study.py")
    spaces = {}
    for s, tab in src.items():
        classes = {}
        for cid, v in formation(tab).items():
            # higher is better in the rating layer: use -E_f (more stable = stronger)
            classes.setdefault(formula_of(cid, idx), []).append(-v["value"])
        classes = {c: sorted(v) for c, v in classes.items() if len(v) >= min_size}
        if len(classes) < 3:
            continue
        ks, _ = setop.fit_space(s, classes, n_events, n_boot, seed)
        if ks is not None:
            spaces[s] = ks
    return spaces, setop.ltw


def pegki_deltas(spaces, ltw, receivers):
    D = {}
    for b in receivers:
        if b not in spaces:
            continue
        for a in spaces:
            if a == b:
                continue
            comp = [ltw.compute_structural_compatibility(spaces[a], spaces[b]),
                    ltw.compute_mapping_efficiency(spaces[a], spaces[b]),
                    ltw.compute_transfer_overhead(spaces[a], spaces[b])]
            D[f"{a}->{b}"] = {"sigma": float(comp[0]), "rho": float(comp[1]), "omega": float(comp[2]),
                              "delta": float(comp[0] + comp[1] - comp[2])}
    return D


def _pegki_draw(args):
    """Transfer scores on MD-noise draw j (j = None: the time-averaged values)."""
    src, idx, receivers, j = args
    srcj = src if j is None else {s_: camp.md_draw(tab, s_, j) for s_, tab in src.items()}
    sp, ltw = pegki_spaces(srcj, idx)
    print(f"[pegki] fit {'time-averaged' if j is None else f'draw {j}'} done", flush=True)
    if j is None:
        return {"classes": {s_: sorted(ks.policy_mus) for s_, ks in sp.items()},
                "deltas": pegki_deltas(sp, ltw, receivers)}
    return {key: d["delta"] for key, d in pegki_deltas(sp, ltw, receivers).items()}


def collect_noise(res):
    per = {}
    for r in res:
        for key, d in r.items():
            per.setdefault(key, []).append(d)
    return {key: {"mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)),
                  "p95": float(np.percentile(v, 95)), "draws": [float(x) for x in v]} for key, v in per.items()}


# ----------------------------------------------------------------- C/D: TENKi and ATEKi

def donors_for(rec, src):
    eng, _, t = rec.partition("@")
    T = int(t[:-1])
    d = [f"{eng}@300K"]
    d += [f"{e}@{T}K" for e in ENGINES if e != eng]
    d += ["mace_relax@0K", "qe_pbe@0K", "qe_hse06@0K"]
    return [x for x in d if x in src and len(formation(src[x])) >= 20]


GRID = np.linspace(0, 1, 41)
MD_NOISE = True   # draw every MD value from its measured standard error per run (campaign.md_draw)


def draw(table, name, s):
    return camp.md_draw(table, name, s) if MD_NOISE else table


def run_arm(arm, rec, donors, pool, X, k, seeds, total, rec_name="receiver"):
    vals, curves = [], []
    for s in range(seeds):
        r = camp.run_campaign(arm, draw(rec, rec_name, s), {d: draw(t, d, s) for d, t in donors.items()},
                              pool, X, k, camp.crc(f"T|{arm}|{k}|{s}"))
        vals.append(camp.summarise(r["curve"], total))
        cx = np.array([0.0] + [p["cost"] / total for p in r["curve"]])
        cy = np.array([0.0] + [p["recall"] for p in r["curve"]])
        curves.append([float(cy[np.searchsorted(cx, g, side="right") - 1]) for g in GRID])
    g = np.array(curves)
    band = {"cost_frac": GRID.tolist(), "mean": g.mean(0).tolist(),
            "p05": np.percentile(g, 5, 0).tolist(), "p95": np.percentile(g, 95, 0).tolist()}
    return {"recall_band": band, **{m: {"mean": float(np.mean([v[m] for v in vals])),
                "p05": float(np.percentile([v[m] for v in vals], 5)),
                "p95": float(np.percentile([v[m] for v in vals], 95))}
            for m in ("early_recall_area", "cost_frac_to_8of10")}}


def tau_matrix(src):
    names = sorted(src, key=lambda s: (s.split("@")[0] not in ENGINES, s.split("@")[0],
                                       int(s.split("@")[1][:-1])))
    M = {}
    for a in names:
        for b in names:
            fa, fb = formation(src[a]), formation(src[b])
            common = sorted(set(fa) & set(fb))
            M[f"{a}|{b}"] = float(kendalltau([fa[c]["value"] for c in common],
                                             [fb[c]["value"] for c in common]).statistic) if len(common) >= 5 else None
    return {"order": names, "tau": M}


def tenki(src, receivers, k, seeds):
    out = {}
    for rec_name in receivers:
        if rec_name not in src:
            continue
        rec = formation(src[rec_name])
        pool = sorted(rec)
        if len(pool) < 40:
            continue
        X = {c: fractions_noase(c) for c in pool}
        total = sum(rec[c]["cost"] for c in pool)
        dn = donors_for(rec_name, src)
        donors = {d: formation(src[d]) for d in dn}
        arms = ["random", "surrogate"] + [f"{m}:{d}" for d in dn for m in ("fixed", "learned")] \
            + ["pooled_fixed", "pooled_learned"]
        res = {a: run_arm(a, rec, donors, pool, X, k, seeds, total, rec_name) for a in arms}
        agree = {d: camp.tau({c: v["value"] for c, v in donors[d].items()},
                             {c: rec[c]["value"] for c in pool}, pool) for d in dn}
        out[rec_name] = {"n_pool": len(pool), "receiver_total_s": total, "donors": dn,
                         "tau_donor_receiver": agree, "arms": res,
                         "top10": sorted(pool, key=lambda c: rec[c]["value"])[:10]}
        print(f"[tenki] {rec_name}: cold {res['surrogate']['early_recall_area']['mean']:.3f} "
              f"pooled_learned {res['pooled_learned']['early_recall_area']['mean']:.3f}", flush=True)
    return out


def ateki(src, receivers, k, seeds, mode="pooled_learned"):
    out = {}
    for rec_name in receivers:
        if rec_name not in src:
            continue
        rec = formation(src[rec_name])
        pool = sorted(rec)
        if len(pool) < 40:
            continue
        X = {c: fractions_noase(c) for c in pool}
        total = sum(rec[c]["cost"] for c in pool)
        names = donors_for(rec_name, src)
        v = {}
        for r_ in range(len(names) + 1):
            for S in itertools.combinations(names, r_):
                vals = []
                for s in range(seeds):
                    rec_s = draw(rec, rec_name, s)
                    if not S:
                        res = camp.run_campaign("surrogate", rec_s, {}, pool, X, k, camp.crc(f"T|surrogate|{k}|{s}"))
                    else:
                        sub = {d: draw(formation(src[d]), d, s) for d in S}
                        arm = mode if len(S) > 1 else "learned:" + S[0]
                        res = camp.run_campaign(arm, rec_s, sub, pool, X, k,
                                                camp.crc(f"T|{mode}|{'+'.join(S)}|{k}|{s}"))
                    vals.append(camp.summarise(res["curve"], total)["early_recall_area"])
                v[frozenset(S)] = float(np.mean(vals))
        n = len(names)
        phi = {}
        for p in names:
            others = [q for q in names if q != p]
            acc = 0.0
            for r_ in range(n):
                for S in itertools.combinations(others, r_):
                    w = math.factorial(r_) * math.factorial(n - r_ - 1) / math.factorial(n)
                    acc += w * (v[frozenset(S) | {p}] - v[frozenset(S)])
            phi[p] = acc
        best = max(v, key=v.get)
        out[rec_name] = {"players": names, "shapley": phi, "v_empty": v[frozenset()],
                         "v_all": v[frozenset(names)], "best_coalition": sorted(best), "v_best": v[best],
                         "efficiency_gap": float(sum(phi.values()) - (v[frozenset(names)] - v[frozenset()]))}
        print(f"[ateki] {rec_name}: " + ", ".join(f"{p} {phi[p]:+.3f}" for p in names), flush=True)
    return out


# ----------------------------------------------------------------- report

ARM_LABEL = {"random": "random order", "surrogate": "no donor (composition surrogate, cold start)",
             "pooled_fixed": "all donors pooled, fixed trust", "pooled_learned": "all donors pooled, learned trust"}


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def write_md(res, path):
    L = ["# Temperature study: context shift inside one source", "",
         "Sources are engine@T. Formation energies are per-engine, per-temperature time-averaged",
         "potential-energy formation energies (no entropy). Generated by `temperature_study.py`.", ""]
    L += ["## A. Size of the shift from 300 K", "",
          "| Source | n common | lost at T | Kendall tau vs 300 K | top-10 overlap | spread ratio | median abs shift (eV/atom) |",
          "|---|---|---|---|---|---|---|"]
    for s, r in res["shift"].items():
        L.append(f"| {label(s)} | {r['n_common']} | {r['lost_at_T']} | {r['tau_vs_300K']:+.3f} | "
                 f"{r['top10_overlap']}/10 | {r['spread_ratio']:.2f} | {r['median_abs_shift_eV_atom']:.4f} |")
    L += ["", "## B. PEGKi directed transfer scores into the hot receivers", "",
          "| Donor -> receiver | sigma | rho | omega | Delta |", "|---|---|---|---|---|"]
    for key, d in sorted(res["pegki"].items(), key=lambda kv: (kv[0].split("->")[1], -kv[1]["delta"])):
        a, b = key.split("->")
        L.append(f"| {label(a)} -> {label(b)} | {d['sigma']:+.3f} | {d['rho']:+.3f} | {d['omega']:+.3f} | {d['delta']:+.3f} |")
    L += ["", "## C. TENKi campaigns at the hot temperatures", ""]
    for rec, r in res["tenki"].items():
        cold = r["arms"]["surrogate"]
        L += [f"### Receiver {label(rec)} ({r['n_pool']} candidates, {r['receiver_total_s']:.0f} s of MD in total)", "",
              "| Arm | early recall area (5-95%) | budget fraction to 8 of 10 | tau donor vs receiver |", "|---|---|---|---|"]
        for arm, m in r["arms"].items():
            d = arm.partition(":")[2]
            t = r["tau_donor_receiver"].get(d)
            name = ARM_LABEL.get(arm) or f"{arm.split(':')[0]} trust, {label(d)}"
            L.append(f"| {name} | "
                     f"{m['early_recall_area']['mean']:.3f} ({m['early_recall_area']['p05']:.3f}-{m['early_recall_area']['p95']:.3f}) | "
                     f"{m['cost_frac_to_8of10']['mean']:.3f} | {fmt(t) if d else ''} |")
        L.append("")
    L += ["## D. ATEKi Shapley credit of the donors (pooled learned trust)", ""]
    for rec, r in res["ateki"].items():
        L += [f"### Receiver {label(rec)}", "", f"v(no donor) = {r['v_empty']:.3f}, v(all donors) = {r['v_all']:.3f}, "
              f"best coalition {', '.join(label(x) for x in r['best_coalition'])} = {r['v_best']:.3f}; "
              f"efficiency gap {r['efficiency_gap']:.1e}", "", "| Donor | Shapley |", "|---|---|"]
        for p, v in sorted(r["shapley"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {label(p)} | {v:+.4f} |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-t", default=str(HERE / "results_T"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--no-reaggregate", action="store_true")
    ap.add_argument("--skip-ateki", action="store_true")
    ap.add_argument("--no-md-noise", action="store_true",
                    help="time-averaged MD values in every run (the mode before the MD-noise rerun)")
    ap.add_argument("--pegki-draws", type=int, default=20,
                    help="MD-noise draws for the transfer scores (0 = skip)")
    ap.add_argument("--workers", type=int, default=10, help="processes for the transfer-score draws")
    args = ap.parse_args()
    global MD_NOISE
    MD_NOISE = not args.no_md_noise
    rt = Path(args.results_t)
    src, idx = load_sources(rt, reaggregate=not args.no_reaggregate)
    print({s: len(formation(t)) for s, t in src.items()}, flush=True)
    receivers = [f"{e}@{T}K" for e in ENGINES for T in HOT if f"{e}@{T}K" in src]
    res = {"sources": {s: {"n_formation": len(formation(t)),
                           "n_cluster": sum(v["property"] == "cluster_binding" for v in t.values()),
                           "median_query_s": float(np.median([v["cost"] for v in t.values()])) if t else None}
                       for s, t in src.items()},
           "shift": shift_stats(src, idx), "clusters": cluster_table(src), "tau_matrix": tau_matrix(src)}
    # PEGKi fits (time-averaged values and the MD-noise draws) run in worker processes
    # while this process runs the TENKi and ATEKi campaigns
    from multiprocessing import Pool
    jobs = [(src, idx, receivers, None)] + [(src, idx, receivers, j) for j in range(args.pegki_draws)]
    with Pool(min(args.workers, len(jobs))) as pool:
        fits = pool.map_async(_pegki_draw, jobs, chunksize=1)
        res["md_noise"] = MD_NOISE
        res["tenki"] = tenki(src, receivers, args.k, args.seeds)
        res["ateki"] = {} if args.skip_ateki else ateki(src, receivers, args.k, args.seeds)
        fits = fits.get()
    res["pegki_classes"] = fits[0]["classes"]
    res["pegki"] = fits[0]["deltas"]
    res["pegki_md_noise"] = collect_noise(fits[1:]) if args.pegki_draws else {}
    # PEGKi as a router: does the top-Delta donor match the measured best single donor?
    route = {}
    for rec, r in res["tenki"].items():
        cand = {d: res["pegki"].get(f"{d}->{rec}", {}).get("delta") for d in r["donors"]}
        cand = {d: x for d, x in cand.items() if x is not None}
        if not cand:
            continue
        pick = max(cand, key=cand.get)
        gains = {d: r["arms"][f"learned:{d}"]["early_recall_area"]["mean"] for d in r["donors"]}
        c8 = {d: r["arms"][f"learned:{d}"]["cost_frac_to_8of10"]["mean"] for d in r["donors"]}
        c8_cold = r["arms"]["surrogate"]["cost_frac_to_8of10"]["mean"]
        route[rec] = {"pegki_pick": pick, "pick_value": gains[pick], "best_measured": max(gains, key=gains.get),
                      "best_value": max(gains.values()),
                      "cold_value": r["arms"]["surrogate"]["early_recall_area"]["mean"],
                      "budget_to_8of10_cold": c8_cold, "budget_to_8of10_pick": c8[pick],
                      "md_seconds_saved_by_pick": (c8_cold - c8[pick]) * r["receiver_total_s"],
                      "spearman_delta_vs_gain": float(spearmanr([cand[d] for d in cand], [gains[d] for d in cand]).statistic)
                      if len(cand) >= 3 else None}
    # the same question on MD-noise draws of the scores: how often is each donor the top-Delta pick
    for rec in route:
        pn = {d: res["pegki_md_noise"].get(f"{d}->{rec}") for d in res["tenki"][rec]["donors"]}
        pn = {d: x for d, x in pn.items() if x}
        if not pn:
            continue
        nd = len(next(iter(pn.values()))["draws"])
        picks = Counter(max(pn, key=lambda d: pn[d]["draws"][j]) for j in range(nd))
        route[rec]["pick_share_over_md_draws"] = {d: n / nd for d, n in picks.most_common()}
        route[rec]["best_measured_pick_share"] = picks.get(route[rec]["best_measured"], 0) / nd
    res["routing"] = route
    out = rt / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "temperature_study.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, out / "temperature_study.md")
    print(json.dumps(route, indent=1))


if __name__ == "__main__":
    main()
