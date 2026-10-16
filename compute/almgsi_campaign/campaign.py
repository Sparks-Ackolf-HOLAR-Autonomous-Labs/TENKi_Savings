"""Stage 5: discovery campaigns on the live Al-Mg-Si calculations.

Question: a group wants the most stable Al-Mg-Si orderings (lowest formation
energy per atom) at a high level of theory (the receiver: QE HSE06 or QE PBE),
and each receiver calculation costs measured CPU time. Lower-cost sources
(MACE-MP relaxation, MD with MACE / NEP89 / MEAM, the MP archive) have already
ranked the pool. How much receiver compute does borrowing their ordering save,
and does learned trust protect the campaign when a donor ranks badly?

Every campaign step evaluates a batch of k candidates at the receiver (a
k-way ranking event: the batch is ranked by the receiver values) and is charged
the measured cost of those calculations. Acquisition score for an unevaluated
candidate c (higher = evaluate sooner):

    score(c) = w * r_donor(c) + (1 - w) * r_surrogate(c)

* r_donor: the donor ordering as an evenly spaced rank score in [0, 1]
  (1 = lowest donor formation energy). Pooled donors average their scores
  with per-donor weights. A candidate the donor cannot answer (for example a
  suggested structure absent from the archive) gets the neutral score 0.5.
* r_surrogate: a composition-kernel estimate from the receiver values
  already measured (Nadaraya-Watson on the Al/Mg/Si fractions with a
  lower-confidence-bound term), rank-normalised. Before any data it is 0.5.
* w: 0 for the no-donor arms; 1 for fixed trust; for learned trust, the
  shrunken Kendall tau between donor and receiver on the evaluated
  candidates, max(0, tau) * n / (n + 8), starting at 0.5.

Arms: ``random``, ``surrogate`` (active learning without a donor),
``fixed:<donor>``, ``learned:<donor>``, ``pooled_fixed``, ``pooled_learned``.

Metrics per campaign, as functions of cumulative receiver cost: recall of
the receiver's 10 lowest-formation-energy candidates, and the Kendall tau of
the campaign's current ranking (measured values first, then unevaluated in
acquisition order) against the full receiver ranking. The full receiver
table is known because every candidate was eventually computed; campaigns
only see values they have paid for.

``--mode replay`` uses the tables in ``results/``. The live mode is in
``live_campaign.py``.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TOPN = 10
HSE_CORES = 32  # CHPC HSE06 single points (LIVE_CAMPAIGN.md)
PBE_CORES = 16  # CHPC PBE relaxations


# ---------------------------------------------------------------- data

def load_index() -> dict[str, dict]:
    return {r["cid"]: r for r in json.loads((HERE / "candidates_index.json").read_text(encoding="utf-8"))}


def fractions(formula_cid: str, index: dict) -> np.ndarray:
    from ase.io import read
    a = read(HERE / "candidates" / f"{formula_cid}.xyz")
    s = a.get_chemical_symbols()
    n = len(s)
    return np.array([s.count("Al") / n, s.count("Mg") / n, s.count("Si") / n])


def load_tables(extra: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """{source: {cid: {value, sem, cost}}} for formation-energy candidates.

    ``extra`` entries have the form ``path:source=name`` (a live_values-style
    file and the source inside it, loaded as ``name``) or ``path=name`` (a file
    that is itself one {cid: record} table, for example the CALPHAD or
    quasi-harmonic receivers)."""
    live = json.loads((RES / "live_values.json").read_text(encoding="utf-8"))
    for e in extra or []:
        spec, _, name = e.partition("=")
        path, _, src = spec.partition(":")
        d = json.loads((HERE / path).read_text(encoding="utf-8"))
        live[name] = d[src] if src else d
    idx = load_index()
    out: dict[str, dict[str, dict]] = {}
    for src, rows in live.items():
        t = {}
        for cid, r in rows.items():
            if cid not in idx or idx[cid]["property"] != "formation_energy":
                continue
            if r.get("status") != "ok" or r.get("value") is None or not math.isfinite(r["value"]):
                continue
            cost = r.get("core_s") or r.get("wall_s") or 0.0
            t[cid] = {"value": float(r["value"]), "sem": float(r.get("sem") or 0.0), "cost": float(cost)}
        if t:
            out[src] = t
    # archive: MP formation energies for the candidates MP holds (lookup cost 0)
    arch = {cid: {"value": float(r["mp_ef"]), "sem": 0.0, "cost": 0.0}
            for cid, r in idx.items() if r.get("mp_ef") is not None and r["property"] == "formation_energy"}
    out["archive_mp"] = arch
    return out


# ---------------------------------------------------------------- helpers

def rank_score(values: dict[str, float], pool: list[str]) -> dict[str, float]:
    """Evenly spaced rank score in [0, 1], 1 = lowest value; missing -> 0.5."""
    have = [c for c in pool if c in values]
    order = sorted(have, key=lambda c: (values[c], c))
    n = len(order)
    sc = {c: 0.5 for c in pool}
    for i, c in enumerate(order):
        sc[c] = 1.0 - i / max(1, n - 1)
    return sc


def tau(a: dict[str, float], b: dict[str, float], cids) -> float:
    cids = [c for c in cids if c in a and c in b]
    if len(cids) < 3:
        return float("nan")
    t = kendalltau([a[c] for c in cids], [b[c] for c in cids]).statistic
    return float(t) if t == t else float("nan")


def surrogate_scores(evald: dict[str, float], pool: list[str], X: dict[str, np.ndarray],
                     bw: float = 0.12, kappa: float = 0.03) -> dict[str, float]:
    todo = [c for c in pool if c not in evald]
    if not evald or not todo:
        return {c: 0.5 for c in todo}
    E = list(evald)
    XE = np.array([X[c] for c in E])
    yE = np.array([evald[c] for c in E])
    ybar = yE.mean()
    pred = {}
    for c in todo:
        d2 = ((XE - X[c]) ** 2).sum(1)
        w = np.exp(-d2 / (2 * bw * bw))
        sw = w.sum()
        mu = (w @ yE + 1.0 * ybar) / (sw + 1.0)
        unc = 1.0 / math.sqrt(sw + 1.0)
        pred[c] = mu - kappa * unc   # lower-confidence bound (lower E_f = better)
    return rank_score(pred, todo)


# ---------------------------------------------------------------- campaign

def run_campaign(arm: str, receiver: dict, donors: dict[str, dict], pool: list[str],
                 X: dict, k: int, seed: int, max_cost: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    jitter = {c: rng.random() * 1e-9 for c in pool}   # seeded tie-break only
    truth = {c: receiver[c]["value"] for c in pool}
    top = set(sorted(pool, key=lambda c: (truth[c], c))[:TOPN])
    kind, _, dname = arm.partition(":")
    use = list(donors) if kind.startswith("pooled") else ([dname] if dname else [])
    dval = {d: {c: donors[d][c]["value"] for c in pool if c in donors[d]} for d in use}
    dscore = {d: rank_score(dval[d], pool) for d in use}
    evald: dict[str, float] = {}
    cost = 0.0
    curve = []
    order_rand = list(rng.permutation(pool))
    while len(evald) < len(pool):
        todo = [c for c in pool if c not in evald]
        if kind == "random":
            batch = [c for c in order_rand if c not in evald][:k]
        else:
            sur = surrogate_scores(evald, pool, X)
            if use:
                wts = {}
                for d in use:
                    if kind.endswith("learned"):
                        n = len(evald)
                        t = tau(dval[d], evald, list(evald)) if n >= 3 else float("nan")
                        wts[d] = 0.5 if not math.isfinite(t) else max(0.0, t) * n / (n + 8)
                    else:
                        wts[d] = 1.0
                wsum = sum(wts.values())
                if kind.startswith("pooled"):
                    w = min(1.0, wsum / len(use)) if kind.endswith("learned") else 1.0
                    r_d = {c: (sum(wts[d] * dscore[d][c] for d in use) / wsum if wsum > 0 else 0.5) for c in todo}
                else:
                    w = wts[use[0]]
                    r_d = {c: dscore[use[0]][c] for c in todo}
            else:
                w, r_d = 0.0, {c: 0.5 for c in todo}
            s = {c: w * r_d[c] + (1 - w) * sur.get(c, 0.5) + jitter[c] for c in todo}
            batch = sorted(todo, key=lambda c: -s[c])[:k]
        for c in batch:
            evald[c] = truth[c]
            cost += receiver[c]["cost"]
        # current ranking: measured in value order, then unevaluated in random-arm order
        rest = [c for c in order_rand if c not in evald]
        ranking = sorted(evald, key=lambda c: evald[c]) + rest
        pos = {c: i for i, c in enumerate(ranking)}
        t_now = tau(pos, truth, pool)
        curve.append({"n": len(evald), "cost": cost, "recall": len(top & set(evald)) / TOPN, "tau": t_now})
        if max_cost is not None and cost >= max_cost:
            break
    return {"arm": arm, "k": k, "seed": seed, "curve": curve}


def summarise(curve: list[dict], total_cost: float, early_frac: float = 0.25) -> dict:
    """Early-window recall area and cost to reach 8 of the 10 lowest."""
    xs = [0.0] + [p["cost"] / total_cost for p in curve]
    ys = [0.0] + [p["recall"] for p in curve]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 >= early_frac:
            break
        x1c = min(x1, early_frac)
        area += y0 * (x1c - x0)   # step function: recall known after the batch completes
    area /= early_frac
    c8 = next((p["cost"] / total_cost for p in curve if p["recall"] >= 0.8), 1.0)
    n8 = next((p["n"] for p in curve if p["recall"] >= 0.8), len(curve))
    return {"early_recall_area": area, "cost_frac_to_8of10": c8, "n_to_8of10": n8}


def crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


def md_draw(table: dict, source: str, run: int) -> dict:
    """One MD-noise draw of a source table for run ``run``.

    Every value with a measured standard error (time-averaged MD energies) is
    drawn from a normal distribution centred on the time average with that
    standard error as its width; values without one (0 K relaxations, DFT,
    archive lookups) are returned unchanged. The draw depends only on
    (source, run), so every arm and coalition of one run sees the same values.
    The campaign seed itself only breaks exact ties (1e-9 jitter), so without
    these draws a campaign with a donor makes the same choices in every run."""
    if not any(r.get("sem", 0.0) > 0 for r in table.values()):
        return table
    rng = np.random.default_rng(crc(f"sem|{source}|{run}"))
    return {c: {**r, "value": r["value"] + (rng.normal(0.0, r["sem"]) if r.get("sem", 0.0) > 0 else 0.0)}
            for c, r in sorted(table.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receiver", default="qe_hse06")
    ap.add_argument("--donors", default="mace_relax,mace_md,gpumd_nep89_md,lammps_meam_md,archive_mp")
    ap.add_argument("--k", default="1,4")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--pool", default="all", choices=["all", "archived"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--shapley", action="store_true", help="also run all pooled donor coalitions")
    ap.add_argument("--cost", default="receiver", choices=["receiver", "combined"],
                    help="combined (HSE06 receiver only): charge each query its HSE06 plus PBE core-seconds "
                         "(the PBE relaxation gives the HSE06 geometry; HSE06 ran on 32 cores, PBE on 16); "
                         "arms and coalitions whose donors include qe_pbe pay PBE for the whole pool before "
                         "the first query. Donor compute of the potentials (under 0.01%%) is not charged")
    ap.add_argument("--no-md-noise", action="store_true",
                    help="use the time-averaged MD values in every run (the only mode before "
                         "the MD-noise rerun; the seed then only breaks ties, so donor runs are identical)")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra source, path:source=name or path=name (relative to this folder)")
    args = ap.parse_args()

    idx = load_index()
    tables = load_tables(args.extra)
    rec = tables[args.receiver]
    if all(r["cost"] <= 0 for r in rec.values()):
        # a lookup receiver (the archive) has no compute cost: charge one unit per query
        rec = {c: {**r, "cost": 1.0} for c, r in rec.items()}
    donors = {d: tables[d] for d in args.donors.split(",") if d in tables and d != args.receiver}
    pool = sorted(rec)
    if args.pool == "archived":
        pool = [c for c in pool if c in tables["archive_mp"]]
    X = {c: fractions(c, idx) for c in pool}
    # elemental structures have E_f = 0 (references) or small positive values (polymorphs);
    # they are not compounds to discover, so the pool holds binaries and ternaries only
    pool = [c for c in pool if (X[c] > 0).sum() >= 2]
    pbe_upfront = 0.0
    rec_query = rec
    if args.cost == "combined":
        if args.receiver != "qe_hse06":
            raise SystemExit("--cost combined is defined for the HSE06 receiver only")
        pbe = tables["qe_pbe"]
        missing = [c for c in pool if c not in pbe]
        if missing:
            raise SystemExit(f"no PBE record for {len(missing)} pool members, e.g. {missing[:3]}")
        # both in core-seconds: HSE06 wall time x 32 cores, PBE wall time x 16 cores
        rec = {c: {**rec[c], "cost": rec[c]["cost"] * HSE_CORES} for c in rec}
        rec_query = {c: {**rec[c], "cost": rec[c]["cost"] + pbe[c]["cost"] * PBE_CORES} for c in pool}
        pbe_upfront = sum(pbe[c]["cost"] * PBE_CORES for c in pool)
    total = sum(rec_query[c]["cost"] for c in pool)

    md_noise = not args.no_md_noise

    def draw(table, name, s):
        return md_draw(table, name, s) if md_noise else table

    def run(arm, sub, k, seed, s):
        """One campaign of run s; with combined cost, a donor set holding qe_pbe pays the pool PBE first.
        Receiver and donors are the run-s MD-noise draws (sources without a standard error are fixed)."""
        pays_pbe = args.cost == "combined" and "qe_pbe" in sub
        r_tab = rec if (args.cost == "receiver" or pays_pbe) else rec_query
        r_tab = draw(r_tab, args.receiver, s)
        sub = {d: draw(t, d, s) for d, t in sub.items()}
        r = run_campaign(arm, r_tab, sub, pool, X, k, seed)
        if pays_pbe:
            r["curve"] = [{**p, "cost": p["cost"] + pbe_upfront} for p in r["curve"]]
        return r
    arms = ["random", "surrogate"] + [f"{m}:{d}" for d in donors for m in ("fixed", "learned")] \
        + (["pooled_fixed", "pooled_learned"] if len(donors) > 1 else [])
    ks = [int(x) for x in args.k.split(",")]
    tag = args.tag or f"{args.receiver}_{args.pool}"
    out = RES / "campaigns" / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = np.linspace(0, 1, 201)
    bands = {}
    for k in ks:
        for arm in arms:
            gcurves = []
            for s in range(args.seeds):
                if arm in ("random", "surrogate"):
                    sub = {}
                elif arm.startswith("pooled"):
                    sub = donors
                else:
                    sub = {arm.split(":", 1)[1]: donors[arm.split(":", 1)[1]]}
                r = run(arm, sub, k, crc(f"{arm}|{k}|{s}"), s)
                sm = summarise(r["curve"], total)
                cx = np.array([0.0] + [p["cost"] / total for p in r["curve"]])
                cy = np.array([0.0] + [p["recall"] for p in r["curve"]])
                gcurves.append([float(cy[np.searchsorted(cx, g, side="right") - 1]) for g in grid])
                rows.append({"arm": arm, "k": k, "seed": s, **sm,
                             "tau_at_10pct": next((p["tau"] for p in r["curve"] if p["cost"] >= 0.1 * total), 1.0),
                             "tau_at_25pct": next((p["tau"] for p in r["curve"] if p["cost"] >= 0.25 * total), 1.0)})
                if s == 0:
                    (out / f"curve_{arm.replace(':', '_')}_k{k}.json").write_text(json.dumps(r["curve"]), encoding="utf-8")
            g = np.array(gcurves)
            bands[f"{arm}|k{k}"] = {"cost_frac": grid.tolist(), "mean": g.mean(0).tolist(),
                                   "p05": np.percentile(g, 5, 0).tolist(), "p95": np.percentile(g, 95, 0).tolist()}
    with open(out / "campaign_runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # donor-receiver agreement on the pool (PEGKi view)
    agree = {d: {"tau_full": tau({c: v["value"] for c, v in donors[d].items()},
                                 {c: rec[c]["value"] for c in pool}, pool),
                 "coverage": sum(c in donors[d] for c in pool) / len(pool),
                 "donor_total_cost": sum(donors[d][c]["cost"] for c in pool if c in donors[d])}
             for d in donors}

    # ATEKi: exact Shapley over donors for pooled_fixed / pooled_learned value
    shap = {}
    coal = {}
    if args.shapley and len(donors) >= 2:
        names = list(donors)
        for mode in ("pooled_fixed", "pooled_learned"):
            v = {}
            per_run = {}
            for r_ in range(len(names) + 1):
                for S in itertools.combinations(names, r_):
                    vals = []
                    for k in ks:
                        for s in range(args.seeds):
                            if not S:
                                arm = "surrogate"
                                res = run(arm, {}, k, crc(f"{arm}|{k}|{s}"), s)
                            else:
                                sub = {d: donors[d] for d in S}
                                arm = mode if len(S) > 1 else ("fixed:" if mode == "pooled_fixed" else "learned:") + S[0]
                                res = run(arm, sub, k, crc(f"{mode}|{'+'.join(S)}|{k}|{s}"), s)
                            vals.append(summarise(res["curve"], total)["early_recall_area"])
                    v[frozenset(S)] = float(np.mean(vals))
                    per_run["+".join(sorted(S))] = vals
            n = len(names)
            phi = {}
            for p in names:
                acc = 0.0
                others = [q for q in names if q != p]
                for r_ in range(n):
                    for S in itertools.combinations(others, r_):
                        wgt = math.factorial(r_) * math.factorial(n - r_ - 1) / math.factorial(n)
                        acc += wgt * (v[frozenset(S) | {p}] - v[frozenset(S)])
                phi[p] = acc
            shap[mode] = {"shapley": phi, "v_empty": v[frozenset()], "v_all": v[frozenset(names)],
                          "best_coalition": max(v, key=v.get) and sorted(max(v, key=v.get)),
                          "v_best": max(v.values())}
            # every coalition value and its per-run values (order: k outer, seed inner), so
            # other semivalues (Banzhaf, Beta Shapley, LOO) can be computed without new campaigns
            coal[mode] = {"players": names, "runs": [f"k{k}|s{s}" for k in ks for s in range(args.seeds)],
                          "value": {"+".join(sorted(S)): x for S, x in v.items()}, "per_run": per_run}
    if coal:
        (out / "coalitions.json").write_text(json.dumps(coal, indent=1), encoding="utf-8")

    summary = {"receiver": args.receiver, "pool": args.pool, "cost": args.cost, "pbe_upfront": pbe_upfront,
               "md_noise": md_noise,
               "n_pool": len(pool),
               "n_top": TOPN, "top_candidates": sorted(pool, key=lambda c: rec[c]["value"])[:TOPN],
               "receiver_total_cost": total, "ks": ks, "seeds": args.seeds,
               "agreement": agree, "shapley": shap, "recall_bands": bands, "arms": {}}
    for k in ks:
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and r["k"] == k]
            summary["arms"][f"{arm}|k{k}"] = {m: {"mean": float(np.mean([r[m] for r in sel])),
                                                  "p05": float(np.percentile([r[m] for r in sel], 5)),
                                                  "p95": float(np.percentile([r[m] for r in sel], 95))}
                                              for m in ("early_recall_area", "cost_frac_to_8of10", "n_to_8of10",
                                                        "tau_at_10pct", "tau_at_25pct")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({a: round(v["early_recall_area"]["mean"], 3) for a, v in summary["arms"].items()}, indent=0))


if __name__ == "__main__":
    main()
