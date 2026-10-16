"""Direct comparison with PRISMS on the computed Al-Mg-Si campaign.

PRISMS (Wijaya et al., arXiv 2610.01827, 2026) screens a candidate pool with
pairwise rankings from several rankers whose fidelity order is given in
advance. Its loop, reimplemented here from the paper (Sections 3-4,
Appendices B.2 and C.2):

* latent value z_i per unmeasured candidate (higher = better; here
  z = -E_f), measured candidates fixed at their receiver value;
* p_ij = sigmoid((z_i - z_j) / tau);
* loss = sum_s lambda1[s] * sum_{pairs ranked by s} BCE(r_ij, p_ij)
         + lambda2 * sum_i (z_i - theta_i)^2,
  theta_i from a random-forest regressor on the measured candidates
  (composition fractions as features);
* comparison graph per round: every unmeasured candidate is joined to all
  measured candidates and to its kU nearest unmeasured candidates in theta;
* ranker g1 ranks every eligible pair; ranker g_l (l > 1) ranks the b_l pairs
  with the largest information gain
  S_ij = log(1 + lambda1[l] / tau^2 * p(1-p) * v^T H^-1 v),
  H the Hessian of the loss at the current estimate, v the incidence vector;
* the k unmeasured candidates with the largest estimate are sent to the
  receiver (greedy top estimate).

Settings: lambda1 = 0.1 for g1 and 10 for g2, g3 (inside the range the paper
reports as best), lambda2 = 0.1, kU = 5, b2 = 100, b3 = 50 pairs per round,
L-BFGS-B with an analytic gradient (300 iterations). tau is the standard
deviation of the measured receiver values (the paper fits tau from measured
designs without stating the procedure). The first batch is random, because
the campaigns here start without measured candidates.

Rankers are the donor potentials of the campaign; a ranker answers a pair by
comparing its own formation energies (time-averaged MD values carry the same
per-run noise draws as ``campaign.py``). PRISMS needs their fidelity order
as an input, so three orders are run:

* ``cost``: ascending median cost per query (MACE-MP-0 relaxation 0.8 s,
  MEAM MD 1.2 s, NEP89 MD 2.2 s), the order a group would assume from price;
* ``class``: classical MEAM < NEP89 < MACE-MP-0, the order a group would
  assume from the model class;
* ``pair``: MEAM MD < MACE-MP-0 only (the two rankers that agree with the
  receiver, without NEP89), a two-fidelity setup a group could choose after a
  first look at the data;
* ``measured``: ascending Kendall tau with the receiver over the whole pool,
  which is known only after the receiver has computed every candidate
  (an upper reference for PRISMS, not a usable setting).

The comparison arms come from ``campaign.run_campaign`` on the same pool,
seeds, and noise draws: surrogate only, the PEGKi-routed donor (MACE-MP-0
relaxation) with learned trust, and the three potentials pooled with learned
trust. Metric as in ``campaign.summarise``: early recall area (recall of the
receiver's 10 lowest candidates over the first 25% of receiver cost) and the
receiver-cost fraction needed to find 8 of the 10.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor

from campaign import (RES, TOPN, crc, fractions, load_index, load_tables, md_draw,
                      run_campaign, summarise)

RANKERS = ["mace_relax", "lammps_meam_md", "gpumd_nep89_md"]
LAMBDA1 = [0.1, 10.0, 10.0]
LAMBDA2 = 0.1
KU = 5
BUDGETS = [None, 100, 50]


def fit(zfix: dict, unm: list, pairs: list, theta: np.ndarray, tau_: float, z0: np.ndarray):
    """L-BFGS-B fit of the unmeasured latent values; pairs = (i, j, r, lam)
    with i, j keys into the unmeasured index (int) or measured (str)."""
    ui = {c: n for n, c in enumerate(unm)}
    I = np.array([ui.get(i, -1) for i, _, _, _ in pairs], dtype=int)
    J = np.array([ui.get(j, -1) for _, j, _, _ in pairs], dtype=int)
    FI = np.array([zfix.get(i, 0.0) for i, _, _, _ in pairs])
    FJ = np.array([zfix.get(j, 0.0) for _, j, _, _ in pairs])
    R = np.array([r for _, _, r, _ in pairs], dtype=float)
    L = np.array([l for _, _, _, l in pairs], dtype=float)

    def f(z):
        zi = np.where(I >= 0, z[np.maximum(I, 0)], FI)
        zj = np.where(J >= 0, z[np.maximum(J, 0)], FJ)
        u = (zi - zj) / tau_
        p = 1.0 / (1.0 + np.exp(-u))
        eps = 1e-12
        loss = float(np.sum(L * -(R * np.log(p + eps) + (1 - R) * np.log(1 - p + eps))))
        loss += LAMBDA2 * float(np.sum((z - theta) ** 2))
        g = L * (p - R) / tau_
        grad = 2 * LAMBDA2 * (z - theta)
        np.add.at(grad, I[I >= 0], g[I >= 0])
        np.subtract.at(grad, J[J >= 0], g[J >= 0])
        return loss, grad

    res = minimize(f, z0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
    z = res.x
    # Hessian of the loss at z
    zi = np.where(I >= 0, z[np.maximum(I, 0)], FI)
    zj = np.where(J >= 0, z[np.maximum(J, 0)], FJ)
    p = 1.0 / (1.0 + np.exp(-(zi - zj) / tau_))
    w = L * p * (1 - p) / tau_ ** 2
    n = len(unm)
    H = 2 * LAMBDA2 * np.eye(n)
    a, b = I >= 0, J >= 0
    np.add.at(H, (I[a], I[a]), w[a])
    np.add.at(H, (J[b], J[b]), w[b])
    ab = a & b
    np.add.at(H, (I[ab], J[ab]), -w[ab])
    np.add.at(H, (J[ab], I[ab]), -w[ab])
    return z, H


def prisms_campaign(receiver: dict, rankers: list[tuple[str, dict]], pool: list, X: dict,
                    k: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    truth = {c: receiver[c]["value"] for c in pool}
    top = set(sorted(pool, key=lambda c: (truth[c], c))[:TOPN])
    measured: dict[str, float] = {}
    cost = 0.0
    curve = []
    queries = [0] * len(rankers)
    first = list(rng.permutation(pool))[:k]
    batch = first
    while True:
        for c in batch:
            measured[c] = truth[c]
            cost += receiver[c]["cost"]
        curve.append({"n": len(measured), "cost": cost, "recall": len(top & set(measured)) / TOPN})
        unm = [c for c in pool if c not in measured]
        if not unm:
            break
        zfix = {c: -v for c, v in measured.items()}
        ys = np.array(list(zfix.values()))
        tau_ = max(float(ys.std()), 1e-3) if len(ys) > 1 else 0.05
        if len(measured) >= 3:
            rf = RandomForestRegressor(n_estimators=100, random_state=seed % 2**31, n_jobs=1)
            rf.fit(np.array([X[c] for c in measured]), ys)
            theta = rf.predict(np.array([X[c] for c in unm]))
        else:
            theta = np.full(len(unm), float(ys.mean()))
        # comparison graph
        order = np.argsort(theta)
        edges = set()
        for c in unm:
            for m in measured:
                edges.add((c, m))
        pos = {unm[o]: r for r, o in enumerate(order)}
        for c in unm:
            r = pos[c]
            nb = sorted(range(max(0, r - KU), min(len(unm), r + KU + 1)),
                        key=lambda q: abs(theta[order[q]] - theta[order[r]]))
            for q in [q for q in nb if q != r][:KU]:
                d = unm[order[q]]
                edges.add(tuple(sorted((c, d))))
        edges = sorted(edges)
        pairs = []
        z = theta.copy()
        H = None
        ui = {c: n for n, c in enumerate(unm)}
        for s, (name, tab) in enumerate(rankers):
            ok = [(i, j) for i, j in edges if i in tab and j in tab]
            if s == 0:
                chosen = ok
            else:
                p_ = {}
                Hinv = np.linalg.inv(H)
                scores = []
                for i, j in ok:
                    v = np.zeros(len(unm))
                    if i in ui:
                        v[ui[i]] += 1
                    if j in ui:
                        v[ui[j]] -= 1
                    zi = z[ui[i]] if i in ui else zfix[i]
                    zj = z[ui[j]] if j in ui else zfix[j]
                    pij = 1 / (1 + math.exp(-(zi - zj) / tau_))
                    ue = float(v @ Hinv @ v)
                    scores.append(math.log1p(LAMBDA1[s] / tau_ ** 2 * pij * (1 - pij) * ue))
                    p_[(i, j)] = pij
                idx = np.argsort(scores)[::-1][:BUDGETS[s]]
                chosen = [ok[q] for q in idx]
            for i, j in chosen:
                r = 1.0 if tab[i]["value"] < tab[j]["value"] else 0.0   # i better = lower E_f
                pairs.append((i, j, r, LAMBDA1[s]))
            queries[s] += len(chosen)
            z, H = fit(zfix, unm, pairs, theta, tau_, z)
        best = np.argsort(-z)[:k]
        batch = [unm[b] for b in best]
    return {"curve": curve, "ranker_queries": queries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receivers", default="qe_pbe,qe_hse06")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    idx = load_index()
    tables = load_tables()
    out = RES / "analysis" / "prisms_comparison"
    out.mkdir(parents=True, exist_ok=True)
    result = {"k": args.k, "seeds": args.seeds, "settings": {
        "lambda1": LAMBDA1, "lambda2": LAMBDA2, "kU": KU, "budgets": BUDGETS[1:],
        "tau": "std of measured receiver values", "first_batch": "random"}, "receivers": {}}
    for recname in args.receivers.split(","):
        rec = tables[recname]
        pool = sorted(c for c in rec if all(c in tables[d] for d in RANKERS))
        X = {c: fractions(c, idx) for c in pool}
        pool = [c for c in pool if (X[c] > 0).sum() >= 2]
        total = sum(rec[c]["cost"] for c in pool)
        rv = {c: rec[c]["value"] for c in pool}
        taus = {d: float(kendalltau([tables[d][c]["value"] for c in pool], [rv[c] for c in pool]).statistic)
                for d in RANKERS}
        orders = {"cost": ["mace_relax", "lammps_meam_md", "gpumd_nep89_md"],
                  "class": ["lammps_meam_md", "gpumd_nep89_md", "mace_relax"],
                  "measured": sorted(RANKERS, key=lambda d: taus[d]),
                  "pair": ["lammps_meam_md", "mace_relax"]}
        arms = {}
        for s in range(args.seeds):
            r_tab = md_draw(rec, recname, s)
            dr = {d: md_draw(tables[d], d, s) for d in RANKERS}
            for oname, order in orders.items():
                for L in (1, 3):
                    if L == 1 and oname != "measured":
                        continue
                    rk = [(d, dr[d]) for d in order[-L:]]
                    name = (f"prisms_single:{order[-1]}" if L == 1 else
                            "prisms_2fid:meam_to_mace" if oname == "pair" else f"prisms_3fid:{oname}")
                    r = prisms_campaign(r_tab, rk, pool, X, args.k, crc(f"prisms|{name}|{s}"))
                    sm = summarise(r["curve"], total)
                    arms.setdefault(name, {"rankers": [d for d, _ in rk], "runs": []})["runs"].append(
                        {**sm, "ranker_queries": r["ranker_queries"]})
            for arm, sub in (("surrogate", {}), ("learned:mace_relax", {"mace_relax": dr["mace_relax"]}),
                             ("pooled_learned", dr), ("fixed:mace_relax", {"mace_relax": dr["mace_relax"]})):
                r = run_campaign(arm, r_tab, sub, pool, X, args.k, crc(f"{arm}|{args.k}|{s}"))
                arms.setdefault(arm, {"rankers": list(sub), "runs": []})["runs"].append(summarise(r["curve"], total))
            print(recname, s, flush=True)
        summ = {}
        for a, d in arms.items():
            e = np.array([r["early_recall_area"] for r in d["runs"]])
            c8 = np.array([r["cost_frac_to_8of10"] for r in d["runs"]])
            boot = np.random.default_rng(0).integers(0, len(e), (2000, len(e)))
            summ[a] = {"rankers": d["rankers"],
                       "early_recall_area": float(e.mean()),
                       "early_ci": [float(np.percentile(e[boot].mean(1), 2.5)), float(np.percentile(e[boot].mean(1), 97.5))],
                       "cost_frac_to_8of10": float(c8.mean()),
                       "c8_ci": [float(np.percentile(c8[boot].mean(1), 2.5)), float(np.percentile(c8[boot].mean(1), 97.5))]}
            if "ranker_queries" in d["runs"][0]:
                summ[a]["mean_ranker_queries"] = np.mean([r["ranker_queries"] for r in d["runs"]], 0).tolist()
        result["receivers"][recname] = {"n_pool": len(pool), "tau_ranker_receiver": taus,
                                        "orders": orders, "arms": summ}
        print(json.dumps({a: (round(v["early_recall_area"], 3), round(v["cost_frac_to_8of10"], 3))
                          for a, v in summ.items()}, indent=0))
    (out / f"prisms_comparison_{args.receivers.replace(',', '_')}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = ["# Direct comparison with PRISMS on the computed Al-Mg-Si pool", "",
             f"k = {args.k} candidates per receiver batch, {args.seeds} seeds (with MD noise draws). "
             "Early = recall area over the first 25% of receiver cost; c8 = receiver-cost fraction to find 8 of the 10 lowest.", ""]
    for recname, R in result["receivers"].items():
        lines += [f"## Receiver {recname} (pool {R['n_pool']})", "",
                  "Kendall tau of each ranker with the receiver: " +
                  ", ".join(f"{d} {t:+.2f}" for d, t in R["tau_ranker_receiver"].items()), "",
                  "| arm | rankers (low to high fidelity) | early recall area [95% CI] | cost to 8 of 10 [95% CI] |",
                  "|---|---|---|---|"]
        for a, v in R["arms"].items():
            lines.append(f"| {a} | {', '.join(v['rankers']) or '-'} | {v['early_recall_area']:.3f} "
                         f"[{v['early_ci'][0]:.3f}, {v['early_ci'][1]:.3f}] | {v['cost_frac_to_8of10']:.3f} "
                         f"[{v['c8_ci'][0]:.3f}, {v['c8_ci'][1]:.3f}] |")
        lines.append("")
    (out / f"prisms_comparison_{args.receivers.replace(',', '_')}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
