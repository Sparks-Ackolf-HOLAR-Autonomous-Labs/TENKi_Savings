"""Collect the numbers the papers cite from the semiconductor campaign
(results/campaigns/*/summary.json and the raw gap tables) into
results/SUMMARY.md and results/summary_numbers.json."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
idx = {r["cid"]: r for r in json.loads((HERE / "candidates_index.json").read_text())}


def gaps(src):
    out = {}
    for p in (RES / src).glob("*.json"):
        if p.name.endswith(".failed.json"):
            continue
        r = json.loads(p.read_text())
        if r.get("status") == "ok" and r.get("gap_eV") is not None:
            out[r["cid"]] = r
    return out


pbe, hse = gaps("qe_pbe"), gaps("qe_hse06")
arch = json.loads((RES / "archive_donors.json").read_text())
N = {"pool": len(idx), "pbe_ok": len(pbe), "hse_ok": len(hse),
     "suggested": sum(r["kind"] == "suggested" for r in idx.values()),
     "hse_suggested": sum(idx[c]["kind"] == "suggested" for c in hse)}


def cmp(a, b):
    cs = sorted(set(a) & set(b))
    x = np.array([a[c] for c in cs]); y = np.array([b[c] for c in cs])
    return {"n": len(cs), "tau": float(kendalltau(x, y).statistic), "mean_shift": float(np.mean(x - y)),
            "mae": float(np.mean(np.abs(x - y)))}


g = lambda t: {c: r["gap_eV"] for c, r in t.items()}
A = {s: {c: v["value"] for c, v in d.items()} for s, d in arch.items()}
checks = {"pbe_vs_mp_gga": cmp(g(pbe), A["mp_gga"]), "hse_vs_expt": cmp(g(hse), A["expt_zhuo"]),
          "pbe_vs_expt": cmp(g(pbe), A["expt_zhuo"]), "hse_vs_pbe": cmp(g(hse), g(pbe)),
          "snumat_hse_vs_hse": cmp(A["snumat_hse"], g(hse)), "mbj_vs_expt": cmp(A["jarvis_mbj"], A["expt_zhuo"])}
cost = {"pbe_core_s": sum(r["core_s"] for r in pbe.values()), "hse_core_s": sum(r["core_s"] for r in hse.values()),
        "pbe_median_core_s": float(np.median([r["core_s"] for r in pbe.values()])),
        "hse_median_core_s": float(np.median([r["core_s"] for r in hse.values()]))}
camps = {}
for d in sorted((RES / "campaigns").glob("*")):
    s = json.loads((d / "summary.json").read_text())
    arms = {a: {"to8": v["cost_frac_to_8of10"]["mean"], "to8_p05": v["cost_frac_to_8of10"]["p05"],
                "to8_p95": v["cost_frac_to_8of10"]["p95"], "early": v["early_recall_area"]["mean"]}
            for a, v in s["arms"].items()}
    camps[d.name] = {"n_pool": s["n_pool"], "top10": s["top10"], "top10_gaps": s["top10_gaps"],
                     "top10_suggested": s["top10_suggested"],
                     "top10_held_by_archive": {src: sum(c in A[src] for c in s["top10"]) for src in A},
                     "agreement": s["agreement"], "arms": arms, "pegki": s.get("pegki"), "shapley": s.get("shapley")}
out = {"counts": N, "checks": checks, "cost": cost, "campaigns": camps}
(RES / "summary_numbers.json").write_text(json.dumps(out, indent=1))
L = ["# Computed semiconductor campaign: numbers", "", f"Counts: {N}", "", "## Checks", ""]
for k, v in checks.items():
    L.append(f"* {k}: n={v['n']}, tau={v['tau']:+.3f}, mean shift={v['mean_shift']:+.3f} eV, MAE={v['mae']:.3f} eV")
L += ["", f"Cost: {cost}", ""]
for k, c in camps.items():
    L += [f"## {k} (pool {c['n_pool']})", "", f"Top 10: {c['top10']}", f"Gaps: {[round(x, 2) for x in c['top10_gaps']]}",
          f"Suggested in top 10: {c['top10_suggested']}", f"Top 10 held by archive: {c['top10_held_by_archive']}", "",
          "| arm | share of receiver cost to 8 of 10 (mean, 5-95%) | early recall |", "|---|---|---|"]
    for a, v in c["arms"].items():
        if a.endswith("k4"):
            L.append(f"| {a} | {v['to8']:.3f} ({v['to8_p05']:.3f}-{v['to8_p95']:.3f}) | {v['early']:.3f} |")
    if c["pegki"]:
        p = c["pegki"]
        L += ["", f"PEGKi pick {p['pick']}, best measured {p['best_measured']}, Spearman {p['spearman_delta_vs_early']:+.2f}"]
        L += [f"* {d}: Delta {v['delta']:+.3f} (sigma {v['sigma']:.3f}, rho {v['rho']:.3f}, omega {v['omega']:.3f}); early {p['learned_k4_early'][d]:.3f}"
              for d, v in p["delta"].items()]
    if c["shapley"]:
        for m, s in c["shapley"].items():
            L += ["", f"Shapley {m}: " + ", ".join(f"{k} {v:+.3f}" for k, v in s["shapley"].items())
                  + f"; v(empty) {s['v_empty']:.3f}, v(all) {s['v_all']:.3f}, best {s['best_coalition']} {s['v_best']:.3f}"]
    L.append("")
(RES / "SUMMARY.md").write_text("\n".join(L))
print("\n".join(L))
