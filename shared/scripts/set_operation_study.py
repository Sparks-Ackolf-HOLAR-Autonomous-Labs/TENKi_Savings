"""
Set-operation studies on the band-gap archive (materials form of the v1
preprint's set-operation experiments)

The v1 PEGKi preprint built five "set-operation" study types on the colour
testbed (Mixbox cap RYB, KM minus Mixbox, Spectral minus RYB, ...): each was a
region of composition space where two engines agree, or where one engine
reaches and the other does not. They served as stratified corroboration of the
transfer score: if engine A owns a region, transfer from A into that region
should score higher than transfer from A into the region it does not own.

On the band-gap archive the natural regions are defined by compounds, not by
colour gamuts. For a donor source A and a receiver source X, on the compounds
(reduced formulas) both sources report:

  * scissor shift s = median over shared compounds of (gap_X - gap_A); a rigid
    shift of the donor's gaps, the "scissor operator" of band-gap practice;
  * X cap_s A (agreement region): receiver records of shared compounds whose
    scissor-corrected donor gap lies within TOL of the receiver gap;
  * X minus_s A (disagreement region): receiver records of shared compounds
    where the scissor correction fails by more than TOL;
  * X minus_c A (coverage difference): receiver records of compounds the donor
    does not report at all; X cap_c A is its complement within X.

Every derived study is a subset of archived receiver records; no value is
created. Each derived study is rated exactly like a base source (Plackett-Luce
over within-source class ranking events, as in learned_transfer_weights.py)
and the committed transfer-score functions give Delta(K_s -> K_t).

Tests
  H1 ownership (the v1 corroboration test): Delta(A -> X cap_s A) exceeds
     Delta(A -> X minus_s A). A donor's class ordering should transfer better
     into the compounds where its gaps (after one rigid shift) agree.
  H2 rerouting: in X minus_s A, the best base donor is reported; if it is not
     A, the disagreement region should be routed elsewhere.
  H3 coverage: Delta(A -> X minus_c A) versus Delta(A -> X cap_c A); a donor's
     class ordering either carries to compounds it never computed or it does
     not.
  Routing: the maximin linear program over base plus derived studies (the v1
     result was that all Nash weight went to set-operation studies).

Uncertainty: 50 record-bootstrap replicates (records resampled within class
for every study); rating SDs are held at their event-bootstrap values.

Usage (from shared/):
    uv run python scripts/set_operation_study.py --universe <path to jsonl>
Outputs set_operation_study.json / .md in --output-dir. ASCII only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import importlib.util
import json

import numpy as np


def _load(name):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ltw = _load("learned_transfer_weights")
trial = ltw.trial
from src.rating.hypergraph_rating import PlackettLuceModel  # noqa: E402
from src.evaluation.phase3.knowledge_spaces import (  # noqa: E402
    KnowledgeSpace, KnowledgeSpaceType)

# (donor A, receiver X): the three fidelity steps with enough shared compounds
PAIRS = [("mp_gga", "expt_zhuo"),
         ("snumat_gga", "snumat_hse"),
         ("jarvis_optb88vdw", "jarvis_mbj")]
COVERAGE = [("mp_gga", "expt_zhuo"), ("mp_gga", "jarvis_mbj"),
            ("jarvis_optb88vdw", "snumat_hse")]
LABEL = {
    "mp_gga": "MP GGA", "mp_r2scan": "MP r2SCAN",
    "mp_gga_expsynth": "MP GGA (ICSD-matched)", "expt_zhuo": "Experiment",
    "jarvis_optb88vdw": "JARVIS OptB88vdW", "jarvis_mbj": "JARVIS TB-mBJ",
    "snumat_gga": "SNUMAT GGA", "snumat_hse": "SNUMAT HSE06"}


def load_records(path):
    """-> list of (source, formula, class, gap) for every usable record."""
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            f = (r.get("x") or {}).get("formula")
            g = (r.get("y") or {}).get("bandgap_eV")
            s = r.get("s")
            if not f or g is None or s is None or float(g) <= trial.MIN_GAP_EV:
                continue
            recs.append((s, f, trial.classify_formula(f), float(g)))
    return recs


def first_value(recs):
    fv = {}
    for s, f, _c, g in recs:
        fv.setdefault(f, {}).setdefault(s, g)
    return fv


def classes_of(recs, source, keep=None):
    out = {}
    for s, f, c, g in recs:
        if s == source and (keep is None or f in keep):
            out.setdefault(c, []).append(g)
    for c in out:
        out[c].sort()
    return out


def restrict_common(a, b, min_size):
    common = sorted(c for c in a if c in b
                    and len(a[c]) >= min_size and len(b[c]) >= min_size)
    return {c: a[c] for c in common}, {c: b[c] for c in common}


def fit_space(name, classes, n_events, n_boot, seed, sd=None):
    edges = trial.within_channel_edges({name: classes}, n_events, 1).get(name)
    if not edges:
        return None, None
    theta = dict(PlackettLuceModel().fit(edges).log_strengths)
    if sd is None:
        rng = np.random.default_rng([seed, len(name)])
        reps = []
        for _ in range(n_boot):
            samp = [edges[j] for j in rng.integers(0, len(edges), len(edges))]
            reps.append(PlackettLuceModel().fit(samp).log_strengths)
        sd = {c: float(np.std([r.get(c, np.nan) for r in reps])) for c in theta}
    maes = {c: float(np.mean(np.abs(np.asarray(classes[c]) - np.median(classes[c]))))
            for c in theta}
    ranks = {c: r + 1 for r, c in enumerate(sorted(theta, key=lambda k: -theta[k]))}
    ks = KnowledgeSpace(
        space_type=KnowledgeSpaceType.K_H, study_type=name,
        policy_mus=dict(theta), policy_sigmas=sd, policy_ranks=ranks,
        policy_maes=maes, policy_mae_stds={}, policy_stat_mus={},
        policy_bc_mus={},
        policy_s_transitive={c: {"bandgap": v} for c, v in theta.items()})
    return ks, sd


def build_studies(recs, tol, min_base, min_derived):
    fv = first_value(recs)
    base = {s: {c: v for c, v in classes_of(recs, s).items() if len(v) >= min_base}
            for s in LABEL}
    derived, meta = {}, {}
    for a, x in PAIRS:
        shared = [f for f, d in fv.items() if a in d and x in d]
        diff = np.array([fv[f][x] - fv[f][a] for f in shared])
        shift = float(np.median(diff))
        res = np.abs(diff - shift)
        agree = {f for f, r in zip(shared, res) if r <= tol}
        dis = {f for f, r in zip(shared, res) if r > tol}
        ca, cd = restrict_common(classes_of(recs, x, agree),
                                 classes_of(recs, x, dis), min_derived)
        na, nd = f"{x}|agree|{a}", f"{x}|disagree|{a}"
        derived[na], derived[nd] = ca, cd
        meta[(a, x, "scissor")] = {
            "agree": na, "disagree": nd, "n_shared": len(shared),
            "scissor_shift_eV": shift, "n_agree": len(agree), "n_disagree": len(dis),
            "classes": sorted(ca),
            "records_agree": int(sum(len(v) for v in ca.values())),
            "records_disagree": int(sum(len(v) for v in cd.values()))}
    for a, x in COVERAGE:
        covered = {f for f, d in fv.items() if a in d}
        xs = {f for f, d in fv.items() if x in d}
        ci, cm = restrict_common(classes_of(recs, x, xs & covered),
                                 classes_of(recs, x, xs - covered), min_derived)
        ni, nm = f"{x}|covered|{a}", f"{x}|uncovered|{a}"
        derived[ni], derived[nm] = ci, cm
        meta[(a, x, "coverage")] = {
            "covered": ni, "uncovered": nm, "classes": sorted(ci),
            "n_compounds_covered": len(xs & covered),
            "n_compounds_uncovered": len(xs - covered),
            "records_covered": int(sum(len(v) for v in ci.values())),
            "records_uncovered": int(sum(len(v) for v in cm.values()))}
    return base, derived, meta


def pretty(name):
    if "|" not in name:
        return LABEL[name]
    x, kind, a = name.split("|")
    sym = {"agree": "agrees with", "disagree": "disagrees with",
           "covered": "also in", "uncovered": "absent from"}[kind]
    return f"{LABEL[x]} where {LABEL[a]} {sym}" if kind in ("agree", "disagree") \
        else f"{LABEL[x]}, compounds {sym} {LABEL[a]}"


def delta_matrix(spaces, donors, receivers):
    D = {}
    for a in donors:
        for b in receivers:
            if a == b or a not in spaces or b not in spaces:
                continue
            if b.split("|")[0] == a.split("|")[0]:
                continue  # same underlying records (a study of the donor's own source)
            x = np.array([
                ltw.compute_structural_compatibility(spaces[a], spaces[b]),
                ltw.compute_mapping_efficiency(spaces[a], spaces[b]),
                -ltw.compute_transfer_overhead(spaces[a], spaces[b])])
            D[(a, b)] = float(x @ ltw.W0)
    return D


def resample(classes, rng):
    return {c: sorted(rng.choice(v, size=len(v), replace=True).tolist())
            for c, v in classes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="output/bandgap_archive_build/bandgap_universe.jsonl")
    ap.add_argument("--tol", type=float, default=0.3,
                    help="scissor-corrected agreement tolerance, eV")
    ap.add_argument("--n-events", type=int, default=300)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--min-derived-class-size", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--n-rep", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="output/set_operation_study")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    universe = (root / args.universe).resolve()
    sha = hashlib.sha256(universe.read_bytes()).hexdigest()[:16]
    out = (root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    recs = load_records(universe)
    base, derived, meta = build_studies(recs, args.tol, args.min_class_size,
                                        args.min_derived_class_size)
    studies = {**base, **derived}
    spaces, sds = {}, {}
    for n in sorted(studies):
        ks, sd = fit_space(n, studies[n], args.n_events, args.n_boot, args.seed)
        if ks is not None and len(ks.policy_mus) >= 3:
            spaces[n], sds[n] = ks, sd
    donors = sorted(base)
    receivers = sorted(spaces)
    D0 = delta_matrix(spaces, donors, receivers)

    rng = np.random.default_rng(args.seed)
    reps = []
    for _ in range(args.n_rep):
        sp = {}
        for n in spaces:
            ks, _ = fit_space(n, resample(studies[n], rng), args.n_events,
                              args.n_boot, args.seed, sd=sds[n])
            if ks is not None:
                sp[n] = ks
        reps.append(delta_matrix(sp, donors, receivers))

    def ci(fn):
        v = np.array([fn(R) for R in reps], float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None

    res = {"config": vars(args), "universe_sha256_16": sha,
           "studies": {n: {"label": pretty(n),
                           "classes": {c: len(v) for c, v in studies[n].items()}}
                       for n in spaces},
           "delta": {f"{a}->{b}": v for (a, b), v in D0.items()},
           "h1_ownership": [], "h2_rerouting": [], "h3_coverage": []}
    for (a, x, kind), m in meta.items():
        if kind == "scissor":
            ag, dg = m["agree"], m["disagree"]
            if (a, ag) not in D0 or (a, dg) not in D0:
                continue
            d = D0[(a, ag)] - D0[(a, dg)]
            res["h1_ownership"].append({
                **m, "donor": a, "receiver": x,
                "delta_into_agree": D0[(a, ag)], "delta_into_disagree": D0[(a, dg)],
                "difference": d,
                "ci95": ci(lambda R: R.get((a, ag), np.nan) - R.get((a, dg), np.nan)),
                "supported": bool(d > 0)})
            for reg in (ag, dg):
                ranked = sorted(((D0[(s, reg)], s) for s in donors if (s, reg) in D0),
                                reverse=True)
                best = ranked[0][1]
                res["h2_rerouting"].append({
                    "receiver_study": reg, "label": pretty(reg), "defining_donor": a,
                    "ranking": [[s, v] for v, s in ranked],
                    "best_donor": best,
                    "best_minus_defining": ranked[0][0] - D0[(a, reg)],
                    "ci95": ci(lambda R, b=best, r=reg: R.get((b, r), np.nan) - R.get((a, r), np.nan))})
        else:
            ci_, cu = m["covered"], m["uncovered"]
            if (a, ci_) not in D0 or (a, cu) not in D0:
                continue
            d = D0[(a, ci_)] - D0[(a, cu)]
            res["h3_coverage"].append({
                **m, "donor": a, "receiver": x,
                "delta_into_covered": D0[(a, ci_)], "delta_into_uncovered": D0[(a, cu)],
                "difference": d,
                "ci95": ci(lambda R: R.get((a, ci_), np.nan) - R.get((a, cu), np.nan))})

    # maximin routing over base plus derived studies
    names = sorted(spaces)
    full = delta_matrix(spaces, names, names)
    M = np.zeros((len(names), len(names)))
    for (a, b), v in full.items():
        M[names.index(a), names.index(b)] = v
    p = ltw.lp_strategy(names, M)
    res["routing_lp_all_studies"] = {n: float(w) for n, w in zip(names, p) if w > 1e-6}
    res["routing_weight_on_derived"] = float(sum(w for n, w in zip(names, p) if "|" in n))
    (out / "set_operation_study.json").write_text(json.dumps(res, indent=2))

    L = ["# Set-operation studies on the band-gap archive", "",
         f"Archive sha256[:16] = `{sha}`; scissor tolerance {args.tol} eV; "
         f"{args.n_rep} record-bootstrap replicates (95% intervals).", "",
         "## H1: does a donor transfer better where its scissor-corrected gaps agree?", "",
         "| donor to receiver | shared compounds | scissor shift (eV) | agree / disagree records | Delta into agree | Delta into disagree | difference [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    for h in res["h1_ownership"]:
        L.append(f"| {LABEL[h['donor']]} to {LABEL[h['receiver']]} | {h['n_shared']} | "
                 f"{h['scissor_shift_eV']:+.2f} | {h['records_agree']} / {h['records_disagree']} | "
                 f"{h['delta_into_agree']:+.3f} | {h['delta_into_disagree']:+.3f} | "
                 f"{h['difference']:+.3f} [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] |")
    L += ["", "## H2: best base donor for each derived region", "",
          "| region | best donor | Delta(best) minus Delta(defining donor) [95% CI] |",
          "|---|---|---|"]
    for h in res["h2_rerouting"]:
        L.append(f"| {h['label']} | {LABEL[h['best_donor']]} | {h['best_minus_defining']:+.3f} "
                 f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] |")
    L += ["", "## H3: does a donor's ordering carry to compounds it never reported?", "",
          "| donor to receiver | compounds covered / uncovered | Delta into covered | Delta into uncovered | difference [95% CI] |",
          "|---|---|---|---|---|"]
    for h in res["h3_coverage"]:
        L.append(f"| {LABEL[h['donor']]} to {LABEL[h['receiver']]} | {h['n_compounds_covered']} / "
                 f"{h['n_compounds_uncovered']} | {h['delta_into_covered']:+.3f} | "
                 f"{h['delta_into_uncovered']:+.3f} | {h['difference']:+.3f} "
                 f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] |")
    L += ["", "## Maximin routing over all studies", "",
          f"Weight on derived (set-operation) studies: {res['routing_weight_on_derived']:.3f}", ""]
    for n, w in sorted(res["routing_lp_all_studies"].items(), key=lambda t: -t[1]):
        L.append(f"- {pretty(n)}: {w:.3f}")
    (out / "set_operation_study.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
