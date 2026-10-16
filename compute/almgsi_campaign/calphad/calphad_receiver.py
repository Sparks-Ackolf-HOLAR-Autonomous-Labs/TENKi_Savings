#!/usr/bin/env python
"""CALPHAD receiver for the computed Al-Mg-Si campaign (COST 507 light-alloy database).

For every formation_energy candidate of the campaign pool (candidates_index.json)
and of the observed-phase index (candidates_index_observed.json), the overall
Al/Mg/Si composition is read from candidates/<cid>.xyz and pycalphad computes the
equilibrium (minimum) molar Gibbs energy of the stable phase assemblage at
T = 300, 453, 803 K and P = 101325 Pa. The receiver value is the Gibbs energy of
formation per atom,

    dG_f(x, T) = G_eq(x, T) - sum_i x_i G_i^ref(T),

with references fcc Al (FCC_A1), hcp Mg (HCP_A3) and diamond Si (DIAMOND_A4) at the
same T and P, converted from J/mol of atoms to eV/atom.

CALPHAD describes phases, not atomic orderings: candidates with the same overall
composition receive identical values. Each distinct composition is computed once.

Phase set: every phase of the database that can be built from {AL, MG, SI, VA}
(pycalphad filter_phases), minus GAS and minus AL12MG17 (the Al-Li-Mg sublattice
model of the gamma phase, which has no Al-Mg end-member parameters; gamma
Al12Mg17 is described by ALMG_GAMMA). A cross-check with only the phases assessed
for Al-Mg-Si and its binaries (LIQUID, FCC_A1, HCP_A3, DIAMOND_A4, MG2SI,
ALMG_BETA, ALMG_EPS, ALMG_GAMMA) is written alongside.

Outputs (results/calphad/):
  calphad_T{T}.json            campaign pool, live_values.json row schema
  calphad_observed_T{T}.json   observed phases not in the campaign pool
  phases.json                  compound Gibbs energies of formation of the
                               named intermetallics at each T
  summary.json                 distinct compositions, phase-set cross-check,
                               Kendall tau versus qe_pbe

Usage: <venv>/bin/python calphad/calphad_receiver.py  (venv with pycalphad 0.11.2 and ase)
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
from ase.io import read
from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TDB = HERE / "COST507_pycalphad.tdb"
OUT = ROOT / "results" / "calphad"
TEMPS = [300, 453, 803]
P = 101325.0
J_PER_EV_MOL = 96485.33212  # J/mol per eV/atom
REF_PHASE = {"AL": "FCC_A1", "MG": "HCP_A3", "SI": "DIAMOND_A4"}
ASSESSED = ["LIQUID", "FCC_A1", "HCP_A3", "DIAMOND_A4", "MG2SI",
            "ALMG_BETA", "ALMG_EPS", "ALMG_GAMMA"]
EXCLUDE = {"GAS", "AL12MG17"}
# Named intermetallics in the database that correspond to observed phases.
# points = site fractions of the ideal stoichiometric configuration, in
# pycalphad's sorted-constituent order restricted to {AL, MG, SI, VA}.
COMPOUNDS = {
    "MG2SI": {"name": "beta Mg2Si", "formula": "Mg2Si", "x": {"MG": 2 / 3, "SI": 1 / 3},
              "cids": ["mp_1367_Mg2Si"]},
    "ALMG_BETA": {"name": "beta Al3Mg2 (Al140Mg89 model)", "formula": "Al140Mg89",
                  "x": {"AL": 140 / 229, "MG": 89 / 229}, "cids": []},
    "ALMG_EPS": {"name": "epsilon Al30Mg23", "formula": "Al30Mg23",
                 "x": {"AL": 30 / 53, "MG": 23 / 53}, "cids": ["mp_17659_Mg23Al30"]},
    "ALMG_GAMMA": {"name": "gamma Al12Mg17", "formula": "Al12Mg17",
                   "x": {"AL": 12 / 29, "MG": 17 / 29}, "cids": ["mp_2151_Mg17Al12"]},
}


def composition(cid):
    at = read(ROOT / "candidates" / f"{cid}.xyz")
    sym = at.get_chemical_symbols()
    n = len(sym)
    return {e: sym.count(e.capitalize()) / n for e in ("AL", "MG", "SI")}, n


def comp_key(x):
    return tuple(round(x[e], 8) for e in ("AL", "MG", "SI"))


def ref_energies(db, T):
    g = {}
    for el, ph in REF_PHASE.items():
        r = calculate(db, [el, "VA"], ph, T=T, P=P, output="GM")
        g[el] = float(np.squeeze(r.GM.values).min()) if np.size(r.GM.values) > 1 \
            else float(np.squeeze(r.GM.values))
    return g


def eq_gibbs(db, x, T, phases):
    """Equilibrium GM (J/mol atoms) and stable phases at overall composition x."""
    els = [e for e in ("AL", "MG", "SI") if x[e] > 1e-12]
    comps = els + ["VA"]
    ph = [p for p in filter_phases(db, comps, phases)]
    conds = {v.T: T, v.P: P, v.N: 1}
    for e in els[1:]:
        conds[v.X(e)] = x[e]
    eq = equilibrium(db, comps, ph, conds, output="GM")
    gm = float(np.squeeze(eq.GM.values))
    names = np.squeeze(eq.Phase.values).ravel()
    fracs = np.squeeze(eq.NP.values).ravel()
    stable = sorted({str(n) for n, f in zip(names, fracs) if str(n) and np.isfinite(f) and f > 1e-6})
    phase_fr = {}
    for n, f in zip(names, fracs):
        if str(n) and np.isfinite(f) and f > 1e-6:
            phase_fr[str(n)] = phase_fr.get(str(n), 0.0) + float(f)
    return gm, stable, phase_fr


def compound_energy(db, phase, x, T):
    """Single-phase minimum GM of a compound phase at its nominal composition."""
    els = [e for e in ("AL", "MG", "SI") if x.get(e, 0) > 0]
    comps = els + ["VA"]
    conds = {v.T: T, v.P: P, v.N: 1}
    for e in els[1:]:
        conds[v.X(e)] = x[e]
    eq = equilibrium(db, comps, [phase], conds, output="GM")
    return float(np.squeeze(eq.GM.values))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = Database(str(TDB))
    all_ph = sorted(set(filter_phases(db, ["AL", "MG", "SI", "VA"], list(db.phases))) - EXCLUDE)

    pool = [c for c in json.loads((ROOT / "candidates_index.json").read_text())
            if c["property"] == "formation_energy"]
    pool_ids = {c["cid"] for c in pool}
    obs = [c for c in json.loads((ROOT / "candidates_index_observed.json").read_text())
           if c["property"] == "formation_energy" and c["cid"] not in pool_ids]

    comps = {}
    for c in pool + obs:
        comps[c["cid"]] = composition(c["cid"])[0]
    keys = {cid: comp_key(x) for cid, x in comps.items()}
    distinct_pool = len({keys[c["cid"]] for c in pool})

    summary = {"database": TDB.name, "phases_all": all_ph, "phases_assessed": ASSESSED,
               "excluded": sorted(EXCLUDE), "n_pool": len(pool),
               "n_pool_distinct_compositions": distinct_pool,
               "n_observed_extra": len(obs), "per_T": {}}
    phases_out = {}
    live = json.loads((ROOT / "results" / "live_values.json").read_text())

    for T in TEMPS:
        gref = ref_energies(db, T)
        cache = {}
        for cid, x in comps.items():
            k = keys[cid]
            if k in cache:
                continue
            t0 = time.time()
            gm, stable, fr = eq_gibbs(db, x, T, all_ph)
            wall = time.time() - t0
            gm_a, stable_a, _ = eq_gibbs(db, x, T, ASSESSED)
            dg = (gm - sum(x[e] * gref[e] for e in gref)) / J_PER_EV_MOL
            dg_a = (gm_a - sum(x[e] * gref[e] for e in gref)) / J_PER_EV_MOL
            cache[k] = dict(value=dg, wall_s=wall, phases=stable, phase_fractions=fr,
                            value_assessed_set=dg_a, phases_assessed_set=stable_a)

        def row(cid):
            r = cache[keys[cid]]
            x = comps[cid]
            return {"property": "formation_energy", "value": r["value"], "sem": 0.0,
                    "wall_s": r["wall_s"], "status": "ok", "phases": r["phases"],
                    "phase_fractions": r["phase_fractions"],
                    "x": {"Al": x["AL"], "Mg": x["MG"], "Si": x["SI"]},
                    "value_assessed_set": r["value_assessed_set"]}

        res = {c["cid"]: row(c["cid"]) for c in pool}
        res_obs = {c["cid"]: row(c["cid"]) for c in obs}
        (OUT / f"calphad_T{T}.json").write_text(json.dumps(res, indent=1))
        (OUT / f"calphad_observed_T{T}.json").write_text(json.dumps(res_obs, indent=1))

        maxdiff = max(abs(r["value"] - r["value_assessed_set"]) for r in cache.values())
        diffset = sum(r["phases"] != r["phases_assessed_set"] for r in cache.values())
        shared = sorted(c for c in set(res) & set(live["qe_pbe"])
                        if live["qe_pbe"][c].get("value") is not None)
        a = [res[c]["value"] for c in shared]
        b = [live["qe_pbe"][c]["value"] for c in shared]
        tau, p = kendalltau(a, b)
        tau_h = None
        if "qe_hse06" in live:
            sh = sorted(c for c in set(res) & set(live["qe_hse06"])
                        if live["qe_hse06"][c].get("value") is not None)
            tau_h = float(kendalltau([res[c]["value"] for c in sh],
                                     [live["qe_hse06"][c]["value"] for c in sh])[0])
        # composition-level comparison: lowest PBE value per distinct composition
        bykey = {}
        for c in shared:
            k = keys[c]
            bykey[k] = min(bykey.get(k, np.inf), live["qe_pbe"][c]["value"])
        kk = sorted(bykey)
        tau_c, p_c = kendalltau([cache[k]["value"] for k in kk], [bykey[k] for k in kk])
        n_liq = sum("LIQUID" in r["phases"] for r in cache.values())
        low = sorted(res, key=lambda c: res[c]["value"])[:10]
        summary["per_T"][T] = {
            "ref_GM_J_per_mol": gref,
            "n_distinct_evaluated": len(cache),
            "wall_s_total": sum(r["wall_s"] for r in cache.values()),
            "max_abs_diff_all_vs_assessed_eV": maxdiff,
            "n_compositions_phase_set_differs": diffset,
            "kendall_tau_vs_qe_pbe": float(tau), "p_value": float(p), "n_shared": len(shared),
            "kendall_tau_vs_qe_hse06": tau_h,
            "kendall_tau_vs_min_qe_pbe_per_composition": float(tau_c),
            "p_value_per_composition": float(p_c), "n_compositions_shared": len(kk),
            "n_compositions_with_liquid": n_liq,
            "lowest10": [{"cid": c, "value": res[c]["value"], "phases": res[c]["phases"],
                          "qe_pbe": live["qe_pbe"].get(c, {}).get("value")} for c in low],
        }

        for ph, info in COMPOUNDS.items():
            x = {e: info["x"].get(e, 0.0) for e in ("AL", "MG", "SI")}
            t0 = time.time()
            gm = compound_energy(db, ph, x, T)
            wall = time.time() - t0
            dg = (gm - sum(x[e] * gref[e] for e in gref)) / J_PER_EV_MOL
            geq, stable, _ = eq_gibbs(db, x, T, all_ph)
            dg_eq = (geq - sum(x[e] * gref[e] for e in gref)) / J_PER_EV_MOL
            d = phases_out.setdefault(ph, {k: info[k] for k in ("name", "formula", "cids")})
            d.setdefault("x", {e.capitalize(): x[e] for e in x})
            d[f"T{T}"] = {"dG_f_eV_per_atom": dg, "wall_s": wall,
                          "above_equilibrium_eV_per_atom": dg - dg_eq,
                          "equilibrium_phases_at_composition": stable}
        print(f"T={T} K: {len(cache)} distinct compositions, tau(PBE)={tau:.3f} "
              f"(n={len(shared)}), tau(HSE06)={tau_h:.3f}, tau(min PBE per comp)={tau_c:.3f} "
              f"(n={len(kk)}, p={p_c:.2g}), liquid in {n_liq}, max|all-assessed|={maxdiff:.2e} eV")
        for c in low:
            print(f"   {c:28s} {res[c]['value']:+.4f}  {'+'.join(res[c]['phases'])}")

    phases_out["_not_in_database"] = ["beta'' Mg5Si6", "beta' Mg9Si5", "U1 MgAl2Si2",
                                      "U2 MgAlSi", "B' / Q-type ternary precipitates"]
    (OUT / "phases.json").write_text(json.dumps(phases_out, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, vv in phases_out.items():
        if not k.startswith("_"):
            print(k, {t: round(vv[f"T{t}"]["dG_f_eV_per_atom"], 4) for t in TEMPS})

if __name__ == "__main__":
    main()
