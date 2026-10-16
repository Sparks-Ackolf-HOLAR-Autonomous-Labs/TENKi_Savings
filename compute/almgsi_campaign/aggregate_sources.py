#!/usr/bin/env python
"""Aggregate per-query source results into campaign property values.

Reads results/<source>/<cid>.json for every source directory present
(mace_relax, mace_md, gpumd_nep89_md, lammps_meam_md, and later qe_pbe /
qe_hse06 in the same format).  Files starting with "_" are metadata and are
skipped, as are directories/files that are not per-query source records
(e.g. results/analysis/).

Properties (always computed within ONE source, never mixing sources):
  formation_energy (eV/atom):
      E_f = e_cand - sum_i x_i e_ref_i
      e_ref from the same source's mp_134_Al (fcc Al), mp_153_Mg (hcp Mg),
      mp_149_Si (diamond Si).  For MD sources every energy is a 300 K, 0 bar
      time-averaged potential energy, so E_f is a 300 K potential-energy
      formation energy (references also at 300 K).
  cluster_binding (eV, positive = attractive):
      E_b = sum_s [E(clu_s1) - E(Al108)] - [E(cluster) - E(Al108)]
      with total energies E = energy_per_atom * natoms of the simulated cell.
SEMs are propagated in quadrature (independent runs).

Outputs: results/table_<source>.csv and results/live_values.json
  {source: {cid: {value, sem, wall_s, status, property}}}
wall_s is the candidate's own query cost; the reference queries are a
one-off shared overhead reported under "_reference_wall_s".
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CAND_DIR = HERE / "candidates"

REFS = {"Al": "mp_134_Al", "Mg": "mp_153_Mg", "Si": "mp_149_Si"}
HOST = "clu_Al108"
SOLUTE_REF = {"Mg": "clu_Mg1", "Si": "clu_Si1"}


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def composition(cid):
    """Element counts of the candidate cell, read from the extxyz file."""
    with open(CAND_DIR / f"{cid}.xyz", "r", encoding="utf-8") as f:
        n = int(f.readline().split()[0])
        f.readline()
        syms = [f.readline().split()[0] for _ in range(n)]
    return Counter(syms), n


def ok(rec):
    return rec is not None and rec.get("status") == "ok" and rec.get("energy_per_atom") is not None


def formation(rec, refs, comp, n):
    if not ok(rec):
        return None, None, "failed" if rec else "missing"
    val = rec["energy_per_atom"]
    var = (rec.get("energy_per_atom_sem") or 0.0) ** 2
    for el, cnt in comp.items():
        r = refs.get(el)
        if not ok(r):
            return None, None, f"ref_{el}_unavailable"
        x = cnt / n
        val -= x * r["energy_per_atom"]
        var += (x * (r.get("energy_per_atom_sem") or 0.0)) ** 2
    return val, math.sqrt(var), "ok"


def total(rec):
    return rec["energy_per_atom"] * rec["natoms"], (rec.get("energy_per_atom_sem") or 0.0) * rec["natoms"]


def binding(rec, recs, comp):
    if not ok(rec):
        return None, None, "failed" if rec else "missing"
    host = recs.get(HOST)
    if not ok(host):
        return None, None, "host_unavailable"
    e_h, s_h = total(host)
    e_c, s_c = total(rec)
    n_sol = 0
    val = -(e_c - e_h)
    var = s_c ** 2
    for el, cnt in comp.items():
        if el == "Al":
            continue
        ref = recs.get(SOLUTE_REF.get(el, ""))
        if not ok(ref):
            return None, None, f"solute_ref_{el}_unavailable"
        e_s, s_s = total(ref)
        val += cnt * (e_s - e_h)
        var += (cnt * s_s) ** 2
        n_sol += cnt
    var += ((1 - n_sol) * s_h) ** 2
    return val, math.sqrt(var), "ok"


def main():
    import argparse
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="aggregate <dir>/<source>/ instead of ./results (temperature study: results_T/T453)")
    RESULTS_DIR = Path(ap.parse_args().results_dir).resolve() if ap.parse_args().results_dir else RESULTS_DIR
    with open(HERE / "candidates_index.json", "r", encoding="utf-8") as f:
        index = json.load(f)
    live = {}
    sources = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir()) if RESULTS_DIR.exists() else []
    for src in sources:
        recs = {}
        for p in sorted((RESULTS_DIR / src).glob("*.json")):
            if p.name.startswith("_") or "." in p.stem:
                continue  # _meta, and superseded attempts (<cid>.attempt1.json, <cid>.qhalf.json)
            r = read_json(p)
            if not isinstance(r, dict) or "cid" not in r or "energy_per_atom" not in r:
                continue  # not a per-query source record (e.g. analysis outputs)
            recs[r["cid"]] = r
        if not recs:
            continue
        refs = {el: recs.get(cid) for el, cid in REFS.items()}
        rows, out = [], {}
        for d in index:
            cid = d["cid"]
            if cid not in recs:
                continue
            rec = recs[cid]
            comp, n = composition(cid)
            if d["property"] == "formation_energy":
                val, sem, st = formation(rec, refs, comp, n)
            elif d["kind"] == "cluster":
                val, sem, st = binding(rec, recs, comp)
            else:
                continue  # cluster references carry no binding value
            out[cid] = {"property": d["property"], "value": val, "sem": sem,
                        "wall_s": rec.get("wall_s"), "status": st}
            rows.append({"cid": cid, "kind": d["kind"], "formula": d["formula"],
                         "property": d["property"], "value": val, "sem": sem,
                         "wall_s": rec.get("wall_s"), "status": st,
                         "energy_per_atom": rec.get("energy_per_atom"),
                         "volume_per_atom": rec.get("volume_per_atom"),
                         "natoms": rec.get("natoms"), "mp_ef": d.get("mp_ef"),
                         "error": rec.get("error")})
        ref_cids = list(REFS.values()) + [HOST] + list(SOLUTE_REF.values())
        out["_reference_wall_s"] = {c: recs[c].get("wall_s") for c in ref_cids if c in recs}
        live[src] = out
        with open(RESULTS_DIR / f"table_{src}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cid"])
            w.writeheader()
            w.writerows(rows)
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"{src}: {len(rows)} values, {n_ok} ok, {len(rows) - n_ok} not ok")
    with open(RESULTS_DIR / "live_values.json", "w", encoding="utf-8") as f:
        json.dump(live, f, indent=1)


if __name__ == "__main__":
    main()
