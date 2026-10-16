#!/usr/bin/env python
"""Aggregate per-query source results into formation energies (Al-Si-Cu pool).

Adapted from ../almgsi_campaign/aggregate_sources.py; same output format, so the
campaign.py-style analysis can be reused. Every candidate in this pool has
property formation_energy (no solute clusters).

Reads results/<source>/<cid>.json for every source directory present
(mace_relax, and qe_pbe once fetched from CHPC). Files starting with "_"
and superseded attempts (<cid>.<tag>.json) are skipped.

  formation_energy (eV/atom):
      E_f = e_cand - sum_i x_i e_ref_i
      e_ref from the SAME source's mp_134_Al (fcc Al), mp_149_Si (diamond Si),
      mp_30_Cu (fcc Cu). Sources are never mixed.

An extra pseudo-source ``archive_mp`` lists the Materials Project formation
energy (index field mp_ef) for every candidate MP holds (archived,
reference and archived_fcc), with wall_s 0 (lookup).

Outputs: results/table_<source>.csv and results/live_values.json
  {source: {cid: {value, sem, wall_s, status, property}}}
plus a short agreement summary (Kendall tau and mean absolute difference of
each source against MP on the MP-held binary compounds, and against qe_pbe
when present) in results/agreement.json.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CAND_DIR = HERE / "candidates"

REFS = {"Al": "mp_134_Al", "Si": "mp_149_Si", "Cu": "mp_30_Cu"}


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def composition(cid):
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


def kendall_tau(x, y):
    """Kendall tau-b (ties handled), no scipy dependency."""
    n = len(x)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif a * b > 0:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + tx) * (c + d + ty))
    return (c - d) / den if den else float("nan")


def compare(a, b, cids):
    common = [c for c in cids if a.get(c) is not None and b.get(c) is not None]
    if len(common) < 3:
        return {"n": len(common)}
    xa = [a[c] for c in common]
    xb = [b[c] for c in common]
    return {"n": len(common), "kendall_tau": round(kendall_tau(xa, xb), 4),
            "mae_eV_atom": round(sum(abs(p - q) for p, q in zip(xa, xb)) / len(common), 4)}


def main():
    import argparse
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir).resolve()
    with open(HERE / "candidates_index.json", "r", encoding="utf-8") as f:
        index = json.load(f)
    live = {}
    sources = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir()) if RESULTS_DIR.exists() else []
    for src in sources:
        recs = {}
        for p in sorted((RESULTS_DIR / src).glob("*.json")):
            if p.name.startswith("_") or "." in p.stem:
                continue
            r = read_json(p)
            if not isinstance(r, dict) or "cid" not in r or "energy_per_atom" not in r:
                continue
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
            val, sem, st = formation(rec, refs, comp, n)
            out[cid] = {"property": d["property"], "value": val, "sem": sem,
                        "wall_s": rec.get("wall_s"), "status": st}
            rows.append({"cid": cid, "kind": d["kind"], "formula": d["formula"],
                         "property": d["property"], "value": val, "sem": sem,
                         "wall_s": rec.get("wall_s"), "status": st,
                         "energy_per_atom": rec.get("energy_per_atom"),
                         "volume_per_atom": rec.get("volume_per_atom"),
                         "natoms": rec.get("natoms"), "mp_ef": d.get("mp_ef"),
                         "error": rec.get("error")})
        out["_reference_wall_s"] = {c: recs[c].get("wall_s") for c in REFS.values() if c in recs}
        live[src] = out
        with open(RESULTS_DIR / f"table_{src}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cid"])
            w.writeheader()
            w.writerows(rows)
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"{src}: {len(rows)} values, {n_ok} ok, {len(rows) - n_ok} not ok")

    # MP archive as a lookup source
    live["archive_mp"] = {d["cid"]: {"property": d["property"], "value": d["mp_ef"], "sem": 0.0,
                                     "wall_s": 0.0, "status": "ok"}
                          for d in index if d.get("mp_ef") is not None}
    with open(RESULTS_DIR / "live_values.json", "w", encoding="utf-8") as f:
        json.dump(live, f, indent=1)

    # agreement summary; compounds only (elemental structures are references)
    compounds = [d["cid"] for d in index if len(composition(d["cid"])[0]) > 1]
    archived = [d["cid"] for d in index if d.get("mp_ef") is not None and d["cid"] in compounds]
    vals = {s: {c: v["value"] for c, v in o.items() if not c.startswith("_") and v.get("status") == "ok"}
            for s, o in live.items()}
    agree = {}
    for s in vals:
        if s == "archive_mp":
            continue
        agree[f"{s}_vs_archive_mp_MPheld_compounds"] = compare(vals[s], vals["archive_mp"], archived)
        if "qe_pbe" in vals and s != "qe_pbe":
            agree[f"{s}_vs_qe_pbe_all_compounds"] = compare(vals[s], vals["qe_pbe"], compounds)
    if "qe_pbe" in vals:
        agree["archive_mp_vs_qe_pbe_MPheld_compounds"] = compare(vals["archive_mp"], vals["qe_pbe"], archived)
    with open(RESULTS_DIR / "agreement.json", "w", encoding="utf-8") as f:
        json.dump(agree, f, indent=1)
    for k, v in agree.items():
        print(k, v)


if __name__ == "__main__":
    main()
