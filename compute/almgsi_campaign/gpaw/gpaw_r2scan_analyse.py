#!/usr/bin/env python
"""Formation energies from GPAW single points (gpaw_r2scan.py output).

For every results/<dir>/ given (default: results/gpaw_r2scan plus the cutoff
scans in results/r2scan_conv/gpaw_*), forms
E_f = E(cell)/N - sum_i x_i E_ref_i with the fcc Al, hcp Mg and diamond Si
cells of the same directory, and prints a table per cell and directory.
Output: results/analysis/gpaw_r2scan_formation.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REFS = {"Al": "mp_134_Al", "Mg": "mp_153_Mg", "Si": "mp_149_Si"}


def comp(formula_counts):
    n = sum(formula_counts.values())
    return {el: c / n for el, c in formula_counts.items()}


def counts(formula):
    import re
    out = Counter()
    for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] += int(k or 1)
    return out


def main():
    dirs = [RES / d for d in sys.argv[1:]] or [d for d in [RES / "gpaw_r2scan", RES / "r2scan_conv" / "gpaw_pbe"]
                                                 if d.exists()] + sorted((RES / "r2scan_conv").glob("gpaw_conv_*"))
    table = {}
    for d in dirs:
        recs = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
        refs = {el: recs[c]["energy_per_atom"] for el, c in REFS.items()
                if c in recs and recs[c]["status"] == "ok"}
        ef = {}
        for cid, r in recs.items():
            if r["status"] != "ok":
                continue
            x = comp(counts(r["formula"]))
            if all(el in refs for el in x):
                ef[cid] = r["energy_per_atom"] - sum(v * refs[el] for el, v in x.items())
        table[d.name] = {"formation_eV_atom": ef, "settings": next(iter(recs.values()))["settings"] if recs else {},
                         "core_s": sum(r.get("core_s") or 0 for r in recs.values())}
    (RES / "analysis").mkdir(exist_ok=True)
    (RES / "analysis" / "gpaw_r2scan_formation.json").write_text(json.dumps(table, indent=1))
    cids = sorted({c for t in table.values() for c in t["formation_eV_atom"]})
    print("cell | " + " | ".join(table))
    for c in cids:
        print(c, "|", " | ".join(f"{1000 * t['formation_eV_atom'][c]:+.1f}" if c in t["formation_eV_atom"] else "-"
                                 for t in table.values()))


if __name__ == "__main__":
    main()
