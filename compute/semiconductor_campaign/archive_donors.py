"""Archived band-gap values for every candidate of the computed semiconductor pool.

Looks each candidate up in the version-locked band-gap archive (the 261,053-record
universe of the retrospective study; manifest checked):

* Materials Project sources (GGA/GGA+U, ICSD-matched subset, r2SCAN): by the
  candidate's MP id (archived candidates only).
* JARVIS (OptB88vdW, TB-mBJ) and SNUMAT (GGA, HSE06): by reduced formula and
  space-group number; the median is taken if several records match.
* Measured gaps (Zhuo et al. 2018): by reduced formula.

Output: results/archive_donors.json, {source: {cid: {"value": gap_eV, "n": k}}}.
A candidate an archive does not hold is simply absent from that source.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from ase.io import read
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
# version-locked band-gap archive (bandgap_universe.jsonl plus its manifest);
# BANDGAP_UNIVERSE overrides, default under the singular_obj root
UNIVERSE = Path(os.environ.get(
    "BANDGAP_UNIVERSE",
    str(HERE.parents[1] / "shared" / "output" / "bandgap_archive_build" / "bandgap_universe.jsonl")))
MANIFEST_HASH = "f2158ecdc78e0229"
BY_ID = ("mp_gga", "mp_gga_expsynth", "mp_r2scan")
BY_SPG = ("jarvis_optb88vdw", "jarvis_mbj", "snumat_gga", "snumat_hse")


def main():
    man = json.loads(UNIVERSE.with_name("bandgap_universe.manifest.json").read_text())
    assert str(man).find(MANIFEST_HASH) >= 0, "archive manifest does not match the version-locked hash"
    idx = json.loads((HERE / "candidates_index.json").read_text())
    want_id = {r["mpid"]: r["cid"] for r in idx if r.get("mpid")}
    keys = {}
    for r in idx:
        s = AseAtomsAdaptor.get_structure(read(HERE / "candidates" / f"{r['cid']}.xyz"))
        spg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
        keys[r["cid"]] = (Composition(r["formula"]).reduced_formula, spg)
    by_key: dict[tuple, list[str]] = {}
    by_formula: dict[str, list[str]] = {}
    for cid, k in keys.items():
        by_key.setdefault(k, []).append(cid)
        by_formula.setdefault(k[0], []).append(cid)
    vals: dict[str, dict[str, list[float]]] = {}
    with UNIVERSE.open() as fh:
        for line in fh:
            rec = json.loads(line)
            s, x, g = rec["s"], rec["x"], rec["y"].get("bandgap_eV")
            if g is None:
                continue
            hits = []
            if s in BY_ID:
                if x.get("src_id") in want_id:
                    hits = [want_id[x["src_id"]]]
            else:
                try:
                    f = Composition(x["formula"]).reduced_formula
                except Exception:  # noqa: BLE001
                    continue
                if s in BY_SPG:
                    hits = by_key.get((f, x.get("spg")), [])
                elif s == "expt_zhuo":
                    hits = [c for c in by_formula.get(f, []) if c.startswith("arc_")]
            for c in hits:
                vals.setdefault(s, {}).setdefault(c, []).append(float(g))
    out = {s: {c: {"value": statistics.median(v), "n": len(v)} for c, v in d.items()} for s, d in vals.items()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "archive_donors.json").write_text(json.dumps(out, indent=1))
    sug = [r["cid"] for r in idx if r["kind"] == "suggested"]
    for s, d in sorted(out.items()):
        print(f"{s:20s} {len(d):4d} candidates; suggested held: {sum(c in d for c in sug)}")


if __name__ == "__main__":
    main()
