"""Candidate pool for the computed semiconductor band-gap campaign.

Two kinds of candidate, written as extended-XYZ plus ``candidates_index.json``:

1. ``archived``: semiconductors and insulators whose gap was measured (Zhuo,
   Mansouri Tehrani and Brgoch 2018 compilation) and whose ground-state
   polymorph is in the Materials Project with at most ``MAX_SITES`` sites in
   the primitive cell. Elements are restricted to non-magnetic main-group and
   d10/d0 metals so that every calculation can be run spin-unpolarised.
2. ``suggested``: ordered 50/50 alloys of pairs of these semiconductors
   (cation or anion substitution in the smallest ordered supercells of the
   zincblende, wurtzite and rocksalt parents: CuAu-I [001] and CuPt [111]
   layering for zincblende and rocksalt, the 4-atom layered cell for
   wurtzite). Each is matched against every MP structure of the same
   composition; only orderings MP does not hold stay ``suggested``. No
   archive holds a gap for these, so only an on-demand calculation can
   answer them.

Archived donor values for every candidate are looked up later
(``archive_donors.py``) from the version-locked band-gap archive.

Usage: python generate_pool.py [--refetch]   (needs MP_API_KEY or MP_API)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

from ase.io import write
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
SINGULAR = HERE.parents[1]
ZHUO = SINGULAR / "shared" / "data" / "expt_compilations" / "zhuo2018_expt_gap.csv"
OUT = HERE / "candidates"
MAX_SITES = 6
GAP_MIN, GAP_MAX = 0.4, 9.0     # measured gap window (eV)
ALLOWED = set("""Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti Cu Zn Ga Ge As Se Br Rb Sr Y Zr
Ag Cd In Sn Sb Te I Cs Ba Hf""".split())

# parent semiconductors for ordered alloys: (pair label, formula A, formula B, prototype)
ALLOY_PAIRS = [
    ("AlGaN", "AlN", "GaN", "wurtzite"), ("GaInN", "GaN", "InN", "wurtzite"),
    ("AlInN", "AlN", "InN", "wurtzite"), ("ZnMgO", "ZnO", "MgO", "wurtzite"),
    ("ZnCdS", "ZnS", "CdS", "wurtzite"), ("ZnCdSe", "ZnSe", "CdSe", "zincblende"),
    ("ZnSSe", "ZnS", "ZnSe", "zincblende"), ("ZnSeTe", "ZnSe", "ZnTe", "zincblende"),
    ("AlGaP", "AlP", "GaP", "zincblende"), ("GaInP", "GaP", "InP", "zincblende"),
    ("AlGaAs", "AlAs", "GaAs", "zincblende"), ("GaPAs", "GaP", "GaAs", "zincblende"),
    ("AlPAs", "AlP", "AlAs", "zincblende"), ("CdSSe", "CdS", "CdSe", "zincblende"),
    ("MgCaO", "MgO", "CaO", "rocksalt"), ("ZnBeO", "ZnO", "BeO", "wurtzite"),
]


def mp_key() -> str:
    return os.environ.get("MP_API_KEY") or os.environ["MP_API"]


def load_zhuo() -> dict[str, float]:
    gaps: dict[str, list[float]] = {}
    with ZHUO.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                comp = Composition(row["formula"])
            except Exception:  # noqa: BLE001
                continue
            if any(abs(v - round(v)) > 1e-6 for v in comp.values()):
                continue   # fractional (disordered) compositions
            gaps.setdefault(comp.reduced_formula, []).append(float(row["expt_gap_eV"]))
    return {f: sum(v) / len(v) for f, v in gaps.items()}


def fetch_mp(formulas: list[str], refetch: bool) -> list[dict]:
    cache = HERE / "mp_semiconductor_raw.json.gz"
    if cache.exists() and not refetch:
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    from mp_api.client import MPRester
    rows = []
    with MPRester(mp_key(), mute_progress_bars=True) as m:
        for i in range(0, len(formulas), 200):
            docs = m.materials.summary.search(
                formula=formulas[i:i + 200],
                fields=["material_id", "formula_pretty", "nsites", "energy_above_hull", "band_gap",
                        "symmetry", "structure", "theoretical", "is_magnetic"])
            for d in docs:
                rows.append(dict(mpid=str(d.material_id), formula=d.formula_pretty, nsites=d.nsites,
                                 ehull=d.energy_above_hull, gap=d.band_gap, spg=d.symmetry.symbol,
                                 spg_number=d.symmetry.number, theoretical=d.theoretical,
                                 magnetic=d.is_magnetic, structure=d.structure.as_dict()))
    with gzip.open(cache, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


PROTO_SPG = {"zincblende": 216, "wurtzite": 186, "rocksalt": 225}


def ground_states(rows: list[dict], spg: int | None = None) -> dict[str, dict]:
    """Lowest-energy polymorph per formula. Polymorphs within 20 meV/atom of the
    minimum count as degenerate and the smallest cell is taken (ZnS and SiC
    polytypes). With ``spg`` only polymorphs of that space group are used."""
    emin: dict[str, float] = {}
    for r in rows:
        f = Composition(r["formula"]).reduced_formula
        e = r["ehull"] if r["ehull"] is not None else 9
        emin[f] = min(emin.get(f, 9), e)
    best: dict[str, dict] = {}
    for r in rows:
        if spg is not None and r["spg_number"] != spg:
            continue
        f = Composition(r["formula"]).reduced_formula
        e = r["ehull"] if r["ehull"] is not None else 9
        key = (spg is None and e > emin[f] + 0.02, r["theoretical"], r["nsites"], e)
        if f not in best or key < best[f]["_key"]:
            best[f] = dict(r, _key=key)
    return best


def ordered_alloys(parent_a: Structure, parent_b: Structure, proto: str) -> list[tuple[str, Structure]]:
    """50/50 orderings of A and B on the smallest supercells of the shared prototype."""
    comp_a, comp_b = parent_a.composition, parent_b.composition
    diff = [el for el in set(comp_a.elements) | set(comp_b.elements) if comp_a[el] != comp_b[el]]
    a_el = [el for el in diff if comp_a[el] > 0][0]
    b_el = [el for el in diff if comp_b[el] > 0][0]
    host = parent_a.copy()
    # Vegard average volume as the starting geometry; every source relaxes
    host.scale_lattice((parent_a.volume / len(parent_a) + parent_b.volume / len(parent_b)) / 2 * len(parent_a))
    out = []
    conv = SpacegroupAnalyzer(host).get_conventional_standard_structure()
    if proto == "wurtzite":
        cells = {"layered_c": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "layered_a": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}
        base = host
    else:
        # primitive fcc-based cell; [001] (CuAu-I) and [111] (CuPt) layerings
        base = SpacegroupAnalyzer(host).get_primitive_standard_structure()
        cells = {"CuAu_001": [[1, 1, 0], [-1, 1, 0], [0, 0, 1]], "CuPt_111": [[1, 0, 0], [0, 1, 0], [0, 0, 2]]}
    del conv
    for name, mat in cells.items():
        sc = base.copy()
        sc.make_supercell(mat)
        sites = [i for i, s in enumerate(sc) if s.specie == a_el]
        if len(sites) < 2:
            continue
        # order by projection along the stacking direction and swap the upper half
        if name == "CuPt_111" or name == "layered_c":
            axis = sc.lattice.matrix[2]
        else:
            axis = sc.lattice.matrix[0]
        proj = sorted(sites, key=lambda i: (sc[i].coords @ axis, i))
        for i in proj[len(proj) // 2:]:
            sc.replace(i, b_el)
        if sc.composition[a_el] == sc.composition[b_el]:
            out.append((name, sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    zhuo = load_zhuo()
    cand_f = [f for f, g in zhuo.items()
              if GAP_MIN <= g <= GAP_MAX and {str(e) for e in Composition(f).elements} <= ALLOWED]
    parents = sorted({p for _, a, b, _ in ALLOY_PAIRS for p in (a, b)})
    rows = fetch_mp(sorted(set(cand_f) | set(parents)), args.refetch)
    gs = ground_states(rows)
    OUT.mkdir(exist_ok=True)
    index = []
    for f in sorted(cand_f):
        r = gs.get(f)
        if r is None or r["nsites"] > MAX_SITES or r["magnetic"] or (r["ehull"] or 0) > 0.05:
            continue
        s = Structure.from_dict(r["structure"])
        cid = f"arc_{f}_{r['mpid']}"
        write(OUT / f"{cid}.xyz", AseAtomsAdaptor.get_atoms(s), format="extxyz")
        index.append(dict(cid=cid, kind="archived", formula=f, mpid=r["mpid"], spg=r["spg"],
                          spg_number=r["spg_number"], nsites=len(s), property="band_gap",
                          expt_gap=zhuo[f], mp_gga_gap=r["gap"]))
    matcher = StructureMatcher()
    by_formula: dict[str, list[Structure]] = {}
    for r in rows:
        by_formula.setdefault(Composition(r["formula"]).reduced_formula, []).append(Structure.from_dict(r["structure"]))
    alloy_rows = []
    for label, fa, fb, proto in ALLOY_PAIRS:
        gp = ground_states(rows, PROTO_SPG[proto])
        if fa not in gp or fb not in gp:
            print("missing parent", label)
            continue
        alloy_rows.append((label, fa, fb, proto, gp))
    # MP entries for the alloy compositions, for the novelty check
    alloy_struct = []
    for label, fa, fb, proto, gp in alloy_rows:
        pa, pb = Structure.from_dict(gp[fa]["structure"]), Structure.from_dict(gp[fb]["structure"])
        for name, sc in ordered_alloys(pa, pb, proto):
            alloy_struct.append((label, fa, fb, proto, name, sc))
    alloy_formulas = sorted({sc.composition.reduced_formula for *_, sc in alloy_struct})
    extra = fetch_alloy_mp(alloy_formulas, args.refetch)
    for r in extra:
        by_formula.setdefault(Composition(r["formula"]).reduced_formula, []).append(Structure.from_dict(r["structure"]))
    for label, fa, fb, proto, name, sc in alloy_struct:
        f = sc.composition.reduced_formula
        held = [s for s in by_formula.get(f, []) if matcher.fit(s, sc)]
        cid = f"sug_{label}_{proto}_{name}"
        write(OUT / f"{cid}.xyz", AseAtomsAdaptor.get_atoms(sc), format="extxyz")
        index.append(dict(cid=cid, kind="archived_match" if held else "suggested", formula=f,
                          parents=[fa, fb], prototype=proto, ordering=name, nsites=len(sc),
                          property="band_gap", expt_gap=None, mp_match=bool(held)))
    (HERE / "candidates_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    kinds = {}
    for r in index:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(len(index), kinds)


def fetch_alloy_mp(formulas: list[str], refetch: bool) -> list[dict]:
    cache = HERE / "mp_alloy_raw.json.gz"
    if cache.exists() and not refetch:
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    from mp_api.client import MPRester
    rows = []
    with MPRester(mp_key(), mute_progress_bars=True) as m:
        docs = m.materials.summary.search(formula=formulas, fields=["material_id", "formula_pretty", "structure", "band_gap"])
        for d in docs:
            rows.append(dict(mpid=str(d.material_id), formula=d.formula_pretty, gap=d.band_gap,
                             structure=d.structure.as_dict()))
    with gzip.open(cache, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


if __name__ == "__main__":
    main()
