"""Stage 1 of the computed Al-Si-Cu pool: build the candidate pool.

Adapted from ../almgsi_campaign/generate_candidates.py (Al-Mg-Si). Two kinds of
candidate, all written as extended-XYZ with an index file; the property is
the formation energy per atom for every candidate (no solute clusters):

1. ``archived``: Materials Project structures in the Al-Si-Cu system with
   at most 16 sites and energy above hull <= 0.1 eV/atom (fetched live with
   the MP API; the fetch is cached in ``mp_alsicu_raw.json.gz``). The MP
   formation energy is stored as ``mp_ef``. The elemental ground states
   (fcc Al, diamond Si, fcc Cu) are written with kind ``reference``.
2. ``suggested``: ordered decorations of small fcc supercells (the Al
   matrix lattice: 4-site cubic, 8-site tetragonal 1x1x2, 8-site
   rhombohedral 2x2x2 primitive) with Al >= 50 at.%, symmetry-distinct
   under pymatgen StructureMatcher. Each is matched against every MP
   structure of the same composition; only the ones MP does not hold stay
   ``suggested``. Decorations that MP does hold are kept as
   ``archived_fcc`` (with the matching mpid and its MP formation energy).

Size control: if the number of suggested decorations exceeds
MAX_SUGGESTED, a deterministic subsample is taken: the suggested list is
sorted by (cell, composition, enumeration order) and every k-th entry is
kept with k = ceil(n / MAX_SUGGESTED). The 4-site cubic decorations are
always kept. The subsample rule and counts are written to
``candidates_meta.json``.

Usage: python generate_candidates.py [--refetch]
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.io import write
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
OUT = HERE / "candidates"
A_AL = 4.04  # fcc Al lattice parameter (A), starting value only; all sources relax
MAX_ARCHIVED_SITES = 16
EHULL_MAX = 0.1
MAX_SUGGESTED = 150
CHEMSYS = ["Al", "Si", "Cu", "Al-Si", "Al-Cu", "Cu-Si", "Al-Si-Cu"]
SOLUTES = ("Si", "Cu")


def fetch_mp(refetch: bool) -> list[dict]:
    cache = HERE / "mp_alsicu_raw.json.gz"
    if cache.exists() and not refetch:
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY") or os.environ["MP_API"]
    with MPRester(key, mute_progress_bars=True) as m:
        docs = m.materials.summary.search(
            chemsys=CHEMSYS,
            fields=["material_id", "formula_pretty", "nsites", "energy_above_hull",
                    "formation_energy_per_atom", "symmetry", "structure", "theoretical"])
    rows = [dict(mpid=str(d.material_id), formula=d.formula_pretty, nsites=d.nsites,
                 ehull=d.energy_above_hull, ef=d.formation_energy_per_atom,
                 spg=d.symmetry.symbol, spg_number=d.symmetry.number,
                 theoretical=d.theoretical, structure=d.structure.as_dict())
            for d in docs]
    rows.sort(key=lambda r: (r["formula"], r["ehull"], r["mpid"]))
    with gzip.open(cache, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


def fcc_supercells() -> list[tuple[str, Atoms]]:
    """Small fcc cells whose decorations give the classic ordered prototypes."""
    prim = bulk("Al", "fcc", a=A_AL)
    conv = bulk("Al", "fcc", a=A_AL, cubic=True)
    return [("conv4", conv),                       # L1_2, L1_0, and ternary 4-site
            ("conv1x1x2", conv.repeat((1, 1, 2))),  # 8-site tetragonal
            ("prim2x2x2", prim.repeat((2, 2, 2)))]  # 8-site rhombohedral


def enumerate_fcc(mp_rows: list[dict]) -> list[dict]:
    matcher = StructureMatcher(ltol=0.25, stol=0.35, angle_tol=6, primitive_cell=True, scale=True,
                               attempt_supercell=True)
    mp_by_comp: dict[str, list] = {}
    for r in mp_rows:
        s = Structure.from_dict(r["structure"])
        mp_by_comp.setdefault(s.composition.reduced_formula, []).append((r["mpid"], r["ef"], s))
    a_el, b_el = SOLUTES
    uniq: list[tuple[Structure, dict]] = []
    for cname, cell in fcc_supercells():
        n = len(cell)
        for n_a in range(0, n // 2 + 1):
            for n_b in range(0, n // 2 + 1 - n_a):
                n_al = n - n_a - n_b
                if n_al < n / 2 or (n_a == 0 and n_b == 0):
                    continue
                for a_sites in itertools.combinations(range(n), n_a):
                    rest = [i for i in range(n) if i not in a_sites]
                    for b_sites in itertools.combinations(rest, n_b):
                        sym = ["Al"] * n
                        for i in a_sites:
                            sym[i] = a_el
                        for i in b_sites:
                            sym[i] = b_el
                        at = cell.copy()
                        at.set_chemical_symbols(sym)
                        s = AseAtomsAdaptor.get_structure(at)
                        if any(s.composition.reduced_composition == u.composition.reduced_composition
                               and matcher.fit(s, u) for u, _ in uniq):
                            continue
                        uniq.append((s, dict(cell=cname)))
    out = []
    for order, (s, meta) in enumerate(uniq):
        rf = s.composition.reduced_formula
        hit, hit_ef = None, None
        for mpid, ef, ms in mp_by_comp.get(rf, []):
            if matcher.fit(s, ms):
                hit, hit_ef = mpid, ef
                break
        out.append(dict(structure=s.get_primitive_structure(), kind="archived_fcc" if hit else "suggested",
                        mp_match=hit, mp_ef=hit_ef, cell=meta["cell"], order=order))
    return out


def subsample(cands: list[dict]) -> tuple[list[dict], dict]:
    sugg = [c for c in cands if c["kind"] == "suggested"]
    info = {"n_suggested_enumerated": len(sugg), "max_suggested": MAX_SUGGESTED, "rule": "none (all kept)"}
    if len(sugg) <= MAX_SUGGESTED:
        return cands, info
    keep_always = [c for c in sugg if c["cell"] == "conv4"]
    rest = sorted((c for c in sugg if c["cell"] != "conv4"),
                  key=lambda c: (c["cell"], c["structure"].composition.reduced_formula, c["order"]))
    k = math.ceil(len(rest) / max(1, MAX_SUGGESTED - len(keep_always)))
    kept = {id(c) for c in keep_always} | {id(c) for c in rest[::k]}
    info["rule"] = (f"all conv4 decorations kept; the other suggested decorations sorted by "
                    f"(cell, reduced formula, enumeration order) and every {k}-th kept")
    out = [c for c in cands if c["kind"] != "suggested" or id(c) in kept]
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    mp_rows = fetch_mp(args.refetch)
    index = []

    # archived MP candidates (plus the three elemental ground states as references)
    for r in mp_rows:
        if r["nsites"] > MAX_ARCHIVED_SITES or r["ehull"] > EHULL_MAX:
            continue
        s = Structure.from_dict(r["structure"])
        cid = f"mp_{r['mpid'].replace('mp-', '')}_{r['formula']}"
        write(OUT / f"{cid}.xyz", AseAtomsAdaptor.get_atoms(s), format="extxyz")
        index.append(dict(cid=cid, kind="reference" if len(s.composition) == 1 and r["ehull"] == 0 else "archived",
                          formula=r["formula"], nsites=r["nsites"], spg=r["spg"],
                          mpid=r["mpid"], mp_ef=r["ef"], mp_ehull=r["ehull"], property="formation_energy"))

    cands, info = subsample(enumerate_fcc(mp_rows))
    n_s = 0
    for i, c in enumerate(cands):
        s = c["structure"]
        cid = f"fcc_{s.composition.reduced_formula}_{c['order']:03d}"
        write(OUT / f"{cid}.xyz", AseAtomsAdaptor.get_atoms(s), format="extxyz")
        n_s += c["kind"] == "suggested"
        d = dict(cid=cid, kind=c["kind"], formula=s.composition.reduced_formula,
                 nsites=len(s), spg=s.get_space_group_info(symprec=0.1)[0],
                 mpid=c["mp_match"], cell=c["cell"], property="formation_energy")
        if c["mp_ef"] is not None:
            d["mp_ef"] = c["mp_ef"]
        index.append(d)

    (HERE / "candidates_index.json").write_text(json.dumps(index, indent=1))
    counts = Counter(r["kind"] for r in index)
    info.update({"n_mp_entries_fetched": len(mp_rows), "counts": dict(counts), "n_suggested_kept": n_s,
                 "chemsys": CHEMSYS, "max_archived_sites": MAX_ARCHIVED_SITES, "ehull_max_eV": EHULL_MAX})
    (HERE / "candidates_meta.json").write_text(json.dumps(info, indent=1))
    print(counts, "suggested kept:", n_s, "of", info["n_suggested_enumerated"])


if __name__ == "__main__":
    main()
