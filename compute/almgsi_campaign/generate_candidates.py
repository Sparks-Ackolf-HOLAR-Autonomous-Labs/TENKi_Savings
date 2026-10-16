"""Stage 1 of the live Al-Mg-Si campaign: build the candidate pool.

Three kinds of candidate, all written as extended-XYZ with an index file:

1. ``archived``: Materials Project structures in the Al-Mg-Si system with
   at most 16 sites and energy above hull <= 0.1 eV/atom (fetched live with
   MP_API; the fetch is cached in ``mp_almgsi_raw.json``). These are the
   candidates a replay over the archive can also see.
2. ``suggested``: ordered decorations of small fcc supercells (the Al
   matrix lattice) with Al >= 50 at.%. These are coherent, on-lattice
   orderings of the kind found in early solute clusters and GP zones. Each
   is matched against every MP structure of the same composition; only the
   ones MP does not hold stay ``suggested`` (the archive cannot answer them).
3. ``cluster``: solute clusters (Mg and Si, 1 to 4 atoms, nearest-neighbour
   geometries) in a 3x3x3 conventional fcc Al supercell (108 sites), plus
   the pure-Al supercell reference. The property for these is the cluster
   binding energy, not the formation energy.

Usage: python generate_candidates.py [--refetch]
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
from pathlib import Path

import numpy as np
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


def fetch_mp(refetch: bool) -> list[dict]:
    cache = HERE / "mp_almgsi_raw.json.gz"
    if cache.exists() and not refetch:
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY") or os.environ["MP_API"]
    with MPRester(key, mute_progress_bars=True) as m:
        docs = m.materials.summary.search(
            chemsys=["Al", "Mg", "Si", "Al-Mg", "Al-Si", "Mg-Si", "Al-Mg-Si"],
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
    cells = [("conv4", conv),                       # L1_2, L1_0, and ternary 4-site
             ("conv1x1x2", conv.repeat((1, 1, 2))),  # 8-site tetragonal
             ("prim2x2x2", prim.repeat((2, 2, 2)))]  # 8-site rhombohedral
    return cells


def enumerate_suggested(mp_rows: list[dict]) -> list[dict]:
    matcher = StructureMatcher(ltol=0.25, stol=0.35, angle_tol=6, primitive_cell=True, scale=True,
                               attempt_supercell=True)
    mp_by_comp: dict[str, list[Structure]] = {}
    for r in mp_rows:
        s = Structure.from_dict(r["structure"])
        mp_by_comp.setdefault(s.composition.reduced_formula, []).append((r["mpid"], s))
    uniq: list[tuple[Structure, dict]] = []
    for cname, cell in fcc_supercells():
        n = len(cell)
        for n_mg in range(0, n // 2 + 1):
            for n_si in range(0, n // 2 + 1 - n_mg):
                n_al = n - n_mg - n_si
                if n_al < n / 2 or (n_mg == 0 and n_si == 0):
                    continue
                for mg_sites in itertools.combinations(range(n), n_mg):
                    rest = [i for i in range(n) if i not in mg_sites]
                    for si_sites in itertools.combinations(rest, n_si):
                        sym = ["Al"] * n
                        for i in mg_sites:
                            sym[i] = "Mg"
                        for i in si_sites:
                            sym[i] = "Si"
                        a = cell.copy()
                        a.set_chemical_symbols(sym)
                        s = AseAtomsAdaptor.get_structure(a)
                        if any(s.composition.reduced_composition == u.composition.reduced_composition
                               and matcher.fit(s, u) for u, _ in uniq):
                            continue
                        uniq.append((s, dict(cell=cname)))
    out = []
    for s, meta in uniq:
        rf = s.composition.reduced_formula
        mp_hit = None
        for mpid, ms in mp_by_comp.get(rf, []):
            if matcher.fit(s, ms):
                mp_hit = mpid
                break
        prim = s.get_primitive_structure()
        out.append(dict(structure=prim, kind="archived_fcc" if mp_hit else "suggested",
                        mp_match=mp_hit, cell=meta["cell"]))
    return out


def solute_clusters() -> list[dict]:
    """Mg/Si clusters on nearest-neighbour sites of a 3x3x3 conventional Al cell."""
    host = bulk("Al", "fcc", a=A_AL, cubic=True).repeat((3, 3, 3))
    pos = host.get_positions()
    centre = np.argmin(np.linalg.norm(pos - host.cell.sum(0) / 2, axis=1))
    d = host.get_distances(centre, range(len(host)), mic=True)
    nn1 = [int(i) for i in np.where(np.abs(d - A_AL / np.sqrt(2)) < 0.05)[0]]
    nn2 = [int(i) for i in np.where(np.abs(d - A_AL) < 0.05)[0]]
    # a nearest-neighbour triangle and tetrahedron containing the centre
    j = nn1[0]
    dj = host.get_distances(j, nn1, mic=True)
    k = nn1[int(np.where(np.abs(dj - A_AL / np.sqrt(2)) < 0.05)[0][0])]
    tri = [centre, j, k]
    common = [m for m in nn1 if m not in (j, k)
              and abs(host.get_distance(m, j, mic=True) - A_AL / np.sqrt(2)) < 0.05
              and abs(host.get_distance(m, k, mic=True) - A_AL / np.sqrt(2)) < 0.05]
    tet = tri + [common[0]]
    specs = [
        ("Al108", []),
        ("Mg1", [(centre, "Mg")]),
        ("Si1", [(centre, "Si")]),
        ("MgSi_1nn", [(centre, "Mg"), (j, "Si")]),
        ("MgSi_2nn", [(centre, "Mg"), (nn2[0], "Si")]),
        ("MgMg_1nn", [(centre, "Mg"), (j, "Mg")]),
        ("SiSi_1nn", [(centre, "Si"), (j, "Si")]),
        ("SiSi_2nn", [(centre, "Si"), (nn2[0], "Si")]),
        ("Mg2Si_tri", [(tri[0], "Mg"), (tri[1], "Mg"), (tri[2], "Si")]),
        ("MgSi2_tri", [(tri[0], "Mg"), (tri[1], "Si"), (tri[2], "Si")]),
        ("Mg2Si2_tet", [(tet[0], "Mg"), (tet[1], "Mg"), (tet[2], "Si"), (tet[3], "Si")]),
        ("Mg3Si_tet", [(tet[0], "Mg"), (tet[1], "Mg"), (tet[2], "Mg"), (tet[3], "Si")]),
        ("MgSi3_tet", [(tet[0], "Mg"), (tet[1], "Si"), (tet[2], "Si"), (tet[3], "Si")]),
    ]
    out = []
    for name, subs in specs:
        a = host.copy()
        for i, el in subs:
            a[i].symbol = el
        out.append(dict(cid=f"clu_{name}", atoms=a, kind="cluster" if subs and len(subs) > 1 else "cluster_ref"))
    return out


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
        a = AseAtomsAdaptor.get_atoms(s)
        write(OUT / f"{cid}.xyz", a, format="extxyz")
        index.append(dict(cid=cid, kind="reference" if len(s.composition) == 1 and r["ehull"] == 0 else "archived",
                          formula=r["formula"], nsites=r["nsites"], spg=r["spg"],
                          mpid=r["mpid"], mp_ef=r["ef"], mp_ehull=r["ehull"], property="formation_energy"))

    # enumerated fcc orderings
    sugg = enumerate_suggested(mp_rows)
    n_s = 0
    for i, c in enumerate(sugg):
        s = c["structure"]
        cid = f"fcc_{s.composition.reduced_formula}_{i:02d}"
        a = AseAtomsAdaptor.get_atoms(s)
        write(OUT / f"{cid}.xyz", a, format="extxyz")
        n_s += c["kind"] == "suggested"
        index.append(dict(cid=cid, kind=c["kind"], formula=s.composition.reduced_formula,
                          nsites=len(s), spg=s.get_space_group_info(symprec=0.1)[0],
                          mpid=c["mp_match"], cell=c["cell"], property="formation_energy"))

    for c in solute_clusters():
        write(OUT / f"{c['cid']}.xyz", c["atoms"], format="extxyz")
        index.append(dict(cid=c["cid"], kind=c["kind"], formula=c["atoms"].get_chemical_formula(),
                          nsites=len(c["atoms"]), property="cluster_binding"))

    (HERE / "candidates_index.json").write_text(json.dumps(index, indent=1))
    from collections import Counter
    print(Counter(r["kind"] for r in index), "suggested:", n_s)


if __name__ == "__main__":
    main()
