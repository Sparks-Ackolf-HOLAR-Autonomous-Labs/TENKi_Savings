#!/usr/bin/env python
"""Experimentally observed Al-Mg-Si phases that the campaign pool lacks.

The campaign pool (candidates_index.json) keeps MP structures of at most 16
sites. The precipitation sequence of 6xxx alloys and the Al-Mg binary edge
include larger phases that are observed experimentally (ICSD-sourced MP
entries, theoretical=False). They are written to a separate index,
candidates_index_observed.json, so the campaign pool and every published
campaign result stay unchanged:

  beta''  Mg5Si6    mp-568306   C2/m   11 sites (precipitate, HRTEM)
  beta'   Mg9Si5    mp-542854   P6_3/m 28 sites (precipitate, HRTEM)
  gamma   Mg17Al12  mp-2151     I-43m  29 sites (Al-Mg equilibrium phase)
  epsilon Mg23Al30  mp-17659    R-3    53 sites (Al-Mg phase)

U1 (MgAl2Si2, mp-3833), U2 (MgAlSi, mp-4499) and beta (Mg2Si, mp-1367) are
already in the campaign pool. Structures come from the cached MP query
mp_almgsi_raw.json.gz (accessed 28 Sept 2026).

Usage: python add_observed_phases.py
Then:  LIVE_INDEX=candidates_index_observed.json python dgx_sources.py --source <s> --all
"""
import gzip
import json
from pathlib import Path

from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
PHASES = {"mp-568306": "beta_dprime", "mp-542854": "beta_prime",
          "mp-2151": "gamma", "mp-17659": "epsilon"}
# pool members that are observed phases (labels only; structures already exist)
IN_POOL = {"mp_3833_Mg(AlSi)2": "U1", "mp_4499_MgAlSi": "U2", "mp_1367_Mg2Si": "beta"}
# references, isolated solutes and solute clusters (stages 0-2 of the sequence; already in the pool)
REF_CIDS = ["mp_134_Al", "mp_153_Mg", "mp_149_Si", "clu_Al108", "clu_Mg1", "clu_Si1",
            "clu_MgMg_1nn", "clu_SiSi_1nn", "clu_SiSi_2nn", "clu_MgSi_1nn", "clu_MgSi_2nn",
            "clu_Mg2Si_tri", "clu_MgSi2_tri", "clu_Mg2Si2_tet", "clu_Mg3Si_tet", "clu_MgSi3_tet"]


def main():
    rows = {r["mpid"]: r for r in json.load(gzip.open(HERE / "mp_almgsi_raw.json.gz"))}
    pool = {d["cid"]: d for d in json.load(open(HERE / "candidates_index.json"))}
    index = []
    for mpid, label in PHASES.items():
        r = rows[mpid]
        assert not r["theoretical"], mpid
        s = Structure.from_dict(r["structure"])
        cid = f"mp_{mpid.replace('mp-', '')}_{r['formula']}"
        write(HERE / "candidates" / f"{cid}.xyz", AseAtomsAdaptor.get_atoms(s), format="extxyz")
        index.append(dict(cid=cid, kind="observed_phase", phase=label, formula=r["formula"],
                          nsites=r["nsites"], spg=r["spg"], mpid=mpid, mp_ef=r["ef"],
                          mp_ehull=r["ehull"], property="formation_energy"))
    for cid, label in IN_POOL.items():
        index.append(dict(pool[cid], kind="observed_phase", phase=label))
    for cid in REF_CIDS:
        index.append(pool[cid])
    (HERE / "candidates_index_observed.json").write_text(json.dumps(index, indent=1))
    print(len(index), "entries")


if __name__ == "__main__":
    main()
