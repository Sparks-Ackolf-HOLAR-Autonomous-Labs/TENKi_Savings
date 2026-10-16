#!/usr/bin/env python
"""r2SCAN single points with GPAW (PAW datasets, libxc) at the QE PBE geometries.

Second route to a meta-GGA receiver for the Al-Mg-Si pool. The
Quantum ESPRESSO route (qe/R2SCAN.md) uses norm-conserving pseudopotentials
and its r2SCAN formation energies drift with the dense-grid cutoff; GPAW uses
PAW datasets and a plane-wave basis, so it is an independent check of whether
the drift belongs to r2SCAN or to the pseudopotential and grid treatment.

For each cell: SCF in GPAW plane-wave mode at the PBE-relaxed structure
(results/qe_pbe/<cid>.relaxed.xyz), functional r2SCAN
(libxc MGGA_X_R2SCAN+MGGA_C_R2SCAN) or PBE (same settings, for a code check
against QE PBE), Gamma-centred k grid at spacing 0.30 1/A (as the HSE06 and
QE r2SCAN stages), Marzari-Vanderbilt smearing 0.272 eV (0.02 Ry), default
GPAW PAW datasets. Output: results/<outdir>/<cid>.json with the total energy;
formation energies are formed in gpaw_r2scan_analyse.py.

Usage (MPI): mpirun -np N python gpaw_r2scan.py --xc r2scan --ecut 800 --outdir gpaw_r2scan --cids a,b,c
Environment: LIVE_ROOT (campaign root with candidates_index.json and results/qe_pbe/).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from ase.io import read
from ase.parallel import parprint, world
from gpaw import GPAW, PW

ROOT = Path(os.environ.get("LIVE_ROOT", Path(__file__).resolve().parent.parent))
XC = {"r2scan": "MGGA_X_R2SCAN+MGGA_C_R2SCAN", "pbe": "PBE"}
KSPACING = 0.30


def kgrid(atoms, spacing=KSPACING):
    rec = 2 * math.pi * np.linalg.inv(atoms.cell.array).T
    return [max(1, math.ceil(np.linalg.norm(b) / spacing)) for b in rec]


def run(cid, xc, ecut, outdir, work):
    out = ROOT / "results" / outdir / f"{cid}.json"
    if out.exists():
        return
    atoms = read(ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz")
    kp = kgrid(atoms)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    status, e = "ok", None
    try:
        atoms.calc = GPAW(mode=PW(ecut), xc=XC[xc], kpts={"size": kp, "gamma": True},
                          occupations={"name": "marzari-vanderbilt", "width": 0.272},
                          convergence={"energy": 1e-6, "density": 1e-5}, maxiter=300,
                          txt=str(work / f"{cid}_{xc}_{ecut:g}.txt"))
        e = float(atoms.get_potential_energy())
    except Exception as exc:  # unconverged SCF and similar
        status = f"failed: {type(exc).__name__}: {exc}"[:300]
    wall = time.perf_counter() - t0
    if world.rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        rec = {"cid": cid, "source": outdir, "status": status, "energy": e,
               "energy_per_atom": e / len(atoms) if e is not None else None, "natoms": len(atoms),
               "formula": atoms.get_chemical_formula(), "wall_s": wall, "core_s": wall * world.size,
               "ncores": world.size, "node": os.uname().nodename,
               "settings": {"code": "GPAW (PAW, plane waves)", "xc": XC[xc], "ecut_eV": ecut, "kpts": kp,
                            "kspacing_invA": KSPACING, "smearing": "marzari-vanderbilt 0.272 eV",
                            "geometry": "QE PBE relaxed (results/qe_pbe)", "setups": "GPAW default PAW datasets"}}
        out.write_text(json.dumps(rec, indent=1))
    parprint(f"GPAW {xc} {ecut:g} {cid} {status} {wall:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xc", choices=list(XC), default="r2scan")
    ap.add_argument("--ecut", type=float, default=800.0)
    ap.add_argument("--outdir", default="gpaw_r2scan")
    ap.add_argument("--cids", required=True, help="comma-separated cell ids")
    a = ap.parse_args()
    work = Path(os.environ.get("GPAW_WORK", ROOT / "work_gpaw"))
    for cid in a.cids.split(","):
        run(cid, a.xc, a.ecut, a.outdir, work)


if __name__ == "__main__":
    main()
