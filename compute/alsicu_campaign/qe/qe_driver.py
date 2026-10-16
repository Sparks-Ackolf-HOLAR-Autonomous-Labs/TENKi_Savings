"""Quantum ESPRESSO driver for the computed Al-Si-Cu pool (CHPC guest partitions).

Adapted from ../../almgsi_campaign/qe/qe_driver.py, PBE only (no HSE06, no
solute clusters). One SLURM array task runs one chunk of candidates
(``chunks.json``). For each candidate it runs a PBE variable-cell relaxation
(``vc-relax``) and writes ``results/qe_pbe/<cid>.json`` (same schema as the
DGX sources), the gzipped pw.x output and the relaxed structure. A
candidate whose result exists is skipped, so a preempted and requeued task
resumes where it stopped. Costs are QE's own CPU and wall seconds.

Settings: QE 7.4, PBE, PseudoDojo NC scalar-relativistic v0.4 PBE standard
(Al 3 e, Si 4 e, Cu 19 e), ecutwfc 90 Ry / ecutrho 360 Ry (Cu needs the
higher cutoff; PseudoDojo normal hint for Cu is about 46 Ha), Monkhorst-Pack
grid from a k-spacing of 0.20 1/A (reciprocal vectors including 2 pi),
Marzari-Vanderbilt smearing 0.02 Ry, BFGS with at most 150 ionic steps.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

ROOT = Path(os.environ.get("LIVE_ROOT", Path(__file__).resolve().parent.parent))
PSEUDO_DIR = ROOT / "pseudos"
PSEUDOS = {"Al": "Al.upf", "Si": "Si.upf", "Cu": "Cu.upf"}
ECUTWFC = 90.0      # Ry
ECUTRHO = 360.0
KSPACING_PBE = 0.20   # 1/A
DEGAUSS = 0.02        # Ry, Marzari-Vanderbilt smearing
NSTEP = 150


def ncores() -> int:
    return int(os.environ.get("SLURM_NTASKS", os.environ.get("QE_NP", "4")))


def kgrid(atoms: Atoms, spacing: float) -> list[int]:
    rec = 2 * math.pi * np.linalg.inv(atoms.cell.array).T
    return [max(1, math.ceil(np.linalg.norm(b) / spacing)) for b in rec]


def run_pw(workdir: Path, prefix: str, atoms: Atoms, input_data: dict, kpts, npool: int):
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / f"{prefix}.in"
    out = workdir / f"{prefix}.out"
    write(inp, atoms, format="espresso-in", input_data=input_data,
          pseudopotentials={el: PSEUDOS[el] for el in set(atoms.get_chemical_symbols())},
          kpts=tuple(kpts), koffset=(0, 0, 0))
    cmd = ["mpirun", "-np", str(ncores()), "pw.x", "-nk", str(npool), "-in", inp.name]
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=workdir, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    text = out.read_text(encoding="utf-8", errors="replace")
    return rc, wall, text, out


def pw_times(text: str):
    m = re.findall(r"PWSCF\s*:\s*(.+?)CPU\s+(.+?)WALL", text)
    if not m:
        return None, None

    def secs(s: str) -> float:
        tot = 0.0
        for val, unit in re.findall(r"([\d.]+)([hms])", s.strip()):
            tot += float(val) * {"h": 3600, "m": 60, "s": 1}[unit]
        return tot
    cpu, wall = m[-1]
    return secs(cpu), secs(wall)


def final_energy_ev(text: str):
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text, re.M)
    return float(e[-1]) * 13.605693122994 if e else None


def base_input(nat: int) -> dict:
    return {
        "control": {"calculation": "vc-relax", "prefix": "pw", "outdir": "./tmp", "pseudo_dir": str(PSEUDO_DIR),
                    "tprnfor": True, "tstress": True, "etot_conv_thr": 1e-5 * nat, "forc_conv_thr": 1e-3,
                    "nstep": NSTEP, "disk_io": "low"},
        "system": {"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO, "occupations": "smearing",
                   "smearing": "mv", "degauss": DEGAUSS},
        "electrons": {"conv_thr": 1e-9 * nat, "mixing_beta": 0.4},
        "ions": {"ion_dynamics": "bfgs"},
        "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 0.5},
    }


def save(source: str, cid: str, rec: dict, out: Path | None, atoms: Atoms | None):
    d = ROOT / "results" / source
    d.mkdir(parents=True, exist_ok=True)
    if out is not None and out.exists():
        with open(out, "rb") as src, gzip.open(d / f"{cid}.pw.out.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    if atoms is not None:
        write(d / f"{cid}.relaxed.xyz", atoms, format="extxyz")
    tmp = d / f"{cid}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(d / f"{cid}.json")


def done(source: str, cid: str) -> bool:
    return (ROOT / "results" / source / f"{cid}.json").exists()


def record(cid, source, atoms, e_ev, rc, wall, text, settings):
    cpu, qe_wall = pw_times(text)
    ok = rc == 0 and e_ev is not None and "JOB DONE" in text
    if "convergence NOT achieved" in text or "The maximum number of steps has been reached" in text:
        ok = False
    rec = {"cid": cid, "source": source, "status": "ok" if ok else "failed",
           "natoms": len(atoms), "energy_per_atom": e_ev / len(atoms) if ok else None,
           "energy_per_atom_sem": 0.0, "volume_per_atom": atoms.get_volume() / len(atoms),
           "wall_s": wall, "qe_cpu_s": cpu, "qe_wall_s": qe_wall, "core_s": wall * ncores(),
           "device": f"cpu:{ncores()}", "node": os.environ.get("SLURMD_NODENAME"),
           "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")), "settings": settings}
    if not ok:
        rec["error"] = f"rc={rc}; tail: " + " | ".join(text.strip().splitlines()[-5:])
    return rec


def pbe_relax(cid: str, atoms: Atoms, work: Path):
    nat = len(atoms)
    inp = base_input(nat)
    kp = kgrid(atoms, KSPACING_PBE)
    npool = 4 if ncores() >= 8 else 1
    rc, wall, text, out = run_pw(work / cid / "pbe", "pbe", atoms, inp, kp, npool)
    try:
        final = read(out, format="espresso-out", index=-1)
    except Exception:  # noqa: BLE001
        final = atoms
    settings = {"functional": "PBE", "calculation": "vc-relax", "ecutwfc_Ry": ECUTWFC,
                "ecutrho_Ry": ECUTRHO, "kpts": kp, "kspacing_inv_A": KSPACING_PBE, "smearing": "mv",
                "degauss_Ry": DEGAUSS, "nstep": NSTEP, "npool": npool,
                "pseudos": "PseudoDojo NC SR v0.4 PBE standard (Al 3 e, Si 4 e, Cu 19 e)"}
    return record(cid, "qe_pbe", final, final_energy_ev(text), rc, wall, text, settings), out, final


def main():
    chunks = json.loads((Path(__file__).resolve().parent / "chunks.json").read_text())
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else "0"))
    group = os.environ.get("CHUNK_GROUP", "formation")
    cids = chunks[group][task]
    work = Path(os.environ.get("QE_WORK", ROOT / "work"))
    print(f"task={task} group={group} cids={cids} node={os.environ.get('SLURMD_NODENAME')} "
          f"restart={os.environ.get('SLURM_RESTART_COUNT', '0')}", flush=True)
    for cid in cids:
        if done("qe_pbe", cid):
            continue
        atoms = read(ROOT / "candidates" / f"{cid}.xyz")
        rec, out, final = pbe_relax(cid, atoms, work)
        save("qe_pbe", cid, rec, out, final)
        shutil.rmtree(work / cid / "pbe" / "tmp", ignore_errors=True)  # wavefunctions, not needed
        print(f"PBE {cid} {rec['status']} wall={rec['wall_s']:.0f}s", flush=True)
    print("TASK DONE", flush=True)


if __name__ == "__main__":
    main()
