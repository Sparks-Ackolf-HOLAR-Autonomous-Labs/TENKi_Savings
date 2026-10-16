"""Quantum ESPRESSO driver for the live Al-Mg-Si campaign (CHPC guest partitions).

One SLURM array task runs one chunk of candidates (``chunks.json``). For each
candidate it runs, in order:

* ``qe_pbe``: PBE relaxation. Formation-energy candidates: variable-cell
  relaxation (``vc-relax``). Cluster cells: positions-only relaxation at the
  fixed cubic cell 3 x a_PBE(Al), where a_PBE(Al) comes from a vc-relax of
  fcc Al run inside the same task (deterministic, about a minute).
* ``qe_hse06``: HSE06 single point at the PBE-relaxed geometry
  (formation-energy candidates only).

PBE results for the whole chunk are written before any HSE06 run starts, so
PBE values become available first. Every stage writes
``results/<source>/<cid>.json`` (same schema as the DGX sources) and is
skipped on restart if that file exists, so a preempted and requeued task
resumes where it stopped. Costs are QE's own CPU and wall seconds.

Pseudopotentials: PseudoDojo NC scalar-relativistic v0.4 PBE standard
(Al 3 e, Mg 10 e, Si 4 e). One set for both functionals, so PBE and HSE06
differ only in the exchange-correlation functional.
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
from ase.build import bulk
from ase.io import read, write

ROOT = Path(os.environ.get("LIVE_ROOT", Path(__file__).resolve().parent))
PSEUDO_DIR = ROOT / "pseudos"
PSEUDOS = {"Al": "Al.upf", "Mg": "Mg.upf", "Si": "Si.upf"}
ECUTWFC = 80.0      # Ry; PseudoDojo hints: Mg normal 42 Ha, Al 20 Ha, Si 18 Ha
ECUTRHO = 320.0
ECUTFOCK = 160.0
KSPACING_PBE = 0.20   # 1/A, reciprocal vectors including 2*pi
KSPACING_HSE = 0.30
# q grid = k grid / HSE_QDIV. The convergence test (hse_convergence.py, job 15870621)
# showed that halving the q grid moves Mg2Si by +0.22 eV/atom, so the default is 1.
HSE_QDIV = int(os.environ.get("HSE_QDIV", "1"))
# k-point pools for HSE06: with one pool, 32 ranks leave some ranks without FFT planes
# on small cells (QE fft_type_set error), so the k points are split over pools.
HSE_NPOOL = int(os.environ.get("HSE_NPOOL", "8"))
KSPACING_CLUSTER = 0.25
DEGAUSS = 0.02        # Ry, Marzari-Vanderbilt smearing


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


def pw_times(text: str) -> tuple[float | None, float | None]:
    m = re.findall(r"PWSCF\s*:\s*(.+?)CPU\s+(.+?)WALL", text)
    if not m:
        return None, None

    def secs(s: str) -> float:
        s = s.strip()
        tot = 0.0
        for val, unit in re.findall(r"([\d.]+)([hms])", s):
            tot += float(val) * {"h": 3600, "m": 60, "s": 1}[unit]
        return tot
    cpu, wall = m[-1]
    return secs(cpu), secs(wall)


def final_energy_ev(text: str) -> float | None:
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text, re.M)
    return float(e[-1]) * 13.605693122994 if e else None


def base_input(calc: str, nat: int) -> dict:
    return {
        "control": {"calculation": calc, "prefix": "pw", "outdir": "./tmp", "pseudo_dir": str(PSEUDO_DIR),
                    "tprnfor": True, "tstress": True, "etot_conv_thr": 1e-5 * nat, "forc_conv_thr": 1e-3,
                    "disk_io": "low"},
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
    (d / f"{cid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


def done(source: str, cid: str) -> bool:
    """True if a result exists. A failed HSE06 record from the first settings
    (no force_symmorphic) is retried once with the revised settings."""
    p = ROOT / "results" / source / f"{cid}.json"
    if not p.exists():
        return False
    if source == "qe_hse06":
        r = json.loads(p.read_text())
        if r["status"] != "ok" and not r["settings"].get("force_symmorphic"):
            p.rename(p.with_name(f"{cid}.attempt1.json"))
            return False
        st = r.get("settings", {})
        if r["status"] != "ok" and st.get("q_div") == 1 and "npool" not in st:
            # failed at the start with one k-point pool (fft_type_set / scale_sym_ops)
            p.rename(p.with_name(f"{cid}.attempt_np1.json"))
            return False
        if r["status"] != "ok" and st.get("npool") not in (None, HSE_NPOOL):
            # failed with another k-point pool count (out of memory with 8
            # pools, or more pools than k-points): retry with HSE_NPOOL
            p.rename(p.with_name(f"{cid}.attempt_npool{st['npool']}.json"))
            return False
        if HSE_QDIV == 1 and st.get("nq") and st.get("nq") != st.get("kpts"):
            # value from the unconverged half q grid: recompute with the full q grid
            p.rename(p.with_name(f"{cid}.qhalf.json"))
            return False
    return True


def record(cid, source, atoms, e_ev, rc, wall, text, settings, extra=None):
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
    if extra:
        rec.update(extra)
    if not ok:
        rec["error"] = f"rc={rc}; tail: " + " | ".join(text.strip().splitlines()[-5:])
    return rec


def pbe_relax(cid: str, atoms: Atoms, fixed_cell: bool, work: Path):
    nat = len(atoms)
    inp = base_input("relax" if fixed_cell else "vc-relax", nat)
    kp = kgrid(atoms, KSPACING_CLUSTER if fixed_cell else KSPACING_PBE)
    npool = 2 if nat > 32 else 4
    rc, wall, text, out = run_pw(work / cid / "pbe", "pbe", atoms, inp, kp, npool)
    try:
        final = read(out, format="espresso-out", index=-1)
    except Exception:  # noqa: BLE001
        final = atoms
    settings = {"functional": "PBE", "calculation": inp["control"]["calculation"], "ecutwfc_Ry": ECUTWFC,
                "ecutrho_Ry": ECUTRHO, "kpts": kp, "smearing": "mv", "degauss_Ry": DEGAUSS,
                "pseudos": "PseudoDojo NC SR v0.4 PBE standard"}
    return record(cid, "qe_pbe", final, final_energy_ev(text), rc, wall, text, settings), out, final


def hse_single_point(cid: str, atoms: Atoms, work: Path):
    nat = len(atoms)
    inp = base_input("scf", nat)
    kp = kgrid(atoms, KSPACING_HSE)
    nq = [max(1, math.ceil(k / HSE_QDIV)) for k in kp]
    inp["system"].update({"input_dft": "hse", "ecutfock": ECUTFOCK, "nqx1": nq[0], "nqx2": nq[1], "nqx3": nq[2],
                          "exxdiv_treatment": "gygi-baldereschi", "x_gamma_extrapolation": True,
                          # drop fractional translations: with EXX they can be incompatible with the
                          # FFT grid (QE error in scale_sym_ops); changes k-point reduction only
                          "force_symmorphic": True})
    # metallic hybrid SCF: gentler local-TF mixing and more steps per EXX cycle
    inp["electrons"].update({"mixing_beta": 0.2, "mixing_mode": "local-TF", "electron_maxstep": 300})
    # single point: QE cannot compute hybrid-functional stress with k-point pools
    inp["control"].update({"tstress": False, "tprnfor": False})
    rc, wall, text, out = run_pw(work / cid / "hse", "hse", atoms, inp, kp, HSE_NPOOL)
    nosym = False
    if "scale_sym_ops" in text:
        # FFT grid incompatible with the symmetry operations even without fractional
        # translations: rerun without symmetry (changes k-point reduction only)
        inp["system"]["nosym"] = True
        nosym = True
        rc, wall, text, out = run_pw(work / cid / "hse", "hse", atoms, inp, kp, HSE_NPOOL)
    settings = {"functional": "HSE06", "calculation": "scf at PBE geometry", "ecutwfc_Ry": ECUTWFC,
                "ecutfock_Ry": ECUTFOCK, "kpts": kp, "nq": nq, "q_div": HSE_QDIV, "npool": HSE_NPOOL, "nosym": nosym, "smearing": "mv", "degauss_Ry": DEGAUSS,
                "force_symmorphic": True, "mixing": "local-TF beta 0.2, maxstep 300",
                "pseudos": "PseudoDojo NC SR v0.4 PBE standard"}
    return record(cid, "qe_hse06", atoms, final_energy_ev(text), rc, wall, text, settings), out


def al_lattice_parameter(work: Path) -> float:
    cache = ROOT / "results" / "qe_pbe_al_lattice.json"
    if cache.exists():
        return json.loads(cache.read_text())["a_A"]
    al = bulk("Al", "fcc", a=4.04)
    rec, out, final = pbe_relax("al_prim_for_lattice", al, fixed_cell=False, work=work)
    a = float((final.get_volume() * 4) ** (1 / 3))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"a_A": a, "energy_per_atom": rec["energy_per_atom"]}))
    tmp.replace(cache)
    return a


def main():
    chunks = json.loads((Path(__file__).resolve().parent / "chunks.json").read_text())
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else "0"))
    group = os.environ.get("CHUNK_GROUP", "formation")
    cids = chunks[group][task]
    index = {r["cid"]: r for r in json.loads((ROOT / os.environ.get("LIVE_INDEX", "candidates_index.json")).read_text())}
    work = Path(os.environ.get("QE_WORK", ROOT / "work"))
    print(f"task={task} group={group} cids={cids} node={os.environ.get('SLURMD_NODENAME')} "
          f"restart={os.environ.get('SLURM_RESTART_COUNT', '0')}", flush=True)
    relaxed: dict[str, Atoms] = {}
    for cid in cids:
        atoms = read(ROOT / "candidates" / f"{cid}.xyz")
        cluster = index[cid]["property"] == "cluster_binding"
        if done("qe_pbe", cid) or os.environ.get("HSE_ONLY"):
            p = ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz"
            relaxed[cid] = read(p) if p.exists() else atoms
            continue
        if cluster:
            a = al_lattice_parameter(work)
            atoms.set_cell(np.eye(3) * 3 * a, scale_atoms=True)
        rec, out, final = pbe_relax(cid, atoms, fixed_cell=cluster, work=work)
        save("qe_pbe", cid, rec, out, final)
        relaxed[cid] = final
        print(f"PBE {cid} {rec['status']} wall={rec['wall_s']:.0f}s", flush=True)
    for cid in cids:
        if os.environ.get("SKIP_HSE"):   # PBE-only pass (HSE06 in a later pass)
            break
        if index[cid]["property"] != "formation_energy" or done("qe_hse06", cid):
            continue
        pbe = json.loads((ROOT / "results" / "qe_pbe" / f"{cid}.json").read_text())
        if pbe["status"] != "ok":
            continue
        rec, out = hse_single_point(cid, relaxed[cid], work)
        save("qe_hse06", cid, rec, out, None)
        print(f"HSE06 {cid} {rec['status']} wall={rec['wall_s']:.0f}s", flush=True)
    print("TASK DONE", flush=True)


if __name__ == "__main__":
    main()
