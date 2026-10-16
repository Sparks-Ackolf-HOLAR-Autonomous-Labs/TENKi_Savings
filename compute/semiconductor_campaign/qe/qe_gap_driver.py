"""Quantum ESPRESSO band-gap driver for the computed semiconductor campaign.

One SLURM array task runs one chunk of candidates (``chunks.json``). For each
candidate, in order:

* ``qe_pbe``: PBE variable-cell relaxation, then a PBE SCF at the relaxed
  geometry with empty bands; the gap is read from the Kohn-Sham eigenvalues.
* ``qe_hse06``: HSE06 SCF at the PBE geometry with the full q grid
  (q grid = k grid); the gap is read the same way.

The gap is CBM - VBM over the Gamma-centred k mesh of that calculation
(band index N_el/2 against N_el/2 + 1), so it covers Gamma and, with even
meshes, the X/L-type zone-boundary points of cubic cells. A negative value
(band overlap at the PBE level, for example InN) is kept as the raw value
and the gap is reported as zero. All calculations are spin-unpolarised and
scalar-relativistic (no spin-orbit coupling), with PseudoDojo NC SR v0.4
PBE standard pseudopotentials for both functionals.

Every stage writes ``results/<source>/<cid>.json`` and is skipped on
restart if that file exists, so a preempted and requeued task resumes.
Costs are QE's own CPU and wall seconds and the core-seconds of the task.
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
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

ROOT = Path(os.environ.get("LIVE_ROOT", Path(__file__).resolve().parent.parent))
PSEUDO_DIR = ROOT / "pseudos"
ECUTWFC = 90.0
ECUTRHO = 360.0
ECUTFOCK = 180.0
KSPACING_RELAX = 0.30
KSPACING_PBE = 0.22
KSPACING_HSE = 0.30
# q grid = k grid / HSE_QDIV. Semiconductors converge much faster in the q grid
# than the Al-Mg-Si metals did (where the full q grid was needed); the setting is
# checked on GaAs, ZnO and MgO by hse_qconv (q = k against q = k/2).
HSE_QDIV = int(os.environ.get("HSE_QDIV", "2"))
HA = 27.211386245988
RY = 13.605693122994


def ncores() -> int:
    return int(os.environ.get("SLURM_NTASKS", os.environ.get("QE_NP", "4")))


def kgrid(atoms: Atoms, spacing: float) -> list[int]:
    """Gamma-centred grid; even in every periodic direction so that zone-boundary
    points (X, L-type) are on the mesh."""
    rec = 2 * math.pi * np.linalg.inv(atoms.cell.array).T
    out = []
    for b in rec:
        n = max(2, math.ceil(np.linalg.norm(b) / spacing))
        out.append(n + (n % 2))
    return out


def zval(el: str) -> float:
    txt = (PSEUDO_DIR / f"{el}.upf").read_text(errors="replace")
    m = re.search(r'z_valence\s*=\s*"\s*([-\d.Ee+]+)', txt)
    return float(m.group(1))


def nelec(atoms: Atoms) -> float:
    return sum(zval(el) for el in atoms.get_chemical_symbols())


def run_pw(workdir: Path, prefix: str, atoms: Atoms, input_data: dict, kpts, npool: int):
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / f"{prefix}.in"
    out = workdir / f"{prefix}.out"
    write(inp, atoms, format="espresso-in", input_data=input_data,
          pseudopotentials={el: f"{el}.upf" for el in set(atoms.get_chemical_symbols())},
          kpts=tuple(kpts), koffset=(0, 0, 0))
    t0 = time.time()
    text = ""
    for np_try in (npool, 1):
        cmd = ["mpirun", "-np", str(ncores()), "pw.x", "-nk", str(np_try), "-in", inp.name]
        with open(out, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=workdir, stdout=fh, stderr=subprocess.STDOUT).returncode
        text = out.read_text(encoding="utf-8", errors="replace")
        if rc == 0 or np_try == 1 or not re.search(r"some nodes have no k-points|npool|fft_type_set", text):
            break
    return rc, time.time() - t0, text, out


def pw_times(text: str):
    m = re.findall(r"PWSCF\s*:\s*(.+?)CPU\s+(.+?)WALL", text)
    if not m:
        return None, None

    def secs(s: str) -> float:
        return sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"([\d.]+)([hms])", s))
    return secs(m[-1][0]), secs(m[-1][1])


def gap_from_xml(save_dir: Path, nocc: int):
    xml = save_dir / "data-file-schema.xml"
    root = ET.parse(xml).getroot()
    eig = []
    for ks in root.iter("ks_energies"):
        e = np.array(ks.find("eigenvalues").text.split(), float) * HA
        eig.append(e)
    eig = np.array(eig)
    vbm = float(eig[:, nocc - 1].max())
    cbm = float(eig[:, nocc].min())
    kv = int(eig[:, nocc - 1].argmax())
    kc = int(eig[:, nocc].argmin())
    return {"gap_raw_eV": cbm - vbm, "gap_eV": max(0.0, cbm - vbm), "vbm_eV": vbm, "cbm_eV": cbm,
            "direct_on_mesh": kv == kc, "nk_irr": len(eig)}


def base_input(calc: str, nat: int, nbnd: int | None) -> dict:
    sysd = {"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO, "occupations": "smearing",
            "smearing": "gaussian", "degauss": 0.005}
    if nbnd:
        sysd["nbnd"] = nbnd
    return {
        "control": {"calculation": calc, "prefix": "pw", "outdir": "./tmp", "pseudo_dir": str(PSEUDO_DIR),
                    "tprnfor": True, "tstress": True, "etot_conv_thr": 1e-5 * nat, "forc_conv_thr": 1e-3,
                    "disk_io": "low"},
        "system": sysd,
        "electrons": {"conv_thr": 1e-9 * nat, "mixing_beta": 0.4, "electron_maxstep": 200},
        "ions": {"ion_dynamics": "bfgs"},
        "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 0.5},
    }


def ok_text(rc: int, text: str) -> bool:
    return (rc == 0 and "JOB DONE" in text and "convergence NOT achieved" not in text
            and "The maximum number of steps has been reached" not in text)


def save(source: str, cid: str, rec: dict, outs: list[Path], atoms: Atoms | None):
    d = ROOT / "results" / source
    d.mkdir(parents=True, exist_ok=True)
    for i, out in enumerate(outs):
        if out is not None and out.exists():
            with open(out, "rb") as src, gzip.open(d / f"{cid}.{out.stem}.out.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
    if atoms is not None:
        write(d / f"{cid}.relaxed.xyz", atoms, format="extxyz")
    tmp = d / f"{cid}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(d / f"{cid}.json")


def done(source: str, cid: str) -> bool:
    """True if a result exists. With RETRY_FAILED=1 a failed record is moved
    aside (``<cid>.failed.json``) and the calculation is run again."""
    p = ROOT / "results" / source / f"{cid}.json"
    if not p.exists():
        return False
    if os.environ.get("RETRY_FAILED") and json.loads(p.read_text())["status"] != "ok":
        p.rename(p.with_name(f"{cid}.failed.json"))
        return False
    return True


def base_record(cid, source, atoms, settings):
    return {"cid": cid, "source": source, "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
            "volume_per_atom": atoms.get_volume() / len(atoms), "device": f"cpu:{ncores()}",
            "node": os.environ.get("SLURMD_NODENAME"),
            "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")), "settings": settings}


def pbe(cid: str, atoms: Atoms, work: Path):
    nat = len(atoms)
    ne = nelec(atoms)
    nocc = int(round(ne)) // 2
    kp = kgrid(atoms, KSPACING_PBE)
    w = work / cid / "pbe"
    inp = base_input("vc-relax", nat, None)
    rc1, wall1, text1, out1 = run_pw(w, "relax", atoms, inp, kgrid(atoms, KSPACING_RELAX), 4)
    relax_nosym = False
    if "checkallsym" in text1:
        # QE symmetry check fails after a cell step ("not orthogonal operation"):
        # rerun the relaxation without symmetry (changes k-point reduction only)
        inp["system"]["nosym"] = True
        relax_nosym = True
        shutil.rmtree(w / "tmp", ignore_errors=True)
        rc1, wall1b, text1, out1 = run_pw(w, "relax", atoms, inp, kgrid(atoms, KSPACING_RELAX), 4)
        wall1 += wall1b
    try:
        final = read(out1, format="espresso-out", index=-1)
    except Exception:  # noqa: BLE001
        final = atoms
    inp2 = base_input("scf", nat, nocc + 8)
    shutil.rmtree(w / "tmp", ignore_errors=True)
    rc2, wall2, text2, out2 = run_pw(w, "scf", final, inp2, kp, 4)
    settings = {"functional": "PBE", "calculation": "vc-relax + scf", "ecutwfc_Ry": ECUTWFC, "ecutrho_Ry": ECUTRHO,
                "kpts": kp, "kpts_relax": kgrid(atoms, KSPACING_RELAX), "relax_nosym": relax_nosym, "smearing": "gaussian 0.005 Ry", "nelec": ne, "spin": "unpolarised", "soc": False,
                "pseudos": "PseudoDojo NC SR v0.4 PBE standard"}
    rec = base_record(cid, "qe_pbe", final, settings)
    ok = ok_text(rc1, text1) and ok_text(rc2, text2) and abs(ne - round(ne)) < 1e-6 and round(ne) % 2 == 0
    if ok:
        rec.update(gap_from_xml(w / "tmp" / "pw.save", nocc))
    c1, wq1 = pw_times(text1)
    c2, wq2 = pw_times(text2)
    rec.update({"status": "ok" if ok else "failed", "wall_s": wall1 + wall2,
                "core_s": (wall1 + wall2) * ncores(), "qe_cpu_s": (c1 or 0) + (c2 or 0),
                "qe_wall_s": (wq1 or 0) + (wq2 or 0), "energy_per_atom": None})
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text2, re.M)
    if e:
        rec["energy_per_atom"] = float(e[-1]) * RY / nat
    if not ok:
        rec["error"] = "relax tail: " + " | ".join(text1.strip().splitlines()[-3:]) + \
                       " || scf tail: " + " | ".join(text2.strip().splitlines()[-3:])
    return rec, [out1, out2], final


def hse(cid: str, atoms: Atoms, work: Path, source: str = "qe_hse06"):
    nat = len(atoms)
    ne = nelec(atoms)
    nocc = int(round(ne)) // 2
    kp = kgrid(atoms, KSPACING_HSE)
    nq = [max(1, k // HSE_QDIV) for k in kp]
    inp = base_input("scf", nat, nocc + 4)
    inp["system"].update({"input_dft": "hse", "ecutfock": ECUTFOCK, "nqx1": nq[0], "nqx2": nq[1], "nqx3": nq[2],
                          "exxdiv_treatment": "gygi-baldereschi", "x_gamma_extrapolation": True,
                          "force_symmorphic": True})
    inp["control"].update({"tstress": False, "tprnfor": False})
    w = work / cid / source
    shutil.rmtree(w / "tmp", ignore_errors=True)
    rc, wall, text, out = run_pw(w, "hse", atoms, inp, kp, 2)
    nosym = False
    if "scale_sym_ops" in text:
        inp["system"]["nosym"] = True
        nosym = True
        shutil.rmtree(w / "tmp", ignore_errors=True)
        rc, wall, text, out = run_pw(w, "hse", atoms, inp, kp, 2)
    settings = {"functional": "HSE06", "calculation": "scf at PBE geometry", "ecutwfc_Ry": ECUTWFC,
                "ecutfock_Ry": ECUTFOCK, "kpts": kp, "nq": nq, "q_div": HSE_QDIV, "nosym": nosym, "smearing": "gaussian 0.005 Ry",
                "nelec": ne, "spin": "unpolarised", "soc": False, "pseudos": "PseudoDojo NC SR v0.4 PBE standard"}
    rec = base_record(cid, source, atoms, settings)
    ok = ok_text(rc, text)
    if ok:
        rec.update(gap_from_xml(w / "tmp" / "pw.save", nocc))
    c, wq = pw_times(text)
    rec.update({"status": "ok" if ok else "failed", "wall_s": wall, "core_s": wall * ncores(),
                "qe_cpu_s": c, "qe_wall_s": wq})
    if not ok:
        rec["error"] = f"rc={rc}; tail: " + " | ".join(text.strip().splitlines()[-5:])
    return rec, [out]


def main():
    chunks = json.loads((ROOT / "qe" / os.environ.get("CHUNKS_FILE", "chunks.json")).read_text())
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else "0"))
    cids = chunks[task]
    work = Path(os.environ.get("QE_WORK", ROOT / "work"))
    print(f"task={task} cids={cids} node={os.environ.get('SLURMD_NODENAME')} "
          f"restart={os.environ.get('SLURM_RESTART_COUNT', '0')}", flush=True)
    relaxed: dict[str, Atoms] = {}
    if os.environ.get("HSE_ONLY"):
        # extra HSE06 workers: only candidates whose PBE result already exists
        cids = [c for c in cids if done("qe_pbe", c)]
    for cid in cids:
        if done("qe_pbe", cid):
            p = ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz"
            relaxed[cid] = read(p) if p.exists() else read(ROOT / "candidates" / f"{cid}.xyz")
            continue
        atoms = read(ROOT / "candidates" / f"{cid}.xyz")
        rec, outs, final = pbe(cid, atoms, work)
        save("qe_pbe", cid, rec, outs, final)
        relaxed[cid] = final
        print(f"PBE {cid} {rec['status']} gap={rec.get('gap_eV')} wall={rec['wall_s']:.0f}s", flush=True)
    if os.environ.get("SKIP_HSE"):
        print("TASK DONE (PBE only)", flush=True)
        return
    for cid in cids:
        if done("qe_hse06", cid):
            continue
        p = json.loads((ROOT / "results" / "qe_pbe" / f"{cid}.json").read_text())
        if p["status"] != "ok":
            continue
        rec, outs = hse(cid, relaxed[cid], work)
        save("qe_hse06", cid, rec, outs, None)
        print(f"HSE06 {cid} {rec['status']} gap={rec.get('gap_eV')} wall={rec['wall_s']:.0f}s", flush=True)
    print("TASK DONE", flush=True)


def qconv():
    """q-grid convergence check: HSE06 gap with q = k and q = k/2 for a few
    candidates (argv: cids). Writes results/hse_qconv/<cid>.q<div>.json."""
    global HSE_QDIV
    work = Path(os.environ.get("QE_WORK", ROOT / "work"))
    for cid in sys.argv[2:]:
        p = ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz"
        atoms = read(p) if p.exists() else read(ROOT / "candidates" / f"{cid}.xyz")
        for div in (1, 2):
            HSE_QDIV = div
            dest = ROOT / "results" / "hse_qconv" / f"{cid}.q{div}.json"
            if dest.exists():
                continue
            rec, outs = hse(cid, atoms, work, source=f"hse_qconv_q{div}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(rec, indent=1))
            print(f"QCONV {cid} q/{div} gap={rec.get('gap_eV')} wall={rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "qconv":
        qconv()
    else:
        main()
