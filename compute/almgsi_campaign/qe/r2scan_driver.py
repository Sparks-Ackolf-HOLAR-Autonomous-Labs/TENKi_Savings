"""r2SCAN single points at the PBE-relaxed geometries (third DFT level).

For every formation-energy candidate in candidates_index.json whose PBE
vc-relax succeeded (results/qe_pbe/<cid>.json status ok), run an SCF with
the r2SCAN meta-GGA (QE input_dft='r2scan', evaluated through libxc:
MGGA_X_R2SCAN 497 + MGGA_C_R2SCAN 498) at the PBE-relaxed structure
(results/qe_pbe/<cid>.relaxed.xyz). Settings follow the HSE06 stage:
same PseudoDojo NC SR v0.4 PBE pseudopotentials, ecutwfc 80 Ry, ecutrho
320 Ry, k spacing 0.30 1/A, Marzari-Vanderbilt smearing 0.02 Ry.

Output: results/qe_r2scan/<cid>.json, same schema as results/qe_hse06/
(energy_per_atom etc.), plus formation_energy_per_atom (eV/atom) against
the r2SCAN energies of fcc Al, hcp Mg and diamond Si at their own PBE
geometries. The three references run first. Existing records are skipped,
so the run is restartable. Run order: references, then cheapest cells first
(natoms x number of k points).

Needs a pw.x linked against libxc (see R2SCAN.md). Environment:
LIVE_ROOT (campaign root holding candidates_index.json, pseudos/,
results/qe_pbe/), QE_NP (MPI ranks), QE_WORK (scratch).
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qe_driver as qd  # noqa: E402

ROOT = qd.ROOT
SOURCE = "qe_r2scan"
REFS = {"Al": "mp_134_Al", "Mg": "mp_153_Mg", "Si": "mp_149_Si"}
KSPACING = qd.KSPACING_HSE
NPOOL = int(os.environ.get("R2SCAN_NPOOL", "4"))
# r2SCAN energies converge slowly with the dense-grid cutoff (see R2SCAN.md);
# both cutoffs are overridable for convergence studies and production runs.
ECUTWFC = float(os.environ.get("R2SCAN_ECUTWFC", qd.ECUTWFC))
ECUTRHO = float(os.environ.get("R2SCAN_ECUTRHO", qd.ECUTRHO))


def res_path(cid: str) -> Path:
    return ROOT / "results" / SOURCE / f"{cid}.json"


def nks(atoms) -> int:
    return math.prod(qd.kgrid(atoms, KSPACING))


def r2scan_single_point(cid: str, atoms, work: Path):
    nat = len(atoms)
    inp = qd.base_input("scf", nat)
    kp = qd.kgrid(atoms, KSPACING)
    inp["system"].update({"input_dft": "r2scan", "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO})
    inp["electrons"].update({"mixing_beta": 0.3, "electron_maxstep": 200})
    npool = NPOOL if math.prod(kp) >= NPOOL else 1
    mixing = "plain beta 0.3, maxstep 200"
    rc, wall, text, out = qd.run_pw(work / cid / "r2scan", "r2scan", atoms, inp, kp, npool)
    tot_wall = wall
    if "JOB DONE" not in text or "convergence NOT achieved" in text:
        # retry: one k-point pool (pool count errors) and gentler local-TF mixing
        npool = 1
        inp["electrons"].update({"mixing_beta": 0.1, "mixing_mode": "local-TF", "electron_maxstep": 400})
        mixing = "local-TF beta 0.1, maxstep 400 (retry)"
        rc, wall, text, out = qd.run_pw(work / cid / "r2scan", "r2scan", atoms, inp, kp, npool)
        tot_wall += wall
    settings = {"functional": "r2SCAN", "input_dft": "r2scan (libxc 497+498)",
                "calculation": "scf at PBE geometry", "ecutwfc_Ry": ECUTWFC, "ecutrho_Ry": ECUTRHO,
                "kpts": kp, "kspacing_invA": KSPACING, "npool": npool, "smearing": "mv",
                "degauss_Ry": qd.DEGAUSS, "mixing": mixing,
                "pseudos": "PseudoDojo NC SR v0.4 PBE standard", "qe": "QE 7.4 + libxc 6.2.2 (DGX build)"}
    rec = qd.record(cid, SOURCE, atoms, qd.final_energy_ev(text), rc, tot_wall, text, settings)
    rec["cpu_s"] = rec.get("qe_cpu_s")
    rec["node"] = rec.get("node") or os.uname().nodename
    return rec, out


def add_formation(cid: str, rec: dict):
    refs = {}
    for el, rc in REFS.items():
        p = res_path(rc)
        if p.exists():
            refs[el] = json.loads(p.read_text())
    if rec["status"] != "ok":
        rec["formation_energy_per_atom"] = None
        return
    comp = Counter(read(ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz").get_chemical_symbols())
    n = sum(comp.values())
    val = rec["energy_per_atom"]
    for el, c in comp.items():
        r = refs.get(el)
        if r is None or r["status"] != "ok":
            rec["formation_energy_per_atom"] = None
            rec["formation_status"] = f"ref_{el}_unavailable"
            return
        val -= c / n * r["energy_per_atom"]
    rec["formation_energy_per_atom"] = val
    rec["formation_status"] = "ok"


def main():
    index = json.loads((ROOT / os.environ.get("LIVE_INDEX", "candidates_index.json")).read_text())
    work = Path(os.environ.get("QE_WORK", ROOT / "work_r2scan"))
    todo = []
    for d in index:
        if d["property"] != "formation_energy":
            continue
        cid = d["cid"]
        pj = ROOT / "results" / "qe_pbe" / f"{cid}.json"
        px = ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz"
        if not pj.exists() or not px.exists() or json.loads(pj.read_text())["status"] != "ok":
            print(f"skip {cid}: no successful PBE geometry", flush=True)
            continue
        atoms = read(px)
        todo.append((0 if cid in REFS.values() else 1, len(atoms) * nks(atoms), cid, atoms))
    todo.sort(key=lambda t: (t[0], t[1], t[2]))
    print(f"{len(todo)} r2SCAN single points, np={qd.ncores()}", flush=True)
    for _, cost, cid, atoms in todo:
        if res_path(cid).exists():
            continue
        rec, out = r2scan_single_point(cid, atoms, work)
        add_formation(cid, rec)
        qd.save(SOURCE, cid, rec, out, None)
        ef = rec.get("formation_energy_per_atom")
        print(f"R2SCAN {cid} nat={len(atoms)} {rec['status']} wall={rec['wall_s']:.0f}s "
              f"Ef={'nan' if ef is None else f'{ef:.4f}'}", flush=True)
    # refresh formation energies (references may have been rerun)
    for _, _, cid, _ in todo:
        p = res_path(cid)
        if p.exists():
            rec = json.loads(p.read_text())
            add_formation(cid, rec)
            p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("R2SCAN DONE", flush=True)


if __name__ == "__main__":
    main()
