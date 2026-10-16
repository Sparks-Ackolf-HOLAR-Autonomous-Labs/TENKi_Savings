"""HSE06 q-grid / k-grid convergence test for the formation energies.

Settings compared, all at the PBE-relaxed geometries:
  base : k-spacing 0.30 1/A, q grid = ceil(k/2)  (the production setting)
  qfull: k-spacing 0.30 1/A, q grid = k grid
  kfine: k-spacing 0.22 1/A, q grid = ceil(k/2)
Results: results/qe_hse06_conv/<setting>/<cid>.json
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qe_driver as Q  # noqa: E402
from ase.io import read  # noqa: E402

CIDS = ["mp_134_Al", "mp_153_Mg", "mp_149_Si", "mp_1367_Mg2Si", "fcc_MgAl3_02"]
SETTINGS = {"qfull": (0.30, 1), "kfine": (0.22, 2), "base": (0.30, 2)}


def run(tag, spacing, qdiv, cid, work):
    out = Q.ROOT / "results" / "qe_hse06_conv" / tag / f"{cid}.json"
    if out.exists():
        return
    atoms = read(Q.ROOT / "results" / "qe_pbe" / f"{cid}.relaxed.xyz")
    Q.KSPACING_HSE = spacing
    nat = len(atoms)
    inp = Q.base_input("scf", nat)
    kp = Q.kgrid(atoms, spacing)
    nq = [max(1, math.ceil(k / qdiv)) for k in kp]
    inp["system"].update({"input_dft": "hse", "ecutfock": Q.ECUTFOCK, "nqx1": nq[0], "nqx2": nq[1], "nqx3": nq[2],
                          "exxdiv_treatment": "gygi-baldereschi", "x_gamma_extrapolation": True,
                          "force_symmorphic": True})
    inp["control"]["tstress"] = False   # QE: hybrid stress is not available with k-point pools
    inp["electrons"].update({"mixing_beta": 0.2, "mixing_mode": "local-TF", "electron_maxstep": 300})
    rc, wall, text, o = Q.run_pw(work / "conv" / tag / cid, "hse", atoms, inp, kp, 8)
    rec = Q.record(cid, "qe_hse06", atoms, Q.final_energy_ev(text), rc, wall, text,
                   {"tag": tag, "kpts": kp, "nq": nq, "kspacing": spacing})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(tag, cid, rec["status"], rec["energy_per_atom"], round(wall), flush=True)


if __name__ == "__main__":
    work = Path(os.environ.get("QE_WORK", Q.ROOT / "work"))
    for tag, (sp, qd) in SETTINGS.items():
        for cid in CIDS:
            run(tag, sp, qd, cid, work)
    print("CONV DONE", flush=True)
