#!/usr/bin/env python
"""DGX-side computed sources for the Al-Si-Cu pool (formation energies).

Adapted from ../almgsi_campaign/dgx_sources.py, reduced to the one source run
for this pool:

  mace_relax   0 K MACE-MP-0 (small) relaxation, positions + full cell (GPU),
               float64, FIRE on a FrechetCellFilter, fmax 0.01 eV/A, at most
               500 steps. Same settings as the Al-Mg-Si campaign.

Every query writes results/mace_relax/<cid>.json with the shared record
schema (cid, source, status, natoms, energy_per_atom, energy_per_atom_sem,
volume_per_atom, wall_s, device, settings, error); wall_s is the query's own
wall-clock time (model loading excluded, reported once in _meta.json). The
elemental references (mp_134_Al fcc Al, mp_149_Si diamond Si, mp_30_Cu fcc
Cu) are relaxed the same way and used by aggregate_sources.py.

Usage (Python environment with ase and mace-torch):
  <venv>/bin/python dgx_sources.py --source mace_relax --all
  <venv>/bin/python dgx_sources.py --source mace_relax --cids mp_134_Al,mp_30_Cu
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAND_DIR = HERE / "candidates"
RESULTS_DIR = Path(os.environ.get("LIVE_RESULTS_DIR", str(HERE / "results")))
SOURCES = ("mace_relax",)


def load_index():
    with open(HERE / os.environ.get("LIVE_INDEX", "candidates_index.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def read_cand(cid):
    from ase.io import read
    return read(str(CAND_DIR / f"{cid}.xyz"), format="extxyz")


def result_path(source, cid):
    return RESULTS_DIR / source / f"{cid}.json"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_record(cid, source):
    return {"cid": cid, "source": source, "status": "failed", "natoms": None,
            "energy_per_atom": None, "energy_per_atom_sem": None,
            "volume_per_atom": None, "wall_s": None, "device": None,
            "settings": {}, "error": None}


_MACE = {}


def get_mace(dtype="float64"):
    if dtype not in _MACE:
        from mace.calculators import mace_mp
        t0 = time.perf_counter()
        calc = mace_mp(model="small", device="cuda", default_dtype=dtype)
        _MACE[dtype] = (calc, time.perf_counter() - t0)
    return _MACE[dtype]


def _mace_meta(source, dtype, load_s):
    meta_p = RESULTS_DIR / source / "_meta.json"
    meta = read_json(meta_p) if meta_p.exists() else {"model_load_s": []}
    meta["model"] = "MACE-MP-0 small (20231210mace128L0_energy_epoch249model)"
    meta["dtype"] = dtype
    meta.setdefault("model_load_s", []).append(round(load_s, 3))
    write_json(meta_p, meta)


def mace_relax_query(cid, calc):
    from ase.filters import FrechetCellFilter
    from ase.io import write
    from ase.optimize import FIRE
    rec = base_record(cid, "mace_relax")
    rec["device"] = "cuda"
    t0 = time.perf_counter()
    try:
        atoms = read_cand(cid)
        settings = {"model": "MACE-MP-0 small", "dtype": "float64", "optimizer": "FIRE",
                    "fmax_eV_A": 0.01, "max_steps": 500,
                    "relax": "positions + full cell (FrechetCellFilter)"}
        atoms.calc = calc
        opt = FIRE(FrechetCellFilter(atoms), logfile=None)
        converged = bool(opt.run(fmax=0.01, steps=500))
        e = float(atoms.get_potential_energy())
        n = len(atoms)
        if not np.isfinite(e):
            raise RuntimeError("non-finite energy")
        settings["converged"] = converged
        settings["n_steps"] = int(opt.get_number_of_steps())
        rec.update({"status": "ok", "natoms": n, "energy_per_atom": e / n,
                    "energy_per_atom_sem": 0.0,
                    "volume_per_atom": float(atoms.get_volume()) / n,
                    "settings": settings,
                    "relaxed_cell": np.asarray(atoms.cell).tolist()})
        atoms.calc = None
        write(str(result_path("mace_relax", cid).with_suffix(".relaxed.xyz")), atoms, format="extxyz")
    except Exception as exc:  # record, never invent
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def run_mace(source, todo):
    calc, load_s = get_mace("float64")
    _mace_meta(source, "float64", load_s)
    print(f"[{source}] model loaded in {load_s:.2f} s (dtype float64)", flush=True)
    for cid in todo:
        if result_path(source, cid).exists():
            continue
        rec = mace_relax_query(cid, calc)
        write_json(result_path(source, cid), rec)
        print(f"[{source}] {cid} {rec['status']} E={rec['energy_per_atom']} "
              f"wall={rec['wall_s']:.2f}s {rec['error'] or ''}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, choices=SOURCES)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cids", help="comma-separated candidate ids")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    index = {d["cid"]: d for d in load_index()}
    cids = list(index) if args.all else [c.strip() for c in args.cids.split(",") if c.strip()]
    for c in cids:
        if c not in index:
            sys.exit(f"unknown cid {c}")
    # elemental references first
    cids.sort(key=lambda c: 0 if index[c]["kind"] == "reference" else 1)
    todo = [c for c in cids if not result_path(args.source, c).exists()]
    print(f"[{args.source}] {len(cids)} requested, {len(todo)} to run", flush=True)
    (RESULTS_DIR / args.source).mkdir(parents=True, exist_ok=True)
    if not todo:
        return
    t0 = time.perf_counter()
    run_mace(args.source, todo)
    print(f"[{args.source}] finished in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
