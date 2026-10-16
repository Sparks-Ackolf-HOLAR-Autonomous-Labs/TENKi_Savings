#!/usr/bin/env python
"""DGX-side calculation sources for the live Al-Mg-Si campaign.

One CLI, four campaign sources (each a different fidelity / cost point)
plus two 0 K diagnostic sources:

  mace_relax      0 K MACE-MP-0 (small) relaxation, positions + cell (GPU)
  mace_md         MACE-MP-0 (small) NPT MD at T (default 300 K), 0 bar (GPU, ASE)
  gpumd_nep89_md  NEP89 foundation potential NPT MD at T (GPU, GPUMD 5.8)
  lammps_meam_md  Jelinek 2012 MEAM NPT MD at T (CPU, LAMMPS python module)
  gpumd_nep89_relax  0 K NEP89 relaxation, positions + full cell (diagnostic)
  lammps_meam_relax  0 K MEAM relaxation, positions + full cell (diagnostic)

MD records carry a "structure" block (rms_disp_A, frac_disp_gt_1A, max_disp_A)
comparing the final MD frame with the input supercell in scaled coordinates.

Temperature study (context shift): --temperature 453 --host-lattice npt
--results-dir results_T/T453 reruns an MD source at another temperature; the
cluster host cell then uses the source's own NPT fcc-Al lattice at that
temperature (thermal expansion included).

Usage:
  python dgx_sources.py --source mace_relax --all
  python dgx_sources.py --source lammps_meam_md --cids mp_134_Al,clu_Al108 --workers 8
  python dgx_sources.py --source gpumd_nep89_md --cids clu_Al108,clu_Mg1 --sample-ps 60

Each query writes results/<source>/<cid>.json (skipped if it already exists,
so runs are resumable).  Files whose names start with "_" in a results
directory are per-process metadata (model load time, host lattice constant),
not candidate results.

Cluster cells (kind cluster / cluster_ref) are 3x3x3 conventional fcc Al
supercells.  Binding energies are differences at a common host cell, so every
source first determines its own pure-Al lattice parameter from clu_Al108 and
all cluster cells are rescaled to it:
  mace_relax, mace_md : 0 K MACE relaxation of clu_Al108 (hydrostatic cell)
  gpumd_nep89_md      : 0 K GPUMD FIRE box minimisation of clu_Al108
  lammps_meam_md      : 0 K LAMMPS box/relax (iso) of clu_Al108
Cluster MD is fixed-cell NVT (Nose-Hoover chain) at that lattice parameter.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAND_DIR = HERE / "candidates"
RESULTS_DIR = Path(os.environ.get("LIVE_RESULTS_DIR", str(HERE / "results")))
POT_DIR = HERE / "potentials"
# scratch for MD/QE working directories (ALMGSI_WORK, default ./work next to this script)
WORK_ROOT = Path(os.environ.get("ALMGSI_WORK", str(HERE / "work")))
if "LIVE_T_K" in os.environ:   # separate scratch per temperature so runs can overlap
    WORK_ROOT = WORK_ROOT / f"T{float(os.environ['LIVE_T_K']):g}"

# GPUMD 5.8 source tree with the built binary (src/gpumd) and the NEP89 potential
GPUMD_ROOT = Path(os.environ.get("GPUMD_ROOT", str(Path.home() / "mat_theory_gpumd" / "GPUMD")))
GPUMD_BIN = GPUMD_ROOT / "src" / "gpumd"
NEP89_FILE = (GPUMD_ROOT / "potentials" / "nep"
              / "nep89_20250409" / "nep89_20250409.txt")
MEAM_LIB = POT_DIR / "library.meam"
MEAM_PAR = POT_DIR / "AlSiMgCuFe.meam"
MEAM_LABEL = {"Al": "AlS", "Mg": "MgS", "Si": "SiS"}

SOURCES = ("mace_relax", "mace_md", "gpumd_nep89_md", "lammps_meam_md",
           "gpumd_nep89_relax", "lammps_meam_relax")
RELAX_DIAG = ("gpumd_nep89_relax", "lammps_meam_relax")
SPECORDER = ["Al", "Mg", "Si"]

# MD protocol (shared by all three MD engines)
T_K = float(os.environ.get("LIVE_T_K", "300"))
# cluster host lattice: "0K" = the source's own 0 K minimisation of clu_Al108
# (original rule); "npt" = the source's own NPT fcc-Al lattice at T_K, read from
# its mp_134_Al result (thermal expansion included; used by the temperature study)
HOST_LATTICE = os.environ.get("LIVE_HOST_LATTICE", "0K")
DT_FS = 2.0
N_EQUIL = 500          # 1 ps
# sampling length: default 3 ps; override with --sample-ps (propagated to
# spawned worker processes through the LIVE_SAMPLE_PS environment variable)
SAMPLE_PS = float(os.environ.get("LIVE_SAMPLE_PS", "3"))
N_SAMPLE = int(round(SAMPLE_PS * 1000.0 / DT_FS))
SAMPLE_EVERY = 10      # steps
N_BLOCKS = 5
MIN_ATOMS = 96
MIN_WIDTH = 9.0        # Angstrom, perpendicular cell width
CLUSTER_KINDS = ("cluster", "cluster_ref")
HOST_REPS = 3          # cluster cells are 3x3x3 conventional fcc


# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------

def load_index():
    # LIVE_INDEX selects an alternative candidate index (for example
    # candidates_index_observed.json, the experimentally observed phases)
    # without touching the campaign pool in candidates_index.json.
    with open(HERE / os.environ.get("LIVE_INDEX", "candidates_index.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def read_cand(cid):
    from ase.io import read
    return read(str(CAND_DIR / f"{cid}.xyz"), format="extxyz")


def cid_seed(cid):
    return zlib.crc32(cid.encode("utf-8")) & 0xFFFFFFFF


def perp_widths(cell):
    cell = np.asarray(cell, dtype=float)
    vol = abs(np.linalg.det(cell))
    w = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        w.append(vol / np.linalg.norm(np.cross(cell[j], cell[k])))
    return np.array(w)


def md_supercell_reps(atoms):
    """Diagonal repeats so that N >= MIN_ATOMS and all perpendicular widths
    >= MIN_WIDTH.  Deterministic; identical for all MD engines."""
    w = perp_widths(atoms.cell)
    reps = [max(1, int(math.ceil(MIN_WIDTH / wi - 1e-9))) for wi in w]
    while len(atoms) * reps[0] * reps[1] * reps[2] < MIN_ATOMS:
        cur = [w[i] * reps[i] for i in range(3)]
        i = int(np.argmin(cur))
        reps[i] += 1
    return tuple(reps)


def make_md_supercell(atoms):
    reps = md_supercell_reps(atoms)
    return atoms.repeat(reps), reps


def block_sem(x, nblocks=N_BLOCKS):
    x = np.asarray(x, dtype=float)
    n = len(x) // nblocks
    if n < 1:
        return float("nan")
    means = x[: n * nblocks].reshape(nblocks, n).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(nblocks))


def set_mb_velocities(atoms, cid):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    rng = np.random.default_rng(cid_seed(cid))
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=rng)
    Stationary(atoms)


def scale_to_host(atoms, a_host):
    """Rescale a 3x3x3 conventional fcc cluster cell to lattice parameter a_host."""
    a_in = float(np.linalg.norm(atoms.cell[0])) / HOST_REPS
    atoms = atoms.copy()
    atoms.set_cell(np.asarray(atoms.cell) * (a_host / a_in), scale_atoms=True)
    return atoms


def disp_indicator(frac0, frac1, cell0):
    """Structural-change indicator between the input (super)cell and the final
    MD frame.  Scaled coordinates remove the cell change; displacements use the
    minimum image, the mean (rigid) translation is removed, and the result is
    expressed in Angstrom with the input cell."""
    d = np.asarray(frac1, dtype=float) - np.asarray(frac0, dtype=float)
    d -= np.round(d)
    d -= d.mean(axis=0)
    d -= np.round(d)
    cart = d @ np.asarray(cell0, dtype=float)
    r = np.linalg.norm(cart, axis=1)
    return {"rms_disp_A": float(np.sqrt(np.mean(r ** 2))),
            "frac_disp_gt_1A": float(np.mean(r > 1.0)),
            "max_disp_A": float(r.max())}


def result_path(source, cid):
    return RESULTS_DIR / source / f"{cid}.json"


def mark_contention(rec):
    """Flag queries whose wall_s was measured while another GPU job shared the
    device (set LIVE_GPU_CONTENDED=1); such timings are rerun later."""
    if os.environ.get("LIVE_GPU_CONTENDED"):
        rec.setdefault("settings", {})["gpu_contended"] = True
    return rec


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")  # unique: parallel shards share _meta.json
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def md_summary(epa, vpa, temps, v0_per_atom):
    """Validate an MD trace and return (ok, reason, stats)."""
    epa = np.asarray(epa, dtype=float)
    vpa = np.asarray(vpa, dtype=float)
    temps = np.asarray(temps, dtype=float)
    if len(epa) == 0:
        return False, "no samples", {}
    if not (np.all(np.isfinite(epa)) and np.all(np.isfinite(vpa))):
        return False, "non-finite energy or volume", {}
    ratio_max = float(vpa.max() / v0_per_atom)
    ratio_min = float(vpa.min() / v0_per_atom)
    if ratio_max > 2.0 or ratio_min < 0.5:
        return False, f"volume change out of bounds (V/V0 in [{ratio_min:.3f},{ratio_max:.3f}])", {}
    if np.any(temps > 5.0 * T_K):
        return False, f"temperature blow-up (max T {temps.max():.0f} K)", {}
    stats = {
        "energy_per_atom": float(epa.mean()),
        "energy_per_atom_sem": block_sem(epa),
        "volume_per_atom": float(vpa.mean()),
        "volume_per_atom_sem": block_sem(vpa),
        "T_mean": float(temps.mean()),
        "n_samples": int(len(epa)),
        "V_over_V0": float(vpa.mean() / v0_per_atom),
    }
    return True, "", stats


HOST_FROM = {"npt": f"same-source NPT fcc Al (mp_134_Al) at {T_K:g} K, a = (4 V/atom)^(1/3)"}


def base_record(cid, source):
    return {"cid": cid, "source": source, "status": "failed", "natoms": None,
            "energy_per_atom": None, "energy_per_atom_sem": None,
            "volume_per_atom": None, "wall_s": None, "device": None,
            "settings": {}, "error": None}


MD_COMMON = {
    "T_K": T_K, "P_bar": 0.0, "timestep_fs": DT_FS, "equil_steps": N_EQUIL,
    "sample_ps": SAMPLE_PS, "sample_steps": N_SAMPLE, "sample_every": SAMPLE_EVERY, "n_blocks": N_BLOCKS,
    "supercell_rule": f"diagonal repeats, N>={MIN_ATOMS}, perpendicular widths>={MIN_WIDTH} A",
    "velocities": f"Maxwell-Boltzmann {T_K:g} K (ase), zero total momentum, seed=crc32(cid)",
    "start": "input structure (unrelaxed)",
}


# ----------------------------------------------------------------------------
# MACE
# ----------------------------------------------------------------------------

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


def mace_relax_query(cid, kind, calc):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    rec = base_record(cid, "mace_relax")
    rec["device"] = "cuda"
    t0 = time.perf_counter()
    try:
        atoms = read_cand(cid)
        settings = {"model": "MACE-MP-0 small", "dtype": "float64", "optimizer": "FIRE",
                    "fmax_eV_A": 0.01, "max_steps": 500}
        if kind in CLUSTER_KINDS and cid != "clu_Al108":
            host = read_json(result_path("mace_relax", "clu_Al108"))
            if host["status"] != "ok":
                raise RuntimeError("clu_Al108 host relaxation failed")
            a_host = host["lattice_a"]
            atoms = scale_to_host(atoms, a_host)
            atoms.calc = calc
            opt_target = atoms
            settings["relax"] = "positions only, cell fixed at MACE-relaxed pure-Al lattice"
            settings["host_lattice_a"] = a_host
        elif cid == "clu_Al108":
            atoms.calc = calc
            opt_target = FrechetCellFilter(atoms, hydrostatic_strain=True)
            settings["relax"] = "positions + hydrostatic cell (FrechetCellFilter hydrostatic_strain=True)"
        else:
            atoms.calc = calc
            opt_target = FrechetCellFilter(atoms)
            settings["relax"] = "positions + full cell (FrechetCellFilter)"
        opt = FIRE(opt_target, logfile=None)
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
        if cid == "clu_Al108":
            rec["lattice_a"] = float(np.linalg.norm(atoms.cell[0])) / HOST_REPS
    except Exception as exc:  # record, never invent
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def npt_host_a(source):
    """Lattice parameter of the source's own NPT fcc-Al run at T_K (mp_134_Al)."""
    r = read_json(result_path(source, "mp_134_Al"))
    if r.get("status") != "ok":
        raise RuntimeError(f"{source} mp_134_Al NPT run failed; no host lattice at {T_K:g} K")
    return (4.0 * r["volume_per_atom"]) ** (1.0 / 3.0)


def mace_host_a():
    if HOST_LATTICE == "npt":
        return npt_host_a("mace_md")
    p = result_path("mace_relax", "clu_Al108")
    if not p.exists():
        calc, load_s = get_mace("float64")
        write_json(p, mace_relax_query("clu_Al108", "cluster_ref", calc))
    host = read_json(p)
    if host["status"] != "ok":
        raise RuntimeError("mace_relax clu_Al108 failed; no host lattice")
    return host["lattice_a"]


MACE_MD_DTYPE = os.environ.get("MACE_MD_DTYPE", "float32")


def mace_md_query(cid, kind, calc, dtype):
    from ase import units
    from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
    rec = base_record(cid, "mace_md")
    rec["device"] = "cuda"
    t0 = time.perf_counter()
    try:
        atoms = read_cand(cid)
        settings = dict(MD_COMMON)
        settings.update({"engine": "ASE 3.29 + MACE-MP-0 small", "dtype": dtype})
        if kind in CLUSTER_KINDS:
            a_host = mace_host_a()
            atoms = scale_to_host(atoms, a_host)
            reps = (1, 1, 1)
            settings.update({"ensemble": "NVT NoseHooverChainNVT (fixed cell)",
                             "tdamp_fs": 100.0, "host_lattice_a": a_host,
                             "host_lattice_from": HOST_FROM.get(HOST_LATTICE, "mace_relax clu_Al108 (0 K)")})
        else:
            atoms, reps = make_md_supercell(atoms)
            settings.update({"ensemble": "NPT IsotropicMTKNPT", "tdamp_fs": 100.0,
                             "pdamp_fs": 1000.0})
        settings["supercell_reps"] = list(reps)
        n = len(atoms)
        v0 = atoms.get_volume() / n
        set_mb_velocities(atoms, cid)
        frac0 = atoms.get_scaled_positions(wrap=False)
        cell0 = np.asarray(atoms.cell).copy()
        atoms.calc = calc
        if kind in CLUSTER_KINDS:
            dyn = NoseHooverChainNVT(atoms, timestep=DT_FS * units.fs, temperature_K=T_K,
                                     tdamp=100.0 * units.fs)
        else:
            dyn = IsotropicMTKNPT(atoms, timestep=DT_FS * units.fs, temperature_K=T_K,
                                  pressure_au=0.0, tdamp=100.0 * units.fs,
                                  pdamp=1000.0 * units.fs)
        dyn.run(N_EQUIL)
        epa, vpa, temps = [], [], []

        def sample():
            epa.append(atoms.get_potential_energy() / n)
            vpa.append(atoms.get_volume() / n)
            temps.append(atoms.get_temperature())

        for _ in range(N_SAMPLE // SAMPLE_EVERY):
            dyn.run(SAMPLE_EVERY)
            sample()
            if not np.isfinite(epa[-1]) or vpa[-1] / v0 > 2.0 or vpa[-1] / v0 < 0.5:
                break
        ok, reason, stats = md_summary(epa, vpa, temps, v0)
        rec["natoms"] = n
        rec["settings"] = settings
        try:
            from ase.io import write as ase_write
            fdir = WORK_ROOT / "mace_md"
            fdir.mkdir(parents=True, exist_ok=True)
            ase_write(str(fdir / f"{cid}_final.xyz"), atoms, format="extxyz")
            rec["structure"] = disp_indicator(frac0, atoms.get_scaled_positions(wrap=False), cell0)
        except Exception as exc:
            rec["structure"] = {"error": f"{type(exc).__name__}: {exc}"}
        if ok:
            rec.update({"status": "ok", "energy_per_atom": stats.pop("energy_per_atom"),
                        "energy_per_atom_sem": stats.pop("energy_per_atom_sem"),
                        "volume_per_atom": stats.pop("volume_per_atom")})
            rec["md_stats"] = stats
        else:
            rec["error"] = reason
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def run_mace(source, todo, index):
    dtype = "float64" if source == "mace_relax" else MACE_MD_DTYPE
    calc, load_s = get_mace(dtype)
    _mace_meta(source, dtype, load_s)
    print(f"[{source}] model loaded in {load_s:.2f} s (dtype {dtype})", flush=True)
    for cid in todo:
        if result_path(source, cid).exists():
            continue  # finished meanwhile by another shard (work stealing)
        kind = index[cid]["kind"]
        if source == "mace_relax":
            rec = mace_relax_query(cid, kind, calc)
        else:
            rec = mark_contention(mace_md_query(cid, kind, calc, dtype))
        write_json(result_path(source, cid), rec)
        print(f"[{source}] {cid} {rec['status']} E={rec['energy_per_atom']} "
              f"wall={rec['wall_s']:.2f}s {rec['error'] or ''}", flush=True)


# ----------------------------------------------------------------------------
# GPUMD / NEP89
# ----------------------------------------------------------------------------

def write_gpumd_model(atoms, path, with_vel=True):
    from ase import units
    cell = np.asarray(atoms.cell)
    lat = " ".join(f"{x:.10f}" for x in cell.reshape(-1))
    props = "species:S:1:pos:R:3" + (":vel:R:3" if with_vel else "")
    pos = atoms.get_positions()
    vel = atoms.get_velocities() * units.fs if with_vel else None  # A/fs
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f'pbc="T T T" lattice="{lat}" properties={props}\n')
        for i, s in enumerate(atoms.get_chemical_symbols()):
            line = f"{s} {pos[i,0]:.10f} {pos[i,1]:.10f} {pos[i,2]:.10f}"
            if with_vel:
                line += f" {vel[i,0]:.10e} {vel[i,1]:.10e} {vel[i,2]:.10e}"
            f.write(line + "\n")


def is_orthogonal(cell, tol=1e-6):
    c = np.asarray(cell)
    return bool(np.all(np.abs(c - np.diag(np.diag(c))) < tol))


def run_gpumd(workdir, timeout=3600):
    with open(workdir / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.run([str(GPUMD_BIN)], cwd=str(workdir), stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode != 0:
        tail = (workdir / "run.log").read_text(encoding="utf-8", errors="replace")[-600:]
        raise RuntimeError(f"gpumd exit {p.returncode}: {tail}")


def gpumd_host_a():
    if HOST_LATTICE == "npt":
        return npt_host_a("gpumd_nep89_md")
    meta_p = RESULTS_DIR / "gpumd_nep89_md" / "_host_lattice.json"
    if meta_p.exists():
        return read_json(meta_p)["lattice_a"]
    wd = WORK_ROOT / "gpumd" / "_host_Al108"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    atoms = read_cand("clu_Al108")
    write_gpumd_model(atoms, wd / "model.xyz", with_vel=False)
    with open(wd / "run.in", "w", encoding="utf-8") as f:
        f.write(f"potential {NEP89_FILE}\n"
                "minimize fire 1.0e-5 20000 1 1\n"
                "ensemble nve\ntime_step 0\n"
                "dump_xyz 1 relaxed.xyz precision double\nrun 1\n")
    t0 = time.perf_counter()
    run_gpumd(wd)
    from ase.io import read
    relaxed = read(str(wd / "relaxed.xyz"), index=-1, format="extxyz")
    a = float(np.linalg.norm(relaxed.cell[0])) / HOST_REPS
    write_json(meta_p, {"lattice_a": a, "method": "GPUMD minimize fire 1e-5 20000 box hydrostatic, clu_Al108",
                        "wall_s": time.perf_counter() - t0})
    return a


def gpumd_query(cid, kind):
    rec = base_record(cid, "gpumd_nep89_md")
    rec["device"] = "cuda"
    t0 = time.perf_counter()
    try:
        atoms = read_cand(cid)
        settings = dict(MD_COMMON)
        settings.update({"engine": "GPUMD 5.8", "potential": "NEP89 nep89_20250409"})
        if kind in CLUSTER_KINDS:
            a_host = gpumd_host_a()
            atoms = scale_to_host(atoms, a_host)
            reps = (1, 1, 1)
            ens = f"nvt_nhc {T_K:g} {T_K:g} 100"
            settings.update({"host_lattice_a": a_host,
                             "host_lattice_from": HOST_FROM.get(HOST_LATTICE, "GPUMD FIRE box minimisation of clu_Al108 (0 K)")})
        else:
            atoms, reps = make_md_supercell(atoms)
            if is_orthogonal(atoms.cell):
                ens = f"npt_scr {T_K:g} {T_K:g} 100 0 70 1000"
            else:
                # npt_scr isotropic mode requires an orthogonal box in GPUMD 5.8
                ens = f"npt_mttk temp {T_K:g} {T_K:g} iso 0 0 tperiod 100 pperiod 1000"
        settings["ensemble"] = ens
        settings["ensemble_units"] = "coupling times in time steps (2 fs); pressure GPa; modulus GPa"
        settings["supercell_reps"] = list(reps)
        n = len(atoms)
        v0 = atoms.get_volume() / n
        set_mb_velocities(atoms, cid)
        wd = WORK_ROOT / "gpumd" / cid
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        write_gpumd_model(atoms, wd / "model.xyz", with_vel=True)
        with open(wd / "run.in", "w", encoding="utf-8") as f:
            f.write(f"potential {NEP89_FILE}\n"
                    f"time_step {DT_FS:g}\n"
                    f"ensemble {ens}\n"
                    f"run {N_EQUIL}\n"
                    f"ensemble {ens}\n"
                    f"dump_thermo {SAMPLE_EVERY}\n"
                    f"dump_xyz {N_SAMPLE} final.xyz precision double\n"
                    f"run {N_SAMPLE}\n")
        frac0 = atoms.get_scaled_positions(wrap=False)
        cell0 = np.asarray(atoms.cell).copy()
        run_gpumd(wd)
        try:
            from ase.io import read as ase_read
            fin = ase_read(str(wd / "final.xyz"), index=-1, format="extxyz")
            rec["structure"] = disp_indicator(frac0, fin.get_scaled_positions(wrap=False), cell0)
        except Exception as exc:
            rec["structure"] = {"error": f"{type(exc).__name__}: {exc}"}
        th = np.loadtxt(wd / "thermo.out", ndmin=2)
        temps = th[:, 0]
        epa = th[:, 2] / n
        vols = []
        for row in th:
            h = row[9:18].reshape(3, 3)
            vols.append(abs(np.linalg.det(h)))
        vpa = np.array(vols) / n
        ok, reason, stats = md_summary(epa, vpa, temps, v0)
        rec["natoms"] = n
        rec["settings"] = settings
        if ok:
            rec.update({"status": "ok", "energy_per_atom": stats.pop("energy_per_atom"),
                        "energy_per_atom_sem": stats.pop("energy_per_atom_sem"),
                        "volume_per_atom": stats.pop("volume_per_atom")})
            rec["md_stats"] = stats
        else:
            rec["error"] = reason
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def run_gpumd_source(todo, index):
    if HOST_LATTICE != "npt" and any(index[c]["kind"] in CLUSTER_KINDS for c in todo):
        print(f"[gpumd] host lattice a = {gpumd_host_a():.5f} A", flush=True)
    for cid in todo:
        rec = mark_contention(gpumd_query(cid, index[cid]["kind"]))
        write_json(result_path("gpumd_nep89_md", cid), rec)
        print(f"[gpumd_nep89_md] {cid} {rec['status']} E={rec['energy_per_atom']} "
              f"wall={rec['wall_s']:.2f}s {rec['error'] or ''}", flush=True)


# ----------------------------------------------------------------------------
# LAMMPS / MEAM
# ----------------------------------------------------------------------------

def _lmp_new():
    from lammps import lammps
    return lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def _lmp_setup(lmp, datafile):
    labels = " ".join(MEAM_LABEL[s] for s in SPECORDER)
    for c in ("units metal", "atom_style atomic", "boundary p p p",
              f"read_data {datafile}",
              "pair_style meam",
              f"pair_coeff * * {MEAM_LIB} AlS SiS MgS CuS FeS {MEAM_PAR} {labels}",
              "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes",
              "thermo_style custom step temp pe vol", "thermo 0"):
        lmp.command(c)


def _lmp_frame(lmp):
    """Positions (ordered by atom id), lower-triangular cell and scaled coords."""
    boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
    h = np.array([[boxhi[0] - boxlo[0], 0.0, 0.0],
                  [xy, boxhi[1] - boxlo[1], 0.0],
                  [xz, yz, boxhi[2] - boxlo[2]]])
    n = lmp.get_natoms()
    x = np.array(lmp.gather_atoms("x", 1, 3)[:], dtype=float).reshape(n, 3)
    frac = (x - np.array(boxlo)) @ np.linalg.inv(h)
    return x, h, frac


def _write_lmp_data(atoms, path, velocities):
    from ase.io.lammpsdata import write_lammps_data
    with open(path, "w", encoding="utf-8") as f:
        write_lammps_data(f, atoms, specorder=SPECORDER, masses=True,
                          velocities=velocities, units="metal", atom_style="atomic",
                          force_skew=True)


def lammps_host_a():
    if HOST_LATTICE == "npt":
        return npt_host_a("lammps_meam_md")
    meta_p = RESULTS_DIR / "lammps_meam_md" / "_host_lattice.json"
    if meta_p.exists():
        return read_json(meta_p)["lattice_a"]
    wd = WORK_ROOT / "lammps" / "_host_Al108"
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    _write_lmp_data(read_cand("clu_Al108"), wd / "data.lmp", velocities=False)
    lmp = _lmp_new()
    _lmp_setup(lmp, wd / "data.lmp")
    lmp.command("fix br all box/relax iso 0.0 vmax 0.001")
    lmp.command("min_style cg")
    for _ in range(3):
        lmp.command("minimize 0 1e-10 20000 200000")
    lx = lmp.extract_box()[1][0] - lmp.extract_box()[0][0]
    pe = lmp.get_thermo("pe")
    lmp.close()
    a = float(lx) / HOST_REPS
    write_json(meta_p, {"lattice_a": a, "E0_per_atom": pe / 108.0,
                        "method": "LAMMPS fix box/relax iso 0 + cg minimize x3, clu_Al108",
                        "wall_s": time.perf_counter() - t0})
    return a


def lammps_query(args):
    cid, kind, a_host = args
    os.environ["OMP_NUM_THREADS"] = "1"
    rec = base_record(cid, "lammps_meam_md")
    rec["device"] = "cpu:1"
    t0 = time.perf_counter()
    lmp = None
    try:
        atoms = read_cand(cid)
        settings = dict(MD_COMMON)
        settings.update({"engine": "LAMMPS 2025-07-22 python module, 1 MPI rank, 1 thread",
                         "potential": "MEAM Jelinek et al. PRB 85 245102 (2012), AlSiMgCuFe.meam"})
        if kind in CLUSTER_KINDS:
            atoms = scale_to_host(atoms, a_host)
            reps = (1, 1, 1)
            fix = f"fix 1 all nvt temp {T_K:g} {T_K:g} 0.2"
            settings.update({"host_lattice_a": a_host,
                             "host_lattice_from": HOST_FROM.get(HOST_LATTICE, "LAMMPS box/relax iso of clu_Al108 (0 K)")})
        else:
            atoms, reps = make_md_supercell(atoms)
            fix = f"fix 1 all npt temp {T_K:g} {T_K:g} 0.2 iso 0 0 2.0"
        settings["ensemble"] = fix
        settings["supercell_reps"] = list(reps)
        n = len(atoms)
        v0 = atoms.get_volume() / n
        set_mb_velocities(atoms, cid)
        wd = WORK_ROOT / "lammps" / cid
        wd.mkdir(parents=True, exist_ok=True)
        _write_lmp_data(atoms, wd / "data.lmp", velocities=True)
        lmp = _lmp_new()
        _lmp_setup(lmp, wd / "data.lmp")
        _, cell0, frac0 = _lmp_frame(lmp)
        lmp.command(f"timestep {DT_FS / 1000.0:g}")
        lmp.command(fix)
        lmp.command(f"run {N_EQUIL}")
        epa, vpa, temps = [], [], []
        for _ in range(N_SAMPLE // SAMPLE_EVERY):
            lmp.command(f"run {SAMPLE_EVERY} pre no post no")
            natoms_now = lmp.get_natoms()
            if natoms_now != n:
                raise RuntimeError(f"atoms lost ({natoms_now} of {n})")
            epa.append(lmp.get_thermo("pe") / n)
            vpa.append(lmp.get_thermo("vol") / n)
            temps.append(lmp.get_thermo("temp"))
            if not np.isfinite(epa[-1]) or vpa[-1] / v0 > 2.0 or vpa[-1] / v0 < 0.5:
                break
        ok, reason, stats = md_summary(epa, vpa, temps, v0)
        rec["natoms"] = n
        rec["settings"] = settings
        try:
            from ase import Atoms
            from ase.io import write as ase_write
            x1, cell1, frac1 = _lmp_frame(lmp)
            ase_write(str(wd / "final.xyz"),
                      Atoms(symbols=atoms.get_chemical_symbols(), positions=x1, cell=cell1, pbc=True),
                      format="extxyz")
            rec["structure"] = disp_indicator(frac0, frac1, cell0)
        except Exception as exc:
            rec["structure"] = {"error": f"{type(exc).__name__}: {exc}"}
        if ok:
            rec.update({"status": "ok", "energy_per_atom": stats.pop("energy_per_atom"),
                        "energy_per_atom_sem": stats.pop("energy_per_atom_sem"),
                        "volume_per_atom": stats.pop("volume_per_atom")})
            rec["md_stats"] = stats
        else:
            rec["error"] = reason
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass
    rec["wall_s"] = time.perf_counter() - t0
    write_json(result_path("lammps_meam_md", cid), rec)
    return cid, rec["status"], rec["energy_per_atom"], rec["wall_s"], rec["error"]


def run_lammps_source(todo, index, workers):
    import multiprocessing as mp
    a_host = None
    if any(index[c]["kind"] in CLUSTER_KINDS for c in todo):
        if HOST_LATTICE == "npt" and not result_path("lammps_meam_md", "mp_134_Al").exists():
            lammps_query(("mp_134_Al", index["mp_134_Al"]["kind"], None))
            todo = [c for c in todo if c != "mp_134_Al"]
        a_host = lammps_host_a()
        print(f"[lammps] host lattice a = {a_host:.5f} A", flush=True)
    jobs = [(c, index[c]["kind"], a_host) for c in todo]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=20) as pool:
        for cid, st, e, w, err in pool.imap_unordered(lammps_query, jobs):
            print(f"[lammps_meam_md] {cid} {st} E={e} wall={w:.2f}s {err or ''}", flush=True)


# ----------------------------------------------------------------------------
# 0 K diagnostic relaxations with the MD potentials (own cell, positions + cell)
# ----------------------------------------------------------------------------

RELAX_FTOL = 1.0e-3   # eV/A (GPUMD generalized force incl. virial term)


def _gpumd_relax_cell(atoms, wd):
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    write_gpumd_model(atoms, wd / "model.xyz", with_vel=False)
    with open(wd / "run.in", "w", encoding="utf-8") as f:
        f.write(f"potential {NEP89_FILE}\n"
                f"minimize fire {RELAX_FTOL:g} 50000 1\n"
                "ensemble nve\ntime_step 0\n"
                "dump_xyz 1 relaxed.xyz precision double force\nrun 1\n")
    run_gpumd(wd)
    from ase.io import read as ase_read
    return ase_read(str(wd / "relaxed.xyz"), index=-1, format="extxyz")


def gpumd_relax_query(cid):
    rec = base_record(cid, "gpumd_nep89_relax")
    rec["device"] = "cuda"
    t0 = time.perf_counter()
    try:
        atoms = read_cand(cid)
        wd = WORK_ROOT / "gpumd_relax" / cid
        cell_used = "own cell"
        first_error = None
        try:
            r = _gpumd_relax_cell(atoms, wd)
        except RuntimeError as exc:
            # GPUMD 5.8 exits silently on some very small cells (e.g. 1-atom
            # primitive fcc); fall back to the MD supercell of the same structure
            first_error = str(exc)[-200:]
            atoms, reps = make_md_supercell(atoms)
            cell_used = f"MD supercell {list(reps)} (own cell crashed GPUMD)"
            r = _gpumd_relax_cell(atoms, wd)
        n = len(atoms)
        e = float(r.get_potential_energy())
        if not np.isfinite(e):
            raise RuntimeError("non-finite energy")
        fmax = float(np.abs(r.get_forces()).max())
        rec.update({"status": "ok", "natoms": n, "energy_per_atom": e / n,
                    "energy_per_atom_sem": 0.0,
                    "volume_per_atom": float(r.get_volume()) / n,
                    "relaxed_cell": np.asarray(r.cell).tolist()})
        rec["settings"] = {"engine": "GPUMD 5.8", "potential": "NEP89 nep89_20250409",
                           "relax": "minimize fire with box change (full 3x3 cell, non-hydrostatic)",
                           "cell_used": cell_used,
                           "force_tolerance_eV_A": RELAX_FTOL, "max_steps": 50000,
                           "final_max_force_eV_A": fmax}
        if first_error:
            rec["settings"]["own_cell_error_tail"] = first_error
        rec["structure"] = disp_indicator(atoms.get_scaled_positions(wrap=False),
                                          r.get_scaled_positions(wrap=False), np.asarray(atoms.cell))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def lammps_relax_query(cid):
    os.environ["OMP_NUM_THREADS"] = "1"
    rec = base_record(cid, "lammps_meam_relax")
    rec["device"] = "cpu:1"
    t0 = time.perf_counter()
    lmp = None
    try:
        atoms = read_cand(cid)
        n = len(atoms)
        wd = WORK_ROOT / "lammps_relax" / cid
        wd.mkdir(parents=True, exist_ok=True)
        _write_lmp_data(atoms, wd / "data.lmp", velocities=False)
        lmp = _lmp_new()
        _lmp_setup(lmp, wd / "data.lmp")
        _, cell0, frac0 = _lmp_frame(lmp)
        lmp.command("thermo_style custom step pe vol press fmax")
        lmp.command("fix br all box/relax tri 0.0 vmax 0.001")
        lmp.command("min_style cg")
        for _ in range(4):
            lmp.command("minimize 0 1e-8 20000 200000")
        e = lmp.get_thermo("pe")
        vol = lmp.get_thermo("vol")
        press = lmp.get_thermo("press")
        fmax = lmp.get_thermo("fmax")
        _, cell1, frac1 = _lmp_frame(lmp)
        if not np.isfinite(e):
            raise RuntimeError("non-finite energy")
        rec.update({"status": "ok", "natoms": n, "energy_per_atom": e / n,
                    "energy_per_atom_sem": 0.0, "volume_per_atom": vol / n,
                    "relaxed_cell": cell1.tolist()})
        rec["settings"] = {"engine": "LAMMPS 2025-07-22 python module, 1 thread",
                           "potential": "MEAM Jelinek et al. PRB 85 245102 (2012)",
                           "relax": "fix box/relax tri 0.0 vmax 0.001 (all six cell dofs) + cg minimize "
                                    "0 1e-8 x4, own cell",
                           "final_fmax_eV_A": fmax, "final_pressure_bar": press}
        rec["structure"] = disp_indicator(frac0, frac1, cell0)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass
    rec["wall_s"] = time.perf_counter() - t0
    write_json(result_path("lammps_meam_relax", cid), rec)
    return cid, rec["status"], rec["energy_per_atom"], rec["wall_s"], rec["error"]


def run_gpumd_relax_source(todo):
    for cid in todo:
        rec = mark_contention(gpumd_relax_query(cid))
        write_json(result_path("gpumd_nep89_relax", cid), rec)
        print(f"[gpumd_nep89_relax] {cid} {rec['status']} E={rec['energy_per_atom']} "
              f"wall={rec['wall_s']:.2f}s {rec['error'] or ''}", flush=True)


def run_lammps_relax_source(todo, workers):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=20) as pool:
        for cid, st, e, w, err in pool.imap_unordered(lammps_relax_query, todo):
            print(f"[lammps_meam_relax] {cid} {st} E={e} wall={w:.2f}s {err or ''}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def set_sample_ps(ps):
    global SAMPLE_PS, N_SAMPLE
    os.environ["LIVE_SAMPLE_PS"] = repr(float(ps))
    SAMPLE_PS = float(ps)
    N_SAMPLE = int(round(SAMPLE_PS * 1000.0 / DT_FS))
    MD_COMMON["sample_ps"] = SAMPLE_PS
    MD_COMMON["sample_steps"] = N_SAMPLE


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, choices=SOURCES)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cids", help="comma-separated candidate ids")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="process pool size (lammps_meam_md only; GPU sources run serially)")
    ap.add_argument("--sample-ps", type=float, default=None,
                    help="MD sampling length in ps after the 1 ps equilibration (default 3)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="MD temperature in K (default 300; propagated to workers via LIVE_T_K)")
    ap.add_argument("--results-dir", default=None,
                    help="write results/<source>/ under this directory instead of ./results")
    ap.add_argument("--host-lattice", choices=["0K", "npt"], default=None,
                    help="cluster host lattice rule (default 0K; npt uses the source's own fcc-Al NPT volume at T)")
    args = ap.parse_args()
    if args.temperature is not None or args.results_dir or args.host_lattice:
        # re-exec so every module-level constant (and spawned workers) see the settings
        env = dict(os.environ)
        if args.temperature is not None:
            env["LIVE_T_K"] = repr(float(args.temperature))
        if args.results_dir:
            env["LIVE_RESULTS_DIR"] = str(Path(args.results_dir).resolve())
        if args.host_lattice:
            env["LIVE_HOST_LATTICE"] = args.host_lattice
        argv = [a for a in sys.argv[1:]]
        for flag in ("--temperature", "--results-dir", "--host-lattice"):
            while flag in argv:
                i = argv.index(flag)
                del argv[i:i + 2]
        argv = [a for a in argv if not a.startswith(("--temperature=", "--results-dir=", "--host-lattice="))]
        os.execve(sys.executable, [sys.executable, __file__] + argv, env)
    if args.sample_ps is not None:
        set_sample_ps(args.sample_ps)

    index = {d["cid"]: d for d in load_index()}
    cids = list(index) if args.all else [c.strip() for c in args.cids.split(",") if c.strip()]
    for c in cids:
        if c not in index:
            sys.exit(f"unknown cid {c}")
    if args.source in RELAX_DIAG:
        # diagnostic 0 K relaxations: formation-energy candidates only
        cids = [c for c in cids if index[c]["property"] == "formation_energy"]
    # host cell first so cluster queries can reuse it
    if HOST_LATTICE == "npt":
        # the NPT fcc-Al run defines the cluster host cell, so it goes first and clusters last
        cids.sort(key=lambda c: (0 if c == "mp_134_Al" else 2 if index[c]["kind"] in CLUSTER_KINDS else 1))
    else:
        cids.sort(key=lambda c: 0 if c == "clu_Al108" else 1)
    todo = [c for c in cids if not result_path(args.source, c).exists()]
    print(f"[{args.source}] {len(cids)} requested, {len(todo)} to run", flush=True)
    (RESULTS_DIR / args.source).mkdir(parents=True, exist_ok=True)
    if not todo:
        return
    t0 = time.perf_counter()
    if args.source in ("mace_relax", "mace_md"):
        run_mace(args.source, todo, index)
    elif args.source == "gpumd_nep89_md":
        run_gpumd_source(todo, index)
    elif args.source == "gpumd_nep89_relax":
        run_gpumd_relax_source(todo)
    elif args.source == "lammps_meam_relax":
        run_lammps_relax_source(todo, args.workers)
    else:
        run_lammps_source(todo, index, args.workers)
    print(f"[{args.source}] finished in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
