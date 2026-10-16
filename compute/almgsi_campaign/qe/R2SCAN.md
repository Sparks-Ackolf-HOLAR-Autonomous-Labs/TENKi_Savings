# r2SCAN single points (third DFT level)

r2SCAN meta-GGA total energies at the PBE-relaxed geometries of the 180
formation-energy candidates, computed on the DGX Spark with Quantum ESPRESSO.
Together with PBE (`results/qe_pbe/`) and HSE06 (`results/qe_hse06/`) this
gives three DFT levels on one candidate pool, one pseudopotential set, and one
geometry per compound.

## Build (aarch64, DGX)

conda-forge `qe` (7.4 and 7.5 for linux-aarch64) is not linked against libxc,
and QE evaluates r2SCAN only through libxc, so pw.x was built from source:

```bash
# toolchain and libraries from conda-forge (micromamba 2.9);
# $QE_LIBXC_PREFIX is the install prefix of this environment and of pw.x
MAMBA_ROOT_PREFIX=<mamba root> micromamba create -y -p $QE_LIBXC_PREFIX -c conda-forge \
  "libxc=6.2.2=cpu*" openmpi=4.1 fftw scalapack libopenblas "libblas=*=*openblas" liblapack \
  cmake make gfortran gcc gxx git pkg-config
# QE 7.4 source: https://github.com/QEF/q-e/archive/refs/tags/qe-7.4.tar.gz, unpacked to q-e-qe-7.4/
# (the commands below run from that folder)
# libxc 6 ships a CMake config whose version check rejects the "5.1.2" request
# (same-major-version rule), so drop the version in QE's config-mode lookup:
sed -i 's/find_package(Libxc 5.1.2 COMPONENTS Fortran CONFIG)/find_package(Libxc COMPONENTS Fortran CONFIG)/' CMakeLists.txt
E=$QE_LIBXC_PREFIX; export PATH=$E/bin:$PATH
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 -DQE_ENABLE_LIBXC=ON \
  -DQE_ENABLE_SCALAPACK=OFF -DQE_ENABLE_OPENMP=OFF -DQE_FFTW_VENDOR=FFTW3 \
  -DBLAS_LIBRARIES=$E/lib/libblas.so -DLAPACK_LIBRARIES=$E/lib/liblapack.so \
  -DCMAKE_PREFIX_PATH=$E -DCMAKE_Fortran_FLAGS="-O2 -fallow-argument-mismatch"
make -j12 pw            # under a minute on the GB10
mkdir -p $E/qe/bin && cp bin/pw.x $E/qe/bin/
```

Runtime environment: `PATH=$QE_LIBXC_PREFIX/qe/bin:$QE_LIBXC_PREFIX/bin:$PATH`,
`LD_LIBRARY_PATH=$QE_LIBXC_PREFIX/lib`. `ldd pw.x` lists `libxc.so.12` and
`libxcf03.so.12`. Check: an fcc Al SCF with `input_dft='r2scan'` prints
`Exchange-correlation= R2SCAN ( 0 0 0 0 0 497 498)`, i.e. libxc
MGGA_X_R2SCAN (497) and MGGA_C_R2SCAN (498).

## Settings

| Setting | Value |
|---|---|
| Functional | r2SCAN, `input_dft='r2scan'` (libxc 6.2.2, ids 497 + 498) |
| Geometry | PBE vc-relaxed structure, `results/qe_pbe/<cid>.relaxed.xyz` (no relaxation) |
| Pseudopotentials | PseudoDojo NC SR v0.4 PBE standard (Al 3 e, Mg 10 e, Si 4 e), as for PBE and HSE06 |
| Cutoffs | ecutwfc 80 Ry, ecutrho 320 Ry by default (as HSE06); overridable with `R2SCAN_ECUTWFC`, `R2SCAN_ECUTRHO`. Not converged for r2SCAN, see "Cutoff convergence" |
| k points | Gamma-centred grid, spacing 0.30 1/A (same as HSE06) |
| Smearing | Marzari-Vanderbilt, 0.02 Ry |
| SCF | conv_thr 1e-9 Ry x natoms; plain mixing beta 0.3, 200 steps; on failure one retry with local-TF beta 0.1, 400 steps, one k-point pool |
| Parallelism | 12 MPI ranks, 4 k-point pools (1 pool when the grid has fewer than 4 points) |

Formation energy: `formation_energy_per_atom = e(cid) - sum_i x_i e_ref_i`
with r2SCAN energies of fcc Al (`mp_134_Al`), hcp Mg (`mp_153_Mg`) and
diamond Si (`mp_149_Si`) at their PBE geometries. The records keep the
`results/qe_hse06/` schema (`energy_per_atom`, `status`, `wall_s`,
`qe_cpu_s`, `core_s`, `device`, `settings`), so `aggregate_sources.py` picks
up `results/qe_r2scan/` as one more source. `mp_1079297_Si` has no
converged PBE geometry and is skipped (179 single points).

Pseudopotential note: meta-GGA in QE requires norm-conserving
pseudopotentials; the PseudoDojo set is PBE-generated, so r2SCAN is applied
with PBE cores. This is the usual practice with PseudoDojo and is the same
inconsistency HSE06 carries here.

## Files

- `qe/r2scan_driver.py`: driver (reuses `qe_driver.py`), restartable: skips existing `results/qe_r2scan/<cid>.json`; references first, then cells by increasing natoms x k points.
- `qe/run_r2scan.sh`: detached launcher; writes `logs/status.log`, `logs/r2scan.log`, and `logs/DONE` at the end.

## Cutoff convergence (2 Oct 2026): production run stopped, cutoff not converged

A first run at ecutwfc 80 / ecutrho 320 Ry completed the three references
and 10 compounds, then was stopped because the r2SCAN formation energies
depend strongly on the dense-grid cutoff. Formation energy of fcc_MgAl_04
(2 atoms, PBE 2.2 meV/atom, HSE06 -7.7 meV/atom), r2SCAN with the same
k grid and pseudopotentials:

| ecutwfc / ecutrho (Ry) | E_f (meV/atom) |
|---|---|
| 80 / 320 | +70.0 |
| 80 / 640 | +10.3 |
| 80 / 960 | -10.3 |
| 80 / 1280 | -18.8 |
| 120 / 480 | +30.3 |
| 160 / 640 | +9.9 |
| SG15 ONCV 1.2 (no core correction), 80 / 320 | -36.9 |
| SG15 ONCV 1.2 (no core correction), 80 / 640 | -5.9 |

The elemental references change by less than 6 meV/atom over the same
range; the compound total energy drifts. The drift follows ecutrho, not
ecutwfc (80/640 and 160/640 agree to 0.4 meV/atom), so it comes from the
real-space integration of the meta-GGA energy on the dense FFT grid, and it
is present without nonlinear core corrections (SG15), so it is not caused by
the PseudoDojo core charges alone. A second symptom: mp_978271_Mg3Al at
80/320 hops between two SCF solutions 53 mRy apart (0.18 eV/atom) and ends
in the upper one.

Records from the stopped run (not valid r2SCAN values) are not included in
this repository (run-directory folder
`results/qe_r2scan_ecutrho320_unconverged/`), nor are the convergence test
inputs and outputs.

Further diagnostics that would precede a production run with this route:
an extended ecutrho scan on fcc_MgAl_04 and mp_978271_Mg3Al to see whether
E_f plateaus (run for fcc_MgAl_04, see "Extended cutoff scan" below); libxc 7
(`libxc=7.*=cpu*` in the same environment, rebuild) in case the libxc 6.2.2
r2SCAN density and tau thresholds are the cause; and r2SCAN-generated or PAW
datasets in another code if the drift persists (the GPAW route below). The
driver and launcher take the cutoffs from `R2SCAN_ECUTRHO` (and optionally
`R2SCAN_ECUTWFC`).

## Running and collecting results

The run directory is a copy of this folder plus `results/qe_pbe/` and
`pseudos/` (`LIVE_ROOT`, by default the folder above `qe/`). Launch detached:

```bash
cd "$LIVE_ROOT"
QE_LIBXC_PREFIX=<libxc QE prefix> nohup bash qe/run_r2scan.sh > logs/run.log 2>&1 < /dev/null &
```

Progress: `grep -c '^R2SCAN ' logs/r2scan.log` (179 when complete);
finished when `logs/DONE` exists. If the run directory is a separate copy,
copy the records back into this folder:

```bash
rsync -a "$LIVE_ROOT/results/qe_r2scan/" results/qe_r2scan/
```

Copy them again after `logs/DONE` appears: the driver's last step rewrites
every record's `formation_energy_per_atom` with the final references.

## Extended cutoff scan and two alternative routes (3 Oct 2026)

Extended dense-grid scan on fcc_MgAl_04 (same settings, 4 ranks on the DGX,
a short test; references rerun at each cutoff):

| ecutrho (Ry) | 1280 | 1600 | 2000 | 2400 |
|---|---|---|---|---|
| E_f (meV/atom) | -18.8 | -20.0 | -23.3 | -24.9 |

The steps shrink (8.5 meV from 960 to 1280 Ry, then 1.2, 3.3 and 1.6 meV) but
do not settle within 1 meV/atom even at 2400 Ry, 7.5 times the PBE density
cutoff. Formation energies in the pool differ by a few meV/atom between
neighbouring compounds, so this route cannot give a usable ordering at an
affordable cost. Data: `results/r2scan_conv/qe_ecutrho_scan.json`.

Two alternative routes replace it:

1. **Archived r2SCAN** (no compute): Materials Project r2SCAN formation
   energies (VASP, PAW) for the 26 pool compounds MP holds,
   `fetch_mp_r2scan.py` -> `results/mp_r2scan/`, compared with the computed
   sources in `mp_r2scan_study.py` -> `results/analysis/mp_r2scan_study.md`.
2. **GPAW r2SCAN** (PAW datasets, libxc 7.1.2) on CHPC guest nodes,
   `gpaw/gpaw_r2scan.py`, `gpaw/gpaw_conv.sbatch`. The conda-forge GPAW build
   is not linked to libxc, so GPAW 25.7.0 was built from source against
   conda-forge libxc in `/scratch/general/vast/$USER/envs/gpaw_xc`
   (siteconfig libraries `xc blas lapack fftw3`; `-lopenblas` fails because
   the env ships only `libopenblas.so.0`). VASP modules exist on CHPC but are
   restricted to a license group that the project account does not belong to.
