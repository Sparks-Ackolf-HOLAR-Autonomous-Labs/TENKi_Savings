# Reproducing the results of TENKi

Every number and figure in the paper can be regenerated at three levels.
Each level starts from the committed outputs of the level below, so a reader
can stop at any level and check the rest against the committed files.

| Level | What it regenerates | Typical time | Hardware |
|---|---|---|---|
| 1 | the figures, from the committed result files | minutes | any computer |
| 2 | the result files (analyses, replayed campaigns, attribution games), from the committed archived and computed records | minutes to about 10 hours per experiment | a multi-core workstation; the 50-seed band-gap runs used 16 cores |
| 3 | the computed records themselves (DFT, MD, machine-learned potentials) | hours to days | a CPU cluster (DFT) and a CUDA GPU (MACE-MP-0, GPUMD) |

Run every command from the repository root unless a step says otherwise.
`FILES.sha256` lists the checksum of every committed file
(`sha256sum -c FILES.sha256` checks the whole folder).

## 1. Software

**Python environments.** The analysis packages are `uv` projects with
lock files; the versions in `uv.lock` are the ones the results were computed
with (Python 3.13, `matplotlib < 3.11`).

```bash
# rating, transfer score, archive build and analyses
uv sync --project shared --extra materials --extra viz
# campaign protocols
uv sync --project tenki/TENKi
# figure scripts only
pip install -r writings/papers_2026/regen_figs/requirements.txt
```

The attribution experiments, where the folder holds them, have their own
project: `uv sync --project ateki`.

**Computed sources (level 3).** These are needed only to recompute the DFT,
MD and potential records. The Python side is pinned in
`compute/almgsi_campaign/requirements_compute.txt` (Python 3.12, MACE-MP-0
through `mace-torch` 0.3.16, LAMMPS Python module 22 July 2025, ase 3.29,
pymatgen 2026.9.24).

| Code | Version and settings | Used for |
|---|---|---|
| Quantum ESPRESSO | 7.4 with OpenMPI 5.0.3 (CHPC modules `quantum-espresso/7.4`, `openmpi/5.0.3`) | PBE relaxations, HSE06 single points, band gaps |
| Pseudopotentials | PseudoDojo NC SR v0.4, PBE, standard accuracy (`nc-sr-04_pbe_standard_upf.tgz` from pseudo-dojo.org); not redistributed here; unpack the element files as `<element>.upf` (for example `Al.upf`, `Mg.upf`, `Si.upf`) into `pseudos/` inside each campaign folder | all QE runs |
| GPAW | 25.7, plane waves, default PAW datasets, libxc 7.1.2 | r2SCAN single points |
| GPUMD | 5.8 with the NEP89 potential `nep89_20250409`; set `GPUMD_ROOT` to the GPUMD source tree with the built `src/gpumd` binary | NEP89 MD and relaxations |
| LAMMPS | 22 July 2025 with the MEAM package, through its Python module; MEAM of Jelinek et al. (2012), files in `compute/almgsi_campaign/potentials/` | MEAM MD and relaxations |
| MACE-MP-0 | `small` model, downloaded by `mace_mp(model="small")` on first use | 0 K relaxations, MD, quasi-harmonic free energies |
| pycalphad | 0.11.2 with the COST 507 database (`compute/almgsi_campaign/calphad/COST507_pycalphad.tdb`) | CALPHAD receiver |

The archived records were fetched once from the Materials Project (API key in
`MP_API_KEY`), JARVIS-DFT and the other sources named in the paper; the
caches are committed, so no step below needs network access unless it says
so.

## 2. Level 1: figures

```bash
python writings/papers_2026/regen_figs/<script>.py
```

Each figure script reads only committed result files (listed per figure in
`README.md`) and draws at print size into `writings/papers_2026/<paper>/figures/`.
Before release every figure script was run from the root of a copy of this
folder, with the pinned figure environment.

## 3. The version-locked band-gap archive

The archive (`shared/output/bandgap_archive_build/bandgap_universe.jsonl`, 261,053
records over eight data sources) is too large for git and is rebuilt from the
committed source caches in `shared/data/`:

```bash
uv run --project shared python compute/bandgap_archive/build_inventory.py
uv run --project shared python compute/bandgap_archive/make_splits.py
```

`build_inventory.py` writes `build_stats.json`, whose `manifest_sha256`
field must begin with `f2158ecdc78e0229`, the archive hash that the
band-gap result files record (for example as `universe_sha256_16`). `make_splits.py` (seed 42) writes the
held-out and visible candidate roles and checks that no candidate leaks
across them.

`compute/bandgap_archive/refetch_channels.py` (subcommands `snumat`, `mp-provenance`,
`r2scan`, `expt-gap`) is the network step that produced the committed caches
(accessed 25 September 2026); it needs `MP_API_KEY` and is not needed to
rebuild the archive. Database versions change over time, so a refetch can
give a different archive hash.

All band-gap scripts read the archive relative to `shared/`; the TENKi and
ATEKi experiment scripts resolve `shared/` from their own location, or from
`PEGKI_ROOT` when it is set.

## 4. Level 2: the band-gap campaigns

The campaign experiments run from `tenki/TENKi/` and read the archive of
section 3 (through `shared/`, or `PEGKI_ROOT`). Every experiment is seeded
(`--seed`, default 42); the defaults are the 20-seed runs of the paper, and
`--help` lists every option. Experiments that checkpoint per receiver and
seed resume after an interruption.

```bash
cd tenki/TENKi
uv run python experiments/11_multi_domain_flip.py
uv run python experiments/24_hyperedge_budget.py                 # k-way events against pairs
uv run python experiments/25_physics_informed_budget.py          # donor orderings as priors
uv run python experiments/26_paired_campaign.py --workers 16     # cold, full and selective reuse
uv run python experiments/26_paired_campaign.py --prior-shape margin \
  --output-dir ../results/paired_campaign_margin
uv run python experiments/27_learned_acquisition.py              # trained acquisition (seed 42)
uv run python experiments/27_learned_acquisition.py --seed 7 \
  --output-dir ../results/learned_acquisition_seeds/seed_7      # likewise seeds 1, 2, 3, 42
uv run python experiments/27b_acquisition_long_eval.py           # 100 seeds, budget 1,920
uv run python experiments/27c_training_seed_spread.py ../results/learned_acquisition_seeds/seed_*
uv run python experiments/28_cross_property_policy.py
uv run python experiments/28b_formation_energy_diagnosis.py
uv run python experiments/29_cost_calibration.py
uv run python experiments/30_exclusion_list_transfer.py --universe <absolute path to the archive> \
  --workers 18 --output-dir ../../compute/shared_queue_experiments/exclusion
uv run python experiments/31_parallel_width.py --universe <absolute path to the archive> \
  --workers 18 --output-dir ../../compute/shared_queue_experiments/width
```

Then, from the repository root,
`python tenki/results/paired_campaign_margin/margin_vs_even.py`,
`python compute/shared_queue_experiments/exclusion/analyze_exclusion.py` and
`python compute/shared_queue_experiments/width/analyze_width.py` write the
summaries. The predictions of experiments 30 and 31
(`PREDICTION.md` in each folder) were committed before those experiments
ran; the analysis scripts grade them.

**Cluster-scale replications (50 seeds).** `compute/chpc/stage1b_trackb.slurm`
runs experiment 24 at k = 3, 4 and 6 and `compute/chpc/stage2_paired.slurm`
runs experiment 26 at k = 4 and 6 and experiment 25, all with `--n-seeds 50`,
into `shared/output/bandgap_cluster_run1/` and `shared/output/bandgap_cluster_run2/`
(about 11 h and 5 h with 16 cores).

`learned_acquisition_salted_seed/` holds the first training run of
experiment 27, whose seed came from Python's per-process `hash()` and cannot
be regenerated; experiment 27b evaluates the weights stored there, and the
seeded reruns are in `learned_acquisition/` and `learned_acquisition_seeds/`.

## The computed Al-Mg-Si campaign (`compute/almgsi_campaign/`)

Run these steps from `compute/almgsi_campaign/` with the compute environment
(`requirements_compute.txt`). Each calculation writes one record
`results/<source>/<candidate>.json` with its settings, status, wall time and
core-seconds, and every script skips candidates whose record exists, so an
interrupted run resumes where it stopped.

**Level 3: the records.**

1. Candidate pool: `python generate_candidates.py` writes `candidates/` and
   `candidates_index.json` from the committed Materials Project cache
   `mp_almgsi_raw.json.gz` (`--refetch` queries the Materials Project again and
   needs `MP_API_KEY`). `python add_observed_phases.py` adds the observed
   precipitate phases (`candidates_index_observed.json`); `python
   vacancy_study.py --make` (with `VAC_N=3`, `4` or `5` for the larger cells)
   writes the vacancy cells.
2. Potentials and MD (one CUDA GPU and a few CPU cores):
   `bash run_gpu_sources.sh` (MACE-MP-0 relaxation, NEP89 MD, MACE-MP-0 MD),
   `bash run_cpu_sources.sh` (MEAM MD), or one source at a time with
   `python dgx_sources.py --source <source> --all`. The MD protocol is
   isotropic NPT at 0 bar, 1 ps equilibration and 3 ps sampling at 2 fs, with
   a fixed random seed per query; the temperature is set with `LIVE_T_K`
   (default 300). `bash run_temperature_study.sh` runs every MD source at 300,
   453 and 803 K into `results_T/T<temp>/`; `chpc_mace_temperature.sbatch` is
   the same MACE-MP-0 MD as a SLURM GPU array.
   `python qha_study.py` computes the MACE-MP-0 quasi-harmonic free energies.
3. DFT (a CPU cluster with Quantum ESPRESSO 7.4; 16 cores per PBE task and
   32 per HSE06 task were used): set `LIVE_ROOT` to a copy of this folder that
   holds `pseudos/`, then submit `qe/live_qe.sbatch` as an array over the chunks
   in `qe/chunks.json` (`qe/make_chunks.py` regenerates them). `qe/qe_driver.py`
   runs the PBE variable-cell relaxation and then the HSE06 single point at the
   PBE geometry (defaults `HSE_QDIV=1`, q grid equal to the k grid, and
   `HSE_NPOOL=8` k-point pools).
   `qe/hse_convergence.py` with `qe/hse_conv.sbatch` is the q-grid convergence
   test. The vacancy and migration cells use `qe/vacancy_qe.sbatch`,
   `qe/vacancy_qe_large.sbatch` and `qe/migration_qe.sbatch` in the same way;
   `qe/run_dgx_qe.sh` and `qe/run_vac_qe.sh` run the same drivers on a single
   workstation. The `submit_*_chpc.sh` and `fetch_*.sh` scripts copy inputs to
   and results from the University of Utah CHPC cluster and are specific to
   that cluster.
4. r2SCAN (GPAW 25.7): `mpirun -np 16 python gpaw/gpaw_r2scan.py --xc r2scan
   --ecut 700 --outdir gpaw_r2scan --cids <list>` over the chunks in
   `gpaw/chunks_pool.txt`, then `python gpaw/gpaw_r2scan_analyse.py`.
   `python fetch_mp_r2scan.py` fetches the archived Materials Project r2SCAN
   formation energies (network, `MP_API_KEY`).
5. CALPHAD: `python calphad/calphad_receiver.py` in an environment with
   pycalphad 0.11.2 and ase.

**Level 2: from records to results.**

```bash
python aggregate_sources.py    # records -> results/table_<source>.csv and live_values.json
python analyze_live.py         # agreement between sources -> results/analysis/
python campaign.py --receiver <receiver> --donors <comma-separated donors> [--pool archived] [--shapley] [--cost combined] [--tag <name>]
```

Each campaign folder `results/campaigns/<name>/` holds a `summary.json` that
records its receiver, its donors (the `agreement` block), pool, cost mode,
batch sizes and number of seeds (20); rerun `campaign.py` with those values
to regenerate the folder. For example, the HSE06 receiver with all five
donors is `python campaign.py --receiver qe_hse06 --donors
mace_relax,gpumd_nep89_md,lammps_meam_md,archive_mp,qe_pbe --tag
qe_hse06_all`; the `*_shapley` folders add `--shapley` (every coalition of
donors is run) and the `*_combined` folders add `--cost combined` (each
HSE06 query also pays its PBE relaxation). MD
values are drawn per run from their measured standard error (`--no-md-noise`
uses the time averages instead). `python live_campaign.py --budget 32 --k 4`
is the on-demand campaign that triggers each calculation when it is first
queried; `python live_campaign.py --replay results_T/T300` replays it over the
tabulated values and `python compare_live_replay.py` compares the two.
`temperature_study.py` (run from `shared/`: `uv run python
../compute/almgsi_campaign/temperature_study.py`) and the other `*_study.py`
scripts listed in `README.md` write the remaining analysis files; each
script's docstring names its inputs and outputs.

The HSE06 values computed with a q grid of half the k grid were not
converged and are left out of this folder; every HSE06 number uses the full
q grid.

## The computed semiconductor band-gap campaign (`compute/semiconductor_campaign/`)

Run from `compute/semiconductor_campaign/`. The pool holds 171 candidates: 139
compounds of the Zhuo et al. measured-gap compilation (measured gap 0.4 to
9 eV, at most 6 atoms in the primitive cell) and 32 ordered 50/50 alloys of
16 semiconductor pairs, of which 24 match no Materials Project entry.

1. `python generate_pool.py` writes `candidates/` and `candidates_index.json`
   from the committed caches `mp_semiconductor_raw.json.gz` and
   `mp_alloy_raw.json.gz` (`--refetch` needs `MP_API_KEY`).
2. DFT: with `pseudos/` filled and `LIVE_ROOT` set to this folder, submit
   `qe/gap_qe.sbatch` as an array over `qe/chunks.json` (16 cores per task).
   `qe/qe_gap_driver.py` runs, per candidate, the PBE variable-cell relaxation
   (k spacing 0.30 1/A), a PBE SCF with empty bands (0.22 1/A) and an HSE06
   SCF at the PBE geometry; ecutwfc 90 Ry, ecutrho 360 Ry, ecutfock 180 Ry,
   Gaussian smearing 0.005 Ry, spin-unpolarised, no spin-orbit coupling. The
   gap is the conduction-band minimum minus the valence-band maximum on the
   Gamma-centred mesh. `qe/hse_qconv.sbatch` is the q-grid check on GaAs, ZnO
   and MgO. The other `qe/chunks_*.json` files are the retry and extra chunks
   of the same run.
3. `python archive_donors.py` looks every candidate up in the band-gap
   archive (section 3; path from `BANDGAP_UNIVERSE`).
4. Campaigns: `python campaign_gap.py --receiver qe_hse06 --pool nohalide
   --shapley`, `--receiver qe_hse06 --pool all`, `--receiver expt_zhuo --pool
   nohalide`, and `--cost combined` for the `*_combined` folders (each HSE06
   query also pays its PBE relaxation). `python summarise.py` collects the
   numbers into `results/SUMMARY.md` and `results/summary_numbers.json`.

## The computed Al-Si-Cu pool (`compute/alsicu_campaign/`)

Run from `compute/alsicu_campaign/`. The pool is built as for Al-Mg-Si: 3
elemental references, 22 archived Materials Project structures, 2 fcc
decorations the Materials Project holds and 141 suggested fcc decorations
(168 candidates, formation energy per atom).

1. `python generate_candidates.py` (cache `mp_alsicu_raw.json.gz`;
   `--refetch` needs `MP_API_KEY`).
2. `bash run_gpu_sources.sh`: MACE-MP-0 0 K relaxation (about 4 minutes on
   one GPU).
3. With `pseudos/` filled (Al, Si, Cu) and `LIVE_ROOT` set, submit
   `qe/live_qe.sbatch` over `qe/chunks.json` (PBE, ecutwfc 90 Ry, ecutrho
   360 Ry, k spacing 0.20 1/A, Marzari-Vanderbilt smearing 0.02 Ry,
   variable-cell relaxation; 16 cores per task).
4. `python aggregate_sources.py` writes the tables and `results/agreement.json`;
   `python campaign.py --receiver qe_pbe --donors mace_relax,archive_mp`
   writes `results/campaigns/qe_pbe_all/`.
