# TENKi

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23413457.svg)](https://doi.org/10.5281/zenodo.23413457)

**Paper.** TENKi: counting the value of borrowed knowledge to a materials campaign in measurements saved (submitted to Machine Learning: Science and Technology (IOP)).

This repository holds the code and data behind the paper: the scripts that draw its figures, the result files they read, the code that produced those results, and the archived or computed inputs that code reads. The manuscript itself is not part of the repository. Every script runs unchanged from the repository root.

**Authors.** Kinston Ackölf, Taylor D. Sparks, Materials Science and Engineering, The University of Utah.

**Citation.** `CITATION.cff` (software citation) and `.zenodo.json` (archive metadata).

## Layout

- `shared/`, `compute/`, `tenki/`: result files and the code that produced them.
- `writings/papers_2026/regen_figs/` and `paper_fig_style.py`: figure scripts (each reads only committed result files and draws at print size into `writings/papers_2026/<paper>/figures/`).

## Running

Figures: run a script from the folder root, for example `python writings/papers_2026/regen_figs/<script>.py` (environment: `writings/papers_2026/regen_figs/requirements.txt`). `REPRODUCE.md` gives the full procedure: software versions, the rebuild of the band-gap archive, the commands that regenerate every result file, and the DFT, MD and potential calculations behind the computed records.

## Figure scripts

| Figure | Script | Files read |
|---|---|---|
| `multi_domain_flip.png` | `writings/papers_2026/regen_figs/tenki_multi_domain_flip.py` | `shared/output/bandgap_cluster_run2/paired_campaign_k4/paired_campaign.json`, `tenki/results/multi_domain_flip.json` |
| `learned_acquisition.png` | `writings/papers_2026/regen_figs/tenki_learned_acquisition.py` | `tenki/results/learned_acquisition/learned_acquisition.json` |
| `cross_property.png` | `writings/papers_2026/regen_figs/tenki_cross_property.py` | `tenki/results/cross_property/cross_property.json`, `tenki/results/cross_property/formation_energy_diagnosis.json` |
| `physics_informed_budget.png` | `writings/papers_2026/regen_figs/tenki_physics_informed_budget.py` | `tenki/results/physics_informed_budget/physics_informed_budget.json` |
| `paired_campaign.png` | `writings/papers_2026/regen_figs/tenki_paired_campaign.py` | `tenki/results/paired_campaign/paired_campaign.json` |
| `live_campaign.png` | `writings/papers_2026/regen_figs/live_almgsi_figs.py` | `compute/almgsi_campaign/results/analysis/source_agreement.json`, `compute/almgsi_campaign/results/campaigns/qe_hse06_all/summary.json`, `compute/almgsi_campaign/results/campaigns/qe_pbe_all/summary.json`, `compute/almgsi_campaign/results_T/analysis/temperature_study.json` |
| `cost_calibration.png` | `writings/papers_2026/regen_figs/tenki_cost_calibration.py` | `tenki/results/cost_calibration/cost_calibration.json` |
| `parallel_width.png` | `compute/shared_queue_experiments/width/analyze_width.py` | `compute/shared_queue_experiments/width/parallel_width_raw.json` |
| `exclusion_list.png` | `compute/shared_queue_experiments/exclusion/analyze_exclusion.py` | `compute/shared_queue_experiments/exclusion/exclusion_list_transfer.json` |
| `semiconductor_campaign.png` | `writings/papers_2026/regen_figs/semiconductor_campaign_figs.py` | `compute/semiconductor_campaign/results/campaigns/expt_zhuo_nohalide/summary.json`, `compute/semiconductor_campaign/results/campaigns/qe_hse06_nohalide/alloy_hits.json`, `compute/semiconductor_campaign/results/campaigns/qe_hse06_nohalide/summary.json` |
| `acquisition_long.png` | `writings/papers_2026/regen_figs/tenki_acquisition_long.py` | `tenki/results/learned_acquisition_long/fold_expt_zhuo.json`, `tenki/results/learned_acquisition_long/fold_jarvis_mbj.json`, `tenki/results/learned_acquisition_long/fold_jarvis_optb88vdw.json`, `tenki/results/learned_acquisition_long/fold_mp_gga.json` and 4 more |
| `prisms_comparison.png` | `writings/papers_2026/regen_figs/tenki_prisms_comparison.py` | `compute/almgsi_campaign/results/analysis/prisms_comparison/prisms_comparison_qe_hse06.json`, `compute/almgsi_campaign/results/analysis/prisms_comparison/prisms_comparison_qe_pbe.json` |
| `flip_budget.png` | `writings/papers_2026/regen_figs/tenki_flip_budget.py` | `tenki/results/flip_budget/flip_budget.json` |
| `donor_roles.png` | `writings/papers_2026/regen_figs/tenki_multi_domain_flip.py` | `shared/output/bandgap_cluster_run2/paired_campaign_k4/paired_campaign.json`, `tenki/results/multi_domain_flip.json` |

## Producing code

The result files above were written by these scripts (local modules they import are included as well):

- `tenki/TENKi/experiments/11_multi_domain_flip.py`
- `tenki/TENKi/experiments/24_hyperedge_budget.py`
- `tenki/TENKi/experiments/25_physics_informed_budget.py`
- `tenki/TENKi/experiments/26_paired_campaign.py`
- `tenki/TENKi/experiments/27_learned_acquisition.py`
- `tenki/TENKi/experiments/27b_acquisition_long_eval.py`
- `tenki/TENKi/experiments/27c_training_seed_spread.py`
- `tenki/TENKi/experiments/28_cross_property_policy.py`
- `tenki/TENKi/experiments/28b_formation_energy_diagnosis.py`
- `tenki/TENKi/experiments/29_cost_calibration.py`
- `tenki/TENKi/experiments/30_exclusion_list_transfer.py`
- `tenki/TENKi/experiments/31_parallel_width.py`
- `tenki/results/paired_campaign_margin/margin_vs_even.py`
- `compute/shared_queue_experiments/width/analyze_width.py`
- `compute/shared_queue_experiments/exclusion/analyze_exclusion.py`
- `compute/bandgap_archive/build_inventory.py`
- `compute/bandgap_archive/make_splits.py`
- `compute/almgsi_archive/curate_channels.py`
- `compute/almgsi_campaign/generate_candidates.py`
- `compute/almgsi_campaign/add_observed_phases.py`
- `compute/almgsi_campaign/dgx_sources.py`
- `compute/almgsi_campaign/aggregate_sources.py`
- `compute/almgsi_campaign/analyze_live.py`
- `compute/almgsi_campaign/campaign.py`
- `compute/almgsi_campaign/live_campaign.py`
- `compute/almgsi_campaign/compare_live_replay.py`
- `compute/almgsi_campaign/temperature_study.py`
- `compute/almgsi_campaign/dft_receiver_study.py`
- `compute/semiconductor_campaign/generate_pool.py`
- `compute/semiconductor_campaign/archive_donors.py`
- `compute/semiconductor_campaign/campaign_gap.py`
- `compute/semiconductor_campaign/summarise.py`
- `compute/alsicu_campaign/generate_candidates.py`
- `compute/alsicu_campaign/dgx_sources.py`
- `compute/alsicu_campaign/aggregate_sources.py`
- `compute/alsicu_campaign/campaign.py`
- `compute/almgsi_campaign/prisms_comparison.py`
- `compute/almgsi_campaign/prisms_sensitivity.py`
- `compute/almgsi_archive/audit_almgsi_coverage.py`
- `compute/almgsi_campaign/rerun_md_noise.py`
- `compute/almgsi_campaign/temperature_coalitions.py`
- `compute/almgsi_campaign/shard_cids.py`

## Notes on inputs

- The version-locked band-gap archive (`shared/output/bandgap_archive_build/bandgap_universe.jsonl`, 261,053 records) is not stored in git; rebuild it from the included source caches with `python compute/bandgap_archive/build_inventory.py` and check it against the manifest hash in `build_stats.json`.
- Computed Al-Mg-Si records (DFT, MACE-MP-0, MD) are under `compute/almgsi_campaign/results*`; the HSE06 values with the half q grid, set aside after the convergence test, are left out.

## Contents by role

- figure data: 559 files
- code: 141 files
- input data: 8338 files
- environment: 6 files
- result named in paper: 81 files

`FILES.sha256` lists every file with its SHA-256 checksum; `CONTENTS.tsv` gives the role of each file.

## Provenance

The folder was assembled from the `singular_obj/` project of the Sparks group repository by `build_paper_repos.py`, which keeps the project's relative layout and gives working folders their public names.

## Licence

Code (Python, shell and SLURM scripts) is released under the MIT License (LICENSE). Data, result files and figures are released under the Creative Commons Attribution 4.0 International licence (LICENSE-CC-BY-4.0.txt). The archived third-party datasets (Materials Project, JARVIS-DFT, SNUMAT, the Zhuo et al. experimental compilation) keep their own terms; their sources are cited in the paper.

## Acknowledgements

Taylor D. Sparks acknowledges support from the Army Research Office Materials Design, under Contract No. W911NF-23-1-0333.
