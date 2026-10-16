# Experiment 26: paired cold/full/selective campaign protocol

Receivers: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse; k=4; budget=240; 50 seeds (matched across conditions); lambda_full=8.0; early window = first 5 of 10 checkpoints.

Reference-policy mapping: cold=random, full=per-channel, selective=transfer-aware, combined=pooled.

## Condition x arm summary (mean over all donor->receiver cells)

| condition | arm | mean T_early | mean T_final | cells>0 (early) |
|---|---|---|---|---|
| combined | kway | +0.1662 | +0.0548 | 100% |
| combined | pair | +0.2338 | +0.0968 | 100% |
| full | kway | +0.1467 | +0.0349 | 79% |
| full | pair | +0.2311 | +0.0783 | 88% |
| selective | kway | +0.1257 | +0.0323 | 77% |
| selective | pair | +0.1961 | +0.0635 | 95% |

## Hyperedge residual vs clique-expansion cyclic energy (cold k-way stream)

| receiver | mean hyperedge residual | clique cyclic fraction |
|---|---|---|
| expt_zhuo | 0.2252 | 0.0665 |
| jarvis_mbj | 0.3067 | 0.1922 |
| jarvis_optb88vdw | 0.3501 | 0.2655 |
| mp_gga | 0.3452 | 0.2728 |
| mp_gga_expsynth | 0.3021 | 0.1858 |
| mp_r2scan | 0.3196 | 0.2134 |
| snumat_gga | 0.3272 | 0.1984 |
| snumat_hse | 0.3071 | 0.1739 |

Full directed matrices with bootstrap CIs: paired_campaign.json.
Each evaluation = one real archived record (no invented observations); donor priors are ordinal-only.
