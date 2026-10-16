# Experiment 26: paired cold/full/selective campaign protocol

Receivers: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse; k=4; budget=240; 20 seeds (matched across conditions); lambda_full=8.0; early window = first 5 of 10 checkpoints.

Reference-policy mapping: cold=random, full=per-channel, selective=transfer-aware, combined=pooled.

## Condition x arm summary (mean over all donor->receiver cells)

| condition | arm | mean T_early | mean T_final | cells>0 (early) |
|---|---|---|---|---|
| combined | kway | +0.1666 | +0.0488 | 100% |
| combined | pair | +0.2339 | +0.0979 | 100% |
| full | kway | +0.1490 | +0.0264 | 80% |
| full | pair | +0.2386 | +0.0823 | 89% |
| selective | kway | +0.1282 | +0.0218 | 84% |
| selective | pair | +0.1968 | +0.0609 | 95% |

## Hyperedge residual vs clique-expansion cyclic energy (cold k-way stream)

| receiver | mean hyperedge residual | clique cyclic fraction |
|---|---|---|
| expt_zhuo | 0.2225 | 0.0681 |
| jarvis_mbj | 0.3061 | 0.1927 |
| jarvis_optb88vdw | 0.3500 | 0.2683 |
| mp_gga | 0.3447 | 0.2687 |
| mp_gga_expsynth | 0.2967 | 0.1793 |
| mp_r2scan | 0.3221 | 0.2214 |
| snumat_gga | 0.3310 | 0.1985 |
| snumat_hse | 0.3133 | 0.1828 |

Full directed matrices with bootstrap CIs: paired_campaign.json.
Each evaluation = one real archived record (no invented observations); donor priors are ordinal-only.
