# Experiment 26: paired cold/full/selective campaign protocol

Receivers: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse; k=4; budget=240; 20 seeds (matched across conditions); lambda_full=8.0; early window = first 5 of 10 checkpoints. Donor prior shape: margin (donor median margins, standardised to the even-prior spread).

Reference-policy mapping: cold=random, full=per-channel, selective=transfer-aware, combined=pooled.

## Condition x arm summary (mean over all donor->receiver cells)

| condition | arm | mean T_early | mean T_final | cells>0 (early) |
|---|---|---|---|---|
| combined | kway | +0.1579 | +0.0470 | 88% |
| combined | pair | +0.2274 | +0.1006 | 88% |
| full | kway | +0.1555 | +0.0284 | 88% |
| full | pair | +0.2450 | +0.0824 | 95% |
| selective | kway | +0.1342 | +0.0256 | 88% |
| selective | pair | +0.2030 | +0.0690 | 95% |

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
Each evaluation = one real archived record (no invented observations); donor priors are relative (standardised donor margins, no absolute eV).
