# Experiment 27: learned hyperedge acquisition (rule vs trained)

Channels: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse
k = 4; budget = 240; 20 eval seeds; CEM 8 iters x 20 pop; leave-one-channel-out.

| arm | final tau | budget to 90% of random-arm final |
|---|---|---|
| pair_random | +0.6443 | 176 |
| kway_random | +0.7153 | 152 |
| kway_active | +0.7354 | 160 |
| kway_fisher | +0.7292 | 120 |
| kway_learned | +0.7537 | 64 |

Learned weights (mean +/- std over folds): coverage -1.25+/-1.56, uncertainty +0.13+/-0.73, closeness -0.53+/-0.75, noise -0.16+/-0.31

Primary (learned >= random at final budget): **PASS**
Secondary (learned faster to 90% of random final: 64 vs 152): **PASS**

Every curve for kway_learned is on a channel the policy never
trained on (LOCO). Training seeds disjoint from eval seeds.
Real archived records only; no invented observations.
