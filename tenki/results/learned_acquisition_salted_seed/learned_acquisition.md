# Experiment 27: learned hyperedge acquisition (rule vs trained)

Channels: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse
k = 4; budget = 240; 20 eval seeds; CEM 8 iters x 20 pop; leave-one-channel-out.

| arm | final tau | budget to 90% of random-arm final |
|---|---|---|
| pair_random | +0.6304 | 240 |
| kway_random | +0.7311 | 120 |
| kway_active | +0.7308 | 144 |
| kway_fisher | +0.7185 | 152 |
| kway_learned | +0.7733 | 80 |

Learned weights (mean +/- std over folds): coverage -0.82+/-1.33, uncertainty -0.08+/-0.50, closeness -0.52+/-0.60, noise -0.17+/-0.57

Primary (learned >= random at final budget): **PASS**
Secondary (learned faster to 90% of random final: 80 vs 120): **PASS**

Every curve for kway_learned is on a channel the policy never
trained on (LOCO). Training seeds disjoint from eval seeds.
Real archived records only; no invented observations.
