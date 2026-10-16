# Experiment 27: learned hyperedge acquisition (rule vs trained)

Channels: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse
k = 4; budget = 240; 20 eval seeds; CEM 8 iters x 20 pop; leave-one-channel-out.

| arm | final tau | budget to 90% of random-arm final |
|---|---|---|
| pair_random | +0.6332 | 240 |
| kway_random | +0.7287 | 128 |
| kway_active | +0.7237 | 160 |
| kway_fisher | +0.7058 | 160 |
| kway_learned | +0.7379 | 96 |

Learned weights (mean +/- std over folds): coverage -0.71+/-1.01, uncertainty +0.29+/-0.56, closeness -0.31+/-0.90, noise -0.04+/-0.98

Primary (learned >= random at final budget): **PASS**
Secondary (learned faster to 90% of random final: 96 vs 128): **PASS**

Every curve for kway_learned is on a channel the policy never
trained on (LOCO). Training seeds disjoint from eval seeds.
Real archived records only; no invented observations.
