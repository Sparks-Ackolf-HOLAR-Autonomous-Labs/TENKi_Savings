# Experiment 27: learned hyperedge acquisition (rule vs trained)

Channels: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse
k = 4; budget = 240; 20 eval seeds; CEM 8 iters x 20 pop; leave-one-channel-out.

| arm | final tau | budget to 90% of random-arm final |
|---|---|---|
| pair_random | +0.6304 | 240 |
| kway_random | +0.7311 | 120 |
| kway_active | +0.7308 | 144 |
| kway_fisher | +0.7185 | 152 |
| kway_learned | +0.7522 | 88 |

Learned weights (mean +/- std over folds): coverage -1.39+/-1.22, uncertainty +0.67+/-0.59, closeness -0.26+/-0.81, noise -0.16+/-0.45

Primary (learned >= random at final budget): **PASS**
Secondary (learned faster to 90% of random final: 88 vs 120): **PASS**

Every curve for kway_learned is on a channel the policy never
trained on (LOCO). Training seeds disjoint from eval seeds.
Real archived records only; no invented observations.
