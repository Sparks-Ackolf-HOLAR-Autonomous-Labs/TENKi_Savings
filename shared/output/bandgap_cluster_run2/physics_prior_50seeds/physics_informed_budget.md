# Experiment 25: physics-informed relative priors under budget

Channels: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse
k = 4; budget = 240; 50 seeds; fixed lambda = 4.0; grid = [0.01, 0.5, 2.0, 8.0, 32.0].

| channel | donor (prior source) | prior quality tau |
|---|---|---|
| expt_zhuo | mp_gga | +0.524 |
| jarvis_mbj | mp_gga | +0.857 |
| jarvis_optb88vdw | mp_gga | +0.786 |
| mp_gga | mp_gga_expsynth | +0.786 |
| mp_gga_expsynth | mp_gga | +0.786 |
| mp_r2scan | mp_gga | +0.714 |
| snumat_gga | mp_gga | +0.357 |
| snumat_hse | mp_gga | +0.357 |

| arm | early mean tau | final tau | mean MM iters/fit |
|---|---|---|---|
| uninformed | +0.5428 | +0.7207 | 48.7 |
| physics_fixed | +0.6980 | +0.7505 | 63.2 |
| physics_adaptive | +0.6445 | +0.7571 | 123.7 |
| wrong_fixed | +0.0461 | +0.6395 | 76.8 |
| wrong_adaptive | +0.4046 | +0.6970 | 340.2 |

Evaluations saved by the physics prior (budget for uninformed to match physics_fixed's first-checkpoint tau): **112** of 240.

Primary (early gain, no final harm): **PASS**
Secondary (adaptive recovers from wrong prior): **PASS**
Tertiary (warm start, fewer MM iterations): **FAIL**

Priors are ordinal-only cross-channel information (donor medians
never cross systems as values). Retrospective querying of the
frozen universe; no invented observations.
