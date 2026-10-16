# Experiment 24: hyperedge query budgets

Channels: expt_zhuo, jarvis_mbj, jarvis_optb88vdw, mp_gga, mp_gga_expsynth, mp_r2scan, snumat_gga, snumat_hse
k = 4; budget = 240 evaluations; 20 seeds; seed base 42.

| arm | final tau (mean over channels) |
|---|---|
| pair_random | +0.6304 |
| kway_random | +0.7311 |
| kway_active | +0.7308 |

Primary (kway_random >= pair_random at final budget): **PASS**
Secondary (active reaches 90% of its final tau earlier than random: 144 vs 120 evaluations): **FAIL**

Each evaluation = one real archived record sampled from the
queried class (retrospective querying; no invented observations).
Plackett-Luce MM fit; k=2 arm is exact Bradley-Terry.
