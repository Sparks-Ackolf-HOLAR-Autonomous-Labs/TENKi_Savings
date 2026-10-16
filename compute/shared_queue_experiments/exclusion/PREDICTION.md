# Exclusion-list transfer (TENKi follow-up T2): prediction registered before the run

Registered 2026-10-04, before any production result of
`tenki/TENKi/experiments/30_exclusion_list_transfer.py` existed (one
single-seed smoke test on the experimental receiver was run to check the
code; it is not part of the result set).

Setting: 8 receivers x 7 donors x 20 matched seeds, budget 240, k = 4 and
pairs, list size m = 2 (exp 30 docstring).

**P1 (primary).** For the experimental receiver (expt_zhuo), averaged over
the four GGA-class donors (mp_gga, mp_gga_expsynth, jarvis_optb88vdw,
snumat_gga), on the k-way arm, the early-budget tau advantage over the cold
campaign is larger for the exclusion list at fixed trust (excl_fixed) than
for the full donor ordering at fixed trust (full_fixed).

Reason: GGA-class databases misorder the wide-gap families against
measured gaps (gap compression at the top of the range), while all donors
agree on which families give the narrowest gaps.

**P2 (secondary).** Averaged over all receivers and donors, the exclusion
list applied to acquisition (excl_acq) does not lower the final-budget tau
relative to the cold campaign (mean final delta >= 0).
