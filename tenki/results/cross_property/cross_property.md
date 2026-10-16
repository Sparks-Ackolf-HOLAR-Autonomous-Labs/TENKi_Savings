# Experiment 28: cross-property transfer of the trained acquisition policy

Policy trained with CEM on ALL band-gap channels of the frozen band-gap archive universe, then frozen and evaluated ZERO-SHOT on channels of other property families.

Frozen weights: coverage -1.262, uncertainty +0.218, closeness +0.001, noise -0.086
k = 4; budget = 240; 20 eval seeds; training seeds disjoint from eval seeds.

| channel | family | classes | records | learned tau | random tau | learned 90% budget | random 90% budget | primary | secondary |
|---|---|---|---|---|---|---|---|---|---|
| bulk_aflow | bulk_modulus | 8 | 5639 | +0.9500 | +0.9071 | 24 | 80 | PASS | PASS |
| bulk_jarvis | bulk_modulus | 8 | 16334 | +0.9333 | +0.8929 | 120 | 120 | PASS | FAIL |
| bulk_mp | bulk_modulus | 8 | 12888 | +0.9333 | +0.9214 | 32 | 128 | PASS | PASS |
| fe_aflow_dHf | formation_energy | 5 | 3197 | +0.6833 | +0.7900 | 192 | 80 | FAIL | FAIL |
| fe_mp_dHf | formation_energy | 4 | 206 | +0.9500 | +0.9333 | 72 | 48 | PASS | FAIL |
| hardness_chen_niu | hardness | 8 | 16334 | +0.9167 | +0.8464 | 64 | 152 | PASS | PASS |

## Per property family

| family | channels | learned tau | random tau | learned 90% budget | random 90% budget | primary | secondary |
|---|---|---|---|---|---|---|---|
| bulk_modulus | 3 | +0.9389 | +0.9071 | 40 | 112 | PASS | PASS |
| formation_energy | 2 | +0.8167 | +0.8617 | 144 | 56 | FAIL | FAIL |
| hardness | 1 | +0.9167 | +0.8464 | 64 | 152 | PASS | PASS |

Primary (transfer does no harm, ALL channels): **FAIL** (5/6 channels)
Secondary (transfer speeds up): 3/6 channels PASS (bulk_aflow, bulk_mp, hardness_chen_niu)

## Skipped channels (honest thinness reporting)

- `hardness_exp` (hardness): only 0 classes with >= 30 records (need >= 4); raw class counts {}

## Band-gap in-family context (exp 27, LOCO)

learned +0.7733 vs random +0.7311 final tau; 90% budget 80 vs 120. (Exp 27 weights are per-LOCO-fold over band-gap channels; this experiment trains one all-channel policy.)

Formation-energy channels are oriented LOWER-is-better (most negative = most stable) through hyperedge_from_scores's higher_is_better flag; both orientations verified in --self-test. Real archived records only; no invented observations. Weights frozen at evaluation time (zero-shot).
