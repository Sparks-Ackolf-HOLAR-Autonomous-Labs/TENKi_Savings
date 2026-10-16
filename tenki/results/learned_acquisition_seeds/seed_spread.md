# Experiment 27c: experiment 27 over five training seeds

k = 4; budget 240; 20 evaluation seeds per run; leave-one-source-out.
Each --seed value changes training draws and evaluation seeds.

| seed | learned final tau | random 4-way | min-spread | Fisher-greedy | pairwise | budget to 90%: learned / random | sources learned >= random |
|---|---|---|---|---|---|---|---|
| 42 | +0.752 | +0.731 | +0.731 | +0.718 | +0.630 | 88 / 120 | 4 of 8 |
| 1 | +0.754 | +0.715 | +0.735 | +0.729 | +0.644 | 64 / 152 | 6 of 8 |
| 2 | +0.738 | +0.729 | +0.724 | +0.706 | +0.633 | 96 / 128 | 5 of 8 |
| 3 | +0.735 | +0.730 | +0.731 | +0.698 | +0.655 | 112 / 136 | 4 of 8 |
| 7 | +0.703 | +0.717 | +0.738 | +0.710 | +0.656 | 96 / 160 | 3 of 8 |

Learned minus random 4-way at the final budget: mean +0.012 (range -0.014 to +0.038); learned at or above random in 4 of 5 runs; faster to 90% of the random arm's final tau in 5 of 5 runs.

Mean closeness weight per run: -0.26, -0.53, -0.31, -0.09, +0.28
