# Parallel-width sweep: summary

Budget 512 evaluations, k = 4, receivers 8, seeds per receiver 30. Threshold = 0.9 x mean final tau of the one-agent random campaign (per receiver). Medians with interquartile range; campaigns that never reach the threshold count at the budget.

| rule | P | evaluations to threshold | rounds to threshold | reached | tau at half budget | final tau | duplicate queries |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| random | 1 | 56 (8-117) | 14 (2-29) | 0.98 | 0.732 | 0.786 | 0.00 |
| random | 2 | 88 (40-216) | 11 (5-27) | 0.93 | 0.706 | 0.777 | 0.01 |
| random | 4 | 80 (48-160) | 5 (3-10) | 0.96 | 0.753 | 0.794 | 0.03 |
| random | 8 | 160 (64-288) | 5 (2-9) | 0.92 | 0.704 | 0.779 | 0.06 |
| random | 16 | 192 (64-256) | 3 (1-4) | 0.94 | 0.740 | 0.785 | 0.12 |
| active | 1 | 94 (20-240) | 24 (5-60) | 0.90 | 0.699 | 0.764 | 0.00 |
| active | 2 | 120 (46-240) | 15 (6-30) | 0.92 | 0.725 | 0.770 | 0.32 |
| active | 4 | 96 (48-192) | 6 (3-12) | 0.95 | 0.745 | 0.787 | 0.55 |
| active | 8 | 160 (88-288) | 5 (3-9) | 0.87 | 0.720 | 0.780 | 0.64 |
| active | 16 | 128 (64-256) | 2 (1-4) | 0.91 | 0.761 | 0.793 | 0.67 |
| active_claim | 1 | 94 (20-240) | 24 (5-60) | 0.90 | 0.699 | 0.764 | 0.00 |
| active_claim | 2 | 104 (40-200) | 13 (5-25) | 0.95 | 0.728 | 0.787 | 0.00 |
| active_claim | 4 | 80 (48-144) | 5 (3-9) | 0.97 | 0.756 | 0.796 | 0.00 |
| active_claim | 8 | 128 (96-224) | 4 (3-7) | 0.93 | 0.749 | 0.786 | 0.03 |
| active_claim | 16 | 128 (64-256) | 2 (1-4) | 0.93 | 0.743 | 0.802 | 0.10 |
