# Exp 27b: acquisition rules, long horizon, per seed

Archive sha256[:16] `f2158ecdc78e0229`; 8 held-out sources x 100 seeds; k = 4; budget 1920 evaluations. kway_learned uses the committed exp 27 fold weights (not retrained).

| arm | tau at 240 (mean) | tau at final (mean) | final p05-p95 |
|---|---|---|---|
| pair_random | +0.646 | +0.805 | +0.571 to +1.000 |
| kway_random | +0.714 | +0.839 | +0.643 to +1.000 |
| kway_active | +0.714 | +0.853 | +0.714 to +1.000 |
| kway_fisher | +0.720 | +0.857 | +0.643 to +1.000 |
| kway_learned | +0.768 | +0.854 | +0.643 to +1.000 |
