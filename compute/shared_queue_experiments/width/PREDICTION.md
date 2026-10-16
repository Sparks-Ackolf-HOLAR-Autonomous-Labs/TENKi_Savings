# Parallel-width sweep (TENKi follow-up T1): prediction registered before the run

Registered 2026-10-04, before any production result of
`tenki/TENKi/experiments/31_parallel_width.py` existed (one single-seed
smoke test on the experimental receiver checked the code only).

Setting: 8 receivers x 30 matched seeds, k = 4, total budget 512
evaluations, widths P = 1, 2, 4, 8, 16, rules random / active /
active_claim (exp 31 docstring).

**W1.** With random acquisition, tau at equal evaluations does not depend
on P (the pooled event stream has the same distribution), so the number
of rounds needed to reach a given tau falls in proportion to 1/P.

**W2.** With uncoordinated active acquisition, the evaluations needed to
reach a given tau rise with P, because agents in one round choose the
same window from the same fit.

**W3.** Claims (active_claim) remove most of the duplication in W2: at
P >= 4 the evaluations needed by active_claim are lower than those of
uncoordinated active.
