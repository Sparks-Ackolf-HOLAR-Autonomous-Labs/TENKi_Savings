# Exclusion-list transfer (TENKi follow-up T2): results

Script `tenki/TENKi/experiments/30_exclusion_list_transfer.py`; full
tables `exclusion_summary.md`; figure `fig_exclusion.png`; per-row data
`exclusion_list_transfer.json`. Prediction registered before the run:
`PREDICTION.md`.

Setting: 8 receivers (eight band-gap data sources) x 7 donors x 20
matched seeds, budget 240 evaluations, 4-way queries and pairs. The donor
sends either its full family ordering, only its two narrowest-gap families
(the exclusion list: "these families did not give wide gaps"), or only its
two widest-gap families (top list). No values move between sources.

## What happened, in lab terms

The exclusion list behaves like a shared logbook of furnace runs that did
not work: it carries little information, but it is almost never wrong.

- **Every donor sends the same exclusion list to the measured-gap
  receiver: pnictides and the "other" group.** The seven computed
  databases disagree about which families have the widest gaps (MP GGA
  puts borides and halides first; SNUMAT puts nitrides first; measured
  gaps put oxides first), but all of them agree on the narrowest.
- **On measured gaps, the exclusion list is a small, safe gain; the full
  ordering is a large gamble.** With 4-way queries at fixed trust, the
  exclusion list raises early ranking quality by +0.024 tau for every
  donor (seed interval about 0.00 to +0.05). The full ordering ranges from
  -0.136 (MP GGA) to +0.118 (SNUMAT), and its average over the four
  GGA-class donors is -0.038. A group whose shortlists come from a
  GGA-level database loses ranking quality by trusting the database's
  order, but not by trusting its list of failures.
- **On computed-gap receivers, the full ordering carries more:** +0.172
  early tau versus +0.100 for the exclusion list (4-way queries), because
  computed databases agree with each other much more than with
  measurement.
- **Not querying the excluded families gives the largest final gain on
  computed receivers** (+0.067 final tau, 4-way): the budget goes to the
  families that remain open. On measured gaps it lowers the final
  Kendall tau by 0.067 only because the list carries no order between the
  two excluded families (they are placed in a fixed order, which is wrong
  for this receiver); the widest-gap families are found more often (top-3
  recovery +0.067).
- **A top list of the same size is the weakest form** (+0.047 early tau,
  computed receivers) and as variable as the full ordering on measured
  gaps: what a donor calls best is where databases disagree.

## Grading of the registered predictions

| prediction | result |
| --- | --- |
| P1: measured-gap receiver, GGA-class donors, 4-way, fixed trust: exclusion list gains more early tau than the full ordering | **Pass**: +0.024 vs -0.038 |
| P2: exclusion applied to acquisition does not lower final tau (all receivers) | **Pass**: +0.051 (4-way), +0.114 (pairs) |

Note on the figure: in the right panel the exclusion-list intervals have
zero width because all four donors send the same list, so the four rows
are identical; the seed-level 95 % interval for those rows is about
+0.00 to +0.05.
