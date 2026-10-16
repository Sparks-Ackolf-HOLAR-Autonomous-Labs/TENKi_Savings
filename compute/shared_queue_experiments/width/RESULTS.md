# Parallel-width sweep (TENKi follow-up T1): results

Script `tenki/TENKi/experiments/31_parallel_width.py`; summary
`width_summary.md`, figure `fig_width.png`, raw curves
`parallel_width_raw.json`. Prediction registered before the run:
`PREDICTION.md`.

Setting: P agents query the same receiver archive at once; each round
costs P x 4 evaluations and one time step; one shared event log, refitted
once per round. 8 receivers (the eight band-gap data sources), 30 matched
seeds, 512 evaluations per campaign. Threshold per receiver: 0.9 x the
final tau of a one-agent random campaign.

## What the numbers say, in lab terms

Think of P identical furnaces loaded at the same time, each with a
four-sample tray, versus one furnace run P times.

| | P = 1 | P = 4 | P = 16 |
| --- | ---: | ---: | ---: |
| random trays: evaluations to threshold (median) | 56 | 80 | 192 |
| random trays: rounds to threshold (median) | 14 | 5 | 3 |
| random trays: tau after 256 evaluations | 0.732 | 0.753 | 0.740 |
| ranked trays, no claims: duplicate trays | 0 | 55 % | 67 % |
| ranked trays with claims: duplicate trays | 0 | 0 % | 10 % |

- **At equal evaluations, width changes nothing.** With random trays the
  ranking quality after 256 evaluations is 0.70 to 0.75 at every width.
- **Wide teams reach the threshold sooner but spend more to get there.**
  Rounds fall from 14 to 3 as width goes from 1 to 16, while evaluations
  rise from 56 to 192. The extra cost comes from committing a whole round
  (64 evaluations at P = 16) before any of its results are seen, so a wide
  team overshoots the point where it could have stopped. This is the
  Agensh trade-off measured on archived band gaps: the one-furnace group
  uses about a third of the samples, the sixteen-furnace group finishes in
  about a fifth of the rounds.
- **Without claims, a ranked-tray rule makes wide teams repeat each other.**
  Agents that read the same fit pick the same "least resolved" tray: 55 %
  of trays are duplicates at P = 4 and 67 % at P = 16. Writing a claim
  before loading removes nearly all of it (0 % and 10 %) and lowers the
  evaluations to threshold at P = 4 (80 vs 96) and P = 8 (128 vs 160); at
  P = 16 both need 128.
- The ranked-tray rule is not better than random trays at P = 1 (94 vs 56
  evaluations), in line with the TENKi finding that hand-written
  acquisition rules do not accelerate random k-way querying.

## Grading of the registered predictions

| prediction | result |
| --- | --- |
| W1: with random trays, tau at equal evaluations does not depend on P; rounds fall about as 1/P | **Pass** for tau at equal evaluations (0.70 to 0.75). Rounds fall 14, 11, 5, 5, 3, close to but slower than 1/P, because a wide round overshoots. |
| W2: without claims, evaluations to threshold rise with P | **Partly.** Duplication rises with P as predicted (0, 32, 55, 64, 67 %), but evaluations to threshold do not rise monotonically (94, 120, 96, 160, 128). |
| W3: claims lower the evaluations needed at P >= 4 | **Pass at P = 4 and 8, tie at P = 16.** |

Spread is wide (interquartile bands in `fig_width.png`), so differences
of 20 to 30 evaluations between rules are within campaign-to-campaign
variation; the duplication shares and the round counts are the robust
readings.
