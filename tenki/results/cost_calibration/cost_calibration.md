# Experiment 29: cost calibration of the paired-campaign logs

Analysis of the committed exp 26 paired-campaign logs (no new campaigns were run). Inputs: 400 (receiver, seed) checkpoint files from paired_campaign_k4; early window = budgets 24..120 of 240 evaluations; conditions full + combined, arms pair + kway; bootstrap CIs over seeds.

## Unit-cost table (planning-grade ASSUMPTIONS, not measured microdata)

Costs are per evaluation, in the channel's NATIVE currency. CPU-core-hours and person/instrument-hours are deliberately NOT converted into each other or into dollars; the dollars column is a labelled illustration only (USD 0.05/CPU-core-hour, USD 75.0/lab-hour, user-adjustable). Override the whole table with --cost-json.

| channel | low | mid | high | unit | provenance |
|---|---|---|---|---|---|
| expt_zhuo | 8 | 40 | 160 | person_instrument_hours_per_evaluation | planning estimate: experimental band gap = powder synthesis + diffuse-reflectance/UV-Vis measurement + analysis per sample; assumption, not measured microdata |
| jarvis_mbj | 10 | 50 | 150 | cpu_core_hours_per_evaluation | planning estimate: TBmBJ is a cheap potential-only correction on a converged density, near-GGA cost per material; assumption, not measured microdata |
| jarvis_optb88vdw | 10 | 40 | 120 | cpu_core_hours_per_evaluation | planning estimate: GGA-level static DFT relaxation+bands at high-throughput database scale, order 10-100 CPU-core-hours per material (typical MP/JARVIS/SNUMAT workflow scale); assumption, not measured microdata (OptB88vdW is a GGA-class vdW functional) |
| mp_gga | 10 | 40 | 120 | cpu_core_hours_per_evaluation | planning estimate: GGA-level static DFT relaxation+bands at high-throughput database scale, order 10-100 CPU-core-hours per material (typical MP/JARVIS/SNUMAT workflow scale); assumption, not measured microdata |
| mp_gga_expsynth | 10 | 40 | 120 | cpu_core_hours_per_evaluation | planning estimate: GGA-level static DFT relaxation+bands at high-throughput database scale, order 10-100 CPU-core-hours per material (typical MP/JARVIS/SNUMAT workflow scale); assumption, not measured microdata (same GGA workflow, experimentally synthesizable subset) |
| mp_r2scan | 20 | 80 | 240 | cpu_core_hours_per_evaluation | planning estimate: r2SCAN meta-GGA static workflow, roughly 2-3x GGA cost per material; assumption, not measured microdata |
| snumat_gga | 10 | 40 | 120 | cpu_core_hours_per_evaluation | planning estimate: GGA-level static DFT relaxation+bands at high-throughput database scale, order 10-100 CPU-core-hours per material (typical MP/JARVIS/SNUMAT workflow scale); assumption, not measured microdata |
| snumat_hse | 300 | 2000 | 8000 | cpu_core_hours_per_evaluation | planning estimate: HSE06 hybrid-functional band structure, 30-100x GGA cost per material; assumption, not measured microdata |

## How evaluations-saved is measured

Per (donor -> receiver) cell and seed: the reuse arm's tau at the end of the early window (budget 120 evaluations) is the target; linear interpolation on the matched-seed cold curve (anchored at budget 0, tau 0) gives the budget at which cold reaches that target. If cold never reaches it, the max budget (240) is used as a conservative floor and the seed is flagged censored. evaluations_saved = cold budget - 120.

Relation to exp 26's transfer matrices: exp 26 reports the MEAN tau gain over the whole early window (T_early), which credits a head start at the very first checkpoints. This script asks a harsher planning question: at the END of the early window, how long would the cold campaign take to reach the reuse arm's tau. A cell can be positive in exp 26 and negative here (e.g. combined -> expt_zhuo): the donated ranking helps immediately, but the cold campaign overtakes it before the early window closes. Both statements are true; they answer different budget questions, and the harsher one is the right basis for pricing hours.

## Headline: expt_zhuo receiver (instrument-hours, the scarcest currency)

| donor | evals saved (mean [95% CI]) | censored | hours saved mid [low..high] | USD (illustration) | breakeven donor records (mid) | donor records the prior needed |
|---|---|---|---|---|---|---|
| __combined__ | -36.2 [-52.0, -19.4] | 4% | -1448 [-290..-5792] h | -108603 | -4.4 | 1680 |
| jarvis_mbj | -56.2 [-68.8, -41.8] | 2% | -2247 [-449..-8986] h | -168489 | -44.9 | 240 |
| jarvis_optb88vdw | -56.1 [-69.6, -41.0] | 2% | -2245 [-449..-8981] h | -168396 | -56.1 | 240 |
| mp_gga | -74.8 [-84.6, -64.4] | 0% | -2992 [-598..-11968] h | -224391 | -74.8 | 240 |
| mp_gga_expsynth | -69.3 [-80.0, -55.9] | 2% | -2771 [-554..-11084] h | -207833 | -69.3 | 240 |
| mp_r2scan | -46.0 [-61.2, -28.9] | 6% | -1841 [-368..-7366] h | -138107 | -23.0 | 240 |
| snumat_gga | +3.9 [-17.6, +26.7] | 20% | +156 [+31..+625] h | +11727 | +3.9 | 240 |
| snumat_hse | +3.9 [-17.7, +26.2] | 20% | +156 [+31..+625] h | +11727 | +0.1 | 240 |

Reading the break-even column: under the FROM-SCRATCH framing, reuse pays if the hours saved on the receiver buy more donor records (at the donor's mid unit cost) than the prior actually needs. The prior needs only enough donor records to rank the donor's material classes; the frozen universe required >= 30 records per class, so 'records needed' = n_donor_classes x 30. Under the ARCHIVED-REUSE framing the donor ranking already exists in a public database (marginal cost ~ 0) and every positive hours-saved figure is pure leverage.

## Sensitivity: cells net-positive across the (low..high) cost ranges

Archived framing: the SIGN of the net saving does not depend on the unit cost at all (cost > 0 multiplies evaluations saved), so the count is stable across the whole low..high range by construction. From-scratch framing: 'worst case' prices the receiver at its LOW unit cost and the donor at its HIGH unit cost.

| condition | arm | cells | net-positive (archived, any costing) | CI-lower > 0 | from-scratch pays (mid) | from-scratch pays (worst case) |
|---|---|---|---|---|---|---|
| combined | kway | 8 | 6 | 3 | 0 | 0 |
| combined | pair | 8 | 7 | 4 | 0 | 0 |
| full | kway | 56 | 29 | 25 | 2 | 0 |
| full | pair | 56 | 35 | 25 | 2 | 1 |

## Donor screen sizes actually present (frozen universe)

| channel | classes (>= min size) | records per class | total |
|---|---|---|---|
| expt_zhuo | 7 | boride: 31, chalcogenide: 2220, halide: 420, nitride: 71, other: 104, oxide: 551, pnictide: 495 | 3892 |
| jarvis_mbj | 8 | boride: 78, carbide: 120, chalcogenide: 1862, halide: 2459, nitride: 473, other: 442, oxide: 3032, pnictide: 584 | 9050 |
| jarvis_optb88vdw | 8 | boride: 133, carbide: 188, chalcogenide: 3826, halide: 6507, nitride: 1022, other: 668, oxide: 9623, pnictide: 824 | 22791 |
| mp_gga | 8 | boride: 313, carbide: 311, chalcogenide: 9487, halide: 21859, nitride: 2945, other: 2644, oxide: 54102, pnictide: 3035 | 94696 |
| mp_gga_expsynth | 8 | boride: 147, carbide: 177, chalcogenide: 3334, halide: 7977, nitride: 1016, other: 424, oxide: 13295, pnictide: 746 | 27116 |
| mp_r2scan | 8 | boride: 61, carbide: 76, chalcogenide: 2630, halide: 4255, nitride: 475, other: 399, oxide: 5517, pnictide: 743 | 14156 |
| snumat_gga | 8 | boride: 40, carbide: 86, chalcogenide: 2167, halide: 2767, nitride: 483, other: 227, oxide: 3737, pnictide: 565 | 10072 |
| snumat_hse | 8 | boride: 44, carbide: 93, chalcogenide: 2268, halide: 2798, nitride: 501, other: 259, oxide: 3866, pnictide: 651 | 10480 |

## Full per-cell table: full reuse, pair arm (hours in the receiver's native currency)

| donor | receiver | evals saved | censored | hours saved mid [low..high] | unit | breakeven recs (mid) | recs needed | pays from scratch (mid/worst) |
|---|---|---|---|---|---|---|---|---|
| expt_zhuo | jarvis_mbj | +22.5 | 22% | +1125 [+225..+3374] | CPU-core-h | +28.1 | 210 | False/False |
| expt_zhuo | jarvis_optb88vdw | +19.4 | 26% | +776 [+194..+2327] | CPU-core-h | +19.4 | 210 | False/False |
| expt_zhuo | mp_gga | -6.6 | 8% | -263 [-66..-789] | CPU-core-h | -6.6 | 210 | False/False |
| expt_zhuo | mp_gga_expsynth | +7.8 | 24% | +311 [+78..+934] | CPU-core-h | +7.8 | 210 | False/False |
| expt_zhuo | mp_r2scan | +47.7 | 40% | +3816 [+954..+11447] | CPU-core-h | +95.4 | 210 | False/False |
| expt_zhuo | snumat_gga | +93.2 | 72% | +3730 [+932..+11189] | CPU-core-h | +93.2 | 210 | False/False |
| expt_zhuo | snumat_hse | +88.9 | 64% | +177744 [+26662..+710976] | CPU-core-h | +4443.6 | 210 | True/False |
| jarvis_mbj | expt_zhuo | -56.2 | 2% | -2247 [-449..-8986] | lab-h | -44.9 | 240 | False/False |
| jarvis_mbj | jarvis_optb88vdw | +46.4 | 42% | +1854 [+464..+5563] | CPU-core-h | +37.1 | 240 | False/False |
| jarvis_mbj | mp_gga | +113.1 | 88% | +4525 [+1131..+13574] | CPU-core-h | +90.5 | 240 | False/False |
| jarvis_mbj | mp_gga_expsynth | +106.4 | 86% | +4258 [+1064..+12773] | CPU-core-h | +85.2 | 240 | False/False |
| jarvis_mbj | mp_r2scan | +89.2 | 78% | +7133 [+1783..+21399] | CPU-core-h | +142.7 | 240 | False/False |
| jarvis_mbj | snumat_gga | -13.1 | 10% | -524 [-131..-1572] | CPU-core-h | -10.5 | 240 | False/False |
| jarvis_mbj | snumat_hse | -26.4 | 8% | -52809 [-7921..-211236] | CPU-core-h | -1056.2 | 240 | False/False |
| jarvis_optb88vdw | expt_zhuo | -56.1 | 2% | -2245 [-449..-8981] | lab-h | -56.1 | 240 | False/False |
| jarvis_optb88vdw | jarvis_mbj | +37.9 | 32% | +1894 [+379..+5682] | CPU-core-h | +47.3 | 240 | False/False |
| jarvis_optb88vdw | mp_gga | +83.0 | 60% | +3320 [+830..+9959] | CPU-core-h | +83.0 | 240 | False/False |
| jarvis_optb88vdw | mp_gga_expsynth | +49.5 | 40% | +1979 [+495..+5936] | CPU-core-h | +49.5 | 240 | False/False |
| jarvis_optb88vdw | mp_r2scan | +68.1 | 52% | +5452 [+1363..+16355] | CPU-core-h | +136.3 | 240 | False/False |
| jarvis_optb88vdw | snumat_gga | -13.9 | 8% | -557 [-139..-1671] | CPU-core-h | -13.9 | 240 | False/False |
| jarvis_optb88vdw | snumat_hse | -20.1 | 10% | -40133 [-6020..-160532] | CPU-core-h | -1003.3 | 240 | False/False |
| mp_gga | expt_zhuo | -74.8 | 0% | -2992 [-598..-11968] | lab-h | -74.8 | 240 | False/False |
| mp_gga | jarvis_mbj | +98.5 | 74% | +4926 [+985..+14778] | CPU-core-h | +123.2 | 240 | False/False |
| mp_gga | jarvis_optb88vdw | +85.4 | 66% | +3418 [+854..+10253] | CPU-core-h | +85.4 | 240 | False/False |
| mp_gga | mp_gga_expsynth | +77.9 | 60% | +3115 [+779..+9344] | CPU-core-h | +77.9 | 240 | False/False |
| mp_gga | mp_r2scan | +57.8 | 46% | +4621 [+1155..+13862] | CPU-core-h | +115.5 | 240 | False/False |
| mp_gga | snumat_gga | -46.5 | 2% | -1862 [-465..-5585] | CPU-core-h | -46.5 | 240 | False/False |
| mp_gga | snumat_hse | -51.6 | 4% | -103277 [-15492..-413109] | CPU-core-h | -2581.9 | 240 | False/False |
| mp_gga_expsynth | expt_zhuo | -69.3 | 2% | -2771 [-554..-11084] | lab-h | -69.3 | 240 | False/False |
| mp_gga_expsynth | jarvis_mbj | +107.8 | 90% | +5392 [+1078..+16176] | CPU-core-h | +134.8 | 240 | False/False |
| mp_gga_expsynth | jarvis_optb88vdw | +64.1 | 54% | +2564 [+641..+7693] | CPU-core-h | +64.1 | 240 | False/False |
| mp_gga_expsynth | mp_gga | +93.2 | 74% | +3728 [+932..+11184] | CPU-core-h | +93.2 | 240 | False/False |
| mp_gga_expsynth | mp_r2scan | +95.4 | 82% | +7630 [+1907..+22890] | CPU-core-h | +190.7 | 240 | False/False |
| mp_gga_expsynth | snumat_gga | -15.3 | 8% | -610 [-153..-1830] | CPU-core-h | -15.3 | 240 | False/False |
| mp_gga_expsynth | snumat_hse | -23.3 | 8% | -46646 [-6997..-186585] | CPU-core-h | -1166.2 | 240 | False/False |
| mp_r2scan | expt_zhuo | -46.0 | 6% | -1841 [-368..-7366] | lab-h | -23.0 | 240 | False/False |
| mp_r2scan | jarvis_mbj | +90.6 | 70% | +4528 [+906..+13584] | CPU-core-h | +56.6 | 240 | False/False |
| mp_r2scan | jarvis_optb88vdw | +82.0 | 64% | +3279 [+820..+9838] | CPU-core-h | +41.0 | 240 | False/False |
| mp_r2scan | mp_gga | +70.2 | 50% | +2807 [+702..+8421] | CPU-core-h | +35.1 | 240 | False/False |
| mp_r2scan | mp_gga_expsynth | +98.4 | 82% | +3936 [+984..+11808] | CPU-core-h | +49.2 | 240 | False/False |
| mp_r2scan | snumat_gga | +8.3 | 20% | +331 [+83..+993] | CPU-core-h | +4.1 | 240 | False/False |
| mp_r2scan | snumat_hse | -2.0 | 10% | -3955 [-593..-15821] | CPU-core-h | -49.4 | 240 | False/False |
| snumat_gga | expt_zhuo | +3.9 | 20% | +156 [+31..+625] | lab-h | +3.9 | 240 | False/False |
| snumat_gga | jarvis_mbj | -34.3 | 4% | -1714 [-343..-5141] | CPU-core-h | -42.8 | 240 | False/False |
| snumat_gga | jarvis_optb88vdw | +13.5 | 20% | +541 [+135..+1623] | CPU-core-h | +13.5 | 240 | False/False |
| snumat_gga | mp_gga | -46.3 | 0% | -1854 [-463..-5561] | CPU-core-h | -46.3 | 240 | False/False |
| snumat_gga | mp_gga_expsynth | -34.9 | 6% | -1396 [-349..-4189] | CPU-core-h | -34.9 | 240 | False/False |
| snumat_gga | mp_r2scan | +9.9 | 16% | +789 [+197..+2367] | CPU-core-h | +19.7 | 240 | False/False |
| snumat_gga | snumat_hse | +112.8 | 86% | +225600 [+33840..+902400] | CPU-core-h | +5640.0 | 240 | True/True |
| snumat_hse | expt_zhuo | +3.9 | 20% | +156 [+31..+625] | lab-h | +0.1 | 240 | False/False |
| snumat_hse | jarvis_mbj | -34.3 | 4% | -1714 [-343..-5141] | CPU-core-h | -0.9 | 240 | False/False |
| snumat_hse | jarvis_optb88vdw | +13.5 | 20% | +541 [+135..+1623] | CPU-core-h | +0.3 | 240 | False/False |
| snumat_hse | mp_gga | -46.3 | 0% | -1854 [-463..-5561] | CPU-core-h | -0.9 | 240 | False/False |
| snumat_hse | mp_gga_expsynth | -34.9 | 6% | -1396 [-349..-4189] | CPU-core-h | -0.7 | 240 | False/False |
| snumat_hse | mp_r2scan | +9.9 | 16% | +789 [+197..+2367] | CPU-core-h | +0.4 | 240 | False/False |
| snumat_hse | snumat_gga | +114.2 | 96% | +4569 [+1142..+13707] | CPU-core-h | +2.3 | 240 | False/False |

Combined-prior cells, the k-way arm, bootstrap CIs and the USD illustration for every cell: cost_calibration.json. All evaluations-saved figures are recomputed from the committed per-seed campaign checkpoints; the only assumptions are the unit costs, which are labelled as such above.
