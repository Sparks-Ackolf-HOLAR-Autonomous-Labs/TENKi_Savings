# Exclusion-list transfer: summary

Deltas are paired against the cold campaign at the same seed (Kendall tau; top-3 = share of the receiver's three widest-gap families in the fitted top three). Early = mean over the first half of the budget checkpoints. Brackets: 95% bootstrap interval over receiver-donor pairs.

P1 (experimental receiver, GGA-class donors, k-way): exclusion list +0.024 vs full ordering -0.038 early tau delta at fixed trust: **PASS**

P2 (exclusion applied to acquisition, final tau delta over all receivers): k-way +0.051, pairs +0.114: **PASS**

## kway, all receivers

| condition | n | early tau delta | final tau delta | early top-3 delta | final top-3 delta |
| --- | ---: | --- | --- | --- | --- |
| full ordering, fixed trust | 56 | +0.149 [+0.113, +0.187] | +0.026 [+0.005, +0.048] | +0.115 [+0.073, +0.159] | +0.056 [+0.026, +0.087] |
| full ordering, learned trust | 56 | +0.128 [+0.098, +0.160] | +0.022 [+0.001, +0.043] | +0.109 [+0.074, +0.146] | +0.059 [+0.031, +0.091] |
| exclusion list, fixed trust | 56 | +0.090 [+0.082, +0.099] | +0.020 [+0.015, +0.025] | +0.019 [+0.017, +0.021] | -0.002 [-0.006, +0.001] |
| exclusion list, learned trust | 56 | +0.076 [+0.067, +0.084] | +0.020 [+0.015, +0.025] | +0.014 [+0.012, +0.016] | -0.000 [-0.004, +0.004] |
| exclusion list, not queried | 56 | +0.125 [+0.110, +0.138] | +0.051 [+0.033, +0.067] | +0.031 [+0.027, +0.036] | +0.046 [+0.034, +0.058] |
| top list, fixed trust | 56 | +0.047 [+0.021, +0.074] | +0.007 [-0.014, +0.028] | +0.037 [+0.006, +0.071] | +0.020 [-0.006, +0.051] |
| top list, learned trust | 56 | +0.043 [+0.026, +0.062] | +0.010 [-0.003, +0.023] | +0.040 [+0.022, +0.060] | +0.030 [+0.015, +0.048] |

## kway, experimental receiver, GGA-class donors

| condition | n | early tau delta | final tau delta | early top-3 delta | final top-3 delta |
| --- | ---: | --- | --- | --- | --- |
| full ordering, fixed trust | 4 | -0.038 [-0.118, +0.064] | -0.065 [-0.123, +0.002] | -0.047 [-0.140, +0.085] | -0.054 [-0.125, +0.033] |
| full ordering, learned trust | 4 | -0.015 [-0.059, +0.046] | -0.036 [-0.062, -0.000] | -0.009 [-0.052, +0.065] | -0.017 [-0.050, +0.038] |
| exclusion list, fixed trust | 4 | +0.024 [+0.024, +0.024] | -0.005 [-0.005, -0.005] | +0.010 [+0.010, +0.010] | +0.000 [+0.000, +0.000] |
| exclusion list, learned trust | 4 | +0.015 [+0.015, +0.015] | +0.005 [+0.005, +0.005] | +0.007 [+0.007, +0.007] | +0.000 [+0.000, +0.000] |
| exclusion list, not queried | 4 | +0.007 [+0.007, +0.007] | -0.067 [-0.067, -0.067] | +0.057 [+0.057, +0.057] | +0.067 [+0.067, +0.067] |
| top list, fixed trust | 4 | -0.016 [-0.150, +0.104] | -0.050 [-0.150, +0.040] | -0.047 [-0.158, +0.065] | -0.050 [-0.142, +0.042] |
| top list, learned trust | 4 | +0.012 [-0.048, +0.064] | -0.015 [-0.042, +0.010] | +0.002 [-0.043, +0.047] | +0.000 [-0.017, +0.021] |

## kway, computed receivers

| condition | n | early tau delta | final tau delta | early top-3 delta | final top-3 delta |
| --- | ---: | --- | --- | --- | --- |
| full ordering, fixed trust | 49 | +0.172 [+0.137, +0.207] | +0.037 [+0.016, +0.059] | +0.137 [+0.095, +0.179] | +0.071 [+0.042, +0.099] |
| full ordering, learned trust | 49 | +0.147 [+0.117, +0.178] | +0.029 [+0.008, +0.052] | +0.125 [+0.090, +0.161] | +0.070 [+0.039, +0.101] |
| exclusion list, fixed trust | 49 | +0.100 [+0.094, +0.106] | +0.023 [+0.018, +0.029] | +0.020 [+0.019, +0.022] | -0.002 [-0.006, +0.001] |
| exclusion list, learned trust | 49 | +0.084 [+0.078, +0.091] | +0.022 [+0.016, +0.028] | +0.015 [+0.013, +0.017] | -0.000 [-0.004, +0.004] |
| exclusion list, not queried | 49 | +0.142 [+0.132, +0.150] | +0.067 [+0.052, +0.081] | +0.028 [+0.023, +0.032] | +0.043 [+0.029, +0.056] |
| top list, fixed trust | 49 | +0.054 [+0.026, +0.082] | +0.013 [-0.009, +0.035] | +0.051 [+0.019, +0.084] | +0.033 [+0.005, +0.062] |
| top list, learned trust | 49 | +0.046 [+0.026, +0.065] | +0.013 [-0.002, +0.027] | +0.047 [+0.027, +0.067] | +0.035 [+0.018, +0.053] |

## pair, all receivers

| condition | n | early tau delta | final tau delta | early top-3 delta | final top-3 delta |
| --- | ---: | --- | --- | --- | --- |
| full ordering, fixed trust | 56 | +0.233 [+0.190, +0.279] | +0.082 [+0.052, +0.114] | +0.206 [+0.156, +0.259] | +0.072 [+0.032, +0.115] |
| full ordering, learned trust | 56 | +0.194 [+0.160, +0.229] | +0.061 [+0.035, +0.089] | +0.166 [+0.129, +0.203] | +0.071 [+0.039, +0.107] |
| exclusion list, fixed trust | 56 | +0.135 [+0.126, +0.142] | +0.070 [+0.063, +0.076] | +0.037 [+0.034, +0.041] | +0.000 [-0.004, +0.004] |
| exclusion list, learned trust | 56 | +0.106 [+0.098, +0.113] | +0.041 [+0.036, +0.047] | +0.034 [+0.031, +0.037] | -0.002 [-0.005, +0.001] |
| exclusion list, not queried | 56 | +0.174 [+0.151, +0.194] | +0.114 [+0.099, +0.128] | +0.067 [+0.060, +0.074] | +0.104 [+0.093, +0.115] |
| top list, fixed trust | 56 | +0.108 [+0.080, +0.137] | +0.016 [-0.011, +0.045] | +0.111 [+0.079, +0.146] | +0.026 [-0.007, +0.060] |
| top list, learned trust | 56 | +0.082 [+0.063, +0.103] | +0.011 [-0.002, +0.025] | +0.080 [+0.059, +0.102] | +0.019 [+0.007, +0.032] |

## pair, experimental receiver, GGA-class donors

| condition | n | early tau delta | final tau delta | early top-3 delta | final top-3 delta |
| --- | ---: | --- | --- | --- | --- |
| full ordering, fixed trust | 4 | -0.016 [-0.106, +0.091] | -0.005 [-0.052, +0.062] | -0.067 [-0.168, +0.078] | -0.088 [-0.200, +0.025] |
| full ordering, learned trust | 4 | +0.006 [-0.042, +0.080] | +0.018 [+0.000, +0.043] | -0.027 [-0.082, +0.058] | -0.017 [-0.054, +0.029] |
| exclusion list, fixed trust | 4 | +0.062 [+0.062, +0.062] | +0.038 [+0.038, +0.038] | +0.020 [+0.020, +0.020] | +0.017 [+0.017, +0.017] |
| exclusion list, learned trust | 4 | +0.033 [+0.033, +0.033] | +0.019 [+0.019, +0.019] | +0.013 [+0.013, +0.013] | +0.017 [+0.017, +0.017] |
| exclusion list, not queried | 4 | -0.032 [-0.032, -0.032] | -0.010 [-0.010, -0.010] | +0.017 [+0.017, +0.017] | +0.033 [+0.033, +0.033] |
| top list, fixed trust | 4 | +0.015 [-0.120, +0.125] | -0.015 [-0.139, +0.090] | -0.038 [-0.172, +0.095] | -0.108 [-0.242, +0.025] |
| top list, learned trust | 4 | +0.022 [-0.045, +0.084] | +0.036 [+0.012, +0.057] | -0.015 [-0.067, +0.037] | +0.008 [-0.008, +0.025] |

## pair, computed receivers

| condition | n | early tau delta | final tau delta | early top-3 delta | final top-3 delta |
| --- | ---: | --- | --- | --- | --- |
| full ordering, fixed trust | 49 | +0.265 [+0.225, +0.306] | +0.092 [+0.059, +0.124] | +0.244 [+0.200, +0.290] | +0.096 [+0.058, +0.135] |
| full ordering, learned trust | 49 | +0.218 [+0.187, +0.250] | +0.066 [+0.038, +0.095] | +0.192 [+0.162, +0.226] | +0.085 [+0.052, +0.121] |
| exclusion list, fixed trust | 49 | +0.146 [+0.142, +0.149] | +0.074 [+0.069, +0.081] | +0.040 [+0.037, +0.043] | -0.002 [-0.007, +0.002] |
| exclusion list, learned trust | 49 | +0.117 [+0.113, +0.120] | +0.044 [+0.040, +0.050] | +0.037 [+0.034, +0.040] | -0.005 [-0.008, -0.002] |
| exclusion list, not queried | 49 | +0.204 [+0.198, +0.210] | +0.132 [+0.123, +0.140] | +0.074 [+0.068, +0.080] | +0.114 [+0.105, +0.124] |
| top list, fixed trust | 49 | +0.119 [+0.090, +0.148] | +0.019 [-0.011, +0.048] | +0.136 [+0.106, +0.166] | +0.048 [+0.017, +0.080] |
| top list, learned trust | 49 | +0.089 [+0.068, +0.110] | +0.007 [-0.008, +0.023] | +0.095 [+0.074, +0.116] | +0.022 [+0.008, +0.036] |

