# Experiment 28b: formation-energy transfer diagnosis

Frozen exp 28 policy weights; k = 4; budget 240; exp 28 run_arm.

| source | seeds | learned tau | random tau | learned minus random (90% CI) |
|---|---|---|---|---|
| aflow_dHf | exp 28 seeds (20) | +0.683 | +0.790 | -0.107 [-0.203, -0.013] |
| aflow_dHf | fresh seeds (200) | +0.783 | +0.774 | +0.009 [-0.026, +0.046] |
| mp_dHf | exp 28 seeds (20) | +0.950 | +0.933 | +0.017 [-0.033, +0.067] |
| mp_dHf | fresh seeds (200) | +0.940 | +0.935 | +0.005 [-0.017, +0.027] |

| source | reference order (median) | fit order, 10,000 random events | ceiling tau |
|---|---|---|---|
| aflow_dHf | Al-Mg, elemental, Al-Mg-Si, Mg-Si, Al-Si | Al-Mg, Al-Mg-Si, elemental, Mg-Si, Al-Si | +0.80 |
| mp_dHf | Al-Mg, Al-Mg-Si, Mg-Si, elemental | Al-Mg, Al-Mg-Si, Mg-Si, elemental | +1.00 |

Class medians and quartiles (eV/atom):

- aflow_dHf elemental: n = 2371, median +0.032, quartiles +0.000 to +0.362
- aflow_dHf Al-Mg: n = 255, median +0.023, quartiles +0.005 to +0.067
- aflow_dHf Al-Si: n = 193, median +0.246, quartiles +0.164 to +0.341
- aflow_dHf Mg-Si: n = 276, median +0.133, quartiles +0.054 to +0.281
- aflow_dHf Al-Mg-Si: n = 102, median +0.118, quartiles +0.086 to +0.178
- mp_dHf elemental: n = 62, median +0.172, quartiles +0.097 to +0.371
- mp_dHf Al-Mg: n = 32, median +0.040, quartiles +0.013 to +0.073
- mp_dHf Mg-Si: n = 100, median +0.142, quartiles +0.079 to +0.182
- mp_dHf Al-Mg-Si: n = 12, median +0.060, quartiles +0.014 to +0.114
