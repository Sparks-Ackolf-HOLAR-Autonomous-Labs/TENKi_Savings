# PEGKi transfer scores into the DFT receivers

| Receiver | Donor | Delta | sigma | rho | omega | learned-trust early share | fixed-trust cost to 8 of 10 | donor pool cost / receiver pool cost |
|---|---|---|---|---|---|---|---|---|
| QE PBE | MACE-MP-0, 0 K | +0.932 | +0.913 | +0.445 | +0.427 | 0.794 | 0.053 | 0.0002 |
| QE PBE | NEP89 MD, 300 K | -0.601 | -0.426 | +0.167 | +0.342 | 0.691 | 0.968 | 0.0010 |
| QE PBE | MEAM MD, 300 K | +0.602 | +0.717 | +0.230 | +0.345 | 0.631 | 0.516 | 0.0002 |

QE PBE: Spearman(Delta, measured early share) = +0.50; top by Delta = MACE-MP-0, 0 K; top measured = MACE-MP-0, 0 K

Over 20 MD-noise draws: top-Delta pick MACE-MP-0, 0 K 100%; Spearman mean +0.50 (range +0.50 to +0.50); Delta 5-95%: MACE-MP-0, 0 K [+0.932, +0.932], NEP89 MD, 300 K [-0.657, -0.549], MEAM MD, 300 K [+0.514, +0.620]

| QE HSE06 | MACE-MP-0, 0 K | +0.817 | +0.909 | +0.458 | +0.550 | 0.659 | 0.106 | 0.0000 |
| QE HSE06 | NEP89 MD, 300 K | -0.762 | -0.405 | +0.170 | +0.527 | 0.475 | 0.921 | 0.0001 |
| QE HSE06 | MEAM MD, 300 K | +0.623 | +0.714 | +0.237 | +0.328 | 0.294 | 0.669 | 0.0000 |
| QE HSE06 | QE PBE | +0.999 | +0.908 | +0.457 | +0.366 | 0.742 | 0.084 | 0.1025 |

QE HSE06: Spearman(Delta, measured early share) = +0.80; top by Delta = QE PBE; top measured = QE PBE

Over 20 MD-noise draws: top-Delta pick QE PBE 100%; Spearman mean +0.78 (range +0.40 to +0.80); Delta 5-95%: MACE-MP-0, 0 K [+0.817, +0.817], NEP89 MD, 300 K [-0.806, -0.678], MEAM MD, 300 K [+0.529, +0.639], QE PBE [+0.999, +0.999]

