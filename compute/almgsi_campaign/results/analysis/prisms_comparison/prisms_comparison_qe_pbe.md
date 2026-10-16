# Direct comparison with PRISMS on the computed Al-Mg-Si pool

k = 4 candidates per receiver batch, 20 seeds (with MD noise draws). Early = recall area over the first 25% of receiver cost; c8 = receiver-cost fraction to find 8 of the 10 lowest.

## Receiver qe_pbe (pool 165)

Kendall tau of each ranker with the receiver: mace_relax +0.87, lammps_meam_md +0.47, gpumd_nep89_md -0.13

| arm | rankers (low to high fidelity) | early recall area [95% CI] | cost to 8 of 10 [95% CI] |
|---|---|---|---|
| prisms_3fid:cost | mace_relax, lammps_meam_md, gpumd_nep89_md | 0.588 [0.535, 0.639] | 0.171 [0.136, 0.210] |
| prisms_3fid:class | lammps_meam_md, gpumd_nep89_md, mace_relax | 0.233 [0.202, 0.267] | 0.392 [0.359, 0.428] |
| prisms_single:mace_relax | mace_relax | 0.744 [0.723, 0.763] | 0.094 [0.089, 0.100] |
| prisms_3fid:measured | gpumd_nep89_md, lammps_meam_md, mace_relax | 0.223 [0.191, 0.255] | 0.783 [0.729, 0.832] |
| prisms_2fid:meam_to_mace | lammps_meam_md, mace_relax | 0.230 [0.199, 0.264] | 0.509 [0.453, 0.566] |
| surrogate | - | 0.412 [0.359, 0.464] | 0.188 [0.171, 0.205] |
| learned:mace_relax | mace_relax | 0.784 [0.784, 0.784] | 0.082 [0.082, 0.082] |
| pooled_learned | mace_relax, lammps_meam_md, gpumd_nep89_md | 0.707 [0.690, 0.721] | 0.124 [0.120, 0.128] |
| fixed:mace_relax | mace_relax | 0.869 [0.869, 0.869] | 0.055 [0.055, 0.055] |
