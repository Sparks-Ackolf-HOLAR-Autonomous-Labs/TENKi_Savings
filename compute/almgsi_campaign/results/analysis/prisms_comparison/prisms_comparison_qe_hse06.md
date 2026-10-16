# Direct comparison with PRISMS on the computed Al-Mg-Si pool

k = 4 candidates per receiver batch, 20 seeds (with MD noise draws). Early = recall area over the first 25% of receiver cost; c8 = receiver-cost fraction to find 8 of the 10 lowest.

## Receiver qe_hse06 (pool 165)

Kendall tau of each ranker with the receiver: mace_relax +0.80, lammps_meam_md +0.44, gpumd_nep89_md -0.13

| arm | rankers (low to high fidelity) | early recall area [95% CI] | cost to 8 of 10 [95% CI] |
|---|---|---|---|
| prisms_3fid:cost | mace_relax, lammps_meam_md, gpumd_nep89_md | 0.489 [0.452, 0.526] | 0.197 [0.171, 0.223] |
| prisms_3fid:class | lammps_meam_md, gpumd_nep89_md, mace_relax | 0.227 [0.195, 0.259] | 0.505 [0.459, 0.550] |
| prisms_single:mace_relax | mace_relax | 0.623 [0.606, 0.638] | 0.129 [0.122, 0.136] |
| prisms_3fid:measured | gpumd_nep89_md, lammps_meam_md, mace_relax | 0.223 [0.192, 0.255] | 0.718 [0.632, 0.793] |
| prisms_2fid:meam_to_mace | lammps_meam_md, mace_relax | 0.256 [0.223, 0.291] | 0.588 [0.514, 0.660] |
| surrogate | - | 0.316 [0.231, 0.396] | 0.248 [0.209, 0.294] |
| learned:mace_relax | mace_relax | 0.547 [0.547, 0.547] | 0.172 [0.172, 0.172] |
| pooled_learned | mace_relax, lammps_meam_md, gpumd_nep89_md | 0.344 [0.321, 0.364] | 0.295 [0.283, 0.307] |
| fixed:mace_relax | mace_relax | 0.715 [0.715, 0.715] | 0.107 [0.107, 0.107] |
