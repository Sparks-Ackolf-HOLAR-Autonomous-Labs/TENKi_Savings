# Computed semiconductor campaign: numbers

Counts: {'pool': 171, 'pbe_ok': 170, 'hse_ok': 170, 'suggested': 24, 'hse_suggested': 24}

## Checks

* pbe_vs_mp_gga: n=119, tau=+0.959, mean shift=-0.004 eV, MAE=0.041 eV
* hse_vs_expt: n=138, tau=+0.660, mean shift=-0.654 eV, MAE=0.910 eV
* pbe_vs_expt: n=138, tau=+0.601, mean shift=-1.588 eV, MAE=1.619 eV
* hse_vs_pbe: n=170, tau=+0.863, mean shift=+0.959 eV, MAE=0.959 eV
* snumat_hse_vs_hse: n=106, tau=+0.912, mean shift=-0.041 eV, MAE=0.131 eV
* mbj_vs_expt: n=112, tau=+0.587, mean shift=-0.493 eV, MAE=0.888 eV

Cost: {'pbe_core_s': 1284335.485092163, 'hse_core_s': 11658194.98633194, 'pbe_median_core_s': 1858.9794540405273, 'hse_median_core_s': 14372.303800582886}

## expt_zhuo_nohalide (pool 111)

Top 10: ['arc_MgO_mp-1265', 'arc_CaO_mp-2605', 'arc_AlN_mp-661', 'arc_SrO_mp-2472', 'arc_MgSe_mp-10760', 'arc_GeO2_mp-470', 'arc_CaS_mp-1672', 'arc_CaSe_mp-1415', 'arc_BaO_mp-1342', 'arc_SrS_mp-1087']
Gaps: [7.74, 7.0, 6.09, 5.73, 5.63, 5.56, 5.4, 4.94, 4.8, 4.78]
Suggested in top 10: []
Top 10 held by archive: {'mp_gga_expsynth': 10, 'jarvis_mbj': 10, 'snumat_hse': 9, 'jarvis_optb88vdw': 10, 'mp_gga': 10, 'snumat_gga': 9, 'expt_zhuo': 10, 'mp_r2scan': 10}

| arm | share of receiver cost to 8 of 10 (mean, 5-95%) | early recall |
|---|---|---|
| random|k4 | 0.802 (0.645-0.973) | 0.089 |
| surrogate|k4 | 0.461 (0.357-0.580) | 0.178 |
| fixed:qe_pbe|k4 | 0.252 (0.252-0.252) | 0.455 |
| learned:qe_pbe|k4 | 0.288 (0.288-0.288) | 0.399 |
| fixed:qe_hse06|k4 | 0.252 (0.252-0.252) | 0.455 |
| learned:qe_hse06|k4 | 0.288 (0.288-0.288) | 0.399 |
| fixed:mp_gga|k4 | 0.252 (0.252-0.252) | 0.469 |
| learned:mp_gga|k4 | 0.288 (0.288-0.288) | 0.399 |
| fixed:mp_r2scan|k4 | 0.252 (0.252-0.252) | 0.455 |
| learned:mp_r2scan|k4 | 0.252 (0.252-0.252) | 0.398 |
| fixed:jarvis_optb88vdw|k4 | 0.252 (0.252-0.252) | 0.441 |
| learned:jarvis_optb88vdw|k4 | 0.396 (0.396-0.396) | 0.399 |
| fixed:jarvis_mbj|k4 | 0.216 (0.216-0.216) | 0.425 |
| learned:jarvis_mbj|k4 | 0.252 (0.252-0.252) | 0.398 |
| fixed:snumat_gga|k4 | 0.288 (0.288-0.288) | 0.383 |
| learned:snumat_gga|k4 | 0.252 (0.252-0.252) | 0.368 |
| fixed:snumat_hse|k4 | 0.288 (0.288-0.288) | 0.397 |
| learned:snumat_hse|k4 | 0.324 (0.324-0.324) | 0.356 |
| pooled_fixed|k4 | 0.252 (0.252-0.252) | 0.426 |
| pooled_learned|k4 | 0.216 (0.216-0.216) | 0.396 |

PEGKi pick qe_hse06, best measured qe_pbe, Spearman +0.93
* qe_pbe: Delta +0.982 (sigma 0.739, rho 0.474, omega 0.232); early 0.399
* qe_hse06: Delta +1.191 (sigma 0.833, rho 0.456, omega 0.098); early 0.399
* mp_gga: Delta +1.005 (sigma 0.847, rho 0.377, omega 0.219); early 0.399
* mp_r2scan: Delta +0.787 (sigma 0.756, rho 0.381, omega 0.350); early 0.398
* jarvis_optb88vdw: Delta +1.063 (sigma 0.770, rho 0.482, omega 0.189); early 0.399
* jarvis_mbj: Delta +0.908 (sigma 0.709, rho 0.353, omega 0.154); early 0.398
* snumat_gga: Delta +0.721 (sigma 0.642, rho 0.477, omega 0.397); early 0.368
* snumat_hse: Delta +0.712 (sigma 0.745, rho 0.355, omega 0.387); early 0.356

Shapley pooled_fixed: qe_pbe +0.060, qe_hse06 +0.063, mp_gga +0.064, jarvis_mbj +0.053, snumat_hse +0.030; v(empty) 0.211, v(all) 0.482, best ['jarvis_mbj', 'qe_hse06'] 0.503

Shapley pooled_learned: qe_pbe +0.049, qe_hse06 +0.053, mp_gga +0.037, jarvis_mbj +0.046, snumat_hse +0.027; v(empty) 0.211, v(all) 0.423, best ['jarvis_mbj', 'qe_pbe'] 0.455

## qe_hse06_all (pool 170)

Top 10: ['arc_KCl_mp-23193', 'arc_NaCl_mp-22862', 'arc_CsCl_mp-22865', 'arc_MgO_mp-1265', 'arc_LiBr_mp-23259', 'arc_RbCl_mp-23295', 'sug_MgCaO_rocksalt_CuPt_111', 'arc_KBr_mp-23251', 'arc_CsBr_mp-22906', 'arc_AlN_mp-661']
Gaps: [6.38, 6.38, 6.36, 6.19, 6.16, 6.16, 5.84, 5.48, 5.45, 5.4]
Suggested in top 10: ['sug_MgCaO_rocksalt_CuPt_111']
Top 10 held by archive: {'mp_gga_expsynth': 9, 'jarvis_mbj': 6, 'snumat_hse': 8, 'jarvis_optb88vdw': 7, 'mp_gga': 9, 'snumat_gga': 8, 'expt_zhuo': 9, 'mp_r2scan': 9}

| arm | share of receiver cost to 8 of 10 (mean, 5-95%) | early recall |
|---|---|---|
| random|k4 | 0.757 (0.422-0.975) | 0.168 |
| surrogate|k4 | 0.199 (0.052-0.509) | 0.429 |
| fixed:qe_pbe|k4 | 0.007 (0.007-0.007) | 0.982 |
| learned:qe_pbe|k4 | 0.033 (0.033-0.033) | 0.930 |
| fixed:mp_gga|k4 | 0.006 (0.006-0.006) | 0.886 |
| learned:mp_gga|k4 | 0.011 (0.011-0.011) | 0.940 |
| fixed:mp_r2scan|k4 | 0.004 (0.004-0.004) | 0.887 |
| learned:mp_r2scan|k4 | 0.038 (0.038-0.038) | 0.894 |
| fixed:jarvis_optb88vdw|k4 | 0.327 (0.302-0.366) | 0.689 |
| learned:jarvis_optb88vdw|k4 | 0.120 (0.120-0.120) | 0.827 |
| fixed:jarvis_mbj|k4 | 0.361 (0.244-0.520) | 0.581 |
| learned:jarvis_mbj|k4 | 0.098 (0.098-0.098) | 0.772 |
| fixed:snumat_gga|k4 | 0.004 (0.004-0.004) | 0.794 |
| learned:snumat_gga|k4 | 0.035 (0.035-0.035) | 0.831 |
| fixed:snumat_hse|k4 | 0.004 (0.004-0.004) | 0.793 |
| learned:snumat_hse|k4 | 0.007 (0.007-0.007) | 0.836 |
| fixed:expt_zhuo|k4 | 0.007 (0.007-0.007) | 0.884 |
| learned:expt_zhuo|k4 | 0.010 (0.010-0.010) | 0.884 |
| pooled_fixed|k4 | 0.060 (0.060-0.060) | 0.862 |
| pooled_learned|k4 | 0.037 (0.037-0.037) | 0.916 |

PEGKi pick expt_zhuo, best measured mp_gga, Spearman -0.05
* qe_pbe: Delta +1.218 (sigma 0.923, rho 0.439, omega 0.143); early 0.930
* mp_gga: Delta +1.113 (sigma 0.914, rho 0.337, omega 0.139); early 0.940
* mp_r2scan: Delta +1.071 (sigma 0.898, rho 0.450, omega 0.277); early 0.894
* jarvis_optb88vdw: Delta +1.139 (sigma 0.926, rho 0.348, omega 0.134); early 0.827
* jarvis_mbj: Delta +1.205 (sigma 0.911, rho 0.453, omega 0.159); early 0.772
* snumat_gga: Delta +0.937 (sigma 0.871, rho 0.382, omega 0.315); early 0.831
* snumat_hse: Delta +1.113 (sigma 0.910, rho 0.500, omega 0.297); early 0.836
* expt_zhuo: Delta +1.314 (sigma 0.890, rho 0.481, omega 0.057); early 0.884

Shapley pooled_fixed: qe_pbe +0.151, mp_gga +0.100, jarvis_mbj -0.026, snumat_hse +0.053, expt_zhuo +0.101; v(empty) 0.500, v(all) 0.879, best ['qe_pbe'] 0.985

Shapley pooled_learned: qe_pbe +0.116, mp_gga +0.095, jarvis_mbj +0.067, snumat_hse +0.063, expt_zhuo +0.094; v(empty) 0.500, v(all) 0.935, best ['expt_zhuo', 'qe_pbe'] 0.968

## qe_hse06_all_combined (pool 170)

Top 10: ['arc_KCl_mp-23193', 'arc_NaCl_mp-22862', 'arc_CsCl_mp-22865', 'arc_MgO_mp-1265', 'arc_LiBr_mp-23259', 'arc_RbCl_mp-23295', 'sug_MgCaO_rocksalt_CuPt_111', 'arc_KBr_mp-23251', 'arc_CsBr_mp-22906', 'arc_AlN_mp-661']
Gaps: [6.38, 6.38, 6.36, 6.19, 6.16, 6.16, 5.84, 5.48, 5.45, 5.4]
Suggested in top 10: ['sug_MgCaO_rocksalt_CuPt_111']
Top 10 held by archive: {'mp_gga_expsynth': 9, 'jarvis_mbj': 6, 'snumat_hse': 8, 'jarvis_optb88vdw': 7, 'mp_gga': 9, 'snumat_gga': 8, 'expt_zhuo': 9, 'mp_r2scan': 9}

| arm | share of receiver cost to 8 of 10 (mean, 5-95%) | early recall |
|---|---|---|
| random|k4 | 0.757 (0.413-0.974) | 0.166 |
| surrogate|k4 | 0.197 (0.053-0.509) | 0.436 |
| fixed:qe_pbe|k4 | 0.105 (0.105-0.105) | 0.587 |
| learned:qe_pbe|k4 | 0.129 (0.129-0.129) | 0.540 |
| fixed:mp_gga|k4 | 0.006 (0.006-0.006) | 0.886 |
| learned:mp_gga|k4 | 0.012 (0.012-0.012) | 0.938 |
| fixed:mp_r2scan|k4 | 0.005 (0.005-0.005) | 0.886 |
| learned:mp_r2scan|k4 | 0.039 (0.039-0.039) | 0.893 |
| fixed:jarvis_optb88vdw|k4 | 0.327 (0.302-0.368) | 0.689 |
| learned:jarvis_optb88vdw|k4 | 0.116 (0.116-0.116) | 0.829 |
| fixed:jarvis_mbj|k4 | 0.361 (0.245-0.513) | 0.580 |
| learned:jarvis_mbj|k4 | 0.096 (0.096-0.096) | 0.776 |
| fixed:snumat_gga|k4 | 0.005 (0.005-0.005) | 0.794 |
| learned:snumat_gga|k4 | 0.036 (0.036-0.036) | 0.833 |
| fixed:snumat_hse|k4 | 0.005 (0.005-0.005) | 0.793 |
| learned:snumat_hse|k4 | 0.008 (0.008-0.008) | 0.837 |
| fixed:expt_zhuo|k4 | 0.007 (0.007-0.007) | 0.883 |
| learned:expt_zhuo|k4 | 0.011 (0.011-0.011) | 0.885 |
| pooled_fixed|k4 | 0.154 (0.154-0.154) | 0.479 |
| pooled_learned|k4 | 0.132 (0.132-0.132) | 0.527 |

PEGKi pick expt_zhuo, best measured mp_gga, Spearman -0.43
* qe_pbe: Delta +1.218 (sigma 0.923, rho 0.439, omega 0.143); early 0.540
* mp_gga: Delta +1.113 (sigma 0.914, rho 0.337, omega 0.139); early 0.938
* mp_r2scan: Delta +1.071 (sigma 0.898, rho 0.450, omega 0.277); early 0.893
* jarvis_optb88vdw: Delta +1.139 (sigma 0.926, rho 0.348, omega 0.134); early 0.829
* jarvis_mbj: Delta +1.205 (sigma 0.911, rho 0.453, omega 0.159); early 0.776
* snumat_gga: Delta +0.937 (sigma 0.871, rho 0.382, omega 0.315); early 0.833
* snumat_hse: Delta +1.113 (sigma 0.910, rho 0.500, omega 0.297); early 0.837
* expt_zhuo: Delta +1.314 (sigma 0.890, rho 0.481, omega 0.057); early 0.885

Shapley pooled_fixed: qe_pbe -0.238, mp_gga +0.099, jarvis_mbj -0.025, snumat_hse +0.053, expt_zhuo +0.100; v(empty) 0.506, v(all) 0.494, best ['expt_zhuo', 'mp_gga'] 0.888

Shapley pooled_learned: qe_pbe -0.276, mp_gga +0.094, jarvis_mbj +0.066, snumat_hse +0.063, expt_zhuo +0.092; v(empty) 0.506, v(all) 0.544, best ['mp_gga'] 0.939

## qe_hse06_nohalide (pool 142)

Top 10: ['arc_MgO_mp-1265', 'sug_MgCaO_rocksalt_CuPt_111', 'arc_AlN_mp-661', 'arc_CaO_mp-2605', 'arc_SrO_mp-2472', 'sug_MgCaO_rocksalt_CuAu_001', 'arc_NaNO2_mp-2964', 'sug_ZnBeO_wurtzite_layered_a', 'arc_BeS_mp-422', 'arc_YCuO2_mp-6972']
Gaps: [6.19, 5.84, 5.4, 5.29, 4.77, 4.74, 4.43, 4.1, 4.1, 3.99]
Suggested in top 10: ['sug_MgCaO_rocksalt_CuPt_111', 'sug_MgCaO_rocksalt_CuAu_001', 'sug_ZnBeO_wurtzite_layered_a']
Top 10 held by archive: {'mp_gga_expsynth': 7, 'jarvis_mbj': 7, 'snumat_hse': 6, 'jarvis_optb88vdw': 7, 'mp_gga': 7, 'snumat_gga': 6, 'expt_zhuo': 7, 'mp_r2scan': 7}

| arm | share of receiver cost to 8 of 10 (mean, 5-95%) | early recall |
|---|---|---|
| random|k4 | 0.727 (0.460-0.946) | 0.092 |
| surrogate|k4 | 0.272 (0.059-0.479) | 0.286 |
| fixed:qe_pbe|k4 | 0.045 (0.045-0.045) | 0.919 |
| learned:qe_pbe|k4 | 0.020 (0.020-0.020) | 0.943 |
| fixed:mp_gga|k4 | 0.288 (0.261-0.317) | 0.651 |
| learned:mp_gga|k4 | 0.051 (0.051-0.051) | 0.858 |
| fixed:mp_r2scan|k4 | 0.294 (0.259-0.354) | 0.658 |
| learned:mp_r2scan|k4 | 0.060 (0.060-0.060) | 0.854 |
| fixed:jarvis_optb88vdw|k4 | 0.296 (0.284-0.322) | 0.646 |
| learned:jarvis_optb88vdw|k4 | 0.073 (0.073-0.073) | 0.829 |
| fixed:jarvis_mbj|k4 | 0.319 (0.238-0.449) | 0.643 |
| learned:jarvis_mbj|k4 | 0.025 (0.025-0.025) | 0.913 |
| fixed:snumat_gga|k4 | 0.277 (0.224-0.365) | 0.562 |
| learned:snumat_gga|k4 | 0.081 (0.081-0.081) | 0.793 |
| fixed:snumat_hse|k4 | 0.334 (0.261-0.409) | 0.550 |
| learned:snumat_hse|k4 | 0.080 (0.080-0.080) | 0.734 |
| fixed:expt_zhuo|k4 | 0.363 (0.344-0.393) | 0.564 |
| learned:expt_zhuo|k4 | 0.042 (0.042-0.042) | 0.846 |
| pooled_fixed|k4 | 0.220 (0.220-0.220) | 0.672 |
| pooled_learned|k4 | 0.081 (0.081-0.081) | 0.800 |

PEGKi pick expt_zhuo, best measured qe_pbe, Spearman +0.57
* qe_pbe: Delta +1.202 (sigma 0.919, rho 0.425, omega 0.142); early 0.943
* mp_gga: Delta +1.093 (sigma 0.906, rho 0.324, omega 0.137); early 0.858
* mp_r2scan: Delta +1.095 (sigma 0.900, rho 0.469, omega 0.274); early 0.854
* jarvis_optb88vdw: Delta +1.114 (sigma 0.921, rho 0.329, omega 0.136); early 0.829
* jarvis_mbj: Delta +1.228 (sigma 0.892, rho 0.462, omega 0.126); early 0.913
* snumat_gga: Delta +0.826 (sigma 0.775, rho 0.396, omega 0.345); early 0.793
* snumat_hse: Delta +0.993 (sigma 0.901, rho 0.427, omega 0.334); early 0.734
* expt_zhuo: Delta +1.298 (sigma 0.859, rho 0.481, omega 0.043); early 0.846

Shapley pooled_fixed: qe_pbe +0.203, mp_gga +0.040, jarvis_mbj +0.040, snumat_hse +0.018, expt_zhuo +0.020; v(empty) 0.377, v(all) 0.697, best ['qe_pbe'] 0.925

Shapley pooled_learned: qe_pbe +0.128, mp_gga +0.093, jarvis_mbj +0.121, snumat_hse +0.026, expt_zhuo +0.088; v(empty) 0.377, v(all) 0.834, best ['jarvis_mbj', 'qe_pbe'] 0.935

## qe_hse06_nohalide_combined (pool 142)

Top 10: ['arc_MgO_mp-1265', 'sug_MgCaO_rocksalt_CuPt_111', 'arc_AlN_mp-661', 'arc_CaO_mp-2605', 'arc_SrO_mp-2472', 'sug_MgCaO_rocksalt_CuAu_001', 'arc_NaNO2_mp-2964', 'sug_ZnBeO_wurtzite_layered_a', 'arc_BeS_mp-422', 'arc_YCuO2_mp-6972']
Gaps: [6.19, 5.84, 5.4, 5.29, 4.77, 4.74, 4.43, 4.1, 4.1, 3.99]
Suggested in top 10: ['sug_MgCaO_rocksalt_CuPt_111', 'sug_MgCaO_rocksalt_CuAu_001', 'sug_ZnBeO_wurtzite_layered_a']
Top 10 held by archive: {'mp_gga_expsynth': 7, 'jarvis_mbj': 7, 'snumat_hse': 6, 'jarvis_optb88vdw': 7, 'mp_gga': 7, 'snumat_gga': 6, 'expt_zhuo': 7, 'mp_r2scan': 7}

| arm | share of receiver cost to 8 of 10 (mean, 5-95%) | early recall |
|---|---|---|
| random|k4 | 0.729 (0.467-0.946) | 0.090 |
| surrogate|k4 | 0.268 (0.058-0.474) | 0.294 |
| fixed:qe_pbe|k4 | 0.139 (0.139-0.139) | 0.531 |
| learned:qe_pbe|k4 | 0.117 (0.117-0.117) | 0.553 |
| fixed:mp_gga|k4 | 0.287 (0.261-0.316) | 0.653 |
| learned:mp_gga|k4 | 0.051 (0.051-0.051) | 0.861 |
| fixed:mp_r2scan|k4 | 0.294 (0.259-0.353) | 0.659 |
| learned:mp_r2scan|k4 | 0.057 (0.057-0.057) | 0.862 |
| fixed:jarvis_optb88vdw|k4 | 0.296 (0.284-0.322) | 0.649 |
| learned:jarvis_optb88vdw|k4 | 0.070 (0.070-0.070) | 0.835 |
| fixed:jarvis_mbj|k4 | 0.322 (0.237-0.458) | 0.643 |
| learned:jarvis_mbj|k4 | 0.025 (0.025-0.025) | 0.913 |
| fixed:snumat_gga|k4 | 0.276 (0.223-0.361) | 0.565 |
| learned:snumat_gga|k4 | 0.077 (0.077-0.077) | 0.802 |
| fixed:snumat_hse|k4 | 0.333 (0.261-0.406) | 0.553 |
| learned:snumat_hse|k4 | 0.077 (0.077-0.077) | 0.741 |
| fixed:expt_zhuo|k4 | 0.358 (0.339-0.388) | 0.568 |
| learned:expt_zhuo|k4 | 0.040 (0.040-0.040) | 0.850 |
| pooled_fixed|k4 | 0.297 (0.297-0.297) | 0.381 |
| pooled_learned|k4 | 0.172 (0.172-0.172) | 0.425 |

PEGKi pick expt_zhuo, best measured jarvis_mbj, Spearman +0.31
* qe_pbe: Delta +1.202 (sigma 0.919, rho 0.425, omega 0.142); early 0.553
* mp_gga: Delta +1.093 (sigma 0.906, rho 0.324, omega 0.137); early 0.861
* mp_r2scan: Delta +1.095 (sigma 0.900, rho 0.469, omega 0.274); early 0.862
* jarvis_optb88vdw: Delta +1.114 (sigma 0.921, rho 0.329, omega 0.136); early 0.835
* jarvis_mbj: Delta +1.228 (sigma 0.892, rho 0.462, omega 0.126); early 0.913
* snumat_gga: Delta +0.826 (sigma 0.775, rho 0.396, omega 0.345); early 0.802
* snumat_hse: Delta +0.993 (sigma 0.901, rho 0.427, omega 0.334); early 0.741
* expt_zhuo: Delta +1.298 (sigma 0.859, rho 0.481, omega 0.043); early 0.850

Shapley pooled_fixed: qe_pbe -0.155, mp_gga +0.053, jarvis_mbj +0.055, snumat_hse +0.024, expt_zhuo +0.034; v(empty) 0.384, v(all) 0.395, best ['expt_zhuo', 'jarvis_mbj', 'mp_gga'] 0.672

Shapley pooled_learned: qe_pbe -0.261, mp_gga +0.093, jarvis_mbj +0.120, snumat_hse +0.030, expt_zhuo +0.089; v(empty) 0.384, v(all) 0.455, best ['jarvis_mbj'] 0.909
