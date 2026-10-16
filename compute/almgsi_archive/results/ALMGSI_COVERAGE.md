# Al-Mg-Si coverage audit (Al-Mg-Si archive step 1)

Generated 2026-09-25 11:11 UTC. Counts are records whose chemical
system is a subset of {Al, Mg, Si}, per source and channel.

## mp_multifidelity  (`shared/data/materials_project/multifidelity.csv`, 95920 rows total)

| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |
|---|---|---|---|---|
| Al-Mg | 1 | band_gap=1, r2scan_band_gap=0, exp_band_gap=0 | 0 | Mg149Al |
| Mg-Si | 2 | band_gap=2, r2scan_band_gap=0, exp_band_gap=1 | 0 | Mg149Si, Mg2Si |
| Si | 19 | band_gap=19, r2scan_band_gap=0, exp_band_gap=4 | 0 | Si |

## mp_electronic  (`shared/data/materials_project/electronic_structure.csv`, 200487 rows total)

| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |
|---|---|---|---|---|
| Al | 10 | band_gap=10 | 10 | Al |
| Al-Mg | 33 | band_gap=33 | 33 | MgAl, MgAl2, MgAl3, Mg3Al, Mg2Al3 |
| Al-Mg-Si | 12 | band_gap=12 | 12 | Mg2(AlSi2)3, Mg4AlSi6, Mg4AlSi3, Mg16Al12Si, Mg(AlSi)2 |
| Al-Si | 3 | band_gap=3 | 3 | Al4Si19, Al4Si, AlSi |
| Mg | 9 | band_gap=9 | 9 | Mg |
| Mg-Si | 685 | band_gap=685 | 685 | Mg2Si3, MgSi2, MgSi, Mg6Si5, Mg9Si5 |
| Si | 43 | band_gap=43 | 43 | Si |

## jarvis_dft3d_bandgap  (`shared/data/jarvis/dft_3d.csv`, 75993 rows total)

| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |
|---|---|---|---|---|
| Al | 4 | optb88vdw_bandgap=4, mbj_bandgap=3, exp_bandgap=0 | 0 | Al |
| Al-Mg | 20 | optb88vdw_bandgap=20, mbj_bandgap=1, exp_bandgap=0 | 0 | MgAl3, MgAl, Mg3Al, Mg7Al, Mg17Al12 |
| Al-Mg-Si | 7 | optb88vdw_bandgap=7, mbj_bandgap=1, exp_bandgap=0 | 0 | MgAlSi, Mg6AlSi, MgAl2Si2, Mg4AlSi6, Mg16Al12Si |
| Al-Si | 3 | optb88vdw_bandgap=3, mbj_bandgap=1, exp_bandgap=0 | 0 | AlSi, Al4Si, Al3Si |
| Mg | 9 | optb88vdw_bandgap=9, mbj_bandgap=7, exp_bandgap=0 | 0 | Mg |
| Mg-Si | 25 | optb88vdw_bandgap=25, mbj_bandgap=6, exp_bandgap=0 | 0 | Mg9Si5, Mg5Si6, Mg2Si, Mg3Si2, MgSi2 |
| Si | 32 | optb88vdw_bandgap=32, mbj_bandgap=25, exp_bandgap=0 | 0 | Si |

## jarvis_dft3d_elastic  (`shared/data/jarvis/dft_3d_elastic.csv`, 16334 rows total)

| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |
|---|---|---|---|---|
| Al-Mg | 2 | bulk_gpa=2, shear_gpa=2, chen_niu_vickers=2, exp_hardness=0 | 0 | Mg3Al, MgAl2 |
| Al-Mg-Si | 1 | bulk_gpa=1, shear_gpa=1, chen_niu_vickers=1, exp_hardness=0 | 0 | MgAl2Si2 |
| Mg | 6 | bulk_gpa=6, shear_gpa=6, chen_niu_vickers=6, exp_hardness=0 | 0 | Mg |
| Mg-Si | 1 | bulk_gpa=1, shear_gpa=1, chen_niu_vickers=1, exp_hardness=0 | 0 | Mg2Si |
| Si | 17 | bulk_gpa=17, shear_gpa=17, chen_niu_vickers=17, exp_hardness=0 | 0 | Si |

## aflow_elastic  (`shared/data/jarvis/aflow_elastic.csv`, 5639 rows total)

| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |
|---|---|---|---|---|
| Al-Mg | 1 | bulk_gpa=1 | 1 | Al8Mg4 |
| Mg-Si | 3 | bulk_gpa=3 | 3 | Mg4Si2, Mg2Si1 |
| Si | 15 | bulk_gpa=15 | 15 | Si34, Si106, Si6, Si82, Si40 |

## mp_elastic  (`shared/data/jarvis/mp_elastic.csv`, 12888 rows total)

| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |
|---|---|---|---|---|
| Al | 2 | bulk_gpa=2 | 2 | Al |
| Al-Mg | 9 | bulk_gpa=9 | 9 | MgAl, MgAl3, Mg17Al12, Mg3Al, Mg13Al14 |
| Al-Mg-Si | 3 | bulk_gpa=3 | 3 | Mg4AlSi6, Mg(AlSi)2, MgAlSi |
| Mg | 3 | bulk_gpa=3 | 3 | Mg |
| Mg-Si | 6 | bulk_gpa=6 | 6 | Mg5Si6, Mg9Si5, Mg2Si, MgSi2 |
| Si | 17 | bulk_gpa=17 | 17 | Si |

## snumat  (`shared/data/jarvis/snumat_composition.csv`, 10481 rows total)

No Al/Mg/Si-subsystem records. 

## Live queries

```json
{
  "oqmd": {
    "Al-Mg": {
      "count": 196
    },
    "Al-Si": {
      "count": 139
    },
    "Mg-Si": {
      "count": 336
    },
    "Al-Mg-Si": {
      "count": 529
    }
  },
  "aflux": {
    "Al-Mg": {
      "sample_ok": true,
      "note": "AFLUX reachable"
    },
    "Al-Si": {
      "error": "RemoteDisconnected: Remote end closed connection without response"
    },
    "Mg-Si": {
      "error": "RemoteDisconnected: Remote end closed connection without response"
    },
    "Al-Mg-Si": {
      "error": "RemoteDisconnected: Remote end closed connection without response"
    }
  }
}
```

## Not yet covered (manual curation queue)

These need manual/curated retrieval (no bulk API); they are the empirical + MD channels of update.md Section 4 and must be inventoried into the o=(x,s,m,y,u,p) schema by hand or scraper.

- NIST SMDDP (6061 / Al-Mg-Si diffusion, phase, mechanical): https://www.nist.gov/programs-projects/asm-structural-materials-data-demonstration-project
- NIST Alloy Data (thermophysical, started with Al-Mg-Si): https://www.nist.gov/mml/acmd/trc/nist-alloy-data
- NIST Interatomic Potentials Repository, Al-Si-Mg-Cu-Fe MEAM (Jelinek 2012): https://www.ctcms.nist.gov/potentials/entry/2012--Jelinek-B-Groh-S-Horstemeyer-M-F-et-al--Al-Si-Mg-Cu-Fe/

## Reading the numbers

DFT databases hold stoichiometric compounds (elemental Al/Mg/Si,
Mg2Si, sparse binaries), NOT 6xxx solid solutions or aged
microstructures. Thin ternary counts here are evidence FOR the
two-track gate: dense-overlap evaluation comes from band-gap archive
(band gap, chemistry-wide), while Al-Mg-Si context shifts need
the NIST empirical channels plus budgeted prospective DFT/MD
(update.md Section 10).
