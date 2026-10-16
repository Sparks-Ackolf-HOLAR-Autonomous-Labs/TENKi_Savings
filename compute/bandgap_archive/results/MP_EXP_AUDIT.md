# Audit: the "MP experimental" band-gap channel

Data snapshot: 2026-09-25 (cache contents and live Materials Project
fetch of that date).

## Question

An internal review called the MP "experimental" channel *"a DFT proxy
wearing a K_E label"*. Some records on Materials Project do have
experimental provenance, so the audit establishes what the channel
actually contains rather than declaring everything DFT.

## Checks

`shared/data/materials_project/multifidelity.csv` carries 95,920 rows;
27,347 have a non-null `exp_band_gap`. Checks:

1. **Value identity.** All 27,347 `exp_band_gap` values equal the same
   row's GGA `band_gap` column (24,010 byte-identical, 3,337 within
   1e-6, 0 different). The values are DFT band gaps, full stop.
2. **Provenance flag.** A fresh fetch of the live MP `summary` endpoint
   (`compute/bandgap_archive/refetch_channels.py mp-provenance`, using the new
   `MP_API` secret) resolved 81,548 of the 95,920 cached material ids
   (85.0%; the remainder are deprecated/merged ids). Against the live
   `theoretical` flag, the 27,347 exp-channel rows split:
   - `theoretical = False` (experimentally synthesized, ICSD-matched
     structure): **27,314**
   - `theoretical = True`: **0**
   - no live record (id deprecated since caching): **33**
   - Live-resolvable materials overall: 27,314 False / 54,234 True.

## Result

The channel is the fallback path of
`fetch_experimental_band_gaps` in
`shared/scripts/mp_multifidelity_phase2.py` (line 218 approximately): the
**DFT GGA gap of materials whose structure is experimentally known**. MP
does carry experimental provenance, but it is provenance about the
*structure* (ICSD matching), not an experimental *band gap*. MP's API
does not serve measured gaps for these 27k materials.

## Verdict and relabelling

- The channel is scientifically legitimate but was mislabelled: it is a
  provenance-filtered theoretical channel (K_T restricted to
  experimentally synthesized materials), not an empirical channel (K_E).
- `build_inventory.py` now emits it as **`mp_gga_expsynth`** with
  `m = {method: DFT, functional: GGA/GGA+U, provenance:
  experimental_structure(theoretical=False)}`. The misleading channel id
  `mp_exp` / method `experiment_compilation` is retired.
- The phase-2 study type `exp_band_gap_empirical (K_E)` in
  `mp_multifidelity_phase2.py` is renamed accordingly (see that script's
  header note).

## The experimental channel that replaces it

`compute/bandgap_archive/refetch_channels.py expt-gap` pulls the Zhuo, Mansouri
Tehrani & Brgoch compilation of measured band gaps:

> Zhuo, Y.; Mansouri Tehrani, A.; Brgoch, J. *Predicting the Band Gaps
> of Inorganic Solids by Machine Learning.* J. Phys. Chem. Lett. 2018,
> 9, 1668-1673. doi:10.1021/acs.jpclett.8b00124
> [Verified against the Crossref record on 2026-09-25: title, authors,
> journal, volume 9, pages 1668-1673 all match.]

- 6,354 records (4,921 distinct reduced formulas; 3,896 records with
  gap > 0), cached at
  `shared/data/expt_compilations/zhuo2018_expt_gap.csv`.
- Formula-keyed only (no structure), so its channel records carry
  `spg = None` and join to other channels on formula (a stated
  limitation), consistent with how the compilation itself is defined.
- Overlap with the MP cache: 1,528 formulas (2,694 Zhuo records) --
  enough for a DFT-vs-experiment paired-transfer axis.
- Selection bias note per DESIGN.md: an empirical compilation is not
  ground truth; it is one more channel with its own m and biases.

## Further outcomes of the same fetch

- All 81,548 live-resolvable MP rows now have **spacegroup numbers**
  (`mp_provenance.csv`), unlocking the formula+spg join key DESIGN.md
  specifies for MP<->JARVIS<->SNUMAT reconciliation (previously MP rows
  had spg=None).
- The `theoretical` flag itself becomes a usable shift axis
  (hypothetical vs synthesized populations at matched chemistry).
