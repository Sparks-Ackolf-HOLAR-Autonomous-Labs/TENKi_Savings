# Al-Mg-Si archive gate report: Al-Mg-Si curated channels

Generated 2026-09-26 by compute/almgsi_archive/curate_channels.py. Records
are in tracka_channels.jsonl (o-schema); this file reports the
matched multi-channel coverage the gate criterion asks for.

## Formation-energy channels (records; unique reduced formulas)

| channel | Al | Mg | Si | Al-Mg | Al-Si | Mg-Si | Al-Mg-Si | unique formulas |
|---|---|---|---|---|---|---|---|---|
| aflow_dHf | 670 | 995 | 706 | 255 | 193 | 276 | 102 | 95 |
| mp_dHf | 10 | 9 | 43 | 32 | 3 | 100 | 12 | 35 |

## Matched overlap and rank agreement

- **aflow_dHf&mp_dHf**: 23 matched formulas ({'Al': 1, 'Al-Mg': 10, 'Al-Mg-Si': 3, 'Al-Si': 3, 'Mg': 1, 'Mg-Si': 4, 'Si': 1})
  Kendall tau (dHf, all matched) = 0.6561; ternary-only = 1.0 (n=3)

## Not curated here (unchanged from ALMGSI_COVERAGE.md)

NIST SMDDP / Alloy Data / Interatomic Potentials MEAM remain a
manual-curation queue (no bulk API); they are the empirical + MD
channels of update.md Section 4 and belong to the budgeted
prospective plan, not to retrospective curation.

## Errors during curation

```json
{
  "oqmd:Al:offset0": "HTTPError: HTTP Error 502: Bad Gateway",
  "oqmd:Mg:offset0": "HTTPError: HTTP Error 502: Bad Gateway",
  "oqmd:Si:offset0": "HTTPError: HTTP Error 502: Bad Gateway",
  "oqmd:Al-Mg:offset0": "HTTPError: HTTP Error 502: Bad Gateway",
  "oqmd:Al-Si:offset0": "HTTPError: HTTP Error 502: Bad Gateway",
  "oqmd:Mg-Si:offset0": "HTTPError: HTTP Error 502: Bad Gateway",
  "oqmd:Al-Mg-Si:offset0": "HTTPError: HTTP Error 502: Bad Gateway"
}
```
