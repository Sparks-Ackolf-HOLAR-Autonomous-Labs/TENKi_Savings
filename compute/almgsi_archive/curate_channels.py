"""
Al-Mg-Si archive step 2 -- Al-Mg-Si channel curation and gate decision input.

The coverage audit (audit_almgsi_coverage.py, ALMGSI_COVERAGE.md) counted
records; this script CURATES the one property the live sources actually
hold at depth for this ternary -- formation energy -- into the
o=(x,s,m,y,u,p) observation schema, and measures the matched
multi-channel overlap the gate criterion asks for
(SYSTEM_CHOICE_ASSESSMENT.md: "at least one property with adequate
matched multi-channel coverage").

Channels curated (chemical system a subset of {Al, Mg, Si}):
  K_oqmd_dHf : OQMD REST, delta_e (eV/atom) + stability, PBE/GGA(+U)
  K_mp_dHf   : Materials Project summary API, formation_energy_per_atom
               + energy_above_hull, GGA/GGA+U/r2SCAN mixed corrections
  K_aflow_dHf: AFLOW AFLUX, enthalpy_formation_atom, PBE
  (any source that is down gets its error recorded, never fabricated)

Matching key: reduced composition (alphabetical element order, counts
divided by gcd). Rank agreement on matched formulas: Kendall tau between
the two channels' formation energies (more negative = more stable, so
lower_is_better -- reported per subsystem and pooled).

Outputs (compute/almgsi_archive/results/):
  tracka_channels.jsonl  : o-schema records, one per (channel, entry)
  almgsi_archive_inventory.json  : counts, overlaps, rank agreement, errors
  TRACKA_GATE.md         : gate report (the decision text lives in
                           SYSTEM_CHOICE_ASSESSMENT.md and the PEGKi draft)

Secrets: reads the Materials Project key from MP_API (or MP_API_KEY).
Never printed. ASCII only; stdlib only (urllib), retry with backoff.

Run:  python3 compute/almgsi_archive/curate_channels.py [--skip-mp] [--skip-oqmd]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUBSYSTEMS = ["Al", "Mg", "Si", "Al-Mg", "Al-Si", "Mg-Si", "Al-Mg-Si"]
TODAY = date.today().isoformat()


def fetch_json(url: str, headers: dict | None = None, tries: int = 4,
               timeout: int = 60):
    last = None
    hdrs = {"User-Agent": "singular_obj-almgsi_archive/1.0 (research curation)"}
    hdrs.update(headers or {})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 -- record, retry, backoff
            last = e
            time.sleep(2 ** attempt)
    raise last


def reduced_formula(formula: str) -> str:
    """'Mg2Si' -> 'Mg2Si1' reduced + alphabetical: canonical match key."""
    counts: dict = {}
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        if not el:
            continue
        n = float(num) if num else 1.0
        counts[el] = counts.get(el, 0.0) + n
    if not counts:
        return formula
    ints = {el: int(round(v * 12)) for el, v in counts.items()}
    g = 0
    for v in ints.values():
        g = math.gcd(g, v)
    g = g or 1
    return "".join(f"{el}{ints[el] // g}" for el in sorted(ints))


def chemsys_of(formula: str) -> str:
    els = sorted(set(re.findall(r"[A-Z][a-z]?", formula)))
    return "-".join(els)


def o_record(channel: str, formula: str, y: dict, provenance: dict) -> dict:
    return {
        "x": {"formula": formula,
              "reduced_formula": reduced_formula(formula),
              "chemsys": chemsys_of(formula)},
        "s": channel,
        "m": provenance.get("method", {}),
        "y": y,
        "u": {"formation_energy_eV_per_atom": "eV/atom",
              "stability_eV_per_atom": "eV/atom"},
        "p": {**provenance, "fetched": TODAY},
    }


def curate_oqmd(errors: dict) -> list:
    records = []
    for comp in SUBSYSTEMS:
        offset, total = 0, None
        while total is None or offset < total:
            url = ("https://oqmd.org/oqmdapi/formationenergy?"
                   f"composition={comp}&fields=name,entry_id,spacegroup,"
                   f"ntypes,delta_e,stability,band_gap&limit=200"
                   f"&offset={offset}")
            try:
                data = fetch_json(url)
            except Exception as e:  # noqa: BLE001
                errors[f"oqmd:{comp}:offset{offset}"] = (
                    f"{type(e).__name__}: {e}")
                break
            rows = data.get("data", [])
            meta = data.get("meta", {})
            total = int(meta.get("data_available", len(rows)))
            for row in rows:
                if row.get("delta_e") is None:
                    continue
                records.append(o_record(
                    "oqmd_dHf", row.get("name", ""),
                    {"formation_energy_eV_per_atom": float(row["delta_e"]),
                     "stability_eV_per_atom":
                         None if row.get("stability") is None
                         else float(row["stability"])},
                    {"source": "OQMD REST oqmdapi/formationenergy",
                     "entry_id": row.get("entry_id"),
                     "spacegroup": row.get("spacegroup"),
                     "method": {"code": "VASP", "functional": "PBE/GGA+U",
                                "database": "OQMD"}}))
            offset += 200
            if not rows:
                break
            time.sleep(1)
        print(f"[oqmd] {comp}: total_available={total}")
    return records


def curate_mp(errors: dict) -> list:
    key = os.environ.get("MP_API") or os.environ.get("MP_API_KEY")
    if not key:
        errors["mp"] = "MP_API secret not present in environment"
        return []
    records = []
    fields = ("material_id,formula_pretty,formation_energy_per_atom,"
              "energy_above_hull,theoretical,symmetry")
    for comp in SUBSYSTEMS:
        page = 1
        while True:
            url = ("https://api.materialsproject.org/materials/summary/?"
                   f"chemsys={urllib.parse.quote(comp)}&_fields={fields}"
                   f"&_limit=500&_page={page}")
            try:
                data = fetch_json(url, headers={"X-API-KEY": key})
            except Exception as e:  # noqa: BLE001
                errors[f"mp:{comp}:page{page}"] = f"{type(e).__name__}: {e}"
                break
            rows = data.get("data", [])
            for row in rows:
                if row.get("formation_energy_per_atom") is None:
                    continue
                sym = row.get("symmetry") or {}
                records.append(o_record(
                    "mp_dHf", row.get("formula_pretty", ""),
                    {"formation_energy_eV_per_atom":
                         float(row["formation_energy_per_atom"]),
                     "stability_eV_per_atom":
                         None if row.get("energy_above_hull") is None
                         else float(row["energy_above_hull"])},
                    {"source": "MP materials/summary",
                     "entry_id": row.get("material_id"),
                     "spacegroup": sym.get("number"),
                     "theoretical": row.get("theoretical"),
                     "method": {"code": "VASP",
                                "functional": "GGA/GGA+U/r2SCAN mixed",
                                "database": "Materials Project"}}))
            if len(rows) < 500:
                break
            page += 1
            time.sleep(1)
        print(f"[mp] {comp}: cumulative={len(records)}")
    return records


def curate_aflux(errors: dict) -> list:
    records = []
    for comp in SUBSYSTEMS:
        species = comp.replace("-", ",")
        n = comp.count("-") + 1
        page = 1
        while True:
            url = ("https://aflow.org/API/aflux/?"
                   f"species({species}),nspecies({n}),paging({page},100),"
                   "enthalpy_formation_atom,compound,spacegroup_relax")
            try:
                data = fetch_json(url, tries=3, timeout=60)
            except Exception as e:  # noqa: BLE001
                errors[f"aflux:{comp}:page{page}"] = (
                    f"{type(e).__name__}: {e}")
                break
            rows = list(data.values()) if isinstance(data, dict) else data
            for row in rows:
                if row.get("enthalpy_formation_atom") is None:
                    continue
                records.append(o_record(
                    "aflow_dHf", row.get("compound", ""),
                    {"formation_energy_eV_per_atom":
                         float(row["enthalpy_formation_atom"]),
                     "stability_eV_per_atom": None},
                    {"source": "AFLOW AFLUX",
                     "entry_id": row.get("auid"),
                     "spacegroup": row.get("spacegroup_relax"),
                     "method": {"code": "VASP", "functional": "PBE",
                                "database": "AFLOW"}}))
            if len(rows) < 100:
                break
            page += 1
            time.sleep(1)
        print(f"[aflux] {comp}: cumulative={len(records)}")
    return records


def kendall_tau(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-oqmd", action="store_true")
    ap.add_argument("--skip-mp", action="store_true")
    ap.add_argument("--skip-aflow", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    errors: dict = {}
    records: list = []
    if not args.skip_oqmd:
        records += curate_oqmd(errors)
    if not args.skip_mp:
        records += curate_mp(errors)
    aflux = {"curated": not args.skip_aflow}
    if not args.skip_aflow:
        records += curate_aflux(errors)

    with open(RESULTS / "tracka_channels.jsonl", "w",
              encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Inventory: per channel x subsystem counts; best (most stable) value
    # per reduced formula for overlap/rank work.
    best: dict = {}
    counts: dict = {}
    for rec in records:
        ch, sub = rec["s"], rec["x"]["chemsys"]
        rf = rec["x"]["reduced_formula"]
        counts.setdefault(ch, {}).setdefault(sub, 0)
        counts[ch][sub] += 1
        v = rec["y"]["formation_energy_eV_per_atom"]
        cur = best.setdefault(ch, {}).get(rf)
        if cur is None or v < cur:
            best[ch][rf] = v

    channels = sorted(best)
    overlap: dict = {}
    agreement: dict = {}
    for i, ca in enumerate(channels):
        for cb in channels[i + 1:]:
            shared = sorted(set(best[ca]) & set(best[cb]))
            per_sub: dict = {}
            for rf in shared:
                per_sub.setdefault(chemsys_of(rf), []).append(rf)
            overlap[f"{ca}&{cb}"] = {
                "matched_formulas": len(shared),
                "by_subsystem": {s: len(v) for s, v in sorted(
                    per_sub.items())},
            }
            if len(shared) >= 3:
                va = [best[ca][rf] for rf in shared]
                vb = [best[cb][rf] for rf in shared]
                agreement[f"{ca}&{cb}"] = {
                    "kendall_tau_all": round(kendall_tau(va, vb), 4),
                    "n": len(shared),
                }
                tern = per_sub.get("Al-Mg-Si", [])
                if len(tern) >= 3:
                    va = [best[ca][rf] for rf in tern]
                    vb = [best[cb][rf] for rf in tern]
                    agreement[f"{ca}&{cb}"]["kendall_tau_ternary"] = round(
                        kendall_tau(va, vb), 4)
                    agreement[f"{ca}&{cb}"]["n_ternary"] = len(tern)

    inventory = {
        "generated": TODAY,
        "records_total": len(records),
        "counts_by_channel_subsystem": counts,
        "unique_reduced_formulas": {c: len(best[c]) for c in channels},
        "overlap": overlap,
        "rank_agreement_dHf_lower_is_better": agreement,
        "aflux": aflux,
        "errors": errors,
    }
    with open(RESULTS / "almgsi_archive_inventory.json", "w",
              encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    lines = [
        "# Al-Mg-Si archive gate report: Al-Mg-Si curated channels",
        "",
        f"Generated {TODAY} by compute/almgsi_archive/curate_channels.py. Records",
        "are in tracka_channels.jsonl (o-schema); this file reports the",
        "matched multi-channel coverage the gate criterion asks for.",
        "",
        "## Formation-energy channels (records; unique reduced formulas)",
        "",
        "| channel | " + " | ".join(SUBSYSTEMS) + " | unique formulas |",
        "|---|" + "---|" * (len(SUBSYSTEMS) + 1),
    ]
    for ch in channels:
        row = [str(counts.get(ch, {}).get(s, 0)) for s in SUBSYSTEMS]
        lines.append(f"| {ch} | " + " | ".join(row)
                     + f" | {len(best[ch])} |")
    lines += ["", "## Matched overlap and rank agreement", ""]
    for pair, ov in overlap.items():
        lines.append(f"- **{pair}**: {ov['matched_formulas']} matched "
                     f"formulas ({ov['by_subsystem']})")
        ag = agreement.get(pair)
        if ag:
            t = f"  Kendall tau (dHf, all matched) = {ag['kendall_tau_all']}"
            if "kendall_tau_ternary" in ag:
                t += (f"; ternary-only = {ag['kendall_tau_ternary']} "
                      f"(n={ag['n_ternary']})")
            lines.append(t)
    lines += [
        "",
        "## Not curated here (unchanged from ALMGSI_COVERAGE.md)",
        "",
        "NIST SMDDP / Alloy Data / Interatomic Potentials MEAM remain a",
        "manual-curation queue (no bulk API); they are the empirical + MD",
        "channels of update.md Section 4 and belong to the budgeted",
        "prospective plan, not to retrospective curation.",
        "",
        "## Errors during curation",
        "",
        "```json",
        json.dumps(errors, indent=2),
        "```",
    ]
    with open(RESULTS / "TRACKA_GATE.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] {RESULTS} ({len(records)} records, "
          f"{len(errors)} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
