"""Al-Mg-Si archive step 1: Al-Mg-Si coverage audit (update.md Section 9.1).

Counts what the channels we can already reach actually contain for the
Al-Mg-Si chemical system (all subsystems: Al, Mg, Si, Al-Mg, Al-Si,
Mg-Si, Al-Mg-Si), producing the overlap-by-(target, channel) table the
gate in SYSTEM_CHOICE_ASSESSMENT.md needs.

Offline sources (bundled caches under shared/data/):
  - MP multifidelity.csv:        band_gap (GGA/GGA+U), r2scan_band_gap, exp_band_gap
  - MP electronic_structure.csv: band_gap (GGA/GGA+U)
  - JARVIS dft_3d.csv:           optb88vdw_bandgap, mbj_bandgap, exp_bandgap
  - JARVIS dft_3d_elastic.csv:   bulk_gpa, shear_gpa, chen_niu_vickers, exp_hardness
  - AFLOW aflow_elastic.csv:     bulk_gpa
  - MP mp_elastic.csv:           bulk_gpa
  - SNUMAT snumat_composition.csv: gga_bandgap, hse_bandgap

Live sources (--live; failures are recorded, never fatal):
  - OQMD REST (formation energies per subsystem)
  - AFLOW AFLUX (entry counts per subsystem)

Usage (from singular_obj/, any Python >= 3.10, stdlib only):
  python compute/almgsi_archive/audit_almgsi_coverage.py [--live] [--out DIR]
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SINGULAR = HERE.parents[1]
DATA = SINGULAR / "shared" / "data"

SYSTEM = {"Al", "Mg", "Si"}
ELEM_RE = re.compile(r"([A-Z][a-z]?)")


def elements_of(formula: str) -> set[str]:
    return set(ELEM_RE.findall(formula or ""))


def subsystem_label(els: set[str]) -> str | None:
    if not els or not els.issubset(SYSTEM):
        return None
    return "-".join(sorted(els))


def is_num(v: str) -> bool:
    if v is None or v == "" or v.lower() in ("na", "nan", "none", "null"):
        return False
    try:
        x = float(v)
    except ValueError:
        return False
    return x == x


def audit_csv(path: Path, formula_col: str, channels: list[str],
              chemsys_col: str | None = None) -> dict:
    """Per-subsystem row counts and per-channel non-null counts + overlaps."""
    out: dict = {"file": str(path.relative_to(SINGULAR)), "total_rows": 0,
                 "subsystems": {}}
    if not path.exists():
        out["error"] = "file not found"
        return out
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            out["total_rows"] += 1
            if chemsys_col and row.get(chemsys_col):
                els = set(row[chemsys_col].split("-"))
            else:
                els = elements_of(row.get(formula_col, ""))
            label = subsystem_label(els)
            if label is None:
                continue
            sub = out["subsystems"].setdefault(label, {
                "rows": 0, "channel_counts": {c: 0 for c in channels},
                "pairwise_overlap": {}, "all_channels": 0,
                "example_formulas": []})
            sub["rows"] += 1
            present = [c for c in channels if is_num(row.get(c))]
            for c in present:
                sub["channel_counts"][c] += 1
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    key = f"{a}&{b}"
                    sub["pairwise_overlap"][key] = sub["pairwise_overlap"].get(key, 0) + 1
            if len(present) == len(channels):
                sub["all_channels"] += 1
            if len(sub["example_formulas"]) < 8:
                fx = row.get(formula_col) or row.get(chemsys_col) or ""
                if fx and fx not in sub["example_formulas"]:
                    sub["example_formulas"].append(fx)
    return out


def fetch_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "almgsi-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def live_oqmd() -> dict:
    """OQMD formation-energy record counts per Al-Mg-Si subsystem."""
    out = {}
    for comp in ["Al-Mg", "Al-Si", "Mg-Si", "Al-Mg-Si"]:
        url = (f"https://oqmd.org/oqmdapi/formationenergy?composition={comp}"
               f"&fields=name,delta_e,stability&limit=1")
        try:
            data = fetch_json(url)
            out[comp] = {"count": data.get("meta", {}).get("data_available",
                         data.get("count", "unknown"))}
        except Exception as e:
            out[comp] = {"error": f"{type(e).__name__}: {e}"}
        time.sleep(1)
    return out


def live_aflux() -> dict:
    """AFLOW AFLUX entry counts for the ternary (paging(0) returns count only)."""
    out = {}
    for label, species in [("Al-Mg", "Al,Mg"), ("Al-Si", "Al,Si"),
                           ("Mg-Si", "Mg,Si"), ("Al-Mg-Si", "Al,Mg,Si")]:
        url = (f"https://aflow.org/API/aflux/?species({species}),nspecies("
               f"{len(species.split(','))}),paging(1,1)")
        try:
            data = fetch_json(url)
            out[label] = {"sample_ok": bool(data), "note": "AFLUX reachable"}
        except Exception as e:
            out[label] = {"error": f"{type(e).__name__}: {e}"}
        time.sleep(1)
    return out


NIST_MANUAL = {
    "note": ("These need manual/curated retrieval (no bulk API); they are the "
             "empirical + MD channels of update.md Section 4 and must be "
             "inventoried into the o=(x,s,m,y,u,p) schema by hand or scraper."),
    "resources": [
        {"name": "NIST SMDDP (6061 / Al-Mg-Si diffusion, phase, mechanical)",
         "url": "https://www.nist.gov/programs-projects/asm-structural-materials-data-demonstration-project"},
        {"name": "NIST Alloy Data (thermophysical, started with Al-Mg-Si)",
         "url": "https://www.nist.gov/mml/acmd/trc/nist-alloy-data"},
        {"name": "NIST Interatomic Potentials Repository, Al-Si-Mg-Cu-Fe MEAM (Jelinek 2012)",
         "url": "https://www.ctcms.nist.gov/potentials/entry/2012--Jelinek-B-Groh-S-Horstemeyer-M-F-et-al--Al-Si-Mg-Cu-Fe/"},
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also query OQMD/AFLUX")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
              "system": sorted(SYSTEM), "sources": {}}

    report["sources"]["mp_multifidelity"] = audit_csv(
        DATA / "materials_project" / "multifidelity.csv", "formula_pretty",
        ["band_gap", "r2scan_band_gap", "exp_band_gap"], chemsys_col="chemsys")
    report["sources"]["mp_electronic"] = audit_csv(
        DATA / "materials_project" / "electronic_structure.csv", "formula_pretty",
        ["band_gap"], chemsys_col="chemsys")
    report["sources"]["jarvis_dft3d_bandgap"] = audit_csv(
        DATA / "jarvis" / "dft_3d.csv", "formula",
        ["optb88vdw_bandgap", "mbj_bandgap", "exp_bandgap"])
    report["sources"]["jarvis_dft3d_elastic"] = audit_csv(
        DATA / "jarvis" / "dft_3d_elastic.csv", "formula",
        ["bulk_gpa", "shear_gpa", "chen_niu_vickers", "exp_hardness"])
    report["sources"]["aflow_elastic"] = audit_csv(
        DATA / "jarvis" / "aflow_elastic.csv", "compound", ["bulk_gpa"])
    report["sources"]["mp_elastic"] = audit_csv(
        DATA / "jarvis" / "mp_elastic.csv", "compound", ["bulk_gpa"])
    report["sources"]["snumat"] = audit_csv(
        DATA / "jarvis" / "snumat_composition.csv", "formula",
        ["gga_bandgap", "hse_bandgap"])

    if args.live:
        report["live_oqmd"] = live_oqmd()
        report["live_aflux"] = live_aflux()
    report["nist_manual_curation"] = NIST_MANUAL
    report["mp_api_note"] = ("Full MP ternary query (all Al-Mg-Si entries incl. "
                             "energies/phase diagram) requires MP_API_KEY; caches "
                             "cover band-gap + elastic channels only.")

    (args.out / "almgsi_coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # Markdown summary
    lines = ["# Al-Mg-Si coverage audit (Al-Mg-Si archive step 1)", "",
             f"Generated {report['generated']}. Counts are records whose chemical",
             "system is a subset of {Al, Mg, Si}, per source and channel.", ""]
    for name, src in report["sources"].items():
        lines.append(f"## {name}  (`{src.get('file', '?')}`, "
                     f"{src.get('total_rows', 0)} rows total)")
        if not src.get("subsystems"):
            lines.append("")
            lines.append("No Al/Mg/Si-subsystem records. " + src.get("error", ""))
            lines.append("")
            continue
        lines.append("")
        lines.append("| Subsystem | Rows | Per-channel non-null | All-channel overlap | Examples |")
        lines.append("|---|---|---|---|---|")
        for label in sorted(src["subsystems"]):
            sub = src["subsystems"][label]
            ch = ", ".join(f"{c}={n}" for c, n in sub["channel_counts"].items())
            ex = ", ".join(sub["example_formulas"][:5])
            lines.append(f"| {label} | {sub['rows']} | {ch} | "
                         f"{sub['all_channels']} | {ex} |")
        lines.append("")
    if "live_oqmd" in report:
        lines.append("## Live queries")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({"oqmd": report["live_oqmd"],
                                 "aflux": report.get("live_aflux", {})}, indent=2))
        lines.append("```")
        lines.append("")
    lines += ["## Not yet covered (manual curation queue)", "",
              report["nist_manual_curation"]["note"], ""]
    for r in NIST_MANUAL["resources"]:
        lines.append(f"- {r['name']}: {r['url']}")
    lines += ["", "## Reading the numbers", "",
              "DFT databases hold stoichiometric compounds (elemental Al/Mg/Si,",
              "Mg2Si, sparse binaries), NOT 6xxx solid solutions or aged",
              "microstructures. Thin ternary counts here are evidence FOR the",
              "two-track gate: dense-overlap evaluation comes from band-gap archive",
              "(band gap, chemistry-wide), while Al-Mg-Si context shifts need",
              "the NIST empirical channels plus budgeted prospective DFT/MD",
              "(update.md Section 10)."]
    (args.out / "ALMGSI_COVERAGE.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    print(f"Wrote {args.out / 'almgsi_coverage.json'} and ALMGSI_COVERAGE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
