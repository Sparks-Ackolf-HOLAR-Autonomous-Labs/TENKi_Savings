"""band-gap archive channel refetch: fix the caches build_inventory.py flagged.

Data fetched 2026-09-25. Four subcommands, each idempotent and
checkpointed so an interrupted run resumes instead of restarting:

  snumat        Rebuild shared/data/jarvis/snumat_composition.csv with real
                formulas. The bundled cache had formula empty for 100% of
                rows because the fetch looked for a `formula` field that the
                JARVIS snumat records do not have; the formula must be
                derived from atoms["elements"]. Also records ICSD_number.
                Needs internet (JARVIS figshare), no API key.

  mp-provenance Fetch material_id -> (spacegroup, theoretical flag, current
                API band gap, formula) for every material in
                multifidelity.csv. Gives MP rows the spg join key DESIGN.md
                wants, and the provenance audit for the mislabelled
                "experimental" channel: the bundled exp_band_gap column is
                byte-identical to the GGA band_gap column (verified 27,347/
                27,347 on 2026-09-25), i.e. it is the DFT gap of materials
                the fetch filtered with theoretical=False. Needs MP_API.

  r2scan        Fetch r2SCAN band gaps via materials.calc_types ->
                tasks(output.bandgap). Prefers r2SCAN Static over Structure
                Optimization; per material takes the median over preferred
                tasks. Fills the (previously 100% null) r2scan_band_gap
                column of multifidelity.csv in place and writes the
                per-task provenance to r2scan_gaps.csv. Needs MP_API.

  expt-gap      Download the Zhuo, Mansouri Tehrani & Brgoch (2018)
                experimental band-gap compilation (J. Phys. Chem. Lett. 9,
                1668-1673, doi:10.1021/acs.jpclett.8b00124) via the matminer
                dataset registry -- a genuinely experimental, formula-keyed
                channel to replace the DFT proxy. No API key.

Usage (from singular_obj/, inside the shared venv with --extra materials):
  python compute/bandgap_archive/refetch_channels.py snumat
  python compute/bandgap_archive/refetch_channels.py mp-provenance
  python compute/bandgap_archive/refetch_channels.py r2scan
  python compute/bandgap_archive/refetch_channels.py expt-gap
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SINGULAR = HERE.parents[1]
DATA = SINGULAR / "shared" / "data"
MULTIFID = DATA / "materials_project" / "multifidelity.csv"
SNUMAT_COMP = DATA / "jarvis" / "snumat_composition.csv"
PROVENANCE = DATA / "materials_project" / "mp_provenance.csv"
R2SCAN_GAPS = DATA / "materials_project" / "r2scan_gaps.csv"
ZHUO_DIR = DATA / "expt_compilations"
ZHUO_CSV = ZHUO_DIR / "zhuo2018_expt_gap.csv"

BATCH = 1000
TASK_BATCH = 250


def _api_key():
    key = os.environ.get("MP_API") or os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API (or MP_API_KEY) not set")
    return key


def _reduced_formula(elements):
    """Alphabetically ordered reduced formula from a per-site element list.

    pymatgen's reduced_formula (electronegativity order) matches MP's
    formula_pretty, which is what the cross-database join uses.
    """
    from pymatgen.core import Composition
    counts = {}
    for el in elements:
        counts[el] = counts.get(el, 0) + 1
    return Composition(counts).reduced_formula


def _classify_by_anion():
    """Import the single canonical classifier (has the NaN-formula fix)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "jpm_phase2", SINGULAR / "shared" / "scripts" /
        "jarvis_prediction_methods_phase2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_by_anion


def _multifid_mids():
    mids = []
    with MULTIFID.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mids.append(row["material_id"])
    return mids


# ---------------------------------------------------------------------------
# snumat
# ---------------------------------------------------------------------------

def cmd_snumat(_args):
    from jarvis.db.figshare import data as jdata
    classify = _classify_by_anion()
    raw = jdata("snumat")
    rows = []
    n_formula = 0
    for entry in raw:
        elements = (entry.get("atoms") or {}).get("elements") or []
        formula = _reduced_formula(elements) if elements else ""
        if formula:
            n_formula += 1
        rows.append({
            "snumat_id": entry.get("SNUMAT_id"),
            "spg_number": entry.get("Space_group_rlx"),
            "formula": formula,
            "composition_class": classify(formula) if formula else "unknown",
            "icsd_number": entry.get("ICSD_number"),
            "gga_bandgap": entry.get("Band_gap_GGA"),
            "hse_bandgap": entry.get("Band_gap_HSE"),
            "gga_optical_bandgap": entry.get("Band_gap_GGA_optical"),
            "hse_optical_bandgap": entry.get("Band_gap_HSE_optical"),
        })
    with SNUMAT_COMP.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"snumat: {len(rows)} rows, {n_formula} with formula "
          f"({100.0 * n_formula / len(rows):.1f}%) -> {SNUMAT_COMP}")
    classes = {}
    for r in rows:
        classes[r["composition_class"]] = classes.get(r["composition_class"], 0) + 1
    print("  composition classes:", dict(sorted(classes.items())))
    return 0


# ---------------------------------------------------------------------------
# mp-provenance
# ---------------------------------------------------------------------------

def cmd_mp_provenance(_args):
    from mp_api.client import MPRester
    mids = _multifid_mids()
    done = set()
    if PROVENANCE.exists():
        with PROVENANCE.open(encoding="utf-8", newline="") as f:
            done = {row["material_id"] for row in csv.DictReader(f)}
        print(f"resume: {len(done)} already fetched")
    todo = [m for m in mids if m not in done]
    write_header = not PROVENANCE.exists()
    n_found = 0
    with MPRester(_api_key(), mute_progress_bars=True) as mpr, \
            PROVENANCE.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["material_id", "formula_pretty", "spg_number",
                        "theoretical", "api_band_gap"])
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=["material_id", "formula_pretty", "symmetry",
                        "theoretical", "band_gap"])
            for d in docs:
                spg = d.symmetry.number if d.symmetry else None
                w.writerow([str(d.material_id), d.formula_pretty, spg,
                            d.theoretical, d.band_gap])
                n_found += 1
            f.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)} "
                  f"found={n_found}", flush=True)
            time.sleep(0.2)
    print(f"mp-provenance: wrote {n_found} new rows -> {PROVENANCE}")
    return 0


# ---------------------------------------------------------------------------
# r2scan
# ---------------------------------------------------------------------------

def cmd_r2scan(_args):
    from mp_api.client import MPRester
    mids = _multifid_mids()
    taskmap_path = DATA / "materials_project" / "r2scan_taskmap.jsonl"

    # Pass 1: material -> r2SCAN task ids (checkpointed per batch)
    seen = set()
    taskmap = {}
    if taskmap_path.exists():
        with taskmap_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seen.add(rec["material_id"])
                if rec["tasks"]:
                    taskmap[rec["material_id"]] = rec["tasks"]
        print(f"resume pass 1: {len(seen)} materials already scanned")
    todo = [m for m in mids if m not in seen]
    with MPRester(_api_key(), mute_progress_bars=True) as mpr:
        with taskmap_path.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                docs = mpr.materials.search(material_ids=batch,
                                            fields=["material_id", "calc_types"])
                got = set()
                for d in docs:
                    mid = str(d.material_id)
                    got.add(mid)
                    tasks = {str(t): str(c) for t, c in (d.calc_types or {}).items()
                             if "r2scan" in str(c).lower()}
                    f.write(json.dumps({"material_id": mid, "tasks": tasks}) + "\n")
                    if tasks:
                        taskmap[mid] = tasks
                for mid in set(batch) - got:  # deprecated/missing ids
                    f.write(json.dumps({"material_id": mid, "tasks": {}}) + "\n")
                f.flush()
                print(f"  pass1 {min(i + BATCH, len(todo))}/{len(todo)} "
                      f"materials_with_r2scan={len(taskmap)}", flush=True)
                time.sleep(0.2)

        # Pass 2: task ids -> band gaps (checkpointed per batch)
        tid_owner = {}
        for mid, tasks in taskmap.items():
            for tid, ctype in tasks.items():
                tid_owner[tid] = (mid, ctype)
        gaps_done = {}
        if R2SCAN_GAPS.exists():
            with R2SCAN_GAPS.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    gaps_done[row["task_id"]] = row
        tids = [t for t in tid_owner if t not in gaps_done]
        print(f"pass 2: {len(tid_owner)} r2SCAN tasks total, "
              f"{len(tids)} still to fetch")
        write_header = not R2SCAN_GAPS.exists()
        with R2SCAN_GAPS.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["task_id", "material_id", "calc_type", "bandgap"])
            for i in range(0, len(tids), TASK_BATCH):
                batch = tids[i:i + TASK_BATCH]
                docs = mpr.materials.tasks.search(
                    task_ids=batch, fields=["task_id", "output.bandgap"])
                got = set()
                for d in docs:
                    tid = str(d.task_id)
                    got.add(tid)
                    mid, ctype = tid_owner.get(tid, ("", ""))
                    bg = getattr(getattr(d, "output", None), "bandgap", None)
                    w.writerow([tid, mid, ctype,
                                "" if bg is None else float(bg)])
                for tid in set(batch) - got:
                    mid, ctype = tid_owner.get(tid, ("", ""))
                    w.writerow([tid, mid, ctype, ""])
                f.flush()
                print(f"  pass2 {min(i + TASK_BATCH, len(tids))}/{len(tids)}",
                      flush=True)
                time.sleep(0.2)

    # Aggregate per material: prefer Static tasks, median gap.
    per_mat = {}
    with R2SCAN_GAPS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["bandgap"] == "" or not row["material_id"]:
                continue
            per_mat.setdefault(row["material_id"], []).append(
                (row["calc_type"], float(row["bandgap"])))

    def agg(entries):
        static = sorted(v for c, v in entries if "static" in c.lower())
        pool = static or sorted(v for _, v in entries)
        return pool[len(pool) // 2] if len(pool) % 2 else \
            0.5 * (pool[len(pool) // 2 - 1] + pool[len(pool) // 2])

    agg_gaps = {mid: agg(entries) for mid, entries in per_mat.items()}
    print(f"r2scan: gaps for {len(agg_gaps)} materials")

    # Fill multifidelity.csv's r2scan_band_gap column in place.
    tmp = MULTIFID.with_suffix(".csv.tmp")
    n_filled = 0
    with MULTIFID.open(encoding="utf-8", newline="") as fin, \
            tmp.open("w", encoding="utf-8", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            g = agg_gaps.get(row["material_id"])
            if g is not None:
                row["r2scan_band_gap"] = repr(g)
                n_filled += 1
            w.writerow(row)
    tmp.replace(MULTIFID)
    print(f"r2scan: filled r2scan_band_gap for {n_filled} rows in "
          f"{MULTIFID.name}")
    return 0


# ---------------------------------------------------------------------------
# expt-gap (Zhuo et al. 2018)
# ---------------------------------------------------------------------------

MATMINER_META = ("https://raw.githubusercontent.com/hackingmaterials/matminer/"
                 "master/matminer/datasets/dataset_metadata.json")


def cmd_expt_gap(_args):
    import requests
    from pymatgen.core import Composition
    meta = requests.get(MATMINER_META, timeout=60).json()
    entry = meta["expt_gap"]
    url = entry["url"]
    print("expt_gap dataset:", entry.get("description", "")[:120])
    print("url:", url)
    ZHUO_DIR.mkdir(parents=True, exist_ok=True)
    blob = requests.get(url, timeout=300).content
    ftype = entry.get("file_type", "json.gz")
    raw_path = ZHUO_DIR / f"expt_gap_raw.{ftype}"
    raw_path.write_bytes(blob)
    import pandas as pd
    if "json" in ftype:
        df = pd.read_json(raw_path, orient="split",
                          compression="gzip" if ftype.endswith("gz") else None)
    else:
        df = pd.read_csv(raw_path)
    print("columns:", list(df.columns), "rows:", len(df))
    formula_col = next(c for c in df.columns
                       if c.lower() in ("formula", "composition"))
    gap_col = next(c for c in df.columns if "gap" in c.lower())
    rows = []
    n_bad = 0
    for idx, r in df.iterrows():
        try:
            reduced = Composition(str(r[formula_col])).reduced_formula
        except Exception:
            n_bad += 1
            continue
        rows.append({"record_id": f"zhuo-{idx}",
                     "formula_published": r[formula_col],
                     "formula": reduced,
                     "expt_gap_eV": float(r[gap_col])})
    with ZHUO_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"expt-gap: {len(rows)} rows ({n_bad} unparseable formulas) "
          f"-> {ZHUO_CSV}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snumat")
    sub.add_parser("mp-provenance")
    sub.add_parser("r2scan")
    sub.add_parser("expt-gap")
    args = ap.parse_args()
    return {"snumat": cmd_snumat, "mp-provenance": cmd_mp_provenance,
            "r2scan": cmd_r2scan, "expt-gap": cmd_expt_gap}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
