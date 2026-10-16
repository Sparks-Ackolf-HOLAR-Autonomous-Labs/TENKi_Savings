"""band-gap archive builder: bundled band-gap caches -> o=(x,s,m,y,u,p) inventory.

Implements the inventory builder of compute/bandgap_archive/DESIGN.md (caches ->
ObservationRecord JSONL plus manifest). Reads the caches
under shared/data/, emits one ObservationRecord per (material, channel)
with a non-null band gap, and freezes a versioned JSONL inventory with
manifest hash. Roles default to QUERYABLE; the split step (held-out
cells per shift context) is a separate, later pass so the freeze itself
stays deterministic and auditable.

Join-key note (refines DESIGN.md): JARVIS and SNUMAT rows carry
spacegroup numbers; the MP caches do not. x therefore carries
formula (reduced, as published), spg (int or None), and source-native id
where available. Cross-database reconciliation sets should be built on
formula-unique materials per database until MP spacegroups are fetched
(needs MP_API_KEY); the manifest records this limitation.

Usage (from singular_obj/, stdlib only, ~1 min):
  python compute/bandgap_archive/build_inventory.py [--out shared/output/bandgap_archive_build]

The JSONL itself (tens of MB) is regenerable and NOT meant for git; the
manifest and the printed stats are.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SINGULAR = HERE.parents[1]
DATA = SINGULAR / "shared" / "data"

# Load observation_record directly by path: importing src.data would pull
# the package __init__ (numpy etc.), and this builder is stdlib-only.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "observation_record",
    SINGULAR / "shared" / "src" / "data" / "observation_record.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["observation_record"] = _mod  # dataclasses resolve via sys.modules
_spec.loader.exec_module(_mod)
Inventory, ObservationRecord = _mod.Inventory, _mod.ObservationRecord

RETRIEVED = "2026-09-25"  # date the bundled caches were audited


def is_num(v):
    if v in (None, "", "na", "nan", "None", "null"):
        return False
    try:
        return float(v) == float(v)
    except ValueError:
        return False


def load_mp_provenance():
    """material_id -> (spg, theoretical) from the live-MP fetch, if present.

    Produced by refetch_channels.py mp-provenance; gives MP rows the
    formula+spg join key and the theoretical flag. Optional: the builder
    still runs (with spg=None on MP rows) when the cache is absent.
    """
    path = DATA / "materials_project" / "mp_provenance.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            spg = row.get("spg_number")
            out[row["material_id"]] = (
                int(float(spg)) if spg and is_num(spg) else None,
                row.get("theoretical") == "True",
            )
    return out


def add_rows(inv, path, spec, stats, spg_lookup=None):
    """spec: (formula_col, spg_col, id_col, [(channel, value_col, m, p_db)])"""
    formula_col, spg_col, id_col, channels = spec
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            formula = (row.get(formula_col) or "").strip() or None
            spg = row.get(spg_col) if spg_col else None
            spg = int(float(spg)) if spg and is_num(spg) else None
            if spg is None and spg_lookup:
                rid0 = (row.get(id_col) or "").strip() if id_col else ""
                spg = spg_lookup.get(rid0, (None, None))[0]
            rid_src = (row.get(id_col) or "").strip() if id_col else ""
            # SNUMAT mostly lacks formulas: keep those rows keyed on the
            # source-native id (channel pairs within one database still
            # need them); drop rows with neither formula nor id.
            if not formula and not rid_src:
                continue
            for channel, col, m, p_db in channels:
                v = row.get(col)
                if not is_num(v):
                    continue
                rec = ObservationRecord(
                    x={"formula": formula, "spg": spg, "src_id": rid_src or None},
                    s=channel,
                    m=m,
                    y={"bandgap_eV": float(v)},
                    p={"db": p_db, "record_id": rid_src or None,
                       "cache_file": str(path.relative_to(SINGULAR)),
                       "retrieved": RETRIEVED},
                )
                try:
                    inv.add(rec)
                    stats[channel] = stats.get(channel, 0) + 1
                except ValueError:
                    stats[f"{channel}__dup"] = stats.get(f"{channel}__dup", 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=SINGULAR / "shared" / "output" / "bandgap_archive_build")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Deterministic rebuild: never append onto a previous freeze.
    for stale in ("bandgap_universe.jsonl", "bandgap_universe.manifest.json"):
        (args.out / stale).unlink(missing_ok=True)
    inv = Inventory(args.out / "bandgap_universe.jsonl")
    stats = {}

    dft = {"method": "DFT", "code": "VASP"}
    mp_prov = load_mp_provenance()
    # NOTE the relabelling (2026-09-25 audit, compute/bandgap_archive/results/
    # MP_EXP_AUDIT.md): the cache's exp_band_gap column is the DFT GGA gap
    # of theoretical=False (experimentally synthesized) materials -- all
    # 27,347 values equal the GGA column, and 27,314/27,347 map exactly to
    # the live theoretical=False set. It is a provenance-filtered K_T
    # channel, NOT an experimental compilation; the old channel id mp_exp
    # with m.method=experiment_compilation is retired. The genuinely
    # experimental channel is expt_zhuo below.
    add_rows(inv, DATA / "materials_project" / "multifidelity.csv",
             ("formula_pretty", None, "material_id", [
                 ("mp_gga", "band_gap",
                  {**dft, "functional": "GGA/GGA+U"}, "Materials Project"),
                 ("mp_r2scan", "r2scan_band_gap",
                  {**dft, "functional": "r2SCAN",
                   "aggregation": "median over r2SCAN tasks, Static preferred"},
                  "Materials Project"),
                 ("mp_gga_expsynth", "exp_band_gap",
                  {**dft, "functional": "GGA/GGA+U",
                   "provenance": "experimental_structure(theoretical=False)"},
                  "Materials Project"),
             ]), stats, spg_lookup=mp_prov)
    zhuo = DATA / "expt_compilations" / "zhuo2018_expt_gap.csv"
    if zhuo.exists():
        add_rows(inv, zhuo,
                 ("formula", None, "record_id", [
                     ("expt_zhuo", "expt_gap_eV",
                      {"method": "experiment_compilation",
                       "source": "Zhuo2018_JPCL",
                       "doi": "10.1021/acs.jpclett.8b00124"},
                      "Zhuo et al. 2018 compilation"),
                 ]), stats)
    add_rows(inv, DATA / "jarvis" / "dft_3d.csv",
             ("formula", "spg_number", "jid", [
                 ("jarvis_optb88vdw", "optb88vdw_bandgap",
                  {**dft, "functional": "OptB88vdW"}, "JARVIS-DFT"),
                 ("jarvis_mbj", "mbj_bandgap",
                  {**dft, "functional": "TBmBJ"}, "JARVIS-DFT"),
                 ("jarvis_exp", "exp_bandgap",
                  {"method": "experiment_compilation"}, "JARVIS-DFT"),
             ]), stats)
    add_rows(inv, DATA / "jarvis" / "snumat_composition.csv",
             ("formula", "spg_number", "snumat_id", [
                 ("snumat_gga", "gga_bandgap",
                  {**dft, "functional": "GGA"}, "SNUMAT"),
                 ("snumat_hse", "hse_bandgap",
                  {**dft, "functional": "HSE06"}, "SNUMAT"),
             ]), stats)

    manifest = inv.freeze(note=(
        "band-gap archive band-gap query universe, built from bundled caches by "
        "compute/bandgap_archive/build_inventory.py. MP rows have spg=None (caches "
        "lack spacegroups; full MP fetch needs MP_API_KEY). KNOWN-EMPTY in "
        "caches: multifidelity.csv r2scan_band_gap and dft_3d.csv "
        "exp_bandgap are 100% null, so mp_r2scan / jarvis_exp channels "
        "need fresh fetches. SNUMAT rows are keyed on snumat_id (formula "
        "field empty). Roles all QUERYABLE until the separate split pass. "
        "Duplicate (x,s,m,record_id) repeats were skipped; counts in "
        "build_stats."))
    print("per-channel records:", {k: v for k, v in sorted(stats.items())})
    print("total:", manifest["n_records"], "sha256:", manifest["sha256"][:16])
    (args.out / "build_stats.json").write_text(
        __import__("json").dumps({"stats": stats,
                                  "manifest_sha256": manifest["sha256"],
                                  "built": time.strftime("%Y-%m-%d %H:%M UTC",
                                                         time.gmtime())},
                                 indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
