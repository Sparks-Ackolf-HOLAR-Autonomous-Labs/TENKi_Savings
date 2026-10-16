"""band-gap archive split pass: assign campaign roles on the frozen universe.

Implements the "Splits and leakage rules" section of DESIGN.md on the
inventory produced by build_inventory.py:

- Contexts: anion family (from formula), crystal system (from spg),
  database of origin (from channel prefix). Each record gets its context
  labels; shifts are defined between context values at protocol time.
- HELD_OUT: a stratified fraction of CANDIDATES per (channel, anion
  family) cell; a held-out candidate is held out in ALL channels (no
  cross-channel leakage of the same material). Candidate identity =
  formula (or src_id where formula is absent, i.e. SNUMAT).
- VISIBLE: a small stratified cold-start sample per anion family.
- Everything else QUERYABLE.

Deterministic under --seed. Writes the role-annotated manifest (via
Inventory.freeze) plus a split report with per-cell counts, and runs a
leakage check (no candidate appears on both sides).

Usage (from singular_obj/, stdlib only):
  python compute/bandgap_archive/make_splits.py [--held-out 0.2] [--visible 0.02]
                                       [--seed 42]
"""

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SINGULAR = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "observation_record",
    SINGULAR / "shared" / "src" / "data" / "observation_record.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["observation_record"] = _mod
_spec.loader.exec_module(_mod)
Inventory, Role = _mod.Inventory, _mod.Role

ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

ANION_FAMILIES = [  # priority order, mirroring the adapters' classify_by_anion
    ("halide", {"F", "Cl", "Br", "I"}),
    ("oxide", {"O"}),
    ("nitride", {"N"}),
    ("chalcogenide", {"S", "Se", "Te"}),
    ("pnictide", {"P", "As", "Sb", "Bi"}),
    ("carbide", {"C"}),
]


def anion_family(formula: str | None) -> str:
    if not formula:
        return "unknown"
    els = {m[0] for m in ELEM_RE.findall(formula)}
    for name, anions in ANION_FAMILIES:
        if els & anions:
            return name
    return "metal_alloy"


def crystal_system(spg: int | None) -> str:
    if spg is None:
        return "unknown"
    for hi, name in [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                     (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
                     (230, "cubic")]:
        if spg <= hi:
            return name
    return "unknown"


def database_of(channel: str) -> str:
    return channel.split("_", 1)[0]


def candidate_key(rec) -> str:
    return rec.x.get("formula") or f"srcid:{rec.x.get('src_id')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=SINGULAR / "shared" /
                    "output" / "bandgap_archive_build" / "bandgap_universe.jsonl")
    ap.add_argument("--held-out", type=float, default=0.2)
    ap.add_argument("--visible", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inv = Inventory(args.universe)
    rng = random.Random(args.seed)

    # Group candidates per (channel, anion family) cell
    cand_channels: dict[str, set[str]] = {}
    cand_family: dict[str, str] = {}
    for rec in inv.records():
        key = candidate_key(rec)
        cand_channels.setdefault(key, set()).add(rec.s)
        cand_family.setdefault(key, anion_family(rec.x.get("formula")))

    # Stratified held-out sample of candidates per family (family is the
    # primary shift axis; holding out by candidate automatically holds
    # out every channel's record for it).
    by_family: dict[str, list[str]] = {}
    for key, fam in cand_family.items():
        by_family.setdefault(fam, []).append(key)
    held, visible = set(), set()
    for fam, keys in sorted(by_family.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_h = max(1, int(len(keys) * args.held_out))
        n_v = max(1, int(len(keys) * args.visible))
        held.update(keys[:n_h])
        visible.update(keys[n_h:n_h + n_v])

    # Assign roles record-by-record
    counts: dict[str, dict[str, int]] = {}
    for rid, rec in inv._records.items():
        key = candidate_key(rec)
        if key in held:
            role = Role.HELD_OUT
        elif key in visible:
            role = Role.VISIBLE
        else:
            role = Role.QUERYABLE
        inv._roles[rid] = role.value
        cell = f"{rec.s}|{cand_family[key]}"
        c = counts.setdefault(cell, {r.value: 0 for r in Role})
        c[role.value] += 1

    # Leakage check: no candidate on both sides
    leaked = held & visible
    qkeys = {candidate_key(r) for r in inv.records(Role.QUERYABLE)}
    leaked |= held & qkeys
    if leaked:
        print(f"LEAKAGE: {len(leaked)} candidates cross the split; aborting")
        return 1

    manifest = inv.freeze(note=(
        f"Split pass (make_splits.py --held-out {args.held_out} --visible "
        f"{args.visible} --seed {args.seed}): held-out stratified by anion "
        "family at candidate level, held out across ALL channels. Context "
        "labels derivable from x (anion family, crystal system) and s "
        "(database of origin)."))

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "seed": args.seed, "held_out_frac": args.held_out,
        "visible_frac": args.visible,
        "n_candidates": len(cand_family),
        "n_held_out_candidates": len(held),
        "n_visible_candidates": len(visible),
        "leakage_check": "PASS (candidate-disjoint roles)",
        "by_role": manifest["by_role"],
        "per_cell": {k: v for k, v in sorted(counts.items())},
    }
    out = args.universe.parent / "split_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"roles: {manifest['by_role']}; candidates held out: "
          f"{len(held)}/{len(cand_family)}; report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
