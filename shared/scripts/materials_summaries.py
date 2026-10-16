"""
materials_summaries.py - make omega live on materials channels.

Phase 3 (src/evaluation/phase3/knowledge_spaces.py) reads per-policy error
statistics from a summaries.json file next to trained_ratings.json:

    [{"policy_type": <name>,
      "overall_stats": {"best_color_distance_mean": <float>,
                        "best_color_distance_std":  <float>}}, ...]

The colour pipeline writes it (best_color_distance = per-policy MAE), but no
materials phase-2 writer ever did, so on every materials channel
policy_maes was empty and omega (transfer overhead) was STRUCTURALLY ZERO:
Delta silently degenerated to sigma + rho. See writings/OMEGA_DECISION.md
for the decision record behind this module.

Two honest MAE-equivalents, chosen per data situation:

  1. write_mae_summaries(out_dir, maes)
       For channels that HAVE real per-policy error numbers
       (jarvis_leaderboard: published MAE per method on a fixed test set).
       Passed through unchanged.

  2. write_dispersion_summaries(out_dir, class_values)
       For class-as-policy channels where each policy is a material class
       holding a list of archived property values. MAE-equivalent =
       mean absolute deviation of the class's values around the class
       median: exactly the expected error of the class's best constant
       prediction under L1 loss. It is a real, per-channel, per-class error
       (how wrong the class label is as a predictor of the property in THIS
       channel), so its source->target relative-standing degradation is a
       meaningful omega. No synthetic numbers are involved.

omega normalises each policy's MAE by the space mean
(compute_transfer_overhead), so raw channel units (eV, GPa) are fine as
long as ONE convention is used per database -- both writers record which
semantics produced the file in a "mae_semantics" field (ignored by the
loader, kept for provenance).

Classes with fewer than MIN_CLASS_N values or zero dispersion are omitted
(omega treats missing/zero policies as no-data rather than perfect).

ASCII only; deterministic; stdlib + numpy only.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

MIN_CLASS_N = 2


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def dispersion_entries(class_values: Dict[str, Sequence[float]]) -> List[dict]:
    """Per-class L1 error of the best constant predictor (median).

    Returns summaries.json entries for classes with >= MIN_CLASS_N values
    and strictly positive dispersion.
    """
    entries = []
    for cn in sorted(class_values):
        vals = np.asarray(list(class_values[cn]), dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < MIN_CLASS_N:
            continue
        med = float(np.median(vals))
        abs_dev = np.abs(vals - med)
        mae = float(np.mean(abs_dev))
        if mae <= 0.0:
            continue
        entries.append({
            "policy_type": cn,
            "overall_stats": {
                "best_color_distance_mean": mae,
                "best_color_distance_std": float(np.std(abs_dev)),
            },
            "n_records": int(len(vals)),
            "class_median": med,
        })
    return entries


def write_dispersion_summaries(
    out_dir,
    class_values: Dict[str, Sequence[float]],
    unit: str = "",
    source: str = "",
) -> int:
    """Write summaries.json with dispersion-based MAE-equivalents.

    Returns the number of policy entries written.
    """
    entries = dispersion_entries(class_values)
    payload = list(entries)
    for e in payload:
        e["mae_semantics"] = "class_dispersion_l1_median"
        if unit:
            e["unit"] = unit
        if source:
            e["source"] = source
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "summaries.json", payload)
    print(f"  Saved: {out_dir / 'summaries.json'} "
          f"({len(payload)} policies, dispersion MAE-equivalent)")
    return len(payload)


def write_mae_summaries(
    out_dir,
    maes: Dict[str, float],
    mae_stds: Optional[Dict[str, float]] = None,
    unit: str = "",
    source: str = "",
) -> int:
    """Write summaries.json from REAL per-policy MAEs (passed through).

    mae_stds is optional; when absent the std key is omitted entirely
    (the Phase 3 loader tolerates missing stds).
    """
    entries = []
    for pn in sorted(maes):
        mae = float(maes[pn])
        if not np.isfinite(mae) or mae <= 0.0:
            continue
        stats = {"best_color_distance_mean": mae}
        if mae_stds and pn in mae_stds and np.isfinite(mae_stds[pn]):
            stats["best_color_distance_std"] = float(mae_stds[pn])
        entry = {
            "policy_type": pn,
            "overall_stats": stats,
            "mae_semantics": "published_test_set_mae",
        }
        if unit:
            entry["unit"] = unit
        if source:
            entry["source"] = source
        entries.append(entry)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "summaries.json", entries)
    print(f"  Saved: {out_dir / 'summaries.json'} "
          f"({len(entries)} policies, published MAEs)")
    return len(entries)


def _self_test() -> int:
    import tempfile
    failures = 0

    # 1. Dispersion semantics: MAE-equivalent is mean |v - median|.
    vals = {"a": [1.0, 2.0, 3.0], "b": [10.0, 10.0, 10.0, 22.0],
            "tiny": [5.0], "flat": [4.0, 4.0]}
    entries = {e["policy_type"]: e for e in dispersion_entries(vals)}
    ok = set(entries) == {"a", "b"}
    exp_a = np.mean([1.0, 0.0, 1.0])
    ok = ok and abs(entries["a"]["overall_stats"]["best_color_distance_mean"] - exp_a) < 1e-12
    exp_b = np.mean([0.0, 0.0, 0.0, 12.0])
    ok = ok and abs(entries["b"]["overall_stats"]["best_color_distance_mean"] - exp_b) < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] dispersion semantics + filtering")
    failures += 0 if ok else 1

    # 2. Round-trip through the Phase 3 loader schema.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_dispersion_summaries(td, vals, unit="eV", source="self_test")
        with open(td / "summaries.json", encoding="utf-8") as f:
            payload = json.load(f)
        # Mirror knowledge_spaces.py lines 184-197 exactly:
        maes, mae_stds = {}, {}
        for entry in payload:
            pt = entry.get("policy_type")
            stats = entry.get("overall_stats", {})
            mae = stats.get("best_color_distance_mean")
            mae_std = stats.get("best_color_distance_std")
            if pt and mae is not None:
                maes[pt] = float(mae)
            if pt and mae_std is not None:
                mae_stds[pt] = float(mae_std)
        ok = set(maes) == {"a", "b"} and all(v > 0 for v in maes.values())
        ok = ok and set(mae_stds) == {"a", "b"}
        print(f"  [{'PASS' if ok else 'FAIL'}] loader-schema round trip")
        failures += 0 if ok else 1

    # 3. Real-MAE writer: pass-through, optional stds, non-finite dropped.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_mae_summaries(td, {"m1": 0.3, "m2": 0.15, "bad": float("nan")},
                            mae_stds={"m1": 0.05}, unit="eV", source="lb")
        with open(td / "summaries.json", encoding="utf-8") as f:
            payload = json.load(f)
        by = {e["policy_type"]: e for e in payload}
        ok = set(by) == {"m1", "m2"}
        ok = ok and by["m1"]["overall_stats"].get("best_color_distance_std") == 0.05
        ok = ok and "best_color_distance_std" not in by["m2"]["overall_stats"]
        print(f"  [{'PASS' if ok else 'FAIL'}] real-MAE pass-through")
        failures += 0 if ok else 1

    # 4. Omega end-to-end sanity on two synthetic spaces sharing classes:
    #    identical relative standings -> omega 0; degraded standing -> omega > 0.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation.phase3.knowledge_spaces import KnowledgeSpace, KnowledgeSpaceType

    def _ks(maes):
        return KnowledgeSpace(
            space_type=KnowledgeSpaceType.K_E, study_type="t",
            policy_mus={}, policy_sigmas={}, policy_ranks={},
            policy_maes=maes, policy_mae_stds={}, policy_stat_mus={},
            policy_bc_mus={})

    from src.evaluation.phase3.transfer_equation import compute_transfer_overhead
    same = compute_transfer_overhead(_ks({"a": 1.0, "b": 2.0}),
                                     _ks({"a": 10.0, "b": 20.0}))
    worse = compute_transfer_overhead(_ks({"a": 1.0, "b": 2.0}),
                                      _ks({"a": 3.0, "b": 1.0}))
    ok = abs(same) < 1e-12 and worse > 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] omega scale-invariance + degradation "
          f"(same={same:.3f}, worse={worse:.3f})")
    failures += 0 if ok else 1

    print(f"self-test: {4 - failures}/4 passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(_self_test())
