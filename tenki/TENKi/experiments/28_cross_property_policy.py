"""
Experiment 28 -- Cross-property transfer of the trained acquisition policy.

The recorded TENKi limit says the trained batch-selection policy of
experiment 27 is proven only within ONE property family: all training
channels and all leave-one-channel-out evaluation channels were band-gap
channels of the frozen band-gap archive universe. Transfer across property
families (gaps -> bulk modulus -> hardness -> formation energy) is
untested. This experiment tests exactly that: train the linear
acquisition policy with CEM on the 8 band-gap channels of the frozen
band-gap archive universe, then evaluate it ZERO-SHOT (weights frozen, no
retraining, no per-channel adaptation) on channels of entirely
different property families.

Target property channels (every record is a real archived value; no
invented observations):
  Bulk modulus (higher_is_better=True, min class size 30):
    bulk_jarvis : shared/data/jarvis/dft_3d_elastic.csv  (formula, bulk_gpa)
    bulk_aflow  : shared/data/jarvis/aflow_elastic.csv   (compound, bulk_gpa)
    bulk_mp     : shared/data/jarvis/mp_elastic.csv      (compound, bulk_gpa)
  Vickers hardness (higher_is_better=True, min class size 30), both from
  dft_3d_elastic.csv:
    hardness_chen_niu : chen_niu_vickers column
    hardness_exp      : exp_hardness column (sparse; included only if it
                        yields >= 4 classes of >= min_class_size records,
                        otherwise recorded as SKIPPED with actual counts)
  Al-Mg-Si formation energy (higher_is_better=FALSE -- more negative is
  more stable is better -- min class size 10), from
  compute/almgsi_archive/results/tracka_channels.jsonl:
    fe_aflow_dHf, fe_mp_dHf
  Formation-energy classes are chemical subsystems ("elemental",
  "Al-Mg", "Al-Si", "Mg-Si", "Al-Mg-Si") from x.chemsys with element
  order normalised. The min class size is 10 rather than 30 because the
  Al-Mg-Si archive is small (~3.4k records over both channels; a
  30-record floor would leave < 4 classes on mp_dHf); each channel is
  included only if >= 4 classes survive, otherwise SKIPPED with counts.

Class labels for the bulk/hardness channels use exp 27's
classify_formula / ANION_PRIORITY (imported, not copied) so entity
semantics match the training universe. The anion_class column present
in the CSVs is deliberately IGNORED: it was produced with different
priority rules and would silently change entity identity.

Orientation: reference ranking per channel = classes by median value,
best first ("best" = highest for bulk/hardness, MOST NEGATIVE for
formation energy). Both the reference and every sampled event pass
through the single canonicalisation point
hyperedge_from_scores(..., higher_is_better=<per-channel flag>).
Exp 27's sample_event/run_arm hard-code higher_is_better=True (band
gaps), so this file carries a minimal oriented mirror of that episode
loop (run_arm below); every selection rule, the feature extractor, the
Plackett-Luce fit, and the tau metric are REUSED from exp 27 unchanged.
Both orientations are verified on a toy input in --self-test
(sign-convention rule: test both orientations).

Policy: identical linear scoring family as exp 27 (features coverage /
uncertainty / closeness / noise, z-scored, top-k by score). Training =
CEM with exp 27's settings (pop 20, 8 iters, fitness = mean tau-AUC over
sampled training channels x training seeds), on ALL 8 band-gap channels
at once. Note exp 27's committed weights are per-LOCO-fold (each fold
held one BAND-GAP channel out); this experiment trains its own single
all-channel policy by default because no target channel is a band-gap
channel, so nothing needs holding out. --weights-json <path> skips
training and loads 4 weights from JSON (a bare list, {"weights": [...]},
or an exp-27-style {"learned_weights_mean": [...]}).
Training seeds (base+500000+...) are disjoint from evaluation seeds
(base+1000*s), mirroring exp 27's convention.

Evaluation per target channel, zero-shot: arms
  pair_random, kway_random, kway_active (min-spread), kway_fisher,
  kway_learned (frozen transferred weights)
k = 4, budget 240, checkpoints every 8 evaluations, 20 eval seeds
(defaults; see CLI). Per-channel checkpoint JSON channel_<name>.json is
written atomically as each channel lands (reruns skip finished
channels, as exp 27 does with folds).

Success criteria (evaluated per channel and per property family)
----------------------------------------------------------------
  Primary  (transfer does no harm): kway_learned final tau >=
            kway_random final tau on the property-family channels the
            policy never saw. Overall PASS = holds on every evaluated
            transfer channel.
  Secondary (transfer speeds up): kway_learned reaches 90% of
            kway_random's FINAL tau at a strictly smaller budget than
            kway_random itself. Reported per channel and per family.
The band-gap in-family numbers from exp 27 are read at runtime from its
committed results JSON (if present) and reported for context only.

Outputs (default results/cross_property/, override --output-dir)
----------------------------------------------------------------
  cross_property.json / .md / .png
  policy_weights.json           (trained weights checkpoint)
  channel_<name>.json           (per-channel checkpoint, atomic)

Run (from tenki/TENKi; PEGKI_ROOT when shared/ is not ../../../shared):
  uv run python experiments/28_cross_property_policy.py
Self-tests (orientation both ways, chemsys classes, loader smoke,
policy containment): --self-test.  Tiny end-to-end run: --smoke.
ASCII only; seeded and reproducible (--seed).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load exp 27 as a module (its filename starts with a digit, so a
# normal import is impossible). Registering it in sys.modules keeps its
# module-level functions picklable for multiprocessing (fork).
# ------------------------------------------------------------------
_EXP27_NAME = "exp27_learned_acquisition"
_spec = importlib.util.spec_from_file_location(
    _EXP27_NAME, os.path.join(_HERE, "27_learned_acquisition.py"))
exp27 = importlib.util.module_from_spec(_spec)
sys.modules[_EXP27_NAME] = exp27
_spec.loader.exec_module(exp27)

# Reused machinery (do not copy; exp 27 is the single source of truth).
classify_formula = exp27.classify_formula
entity_features = exp27.entity_features
pick_learned_subset = exp27.pick_learned_subset
pick_active_subset = exp27.pick_active_subset
pick_fisher_subset = exp27.pick_fisher_subset
load_universe = exp27.load_universe
PlackettLuceModel = exp27.PlackettLuceModel
hyperedge_from_scores = exp27.hyperedge_from_scores
_kendall_tau = exp27._kendall_tau
FEATURES = exp27.FEATURES

ARMS = ["pair_random", "kway_random", "kway_active", "kway_fisher",
        "kway_learned"]
# Same visual style as exp 27's plot.
STYLES = {"pair_random": ("tab:gray", "--"),
          "kway_random": ("tab:blue", "-"),
          "kway_active": ("tab:red", ":"),
          "kway_fisher": ("tab:orange", "-."),
          "kway_learned": ("tab:green", "-")}
FE_CHANNELS = ("aflow_dHf", "mp_dHf")
FE_ELEMENTS = {"Al", "Mg", "Si"}


def _pegki_root() -> Path:
    env = os.environ.get("PEGKI_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(_HERE) / ".." / ".." / ".." / "shared").resolve()


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------------
# Orientation-aware episode pieces. exp 27's sample_event and run_arm
# hard-code higher_is_better=True (band gaps); rather than modify
# exp 27, this file mirrors its episode loop with the per-channel
# orientation flag passed through the single canonicalisation point
# (hyperedge_from_scores). With higher_is_better=True these functions
# are behaviourally identical to exp 27's (same RNG call sequence).
# ------------------------------------------------------------------

def reference_ranking(classes: dict, higher_is_better: bool = True) -> list:
    med = {c: float(np.median(v)) for c, v in classes.items()}
    sign = -1.0 if higher_is_better else 1.0
    return sorted(med, key=lambda c: (sign * med[c], c))


def sample_event(classes: dict, subset: list, rng: np.random.Generator,
                 higher_is_better: bool = True):
    """One real archived value per participating class -> Hyperedge."""
    scores = {c: float(classes[c][rng.integers(0, len(classes[c]))])
              for c in subset}
    return hyperedge_from_scores(scores, higher_is_better=higher_is_better)


def run_arm(classes: dict, arm: str, kway: int, budget: int,
            checkpoints: list, seed: int, weights=None,
            higher_is_better: bool = True):
    """Oriented mirror of exp27.run_arm; all pickers/fit/tau reused."""
    rng = np.random.default_rng(seed)
    names = sorted(classes)
    ref = reference_ranking(classes, higher_is_better)
    k = 2 if arm == "pair_random" else kway
    counts = {n: 0 for n in names}
    edges, taus, spent, ci = [], [], 0, 0
    model = PlackettLuceModel()
    while spent < budget and ci < len(checkpoints):
        if arm == "kway_active":
            subset = pick_active_subset(model, names, k, rng)
        elif arm == "kway_fisher":
            subset = pick_fisher_subset(model, names, k, rng)
        elif arm == "kway_learned":
            subset = pick_learned_subset(weights, model, names, counts,
                                         k, rng)
        else:
            subset = list(rng.choice(names, size=k, replace=False))
        e = sample_event(classes, subset, rng, higher_is_better)
        spent += k
        for n in subset:
            counts[n] += 1
        if e is not None:
            edges.append(e)
        while ci < len(checkpoints) and spent >= checkpoints[ci]:
            if edges:
                model = PlackettLuceModel()
                model.fit(edges)
                taus.append(_kendall_tau(model.ranking(), ref))
            else:
                taus.append(0.0)
            ci += 1
    while len(taus) < len(checkpoints):
        taus.append(taus[-1] if taus else 0.0)
    return taus


# ------------------------------------------------------------------
# Target-channel loaders (real archived values only)
# ------------------------------------------------------------------

def _keep_classes(raw: dict, min_class_size: int):
    """raw {class: [values]} -> ({class: sorted array}, {class: count})."""
    kept = {c: np.asarray(sorted(v), dtype=float)
            for c, v in raw.items() if len(v) >= min_class_size}
    counts = {c: len(v) for c, v in raw.items()}
    return kept, counts


def load_csv_channel(path: Path, formula_col: str, value_col: str,
                     min_class_size: int):
    """CSV -> ({class: values}, {class: raw count}). Drops null/<=0.

    Classes come from exp 27's classify_formula so entity semantics
    match the training universe. The CSVs also carry an anion_class
    column, but it was produced with DIFFERENT priority rules -- it is
    deliberately not used.
    """
    raw: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = row.get(value_col)
            if v is None or v.strip() == "":
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            if val <= 0.0:
                continue
            formula = row.get(formula_col) or ""
            if not formula:
                continue
            raw.setdefault(classify_formula(formula), []).append(val)
    return _keep_classes(raw, min_class_size)


def chemsys_from_formula(formula: str) -> str:
    """'Al2Mg3' -> 'Al-Mg' (element-order-normalised chemsys)."""
    elems = sorted(set(re.findall(r"[A-Z][a-z]?", formula)))
    return "-".join(elems)


def chemsys_class(chemsys: str):
    """Normalised chemsys -> subsystem class, or None if outside Al-Mg-Si.

    'Al'|'Mg'|'Si' -> 'elemental'; binaries -> 'Al-Mg'|'Al-Si'|'Mg-Si';
    ternary -> 'Al-Mg-Si'. Element order in the input is normalised.
    """
    elems = sorted(set(e for e in str(chemsys).split("-") if e))
    if not elems or not set(elems) <= FE_ELEMENTS:
        return None
    if len(elems) == 1:
        return "elemental"
    return "-".join(elems)


def load_fe_channels(path: Path, min_class_size: int):
    """tracka_channels.jsonl -> {channel: (classes, raw counts)}.

    Values are y.formation_energy_eV_per_atom (eV/atom, real archived
    records; zero/negative values are physical and kept). Orientation
    for these channels is LOWER IS BETTER (more negative = more
    stable), handled by the higher_is_better=False flag downstream.
    """
    raw: dict = {c: {} for c in FE_CHANNELS}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec.get("s")
            if s not in FE_CHANNELS:
                continue
            y = (rec.get("y") or {}).get("formation_energy_eV_per_atom")
            if y is None:
                continue
            x = rec.get("x") or {}
            chemsys = x.get("chemsys") or chemsys_from_formula(
                x.get("formula") or "")
            cls = chemsys_class(chemsys)
            if cls is None:
                continue
            raw[s].setdefault(cls, []).append(float(y))
    return {s: _keep_classes(raw[s], min_class_size) for s in FE_CHANNELS}


def build_target_channels(args):
    """-> (evaluated {name: spec}, skipped {name: record}).

    spec = {classes, family, higher_is_better, min_class_size,
            class_sizes (kept), raw_class_counts (pre-threshold)}.
    A channel is evaluated only if >= 4 classes survive its class-size
    floor; otherwise it is recorded as skipped WITH the actual counts
    (honest thinness reporting; no channel silently disappears).
    """
    root = _pegki_root()
    jarvis_dir = (Path(args.jarvis_dir) if args.jarvis_dir
                  else root / "data" / "jarvis")
    almgsi_archive = (Path(args.tracka_jsonl) if args.tracka_jsonl
              else (root.parent / "compute" / "almgsi_archive" / "results"
                    / "tracka_channels.jsonl"))
    csv_specs = [
        ("bulk_jarvis", jarvis_dir / "dft_3d_elastic.csv", "formula",
         "bulk_gpa", "bulk_modulus"),
        ("bulk_aflow", jarvis_dir / "aflow_elastic.csv", "compound",
         "bulk_gpa", "bulk_modulus"),
        ("bulk_mp", jarvis_dir / "mp_elastic.csv", "compound",
         "bulk_gpa", "bulk_modulus"),
        ("hardness_chen_niu", jarvis_dir / "dft_3d_elastic.csv",
         "formula", "chen_niu_vickers", "hardness"),
        ("hardness_exp", jarvis_dir / "dft_3d_elastic.csv", "formula",
         "exp_hardness", "hardness"),
    ]
    evaluated, skipped = {}, {}

    def admit(name, family, hib, mcs, kept, counts, source):
        rec = {"family": family, "higher_is_better": hib,
               "min_class_size": mcs, "source": source,
               "raw_class_counts": counts,
               "n_classes_surviving": len(kept)}
        if len(kept) >= 4:
            rec["classes"] = kept
            rec["class_sizes"] = {c: int(len(v)) for c, v in kept.items()}
            evaluated[name] = rec
        else:
            rec["reason"] = (f"only {len(kept)} classes with >= {mcs} "
                             f"records (need >= 4)")
            skipped[name] = rec

    for name, path, fcol, vcol, family in csv_specs:
        if not path.exists():
            skipped[name] = {"family": family, "source": str(path),
                             "reason": "file not found",
                             "raw_class_counts": {}}
            continue
        kept, counts = load_csv_channel(path, fcol, vcol,
                                        args.min_class_size)
        admit(name, family, True, args.min_class_size, kept, counts,
              str(path))

    if not almgsi_archive.exists():
        for s in FE_CHANNELS:
            skipped[f"fe_{s}"] = {"family": "formation_energy",
                                  "source": str(almgsi_archive),
                                  "reason": "file not found",
                                  "raw_class_counts": {}}
    else:
        fe = load_fe_channels(almgsi_archive, args.fe_min_class_size)
        for s in FE_CHANNELS:
            kept, counts = fe[s]
            admit(f"fe_{s}", "formation_energy", False,
                  args.fe_min_class_size, kept, counts, str(almgsi_archive))
    return evaluated, skipped


# ------------------------------------------------------------------
# Training (single all-channel policy on the band-gap universe)
# ------------------------------------------------------------------

def train_policy(channels: list, args, pool) -> dict:
    """CEM over the 4 feature weights on ALL band-gap channels.

    Mirrors exp27.train_fold (same population/elite/sigma-floor rules,
    same fitness = mean tau-AUC via exp27._episode_auc in the pool,
    same disjoint training-seed convention base+500000+1000*it+s) with
    two deliberate differences, both documented here:
      1. no held-out channel -- the evaluation channels of this
         experiment are different PROPERTY FAMILIES, so nothing needs
         holding out of the band-gap training set; and
      2. the CEM RNG is seeded with a fixed offset (args.seed + 271828)
         instead of exp 27's `args.seed + hash(held_out) % 100000`,
         because str hash() is salted per interpreter (PYTHONHASHSEED)
         and would break run-to-run reproducibility.
    """
    rng = np.random.default_rng(args.seed + 271828)
    dim = len(FEATURES)
    mu, sigma = np.zeros(dim), np.ones(dim)
    n_elite = max(2, args.cem_pop // 4)
    history = []
    for it in range(args.cem_iters):
        cand = mu + sigma * rng.standard_normal((args.cem_pop, dim))
        chans = [str(c) for c in rng.choice(
            channels,
            size=min(args.train_channels_per_iter, len(channels)),
            replace=False)]
        seeds = [args.seed + 500000 + 1000 * it + s
                 for s in range(args.train_seeds)]
        tasks = [(tuple(c), ch, sd, args.kway, args.budget,
                  args.budget_step)
                 for c in cand for ch in chans for sd in seeds]
        aucs = pool.map(exp27._episode_auc, tasks)
        per_cand = np.asarray(aucs).reshape(
            args.cem_pop, len(chans) * len(seeds)).mean(axis=1)
        elite = cand[np.argsort(-per_cand)[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        history.append({"iter": it, "best_auc": float(per_cand.max()),
                        "mean_auc": float(per_cand.mean()),
                        "mu": mu.tolist()})
        print(f"[train] CEM iter {it + 1}/{args.cem_iters}: "
              f"best_auc={per_cand.max():+.4f} "
              f"mean_auc={per_cand.mean():+.4f}")
    return {"weights": mu.tolist(), "feature_names": FEATURES,
            "trained_on_channels": list(channels),
            "trained_on_family": "bandgap",
            "history": history}


def load_weights_json(path: Path) -> list:
    """Accept a bare 4-list, {"weights": [...]}, or an exp-27-style
    results JSON ({"learned_weights_mean": [...]})."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        w = obj
    elif isinstance(obj, dict) and "weights" in obj:
        w = obj["weights"]
    elif isinstance(obj, dict) and "learned_weights_mean" in obj:
        w = obj["learned_weights_mean"]
    else:
        raise ValueError(f"no usable weights in {path}")
    w = [float(v) for v in w]
    if len(w) != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} weights, got {len(w)}")
    return w


# ------------------------------------------------------------------
# Parallel zero-shot evaluation
# ------------------------------------------------------------------

_EVAL_DATA = None  # {name: (classes, higher_is_better)}; set per worker


def _eval_init(payload):
    global _EVAL_DATA
    _EVAL_DATA = payload


def _eval_task(task):
    name, arm, seed, kway, budget, step, weights = task
    classes, hib = _EVAL_DATA[name]
    checkpoints = list(range(step, budget + 1, step))
    w = np.asarray(weights, dtype=float) if arm == "kway_learned" else None
    return run_arm(classes, arm, kway, budget, checkpoints, seed,
                   weights=w, higher_is_better=hib)


def _speed_to(target: float, mean_curve: list, checkpoints: list) -> int:
    for i, v in enumerate(mean_curve):
        if v >= target:
            return checkpoints[i]
    return checkpoints[-1]


def evaluate_channel(name, spec, weights, args, checkpoints, pool) -> dict:
    t0 = time.time()
    tasks = [(name, arm, args.seed + 1000 * s, args.kway, args.budget,
              args.budget_step, tuple(weights))
             for arm in ARMS for s in range(args.seeds)]
    rows = pool.map(_eval_task, tasks)
    curves, curves_std = {}, {}
    for ai, arm in enumerate(ARMS):
        block = np.asarray(rows[ai * args.seeds:(ai + 1) * args.seeds])
        curves[arm] = block.mean(axis=0).tolist()
        curves_std[arm] = block.std(axis=0).tolist()
    final = {arm: curves[arm][-1] for arm in ARMS}
    target = 0.9 * final["kway_random"]
    speed = {arm: _speed_to(target, curves[arm], checkpoints)
             for arm in ARMS}
    primary = final["kway_learned"] >= final["kway_random"] - 1e-9
    secondary = speed["kway_learned"] < speed["kway_random"]
    return {
        "channel": name,
        "family": spec["family"],
        "higher_is_better": spec["higher_is_better"],
        "min_class_size": spec["min_class_size"],
        "source": spec["source"],
        "class_sizes": spec["class_sizes"],
        "n_records": int(sum(spec["class_sizes"].values())),
        "reference_ranking": reference_ranking(
            spec["classes"], spec["higher_is_better"]),
        "checkpoints": checkpoints,
        "curves": curves,
        "curves_std": curves_std,
        "final_tau": final,
        "budget_to_90pct_of_random_final": speed,
        "criteria": {
            "primary_learned_ge_random_at_final_budget": bool(primary),
            "secondary_learned_faster_to_90pct_of_random_final":
                bool(secondary),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }


# ------------------------------------------------------------------
# Band-gap in-family context (exp 27 committed results, if present)
# ------------------------------------------------------------------

def load_bandgap_context():
    candidates = [
        Path(_HERE).parent / "results" / "learned_acquisition"
        / "learned_acquisition.json",
        Path(_HERE).parent.parent / "results"
        / "learned_acquisition" / "learned_acquisition.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {"source": str(p),
                    "final_tau": d.get("final_tau"),
                    "budget_to_90pct_of_random_final":
                        d.get("budget_to_90pct_of_random_final"),
                    "learned_weights_mean_over_loco_folds":
                        d.get("learned_weights_mean")}
    return None


# ------------------------------------------------------------------
# Self-tests
# ------------------------------------------------------------------

def self_test() -> int:
    ok = True

    # (1) Orientation, both ways, on a toy LOWER-is-better channel
    # (sign-convention rule: test both orientations for every
    # new domain adapter). Constant per-class values make the sampled
    # events deterministic.
    toy = {"stable": np.asarray([-3.0, -3.0, -3.0]),
           "meta": np.asarray([-1.0, -1.0, -1.0]),
           "loose": np.asarray([2.0, 2.0, 2.0])}
    ref_lo = reference_ranking(toy, higher_is_better=False)
    ref_hi = reference_ranking(toy, higher_is_better=True)
    t = (ref_lo == ["stable", "meta", "loose"]
         and ref_hi == ["loose", "meta", "stable"])
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] orientation: reference ranking "
          f"lo={ref_lo} hi={ref_hi}")
    rng = np.random.default_rng(0)
    e_lo = sample_event(toy, list(toy), rng, higher_is_better=False)
    e_hi = sample_event(toy, list(toy), rng, higher_is_better=True)
    t = (e_lo.ranking == ("stable", "meta", "loose")
         and e_hi.ranking == ("loose", "meta", "stable"))
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] orientation: sampled events "
          f"lo={e_lo.ranking} hi={e_hi.ranking}")
    # End-to-end: an episode on the noiseless lower-is-better toy must
    # recover the oriented reference exactly (tau = +1).
    taus = run_arm(toy, "kway_random", 3, 30, [12, 24, 30], seed=1,
                   higher_is_better=False)
    t = abs(taus[-1] - 1.0) < 1e-12
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] orientation: lower-is-better "
          f"episode final tau {taus[-1]:+.4f} (expect +1)")

    # (2) Chemsys subsystem classifier.
    cases = [(chemsys_from_formula("Al2Mg3"), "Al-Mg"),
             (chemsys_from_formula("Al3Mg4Si2"), "Al-Mg-Si"),
             (chemsys_class("Al-Mg"), "Al-Mg"),
             (chemsys_class("Si-Al"), "Al-Si"),
             (chemsys_class("Mg-Si"), "Mg-Si"),
             (chemsys_class("Al"), "elemental"),
             (chemsys_class("Al-Mg-Si"), "Al-Mg-Si"),
             (chemsys_class("Al-Fe"), None)]
    t = all(got == want for got, want in cases)
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] chemsys classes: "
          + ", ".join(f"{got}=={want}" for got, want in cases))

    # (3) Loader smoke on the real CSVs: bulk channels must each yield
    # >= 4 classes. Skipped (not failed) if the data files are absent.
    class _A:
        min_class_size = 30
        fe_min_class_size = 10
        jarvis_dir = None
        tracka_jsonl = None
    jdir = _pegki_root() / "data" / "jarvis"
    if jdir.exists():
        evaluated, skipped = build_target_channels(_A())
        for name in ["bulk_jarvis", "bulk_aflow", "bulk_mp"]:
            t = (name in evaluated
                 and len(evaluated[name]["classes"]) >= 4)
            ok &= t
            sizes = (evaluated[name]["class_sizes"] if name in evaluated
                     else skipped.get(name, {}).get("raw_class_counts"))
            print(f"[{'PASS' if t else 'FAIL'}] loader {name}: "
                  f"classes {sizes}")
        for name in sorted(set(list(evaluated) + list(skipped))
                           - {"bulk_jarvis", "bulk_aflow", "bulk_mp"}):
            if name in evaluated:
                print(f"[info] loader {name}: evaluated, classes "
                      f"{evaluated[name]['class_sizes']}")
            else:
                print(f"[info] loader {name}: skipped "
                      f"({skipped[name]['reason']}); raw counts "
                      f"{skipped[name]['raw_class_counts']}")
    else:
        print(f"[skip] loader smoke: data dir not found ({jdir})")

    # (4) Policy containment: noise-only weights = a valid random
    # subset (exp 27's containment property, via the reused picker),
    # and different RNG streams do differ.
    names = ["a", "b", "c", "d", "e"]
    model = PlackettLuceModel()
    counts = {n: 1 for n in names}
    subs = [tuple(pick_learned_subset(np.array([0, 0, 0, 1.0]), model,
                                      names, counts, 3,
                                      np.random.default_rng(s)))
            for s in range(10)]
    t = (all(len(set(s)) == 3 and set(s) <= set(names) for s in subs)
         and len(set(subs)) > 1)
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] containment: noise-only weights "
          f"-> valid, varying random subsets ({len(set(subs))}/10 unique)")

    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="TENKi exp 28: cross-property transfer of the "
                    "trained acquisition policy")
    ap.add_argument("--universe",
                    default="output/bandgap_archive_build/bandgap_universe.jsonl",
                    help="band-gap training universe, rel. to PEGKI_ROOT")
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--budget-step", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=20,
                    help="evaluation seeds per (channel, arm)")
    ap.add_argument("--min-class-size", type=int, default=30,
                    help="floor for band-gap/bulk/hardness classes")
    ap.add_argument("--fe-min-class-size", type=int, default=10,
                    help="floor for the small Al-Mg-Si formation-energy "
                         "channels (see docstring)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cem-pop", type=int, default=20)
    ap.add_argument("--cem-iters", type=int, default=8)
    ap.add_argument("--train-seeds", type=int, default=2)
    ap.add_argument("--train-channels-per-iter", type=int, default=4)
    ap.add_argument("--weights-json", default=None,
                    help="skip training; load 4 policy weights from JSON")
    ap.add_argument("--jarvis-dir", default=None,
                    help="override PEGKI_ROOT/data/jarvis")
    ap.add_argument("--almgsi_archive-jsonl", default=None,
                    help="override <root>/compute/almgsi_archive/results/"
                         "tracka_channels.jsonl")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = min(14, cpu_count)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 2 seeds, budget 48, reduced CEM")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.smoke:
        args.seeds = 2
        args.budget = 48
        args.cem_iters = 2
        args.cem_pop = 8
        args.train_seeds = 1
        args.train_channels_per_iter = 2
        print("[smoke] seeds=2 budget=48 cem_iters=2 cem_pop=8 "
              "train_seeds=1 train_channels_per_iter=2")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(_HERE).parent / "results" / "cross_property")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list(range(args.budget_step, args.budget + 1,
                             args.budget_step))
    n_workers = args.workers or min(14, os.cpu_count() or 1)

    # ----- target channels (must exist before spending training time)
    evaluated, skipped = build_target_channels(args)
    for name in sorted(evaluated):
        spec = evaluated[name]
        print(f"[load] {name} ({spec['family']}, "
              f"{'higher' if spec['higher_is_better'] else 'LOWER'}"
              f" is better): {len(spec['classes'])} classes, "
              f"{sum(spec['class_sizes'].values())} records "
              f"{spec['class_sizes']}")
    for name in sorted(skipped):
        print(f"[skip-channel] {name}: {skipped[name]['reason']}; "
              f"raw counts {skipped[name]['raw_class_counts']}")
    if not evaluated:
        print("[error] no target channel survived its class-size floor")
        return 1

    # ----- policy weights: --weights-json > checkpoint > train
    weights_file = out_dir / "policy_weights.json"
    if args.weights_json:
        w = load_weights_json(Path(args.weights_json))
        policy = {"weights": w, "feature_names": FEATURES,
                  "source": str(Path(args.weights_json).resolve()),
                  "trained_on_family": "bandgap (external)"}
        print(f"[weights] loaded from {args.weights_json}: "
              + ", ".join(f"{n}={v:+.3f}" for n, v in zip(FEATURES, w)))
    elif weights_file.exists():
        with open(weights_file, "r", encoding="utf-8") as f:
            policy = json.load(f)
        print(f"[weights] reusing checkpoint {weights_file}: "
              + ", ".join(f"{n}={v:+.3f}"
                          for n, v in zip(FEATURES, policy["weights"])))
    else:
        universe = _pegki_root() / args.universe
        if not universe.exists():
            print(f"[error] universe not found: {universe}")
            return 1
        per_channel = load_universe(universe, args.min_class_size)
        train_channels = sorted(per_channel)
        print(f"[train] {len(train_channels)} band-gap channels: "
              + ", ".join(train_channels))
        if not train_channels:
            return 1
        t0 = time.time()
        with Pool(n_workers, initializer=exp27._pool_init,
                  initargs=(str(universe), args.min_class_size)) as pool:
            policy = train_policy(train_channels, args, pool)
        policy["train_elapsed_s"] = round(time.time() - t0, 2)
        _write_json_atomic(weights_file, policy)
        print(f"[train] done in {policy['train_elapsed_s']}s; weights "
              + ", ".join(f"{n}={v:+.3f}"
                          for n, v in zip(FEATURES, policy["weights"])))
    weights = [float(v) for v in policy["weights"]]

    # ----- zero-shot evaluation, per channel, checkpointed
    payload = {name: (spec["classes"], spec["higher_is_better"])
               for name, spec in evaluated.items()}
    per_channel_results = {}
    with Pool(n_workers, initializer=_eval_init,
              initargs=(payload,)) as pool:
        for name in sorted(evaluated):
            ck = out_dir / f"channel_{name}.json"
            if ck.exists():
                with open(ck, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                print(f"[skip] {name} already evaluated ({ck.name})")
            else:
                rec = evaluate_channel(name, evaluated[name], weights,
                                       args, checkpoints, pool)
                _write_json_atomic(ck, rec)
                print(f"[done] {name}: learned "
                      f"{rec['final_tau']['kway_learned']:+.4f} vs "
                      f"random {rec['final_tau']['kway_random']:+.4f}; "
                      f"90pct budget {rec['budget_to_90pct_of_random_final']['kway_learned']}"
                      f" vs {rec['budget_to_90pct_of_random_final']['kway_random']}"
                      f" ({rec['elapsed_s']}s)")
            per_channel_results[name] = rec

    # ----- aggregate per property family and overall
    families = sorted({r["family"] for r in per_channel_results.values()})
    per_family = {}
    for fam in families:
        chans = sorted(n for n, r in per_channel_results.items()
                       if r["family"] == fam)
        agg = {}
        for arm in ARMS:
            stack = np.stack([np.asarray(per_channel_results[n]
                                         ["curves"][arm])
                              for n in chans])
            agg[arm] = {"mean": stack.mean(axis=0).tolist(),
                        "std": stack.std(axis=0).tolist()}
        final = {arm: agg[arm]["mean"][-1] for arm in ARMS}
        target = 0.9 * final["kway_random"]
        speed = {arm: _speed_to(target, agg[arm]["mean"], checkpoints)
                 for arm in ARMS}
        per_family[fam] = {
            "channels": chans,
            "aggregate": agg,
            "final_tau": final,
            "budget_to_90pct_of_random_final": speed,
            "criteria": {
                "primary_learned_ge_random_at_final_budget":
                    bool(final["kway_learned"]
                         >= final["kway_random"] - 1e-9),
                "secondary_learned_faster_to_90pct_of_random_final":
                    bool(speed["kway_learned"] < speed["kway_random"]),
            },
        }

    chan_names = sorted(per_channel_results)
    primary_by_channel = {
        n: per_channel_results[n]["criteria"]
        ["primary_learned_ge_random_at_final_budget"]
        for n in chan_names}
    secondary_by_channel = {
        n: per_channel_results[n]["criteria"]
        ["secondary_learned_faster_to_90pct_of_random_final"]
        for n in chan_names}
    overall = {
        "primary_transfer_no_harm_all_channels":
            bool(all(primary_by_channel.values())),
        "primary_pass_channels":
            [n for n in chan_names if primary_by_channel[n]],
        "secondary_transfer_speedup_channels":
            [n for n in chan_names if secondary_by_channel[n]],
        "secondary_pass_all": bool(all(secondary_by_channel.values())),
        "secondary_pass_any": bool(any(secondary_by_channel.values())),
    }
    bandgap_context = load_bandgap_context()

    results = {
        "config": {k: v for k, v in vars(args).items()
                   if k not in ("self_test",)},
        "checkpoints": checkpoints,
        "policy": policy,
        "channels_evaluated": chan_names,
        "channels_skipped": {
            n: {k: v for k, v in rec.items() if k != "classes"}
            for n, rec in skipped.items()},
        "per_channel": {
            n: {k: v for k, v in per_channel_results[n].items()}
            for n in chan_names},
        "per_family": per_family,
        "criteria": overall,
        "bandgap_in_family_context_exp27": bandgap_context,
    }
    _write_json_atomic(out_dir / "cross_property.json", results)

    # ----- plot: tau-vs-budget per evaluated channel (exp 27 style)
    n = len(chan_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 3.5 * nrows),
                             squeeze=False)
    for i, name in enumerate(chan_names):
        ax = axes[i // ncols][i % ncols]
        rec = per_channel_results[name]
        for arm in ARMS:
            m = np.asarray(rec["curves"][arm])
            s = np.asarray(rec["curves_std"][arm])
            c, ls = STYLES[arm]
            ax.plot(checkpoints, m, ls, color=c, label=arm)
            ax.fill_between(checkpoints, m - s, m + s, color=c,
                            alpha=0.10)
        orient = "" if rec["higher_is_better"] else ", lower is better"
        ax.set_title(f"{name} ({rec['family']}{orient})", fontsize=9)
        ax.grid(alpha=0.3)
        if i // ncols == nrows - 1:
            ax.set_xlabel("cumulative evaluations")
        if i % ncols == 0:
            ax.set_ylabel("Kendall tau vs reference")
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Zero-shot cross-property transfer of the band-gap-"
                 f"trained policy (k={args.kway}, {args.seeds} seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "cross_property.png", dpi=150)
    plt.close(fig)

    # ----- markdown summary
    lines = [
        "# Experiment 28: cross-property transfer of the trained "
        "acquisition policy",
        "",
        "Policy trained with CEM on ALL band-gap channels of the frozen "
        "band-gap archive universe, then frozen and evaluated ZERO-SHOT on "
        "channels of other property families.",
        "",
        "Frozen weights: "
        + ", ".join(f"{nme} {v:+.3f}"
                    for nme, v in zip(FEATURES, weights)),
        f"k = {args.kway}; budget = {args.budget}; {args.seeds} eval "
        f"seeds; training seeds disjoint from eval seeds.",
        "",
        "| channel | family | classes | records | learned tau | "
        "random tau | learned 90% budget | random 90% budget | "
        "primary | secondary |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nme in chan_names:
        r = per_channel_results[nme]
        lines.append(
            f"| {nme} | {r['family']} | {len(r['class_sizes'])} | "
            f"{r['n_records']} | {r['final_tau']['kway_learned']:+.4f} | "
            f"{r['final_tau']['kway_random']:+.4f} | "
            f"{r['budget_to_90pct_of_random_final']['kway_learned']} | "
            f"{r['budget_to_90pct_of_random_final']['kway_random']} | "
            f"{'PASS' if primary_by_channel[nme] else 'FAIL'} | "
            f"{'PASS' if secondary_by_channel[nme] else 'FAIL'} |")
    lines += ["", "## Per property family", "",
              "| family | channels | learned tau | random tau | "
              "learned 90% budget | random 90% budget | primary | "
              "secondary |",
              "|---|---|---|---|---|---|---|---|"]
    for fam in families:
        f = per_family[fam]
        lines.append(
            f"| {fam} | {len(f['channels'])} | "
            f"{f['final_tau']['kway_learned']:+.4f} | "
            f"{f['final_tau']['kway_random']:+.4f} | "
            f"{f['budget_to_90pct_of_random_final']['kway_learned']} | "
            f"{f['budget_to_90pct_of_random_final']['kway_random']} | "
            f"{'PASS' if f['criteria']['primary_learned_ge_random_at_final_budget'] else 'FAIL'} | "
            f"{'PASS' if f['criteria']['secondary_learned_faster_to_90pct_of_random_final'] else 'FAIL'} |")
    lines += [
        "",
        f"Primary (transfer does no harm, ALL channels): "
        f"**{'PASS' if overall['primary_transfer_no_harm_all_channels'] else 'FAIL'}** "
        f"({len(overall['primary_pass_channels'])}/{n} channels)",
        f"Secondary (transfer speeds up): "
        f"{len(overall['secondary_transfer_speedup_channels'])}/{n} "
        f"channels PASS "
        f"({', '.join(overall['secondary_transfer_speedup_channels']) or 'none'})",
        "",
        "## Skipped channels (honest thinness reporting)",
        "",
    ]
    if skipped:
        for nme in sorted(skipped):
            rec = skipped[nme]
            lines.append(f"- `{nme}` ({rec.get('family')}): "
                         f"{rec['reason']}; raw class counts "
                         f"{rec['raw_class_counts']}")
    else:
        lines.append("- none")
    if bandgap_context:
        ft = bandgap_context["final_tau"]
        sp = bandgap_context["budget_to_90pct_of_random_final"]
        lines += [
            "",
            "## Band-gap in-family context (exp 27, LOCO)",
            "",
            f"learned {ft['kway_learned']:+.4f} vs random "
            f"{ft['kway_random']:+.4f} final tau; 90% budget "
            f"{sp['kway_learned']} vs {sp['kway_random']}. "
            "(Exp 27 weights are per-LOCO-fold over band-gap channels; "
            "this experiment trains one all-channel policy.)",
        ]
    lines += [
        "",
        "Formation-energy channels are oriented LOWER-is-better (most "
        "negative = most stable) through hyperedge_from_scores's "
        "higher_is_better flag; both orientations verified in "
        "--self-test. Real archived records only; no invented "
        "observations. Weights frozen at evaluation time (zero-shot).",
    ]
    with open(out_dir / "cross_property.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[save] {out_dir}")
    print("primary (no harm, all channels): "
          + ("PASS" if overall["primary_transfer_no_harm_all_channels"]
             else "FAIL"))
    print(f"secondary (speedup): "
          f"{len(overall['secondary_transfer_speedup_channels'])}/{n} "
          f"channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
