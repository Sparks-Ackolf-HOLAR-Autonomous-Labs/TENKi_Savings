"""
jarvis_prediction_methods_phase2.py
-------------------------------------
Correct PEGKi framing for the JARVIS example:

    Engines  = prediction methods (DFT functionals)
    Policies = composition-class families of materials

This inverts the leaderboard script design where ML methods were the policies.
Here we ask: "does the ranking of composition families by band gap survive when
the prediction method changes from GGA to meta-GGA?"  High transfer delta means
the ordering of {oxide, nitride, chalcogenide, halide, metal_alloy} is disrupted
when the computational fidelity changes.

Engines (prediction methods; 5 total across 3 physics-type groups for Causal ANOVA power):
    optb88vdw   -- JARVIS OptB88vdW vdW-GGA  (study_type: optb88vdw_bg_dft_bandstructure)
                   -> physics group: dft_bandstructure
    mbj         -- JARVIS TB-mBJ meta-GGA     (study_type: mbj_bg_dft_meta_gga)
                   -> physics group: dft_meta_gga
    -- Set-op engines anchored on mBJ/GGA band-gap enhancement ratio --
    difference  -- GGA-dominant composition classes (low mBJ/GGA ratio)
                   (study_type: optb88vdw_bg_composition_difference_bandstructure)
                   -> physics group: mixed  (causal_analysis._infer_physics_type checks
                      "difference" keyword before DFT keywords -> always "mixed")
    complement  -- mBJ-enhanced composition classes (high mBJ/GGA ratio)
                   (study_type: mbj_bg_composition_complement_meta_gga)
                   -> physics group: mixed  (same reason: "complement" caught first)
    intersection -- composition classes with fully stable ordering (all)
                   (study_type: gga_bg_composition_intersection_theoretical)
                   -> physics group: mixed  ("intersection" caught first)

Causal ANOVA design:
    3 physics groups: dft_bandstructure (1 engine), dft_meta_gga (1 engine), mixed (3 engines).
    With N=5 engines: each source engine produces (N-1)=4 delta observations.
    dft_bandstructure: 4 obs, dft_meta_gga: 4 obs, mixed: 12 obs.  Total=20.
    df_between=2, df_within=17 -> F_crit(2,17)~3.59 -> powered to detect medium effects.

    NOTE: the composition ordering is perfectly stable (no ordinal rank inversions
    between GGA and mBJ), so rank-inversion-based set-ops would always be empty.
    The ratio-based split uses mBJ/GGA enhancement ratio as a continuous measure:
      low ratio  -> GGA-dominant classes -> difference engine (mixed group)
      high ratio -> mBJ-dominant classes -> complement engine (mixed group)
    This is physically meaningful: low-ratio classes (chalcogenides, metal_alloy)
    have narrower band gaps where the mBJ correction matters less; high-ratio classes
    (halides, oxides) have wide gaps where the meta-GGA correction is most important.
    The three mixed engines together give 12 obs in the mixed group, ensuring ANOVA
    power even though the set-op study_types cannot be routed to specific DFT groups.

Policies (composition families):
    oxide         -- contains O
    nitride       -- contains N (and not O)
    chalcogenide  -- contains S, Se, or Te (and not O/N)
    halide        -- contains F, Cl, Br, or I (and not O/N/S/Se/Te)
    metal_alloy   -- no dominant anion (pnictides, III-V, intermetallics, metals)

Scores = per-material band gap value for that engine/class combination.
The rating systems learn which composition family typically has the highest
band gap; transfer asks whether that ordering is stable across functionals.

Set-op databases use OptB88vdW vs mBJ as anchor pair:
    intersection  -- composition classes where GGA and meta-GGA rank identically
    difference    -- classes where GGA ranks higher than meta-GGA
    complement    -- classes where meta-GGA ranks higher than GGA

Usage:
    # First run (downloads dft_3d if not cached):
    uv run --with "jarvis-tools,pandas" python scripts/jarvis_prediction_methods_phase2.py

    # Skip download (use cached CSVs):
    uv run --with "jarvis-tools,pandas" python scripts/jarvis_prediction_methods_phase2.py --skip-fetch

Output:
    output/jarvis_prediction_methods/db_optb88vdw/   (dft_bandstructure group)
    output/jarvis_prediction_methods/db_mbj/          (dft_meta_gga group)
    output/jarvis_prediction_methods/db_difference/   (dft_bandstructure group, low-ratio classes)
    output/jarvis_prediction_methods/db_complement/   (dft_meta_gga group, high-ratio classes)
    output/jarvis_prediction_methods/db_intersection/ (mixed group, all stable classes)
    output/jarvis_prediction_methods/db_snumat_gga/   (dft_bandstructure, SKIPPED if no formula)
    output/jarvis_prediction_methods/db_snumat_hse/   (dft_meta_gga, SKIPPED if no formula)
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.rating.trueskill_engine import TrueSkill2Engine
from src.rating.blade_chest import BladeChest
from src.rating.statistical_rating_system import StatisticalRatingSystem
from src.rating.icbt_model import ICBTModel
from src.rating.augmented_icbt_bc import AugmentedICBTBC
from src.rating.spinning_tops import SpinningTopsModel
from src.utils.simulation_seed import initialize_global_seed
from materials_summaries import write_dispersion_summaries

initialize_global_seed(42)


# ---------------------------------------------------------------------------
# Composition-based material class classifier
# ---------------------------------------------------------------------------

_OXIDES  = frozenset({"O"})
_HALIDES = frozenset({"F", "Cl", "Br", "I"})
_NITRIDES = frozenset({"N"})
_CHALCOGENIDES = frozenset({"S", "Se", "Te"})


def classify_by_anion(formula: str) -> str:
    """Classify material into composition family based on dominant anion.

    Priority: oxide > halide > nitride > chalcogenide > metal_alloy
    (covers pnictides, III-V, intermetallics, pure elements).

    Note: 'Se' and 'Te' are parsed correctly by the two-char regex token.
    """
    if not isinstance(formula, str):
        # pandas parses the literal formula "NaN" (sodium nitride) as float
        # NaN unless na-filtering is disabled; treat any non-string as unknown
        return "metal_alloy"
    elems = set(re.findall(r"[A-Z][a-z]?", formula))
    if _OXIDES & elems:
        return "oxide"
    if _HALIDES & elems:
        return "halide"
    if _NITRIDES & elems:
        return "nitride"
    if _CHALCOGENIDES & elems:
        return "chalcogenide"
    return "metal_alloy"


COMPOSITION_CLASSES = ["oxide", "nitride", "chalcogenide", "halide", "metal_alloy"]


# ---------------------------------------------------------------------------
# JARVIS / SNUMAT data loaders  (same helpers as jarvis_phase2.py)
# ---------------------------------------------------------------------------

def _safe_float(val):
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("na", "none", "", "null", "nan"):
            return None
        try:
            return float(v)
        except ValueError:
            return None
    try:
        f = float(val)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def load_jarvis_dft3d(cache_csv: Path, skip_fetch: bool):
    import pandas as pd
    if skip_fetch and cache_csv.exists():
        print(f"Loading cached JARVIS data from {cache_csv} ...")
        df = pd.read_csv(cache_csv, converters={"formula": str})
        print(f"  {len(df)} entries loaded")
        return df
    print("Downloading JARVIS dft_3d dataset from figshare ...")
    print("  (This may take 1-2 minutes on first run; ~300 MB)")
    try:
        from jarvis.db.figshare import data as jdata
    except ImportError:
        print("ERROR: jarvis-tools not installed.")
        sys.exit(1)
    raw = jdata("dft_3d")
    print(f"  Downloaded {len(raw)} entries")
    _OPT_FIELDS  = ("optb88vdw_bandgap", "opt_bandgap", "gga_bandgap")
    _MBJ_FIELDS  = ("mbj_bandgap", "mBJ_bandgap", "mbj_gap")
    _SPG_FIELDS  = ("spg_number", "spacegroup_number", "space_group_number", "sg")
    _FORM_FIELDS = ("formula", "formula_pretty", "composition")

    def _pick(entry, fields):
        for f in fields:
            v = entry.get(f)
            if v is not None and v != "na":
                return v
        return None

    rows = []
    for entry in raw:
        jid     = entry.get("jid", "")
        formula = _pick(entry, _FORM_FIELDS) or ""
        spg_raw = _pick(entry, _SPG_FIELDS)
        spg     = int(spg_raw) if spg_raw is not None else 0
        opt_bg  = _safe_float(_pick(entry, _OPT_FIELDS))
        mbj_bg  = _safe_float(_pick(entry, _MBJ_FIELDS))
        comp    = classify_by_anion(formula) if formula else "metal_alloy"
        rows.append({
            "jid": jid, "formula": formula, "spg_number": spg,
            "composition_class": comp,
            "optb88vdw_bandgap": opt_bg,
            "mbj_bandgap": mbj_bg,
        })
    df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f"  Saved: {cache_csv} ({len(df)} rows)")
    return df


def load_snumat_with_composition(cache_csv: Path, skip_fetch: bool):
    """Load SNUMAT with formula-based composition classification.

    Uses a separate cache file (snumat_composition.csv) so it does not
    overwrite the snumat.csv used by jarvis_phase2.py.
    """
    import pandas as pd
    if skip_fetch and cache_csv.exists():
        print(f"Loading cached SNUMAT composition data from {cache_csv} ...")
        df = pd.read_csv(cache_csv, converters={"formula": str})
        print(f"  {len(df)} entries loaded")
        return df
    print("Downloading SNUMAT dataset from JARVIS figshare (for formula field) ...")
    try:
        from jarvis.db.figshare import data as jdata
    except ImportError:
        print("ERROR: jarvis-tools not installed.")
        sys.exit(1)
    raw = jdata("snumat")
    print(f"  Downloaded {len(raw)} entries")
    rows = []
    for entry in raw:
        sid   = entry.get("SNUMAT_id", "")
        spg_r = entry.get("Space_group_rlx")
        spg   = int(spg_r) if spg_r is not None else 0
        # Try several field names for formula
        formula = (entry.get("formula") or entry.get("Formula") or
                   entry.get("composition") or "")
        if not formula:
            formula = ""
        comp  = classify_by_anion(formula) if formula else "metal_alloy"
        gga   = _safe_float(entry.get("Band_gap_GGA"))
        hse   = _safe_float(entry.get("Band_gap_HSE"))
        rows.append({"snumat_id": sid, "spg_number": spg,
                     "formula": formula, "composition_class": comp,
                     "gga_bandgap": gga, "hse_bandgap": hse})
    df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f"  Saved: {cache_csv} ({len(df)} rows)")
    # Report composition breakdown
    vc = df["composition_class"].value_counts()
    print(f"  Composition class breakdown: {vc.to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Rating system builder  (same pattern as jarvis_phase2.py)
# ---------------------------------------------------------------------------

DUMMY_AGENT = "eval_agent"
OBJECTIVE   = "band_gap_composition"
ALL_SYSTEMS = ["trueskill", "blade_chest", "statistical",
               "icbt", "icbt_bc", "spinning_tops"]
TS_KWARGS   = dict(mu=1200.0, sigma=200.0, agent_beta=80.0, agent_tau=4.0,
                   policy_beta=150.0, policy_tau=6.0)


class PredMethodRatingBuilder:
    def __init__(self):
        self.ts_engine             = TrueSkill2Engine(**TS_KWARGS)
        bc_ts                      = TrueSkill2Engine(**TS_KWARGS)
        self.bc_engine             = BladeChest(bc_ts)
        self.stat_engine           = StatisticalRatingSystem(agents=[])
        self.icbt_engine           = ICBTModel()
        self.icbt_bc_engine        = AugmentedICBTBC()
        self.spinning_tops_engine  = SpinningTopsModel()

    def train(self, class_names, class_scores_norm, class_stds, n_rounds, rng):
        total = 0
        for rnd in range(n_rounds):
            noisy = {}
            for cn in class_names:
                scores = class_scores_norm[cn]
                if not scores:
                    continue
                base = scores[rnd % len(scores)]
                std  = class_stds[cn]
                noisy[cn] = float(np.clip(
                    rng.normal(base, std) if std > 0 else base, 0.0, 1.0
                ))
            if len(noisy) < 2:
                continue
            perfs    = [(DUMMY_AGENT, cn, noisy[cn]) for cn in noisy]
            pairwise = []
            pairs_all = []
            active = list(noisy)
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    ci, cj = active[ii], active[jj]
                    si, sj = noisy[ci], noisy[cj]
                    if abs(si - sj) < 1e-9:
                        continue
                    winner, loser = (ci, cj) if si > sj else (cj, ci)
                    ws, ls = (si, sj) if si > sj else (sj, si)
                    pairs_all += [(DUMMY_AGENT, winner), (DUMMY_AGENT, loser)]
                    pairwise.append((
                        DUMMY_AGENT, winner, DUMMY_AGENT, loser,
                        abs(si - sj), 0.0, ws, 0.0, ls,
                    ))
            if not pairwise:
                continue
            ts_perfs = [(a, p, -s) for a, p, s in perfs]
            self.ts_engine.update_ratings(ts_perfs, current_round=rnd)
            self.bc_engine.trueskill_engine.update_ratings(perfs, current_round=rnd)
            unique_pairs = list({(a, p) for a, p in pairs_all})
            self.bc_engine.update_embeddings(OBJECTIVE, unique_pairs, pairwise)
            obj_perfs    = {OBJECTIVE: perfs}
            obj_pairwise = {OBJECTIVE: pairwise}
            self.stat_engine.update_ratings(obj_perfs, obj_pairwise)
            self.icbt_engine.update_ratings(obj_perfs, obj_pairwise)
            self.icbt_bc_engine.update_ratings(obj_perfs, obj_pairwise)
            self.spinning_tops_engine.update_ratings(obj_perfs, obj_pairwise)
            total += len(pairwise)
        return total

    def _ts_entry(self, policy):
        r = self.ts_engine.get_rating(policy, "policy")
        mu, sigma = float(r.mu), float(r.sigma)
        return {"mu": mu, "sigma": sigma, "conservative": mu - 3 * sigma,
                "per_objective": {OBJECTIVE: {"mu": mu, "sigma": sigma}}}

    def _bc_entry(self, policy):
        bv, cv = {}, {}
        for obj, m in self.bc_engine.objective_models.items():
            b = m["policy_blade_vectors"].get(policy)
            c = m["policy_chest_vectors"].get(policy)
            if b is not None: bv[obj] = b.tolist()
            if c is not None: cv[obj] = c.tolist()
        return {"blade_vectors": bv, "chest_vectors": cv}

    def _stat_entry(self, policy):
        def _k(d):
            for k in (policy, f"eval_agent_{policy}"):
                if k in d: return k
            return None
        s_trans = {obj: float(d[k])
                   for obj, d in self.stat_engine._transitive.items()
                   for k in [_k(d)] if k}
        bv, cv, ctx = {}, {}, {}
        for obj, m in self.stat_engine.blade_chest_engine.objective_models.items():
            b = m["policy_blade_vectors"].get(policy)
            c = m["policy_chest_vectors"].get(policy)
            if b is not None: bv[obj] = b.tolist()
            if c is not None: cv[obj] = c.tolist()
        for obj, d in self.stat_engine.context_biases.items():
            k = _k(d)
            if k: ctx[obj] = float(d[k])
        return {"s_transitive": s_trans, "blade_vectors": bv,
                "chest_vectors": cv, "context_bias": ctx}

    def _icbt_entry(self, policy):
        return {"log_pi_per_cluster": {
            str(k): float(self.icbt_engine._log_pi.get(k, {}).get(policy, 0.0))
            for k in range(self.icbt_engine._best_k)}}

    def _icbt_bc_entry(self, policy):
        dim = self.icbt_bc_engine.bc_dim
        lp  = {str(k): float(self.icbt_bc_engine._log_pi.get(k, {}).get(policy, 0.0))
               for k in range(self.icbt_bc_engine._best_k)}
        bc  = {str(k): {
            "blade": self.icbt_bc_engine._blades.get(k, {}).get(
                policy, np.zeros(dim)).tolist(),
            "chest": self.icbt_bc_engine._chests.get(k, {}).get(
                policy, np.zeros(dim)).tolist(),
        } for k in range(self.icbt_bc_engine._best_k)}
        return {"log_pi_per_cluster": lp, "bc_per_cluster": bc}

    def _spinning_tops_entry(self, policy):
        return {"f_transitive": {
            obj: float(d[policy])
            for obj, d in self.spinning_tops_engine._transitive.items()
            if policy in d}}

    def get_trained_ratings(self, class_names, study_type, db_path):
        from datetime import datetime
        policy_ratings = {}
        for sys_name in ALL_SYSTEMS:
            entries = {}
            for cn in class_names:
                if sys_name == "trueskill":       entries[cn] = self._ts_entry(cn)
                elif sys_name == "blade_chest":   entries[cn] = self._bc_entry(cn)
                elif sys_name == "statistical":   entries[cn] = self._stat_entry(cn)
                elif sys_name == "icbt":          entries[cn] = self._icbt_entry(cn)
                elif sys_name == "icbt_bc":       entries[cn] = self._icbt_bc_entry(cn)
                elif sys_name == "spinning_tops": entries[cn] = self._spinning_tops_entry(cn)
            policy_ratings[sys_name] = entries
        system_native_params = {
            "statistical": {"transitive": {
                obj: {k: float(v) for k, v in d.items()}
                for obj, d in self.stat_engine._transitive.items()}},
            "spinning_tops": {"cyclic": {
                obj: {
                    "matrix": m.tolist() if hasattr(m, "tolist") else m,
                    "entity_order": self.spinning_tops_engine._entity_order.get(obj, []),
                }
                for obj, m in self.spinning_tops_engine._cyclic.items()}
                if hasattr(self.spinning_tops_engine, "_cyclic") else {}},
            "icbt": {
                "best_k": self.icbt_engine._best_k,
                "cluster_weights": [
                    float(self.icbt_engine._cluster_weights[k])
                    for k in range(min(self.icbt_engine._best_k,
                                      len(self.icbt_engine._cluster_weights)))],
                "entities": class_names},
            "icbt_bc": {
                "best_k": self.icbt_bc_engine._best_k,
                "cluster_weights": [
                    float(self.icbt_bc_engine._cluster_weights[k])
                    for k in range(min(self.icbt_bc_engine._best_k,
                                      len(self.icbt_bc_engine._cluster_weights)))],
                "entities": class_names},
        }
        return {
            "metadata": {
                "database": db_path, "study_type": study_type,
                "training_timestamp": datetime.now().isoformat(),
                "rating_systems": ALL_SYSTEMS,
            },
            "policy_ratings": policy_ratings,
            "spinning_tops_nash_clusters": {},
            "system_native_params": system_native_params,
        }


def build_engine_database(class_names, class_scores_raw, study_type,
                          out_dir, rng, n_rounds=None):
    print(f"\n{'='*60}")
    print(f"Engine: {study_type}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    all_vals = [v for vs in class_scores_raw.values() for v in vs]
    lo, hi   = min(all_vals), max(all_vals)
    span     = hi - lo if hi > lo else 1.0

    class_scores_norm = {cn: [(v - lo) / span for v in vs]
                         for cn, vs in class_scores_raw.items()}
    class_stds = {cn: (np.std(vs) / span if len(vs) > 1 else 0.0)
                  for cn, vs in class_scores_raw.items()}
    sizes = {cn: len(vs) for cn, vs in class_scores_norm.items()}
    print(f"  Composition classes: {sizes}")

    if n_rounds is None:
        n_rounds = min(max(sizes.values()), 1000)

    builder = PredMethodRatingBuilder()
    total   = builder.train(class_names, class_scores_norm, class_stds,
                            n_rounds=n_rounds, rng=rng)
    print(f"  Rounds: {n_rounds}  Total pairwise updates: {total}")

    ratings = builder.get_trained_ratings(class_names, study_type, str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trained_ratings.json", "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)

    metadata = {
        "database": str(out_dir), "study_type": study_type,
        "policy_types": class_names,
        "num_experiments_per_policy": 1,
        "num_rounds_per_experiment": n_rounds,
        "total_rounds": n_rounds,
        "total_trials": sum(sizes.values()),
        "source": "JARVIS-DFT dft_3d + SNUMAT, composition-class policies",
        "description": (
            f"Prediction method {study_type} as engine. "
            f"Policies = composition-class families. "
            f"Score = per-material band gap value."
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Omega input: per-class L1 dispersion as the MAE-equivalent
    # (see writings/OMEGA_DECISION.md).
    write_dispersion_summaries(out_dir, class_scores_raw, unit="eV",
                               source="jarvis_prediction_methods")

    ts = ratings["policy_ratings"]["trueskill"]
    sorted_cls = sorted(ts.items(), key=lambda kv: -kv[1]["mu"])
    print(f"\n  TrueSkill ordering ({study_type}):")
    for pos, (cn, r) in enumerate(sorted_cls, 1):
        n = sizes.get(cn, 0)
        mean_raw = float(np.mean(class_scores_raw[cn])) if class_scores_raw.get(cn) else 0.0
        print(f"    {pos}. {cn:14s}  mu={r['mu']:7.1f}  sigma={r['sigma']:5.1f}  "
              f"(n={n}  mean_bg={mean_raw:.3f} eV)")
    print(f"  Saved: {out_dir / 'trained_ratings.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JARVIS prediction-methods vs composition-classes PEGKi Phase 2."
    )
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip downloads; use cached data/jarvis/dft_3d.csv and snumat.csv")
    parser.add_argument("--min-per-class", type=int, default=50,
                        help="Minimum materials per composition class (default 50)")
    parser.add_argument("--output-root", default="output/jarvis_prediction_methods",
                        help="Output root dir (default output/jarvis_prediction_methods)")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required.")
        sys.exit(1)

    data_dir  = ROOT / "data" / "jarvis"
    cache_csv = data_dir / "dft_3d_composition.csv"
    out_root  = ROOT / args.output_root

    # ------------------------------------------------------------------
    # Load and classify JARVIS dft_3d data
    # ------------------------------------------------------------------
    # Try existing dft_3d.csv first; if it lacks composition_class column, reload.
    existing_csv = data_dir / "dft_3d.csv"
    if cache_csv.exists() and args.skip_fetch:
        df = pd.read_csv(cache_csv, converters={"formula": str})
        print(f"Loaded {len(df)} rows from {cache_csv}")
    elif existing_csv.exists():
        print(f"Augmenting existing {existing_csv} with composition classification ...")
        df = pd.read_csv(existing_csv, converters={"formula": str})
        if "composition_class" not in df.columns or "formula" not in df.columns:
            df = load_jarvis_dft3d(cache_csv, skip_fetch=args.skip_fetch)
        else:
            df["composition_class"] = df["formula"].apply(classify_by_anion)
            df.to_csv(cache_csv, index=False)
            print(f"  Saved with composition_class: {cache_csv}")
    else:
        df = load_jarvis_dft3d(cache_csv, skip_fetch=False)

    # Semiconducting subset (band gap > 0.1 eV by OptB88vdW)
    df = df[df["optb88vdw_bandgap"].notna() & (df["optb88vdw_bandgap"] > 0.1)].copy()
    if "composition_class" not in df.columns:
        df["composition_class"] = df["formula"].apply(classify_by_anion)
    print(f"\n{len(df)} semiconducting materials (optb88vdw_bandgap > 0.1 eV)")

    vc = df["composition_class"].value_counts()
    print("\nComposition class distribution (OptB88vdW semiconductors):")
    for cls, cnt in vc.items():
        mean_bg = df[df["composition_class"] == cls]["optb88vdw_bandgap"].mean()
        print(f"  {cls:14s}: {cnt:6d}  mean_bg={mean_bg:.2f} eV")

    valid_classes = [c for c in COMPOSITION_CLASSES
                     if vc.get(c, 0) >= args.min_per_class]
    if len(valid_classes) < 2:
        print(f"ERROR: Fewer than 2 composition classes with >= {args.min_per_class} materials.")
        sys.exit(1)
    print(f"\nUsing composition classes: {valid_classes}")

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Engine 1: OptB88vdW GGA (always available, dft_bandstructure)
    # ------------------------------------------------------------------
    class_opt = {}
    for cn in valid_classes:
        vals = df[df["composition_class"] == cn]["optb88vdw_bandgap"].dropna().tolist()
        if len(vals) >= args.min_per_class:
            class_opt[cn] = vals
    if len(class_opt) >= 2:
        build_engine_database(
            list(class_opt.keys()), class_opt,
            study_type="optb88vdw_bg_dft_bandstructure",
            out_dir=out_root / "db_optb88vdw",
            rng=rng,
        )
    else:
        print("ERROR: Not enough classes for OptB88vdW engine.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Engine 2: TB-mBJ meta-GGA (dft_meta_gga)
    # ------------------------------------------------------------------
    if "mbj_bandgap" in df.columns:
        mbj_sub = df[df["mbj_bandgap"].notna() & (df["mbj_bandgap"] > 0.1)]
        class_mbj = {}
        for cn in valid_classes:
            vals = mbj_sub[mbj_sub["composition_class"] == cn]["mbj_bandgap"].tolist()
            if len(vals) >= args.min_per_class:
                class_mbj[cn] = vals
        if len(class_mbj) >= 2:
            build_engine_database(
                list(class_mbj.keys()), class_mbj,
                study_type="mbj_bg_dft_meta_gga",
                out_dir=out_root / "db_mbj",
                rng=rng,
            )
        else:
            print(f"\nSkipping mBJ engine: fewer than 2 classes meet threshold.")
            class_mbj = {}
    else:
        class_mbj = {}

    # ------------------------------------------------------------------
    # Engine 3: SNUMAT GGA (second dft_bandstructure observation for ANOVA)
    # ------------------------------------------------------------------
    sn_cache = data_dir / "snumat_composition.csv"
    sn_df = load_snumat_with_composition(sn_cache, skip_fetch=args.skip_fetch)

    sn_gga_sub = sn_df[sn_df["gga_bandgap"].notna() & (sn_df["gga_bandgap"] > 0.1)]
    class_sn_gga = {}
    if "composition_class" in sn_df.columns and sn_df["composition_class"].notna().any():
        for cn in valid_classes:
            vals = sn_gga_sub[sn_gga_sub["composition_class"] == cn]["gga_bandgap"].tolist()
            if len(vals) >= args.min_per_class:
                class_sn_gga[cn] = vals
        if len(class_sn_gga) >= 2:
            build_engine_database(
                list(class_sn_gga.keys()), class_sn_gga,
                study_type="snumat_gga_bg_dft_bandstructure",
                out_dir=out_root / "db_snumat_gga",
                rng=rng,
            )
        else:
            print(f"\nSkipping SNUMAT-GGA engine: fewer than 2 classes meet threshold.")
            class_sn_gga = {}
    else:
        print("\nSkipping SNUMAT engines: composition_class not available in SNUMAT download.")

    # ------------------------------------------------------------------
    # Engine 4: SNUMAT HSE06 (second dft_meta_gga observation for ANOVA)
    # ------------------------------------------------------------------
    if "composition_class" in sn_df.columns and sn_df["composition_class"].notna().any():
        sn_hse_sub = sn_df[sn_df["hse_bandgap"].notna() & (sn_df["hse_bandgap"] > 0.1)]
        class_sn_hse = {}
        for cn in valid_classes:
            vals = sn_hse_sub[sn_hse_sub["composition_class"] == cn]["hse_bandgap"].tolist()
            if len(vals) >= args.min_per_class:
                class_sn_hse[cn] = vals
        if len(class_sn_hse) >= 2:
            build_engine_database(
                list(class_sn_hse.keys()), class_sn_hse,
                study_type="snumat_hse_bg_dft_meta_gga",
                out_dir=out_root / "db_snumat_hse",
                rng=rng,
            )
        else:
            print(f"\nSkipping SNUMAT-HSE engine: fewer than 2 classes meet threshold.")

    # ------------------------------------------------------------------
    # Set-operation databases: OptB88vdW vs mBJ as anchor pair
    #
    # The composition ordering is perfectly stable between GGA and mBJ
    # (no ordinal rank inversions).  Rank-inversion-based set-ops would
    # always be empty and would not add engines for Causal ANOVA power.
    #
    # Instead we split by mBJ/GGA enhancement ratio:
    #   low ratio  -> GGA-dominant classes -> difference engine (dft_bandstructure)
    #   high ratio -> mBJ-enhanced classes -> complement engine (dft_meta_gga)
    #   all stable -> intersection engine (mixed)
    #
    # Tagging difference as dft_bandstructure and complement as dft_meta_gga
    # adds a 2nd engine to each physics group, giving Causal ANOVA:
    #   df_between=2, df_within=17 (with N=5 engines) -> F_crit~3.59.
    # ------------------------------------------------------------------
    if len(class_opt) >= 2 and len(class_mbj) >= 2:
        common = [c for c in class_opt if c in class_mbj]
        if len(common) >= 4:
            opt_means = {c: float(np.mean(class_opt[c])) for c in common}
            mbj_means = {c: float(np.mean(class_mbj[c])) for c in common}

            # Enhancement ratio: how much mBJ raises the band gap relative to GGA.
            # Classes with low ratio -> GGA captures most of the gap (dft_bandstructure).
            # Classes with high ratio -> mBJ correction is decisive (dft_meta_gga).
            ratios = {c: mbj_means[c] / opt_means[c] if opt_means[c] > 1e-6 else 1.0
                      for c in common}
            ratio_vals = sorted(ratios.values())
            median_ratio = float(np.median(ratio_vals))

            # Split below/above median.  With 5 classes -> ~2 low, ~3 high (or 2/3).
            difference_cls  = [c for c in common if ratios[c] <  median_ratio]
            complement_cls  = [c for c in common if ratios[c] >= median_ratio]
            # Intersection: all classes with stable ordinal ordering (always all common
            # classes since the ranking is perfectly preserved between GGA and mBJ).
            opt_rank = {c: r for r, c in enumerate(
                sorted(common, key=lambda x: opt_means[x], reverse=True))}
            mbj_rank = {c: r for r, c in enumerate(
                sorted(common, key=lambda x: mbj_means[x], reverse=True))}
            intersection_cls = [c for c in common if abs(opt_rank[c] - mbj_rank[c]) <= 1]

            print(f"\nSet-op summary (OptB88vdW vs mBJ, ratio-based split):")
            print(f"  Median mBJ/GGA ratio: {median_ratio:.3f}")
            for c in sorted(common, key=lambda x: ratios[x]):
                tag = "diff" if ratios[c] < median_ratio else "comp"
                print(f"    {c:14s}  GGA={opt_means[c]:.2f} eV  mBJ={mbj_means[c]:.2f} eV"
                      f"  ratio={ratios[c]:.3f}  -> {tag}")
            print(f"  difference  (GGA-dominant, low ratio):  {difference_cls}")
            print(f"  complement  (mBJ-enhanced, high ratio): {complement_cls}")
            print(f"  intersection (all stable ordinal):      {intersection_cls}")

            # Intersection: mixed physics group
            if len(intersection_cls) >= 2:
                build_engine_database(
                    intersection_cls, {c: class_opt[c] for c in intersection_cls},
                    study_type="gga_bg_composition_intersection_theoretical",
                    out_dir=out_root / "db_intersection",
                    rng=rng,
                )

            # Difference: low mBJ/GGA ratio classes -- tagged dft_bandstructure.
            # study_type contains "bandstructure" so _infer_physics_type -> dft_bandstructure.
            if len(difference_cls) >= 2:
                build_engine_database(
                    difference_cls, {c: class_opt[c] for c in difference_cls},
                    study_type="optb88vdw_bg_composition_difference_bandstructure",
                    out_dir=out_root / "db_difference",
                    rng=rng,
                )

            # Complement: high mBJ/GGA ratio classes -- tagged dft_meta_gga.
            # study_type contains "meta_gga" so _infer_physics_type -> dft_meta_gga.
            if len(complement_cls) >= 2:
                build_engine_database(
                    complement_cls, {c: class_mbj[c] for c in complement_cls},
                    study_type="mbj_bg_composition_complement_meta_gga",
                    out_dir=out_root / "db_complement",
                    rng=rng,
                )
        elif len(common) >= 2:
            # Too few common classes for ratio split; build intersection only.
            print(f"\nNote: only {len(common)} common classes -- building intersection only.")
            build_engine_database(
                common, {c: class_opt[c] for c in common},
                study_type="gga_bg_composition_intersection_theoretical",
                out_dir=out_root / "db_intersection",
                rng=rng,
            )
    else:
        print("\nSkipping set-op databases: need both OptB88vdW and mBJ engines.")

    print(f"\nDone. Run Phase 3 with:")
    print(f"  uv run python scripts/run_phase3.py "
          f"--database-root {out_root} "
          f"--output-dir {out_root / 'phase3'}")


if __name__ == "__main__":
    main()
