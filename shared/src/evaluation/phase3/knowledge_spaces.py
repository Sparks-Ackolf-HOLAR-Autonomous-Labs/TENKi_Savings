"""
Knowledge Space Definitions -- Section 10.1

Implements K_T (theoretical), K_E (empirical), K_H (hybrid), K_H_OVERLAP (validation)
knowledge spaces from policy database trained ratings.
"""

from enum import Enum
from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class KnowledgeSpaceType(Enum):
    K_T = "theoretical"
    K_E = "empirical"
    K_H = "hybrid"
    K_H_OVERLAP = "hybrid_overlap"


# Known study types -- extend here when new engines are added.
# Unknown study types are handled by _infer_ks_type().
STUDY_TYPE_TO_KS = {
    # Original 4
    'uv_vis_only':    KnowledgeSpaceType.K_T,   # colors only UV-vis physics can produce
    'mixbox':         KnowledgeSpaceType.K_E,   # K-M pigment model (empirical)
    'spectral_gamut': KnowledgeSpaceType.K_H,   # physics + simulation (hybrid)
    'overlap_gamut':  KnowledgeSpaceType.K_H_OVERLAP,  # consensus region (validation)
    # New engines
    'km_paint':       KnowledgeSpaceType.K_E,   # measured OSF oil pigment K/S data
    'ryb_paint':      KnowledgeSpaceType.K_T,   # artist RYB color-theory model
    # Single-engine aliases (used when output dir is db_spectral, db_km, etc.)
    'spectral':       KnowledgeSpaceType.K_H,
    'km':             KnowledgeSpaceType.K_E,
    'ryb':            KnowledgeSpaceType.K_T,
    'mixbox_engine':  KnowledgeSpaceType.K_E,
}


def _infer_ks_type(study_type: str) -> KnowledgeSpaceType:
    """Infer KnowledgeSpaceType from study_type string when not in STUDY_TYPE_TO_KS.

    Rules (applied in order, first match wins):
      - contains 'intersection', 'overlap', or 'consensus'  -> K_H_OVERLAP
      - contains 'difference', 'complement', 'minus',
        'exclusive', 'only', or 'uv_vis'                   -> K_T
      - contains 'union'                                    -> K_H
      - contains 'spectral', 'physics', or 'hybrid'        -> K_H
      - contains 'empirical', 'paint', 'pigment', or 'km'  -> K_E
      - contains 'theoretical', 'ryb', 'artist', 'theory'  -> K_T
      - default                                             -> K_H
    """
    s = study_type.lower()
    if any(k in s for k in ('intersection', 'overlap', 'consensus')):
        return KnowledgeSpaceType.K_H_OVERLAP
    if any(k in s for k in ('difference', 'complement', 'minus', 'exclusive', 'only', 'uv_vis')):
        return KnowledgeSpaceType.K_T
    if 'union' in s:
        return KnowledgeSpaceType.K_H
    if any(k in s for k in ('spectral', 'physics', 'hybrid')):
        return KnowledgeSpaceType.K_H
    if any(k in s for k in ('empirical', 'paint', 'pigment', '_km', 'kubelka')):
        return KnowledgeSpaceType.K_E
    if any(k in s for k in ('theoretical', 'ryb', 'artist', 'theory')):
        return KnowledgeSpaceType.K_T
    return KnowledgeSpaceType.K_H


def _ks_type_for(study_type: str) -> KnowledgeSpaceType:
    """Look up or infer KnowledgeSpaceType for a study_type string."""
    return STUDY_TYPE_TO_KS.get(study_type, _infer_ks_type(study_type))


@dataclass
class KnowledgeSpace:
    """A single knowledge space derived from one policy database study type."""
    space_type: KnowledgeSpaceType
    study_type: str
    policy_mus: Dict[str, float]       # TrueSkill mu per policy
    policy_sigmas: Dict[str, float]    # TrueSkill sigma per policy (uncertainty weight in rho)
    policy_ranks: Dict[str, int]       # rank 1 = best; backward compat
    policy_maes: Dict[str, float]      # mean best_color_distance per policy (omega)
    policy_mae_stds: Dict[str, float]  # std of best_color_distance per policy
    policy_stat_mus: Dict[str, float]  # Statistical mu -- legacy fallback for sigma/rho
    policy_bc_mus: Dict[str, float]    # BC mu -- legacy fallback for bc_spectral
    # Per-experiment MAE data (loaded from round files when available).
    policy_exp_maes: Dict[str, List[float]] = field(default_factory=dict)
    engine_exp_maes: List[float] = field(default_factory=list)
    # Native params from new Phase 2 format (replacing fake mu/sigma projections).
    # policy_s_transitive[policy][obj] = Hodge potential f_i (Statistical model).
    # Used in transfer equation sigma/rho (pairwise differences, not log-ratios).
    policy_s_transitive: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # policy_context_bias[policy][obj] = gamma_i (residual context bias, Statistical).
    # Used in context bias portability analysis.
    policy_context_bias: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # policy_blade_vectors[policy][obj] = blade vector (list of floats).
    # policy_chest_vectors[policy][obj] = chest vector (list of floats).
    # Used in BC pairwise score matrix for spectral analysis.
    policy_blade_vectors: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    policy_chest_vectors: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # policy_f_transitive[policy][obj] = Hodge potential f_i (Spinning Tops).
    policy_f_transitive: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # system_native_params: system-level params not per-policy.
    # Keys: "spinning_tops" -> {cyclic: {obj: {entity_order, matrix}}},
    #       "icbt"/"icbt_bc" -> {best_k, cluster_weights, entities}.
    system_native_params: Dict = field(default_factory=dict)
    # policy_log_pi[policy][str(k)] = log_pi value under cluster k (icbt_bc preferred).
    # Used in ICBT cluster membership portability analysis.
    policy_log_pi: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # policy_train_times[policy] = mean duration_seconds per round (from Phase 1 round files).
    # Used in training cost / cost-adjusted transfer delta analysis.
    policy_train_times: Dict[str, float] = field(default_factory=dict)


def _strip_dir_prefix(name: str) -> str:
    """Strip known directory prefixes to get a canonical study_type key."""
    for prefix in ('policy_database_', 'db_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_knowledge_spaces(database_root: str = "output") -> Dict[str, KnowledgeSpace]:
    """Scan database_root for any subdirectory containing trained_ratings.json.

    Accepts any naming convention (policy_database_*, db_*, or bare study_type
    names) so Phase 3 works regardless of what --output was passed to Phase 2.

    The canonical study_type key is taken from metadata.json if present;
    otherwise the directory name is used (with common prefixes stripped).

    Returns {study_type: KnowledgeSpace}.
    """
    root = Path(database_root)
    spaces = {}

    for ratings_file in sorted(root.glob("*/trained_ratings.json")):
        db_dir = ratings_file.parent
        meta_file = db_dir / "metadata.json"

        # Derive study_type: prefer metadata, fall back to stripped dir name
        meta_study_type = None
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            meta_study_type = meta.get("study_type") or None

        dir_study_type = _strip_dir_prefix(db_dir.name)

        # Use metadata value only if it is more specific than the directory name.
        # A metadata value of 'spectral_gamut' on a dir called 'db_study_a_...'
        # is the schema default that was never overridden -- prefer the dir name.
        _DEFAULT_META = "spectral_gamut"
        if meta_study_type and meta_study_type != dir_study_type:
            if meta_study_type == _DEFAULT_META and dir_study_type != _DEFAULT_META:
                # Metadata was never explicitly set; dir name is the real key.
                study_type = dir_study_type
            elif meta_study_type in STUDY_TYPE_TO_KS or len(meta_study_type) >= len(dir_study_type):
                # Metadata is a known specific type or longer (more descriptive) than dir.
                study_type = meta_study_type
            else:
                study_type = dir_study_type
        else:
            study_type = dir_study_type

        with open(ratings_file, encoding="utf-8") as f:
            ratings = json.load(f)

        ts_ratings = ratings.get("policy_ratings", {}).get("trueskill", {})
        if not ts_ratings:
            continue

        mus = {p: v["mu"] for p, v in ts_ratings.items()}
        sigmas = {p: v["sigma"] for p, v in ts_ratings.items()}

        sorted_policies = sorted(mus, key=lambda p: mus[p], reverse=True)
        ranks = {p: i + 1 for i, p in enumerate(sorted_policies)}

        # Load per-policy MAE mean and std from summaries.json (used in omega).
        maes: Dict[str, float] = {}
        mae_stds: Dict[str, float] = {}
        summaries_file = db_dir / "summaries.json"
        if summaries_file.exists():
            with open(summaries_file, encoding="utf-8") as f:
                summaries = json.load(f)
            if isinstance(summaries, list):
                for entry in summaries:
                    pt = entry.get("policy_type")
                    stats = entry.get("overall_stats", {})
                    mae = stats.get("best_color_distance_mean")
                    mae_std = stats.get("best_color_distance_std")
                    if pt and mae is not None:
                        maes[pt] = float(mae)
                    if pt and mae_std is not None:
                        mae_stds[pt] = float(mae_std)

        # --- Statistical: legacy scalar + native Hodge params ---
        stat_mus: Dict[str, float] = {}
        bc_mus: Dict[str, float] = {}
        policy_s_transitive: Dict[str, Dict[str, float]] = {}
        policy_context_bias: Dict[str, Dict[str, float]] = {}
        stat_ratings = ratings.get("policy_ratings", {}).get("statistical", {})
        for p, v in stat_ratings.items():
            if "mu" in v:
                stat_mus[p] = float(v["mu"])
            if "s_transitive" in v:
                policy_s_transitive[p] = {k: float(f) for k, f in v["s_transitive"].items()}
            if "context_bias" in v:
                policy_context_bias[p] = {k: float(g) for k, g in v["context_bias"].items()}

        # --- Blade-Chest: legacy scalar + native blade/chest vectors ---
        policy_blade_vectors: Dict[str, Dict[str, List[float]]] = {}
        policy_chest_vectors: Dict[str, Dict[str, List[float]]] = {}
        bc_ratings = ratings.get("policy_ratings", {}).get("blade_chest", {})
        for p, v in bc_ratings.items():
            if "mu" in v:
                bc_mus[p] = float(v["mu"])
            if "blade_vectors" in v:
                policy_blade_vectors[p] = {k: list(vec) for k, vec in v["blade_vectors"].items()}
            if "chest_vectors" in v:
                policy_chest_vectors[p] = {k: list(vec) for k, vec in v["chest_vectors"].items()}

        # --- Spinning Tops: f_transitive per policy ---
        policy_f_transitive: Dict[str, Dict[str, float]] = {}
        spinning_tops_ratings = ratings.get("policy_ratings", {}).get("spinning_tops", {})
        for p, v in spinning_tops_ratings.items():
            if "f_transitive" in v:
                policy_f_transitive[p] = {k: float(f) for k, f in v["f_transitive"].items()}

        # --- ICBT per-policy log_pi (icbt_bc preferred over icbt) ---
        policy_log_pi: Dict[str, Dict[str, float]] = {}
        for sys_key in ("icbt_bc", "icbt"):
            icbt_ratings = ratings.get("policy_ratings", {}).get(sys_key, {})
            if icbt_ratings:
                for p, v in icbt_ratings.items():
                    lp = v.get("log_pi_per_cluster", {})
                    if lp:
                        policy_log_pi[p] = {str(k): float(val) for k, val in lp.items()}
                break  # stop after first non-empty source

        # --- System-level native params (cyclic matrices, ICBT cluster data) ---
        system_native_params: Dict = ratings.get("system_native_params", {})

        # Load per-experiment MAE from round files.
        # Each experiment = one target color (shared across all engines).
        # Accumulate per policy: {policy: {exp_id: best_color_distance}}
        policy_exp_maes: Dict[str, List[float]] = {}
        engine_exp_maes: List[float] = []
        policy_train_times: Dict[str, float] = {}
        policies_dir = db_dir / "policies"
        if policies_dir.exists():
            exp_accum: Dict[str, Dict[int, float]] = {}  # {policy: {exp_id: mae}}
            dur_accum: Dict[str, List[float]] = {}       # {policy: [duration_seconds, ...]}
            for policy_dir in sorted(policies_dir.iterdir()):
                if not policy_dir.is_dir():
                    continue
                pname = policy_dir.name
                exp_accum[pname] = {}
                dur_accum[pname] = []
                for exp_dir in sorted(policy_dir.glob("experiment_*")):
                    round_file = exp_dir / "round_000.json"
                    if not round_file.exists():
                        continue
                    try:
                        with open(round_file, encoding="utf-8") as rf:
                            rd = json.load(rf)
                        exp_id = int(rd.get("experiment_id", exp_dir.name.replace("experiment_", "")))
                        mae = rd.get("best_color_distance")
                        if mae is not None:
                            exp_accum[pname][exp_id] = float(mae)
                        dur = rd.get("duration_seconds")
                        if dur is not None:
                            dur_accum[pname].append(float(dur))
                    except Exception:
                        pass
            # Convert to ordered lists and build per-engine mean
            all_exp_ids = sorted(set(
                eid for pd in exp_accum.values() for eid in pd.keys()
            ))
            for pname, exp_dict in exp_accum.items():
                policy_exp_maes[pname] = [exp_dict.get(eid, float("nan")) for eid in all_exp_ids]
            # Per-experiment mean across all policies
            if all_exp_ids and policy_exp_maes:
                arr = np.array([policy_exp_maes[p] for p in policy_exp_maes], dtype=float)
                engine_exp_maes = list(np.nanmean(arr, axis=0))
            # Mean training time per policy (across all experiments in this DB)
            for pname, durs in dur_accum.items():
                if durs:
                    policy_train_times[pname] = float(np.mean(durs))

        spaces[study_type] = KnowledgeSpace(
            space_type=_ks_type_for(study_type),
            study_type=study_type,
            policy_mus=mus,
            policy_sigmas=sigmas,
            policy_ranks=ranks,
            policy_maes=maes,
            policy_mae_stds=mae_stds,
            policy_stat_mus=stat_mus,
            policy_bc_mus=bc_mus,
            policy_exp_maes=policy_exp_maes,
            engine_exp_maes=engine_exp_maes,
            policy_s_transitive=policy_s_transitive,
            policy_context_bias=policy_context_bias,
            policy_blade_vectors=policy_blade_vectors,
            policy_chest_vectors=policy_chest_vectors,
            policy_f_transitive=policy_f_transitive,
            system_native_params=system_native_params,
            policy_log_pi=policy_log_pi,
            policy_train_times=policy_train_times,
        )

    return spaces


def build_performance_vectors(
    spaces: Dict[str, KnowledgeSpace],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Build an N-dimensional performance vector per policy from all available study types.

    Dimensions = sorted(spaces.keys()) so the order is deterministic and the
    caller always knows what each dimension means via the returned labels list.

    Policies missing from a study type receive that study type's mean mu.

    Returns:
        vectors:    {policy_name: np.ndarray shape(N,)}
        dimensions: list[str] -- study_type label for each dimension
    """
    dimensions: List[str] = sorted(spaces.keys())

    all_policies: set = set()
    for ks in spaces.values():
        all_policies.update(ks.policy_mus.keys())

    vectors: Dict[str, np.ndarray] = {}
    for policy in sorted(all_policies):
        vec = []
        for st in dimensions:
            ks = spaces[st]
            mean_mu = float(np.mean(list(ks.policy_mus.values())))
            vec.append(ks.policy_mus.get(policy, mean_mu))
        vectors[policy] = np.array(vec, dtype=float)

    return vectors, dimensions
