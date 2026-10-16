"""
General Transfer Equation -- Sections 10.3-4

All components are based on within-engine pairwise comparisons (log-ratios between
policies). Absolute performance values are never compared across engines because
each engine's MAE scale is local to its own physics and target distribution.
Only ratios between policies within the same engine are cross-engine comparable.

    - sigma(K_s -> K_t): Pearson correlation of pairwise comparison vectors.
      PRIMARY signal: policy_s_transitive (Hodge potential from the
      Statistical model; captures transitive skill and, via s_transitive's
      construction, the intransitive Blade-Chest component). Potentials are
      linear-scale, so pairwise DIFFERENCES s_j - s_i are used.
      FALLBACK (legacy DBs only): positive policy_stat_mus via pairwise
      log-ratios log(stat_mu_j / stat_mu_i).
      High sigma = same policies dominate by similar margins in both.
      Weighted by source reliability = 1/(1 + mean_ts_sigma_s): lower TrueSkill
      sigma = tighter comparison estimates = more trustworthy transfer signal.
      Directional: reliability depends only on source.

    - mu: REMOVED. Absolute performance is engine-local and cannot be compared
      across databases. beta is fixed at 0 and mu always returns 0.

    - rho(K_s -> K_t): OLS slope of log_ratio_target ~ log_ratio_source,
      using the same Statistical mu log-ratio vectors as sigma.
      slope~1 means ratio gaps transfer at full magnitude (ideal).
      slope<1 means source over-discriminates (ratios compress in target).
      slope>1 means source under-discriminates (ratios amplify in target).
      Mapped to [0,1]: 1/(1+|slope-1|). Weighted by TrueSkill sigma_t/(sigma_s+sigma_t)
      for directionality: TrueSkill sigma is a valid relative uncertainty measure
      across engines because all databases share the same prior.

    - omega(K_s -> K_t): mean positive log-ratio of relative standing degradation.
      relative(p, engine) = MAE_p / mean_MAE_engine (within-engine comparison).
      For each policy: log(relative_t / relative_s) if positive (degradation only).
      Mean over all policies gives a continuous, threshold-free overhead measure.
      Higher omega = on average, policies that were relatively good in source
      become relatively worse in target.

    - Delta(K_s -> K_t) = alpha*sigma + gamma*rho - delta*omega
      (beta term removed; mu always 0)
    - Conjecture verification (2.1 Asymmetry, 2.2 Hybrid Superiority)
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional, Tuple

from .knowledge_spaces import KnowledgeSpace


def _mean_s_transitive(policy_s_transitive: Dict) -> Dict[str, float]:
    """Average s_transitive (Hodge potential) across objectives per policy.

    Returns {policy: mean_f} for policies that have at least one objective value.
    """
    result = {}
    for p, obj_dict in policy_s_transitive.items():
        vals = [v for v in obj_dict.values() if v is not None]
        if vals:
            result[p] = float(np.mean(vals))
    return result


def _diff_vectors(vals_s: Dict[str, float], vals_t: Dict[str, float]) -> tuple:
    """Build pairwise difference vectors for common policies.

    For every ordered pair (i,j) with i!=j: val_j - val_i in source and target.
    Appropriate for linear-scale Hodge potentials (unlike log-ratios, works for
    zero and negative values).
    Returns (arr_s, arr_t, common) or (None, None, []) if insufficient data.
    """
    common = sorted(set(vals_s) & set(vals_t))
    if len(common) < 2:
        return None, None, common
    diff_s, diff_t = [], []
    for p_i in common:
        for p_j in common:
            if p_i == p_j:
                continue
            diff_s.append(vals_s[p_j] - vals_s[p_i])
            diff_t.append(vals_t[p_j] - vals_t[p_i])
    return np.array(diff_s), np.array(diff_t), common


def _log_ratio_vectors(vals_s: Dict, vals_t: Dict) -> tuple:
    """Build pairwise log-ratio vectors for common policies.

    For every ordered pair (i,j) with i!=j: log(val_j/val_i) in source and target.
    Returns (arr_s, arr_t, common) or (None, None, []) if not computable.
    All values must be positive.
    """
    common = sorted(set(vals_s) & set(vals_t))
    if len(common) < 2:
        return None, None, common
    if any(vals_s[p] <= 0 for p in common) or any(vals_t[p] <= 0 for p in common):
        return None, None, common
    log_r_s, log_r_t = [], []
    for p_i in common:
        for p_j in common:
            if p_i == p_j:
                continue
            log_r_s.append(np.log(vals_s[p_j] / vals_s[p_i]))
            log_r_t.append(np.log(vals_t[p_j] / vals_t[p_i]))
    return np.array(log_r_s), np.array(log_r_t), common


def compute_structural_compatibility(ks_s: KnowledgeSpace, ks_t: KnowledgeSpace) -> float:
    """sigma(K_s -> K_t): Pearson correlation of pairwise comparison vectors.

    Primary signal: policy_s_transitive (Hodge potential from the Statistical model).
    Hodge potentials are linear-scale, so we use pairwise differences s_j - s_i
    rather than log-ratios. This is the natural operation for additive potentials
    and works correctly for zero/negative values.

    Fallback: policy_stat_mus (legacy format, positive scalars, log-ratios).

    High sigma = the same policies dominate others by similar margins in both
    source and target, including the intransitive Blade-Chest component encoded
    in s_transitive.

    Reliability weight = 1/(1 + mean_ts_sigma_s): lower TrueSkill sigma means
    tighter comparison estimates and more trustworthy transfer signal.
    """
    # Prefer native s_transitive (no artificial scaling)
    s_mean_s = _mean_s_transitive(ks_s.policy_s_transitive)
    s_mean_t = _mean_s_transitive(ks_t.policy_s_transitive)
    if s_mean_s and s_mean_t:
        arr_s, arr_t, common = _diff_vectors(s_mean_s, s_mean_t)
    else:
        arr_s, arr_t, common = _log_ratio_vectors(ks_s.policy_stat_mus, ks_t.policy_stat_mus)

    if arr_s is None or np.std(arr_s) < 1e-10 or np.std(arr_t) < 1e-10:
        return 0.0
    r, _ = pearsonr(arr_s, arr_t)
    if np.isnan(r):
        return 0.0

    ts_sigmas = [ks_s.policy_sigmas[p] for p in common if p in ks_s.policy_sigmas]
    mean_sigma_s = float(np.mean(ts_sigmas)) if ts_sigmas else 1.0
    return float(r) / (1.0 + mean_sigma_s)


def compute_source_quality(
    ks_s: KnowledgeSpace,
    all_spaces: Optional[Dict[str, KnowledgeSpace]] = None,
) -> float:
    """mu: REMOVED. Always returns 0.

    Absolute performance is engine-local -- each engine's MAE scale depends on
    its own physics, gamut, and target distribution and cannot be meaningfully
    compared across engines. The beta weight is fixed at 0 in compute_transfer_delta.
    """
    return 0.0


def compute_mapping_efficiency(ks_s: KnowledgeSpace, ks_t: KnowledgeSpace) -> float:
    """rho(K_s -> K_t): OLS slope of target_diffs ~ source_diffs.

    Uses the same comparison vectors as sigma (s_transitive differences, or
    stat_mus log-ratios as fallback). OLS slope measures whether gap magnitudes
    transfer at full scale:
        slope ~ 1  => source gaps predict target gaps 1:1 (ideal)
        slope < 1  => source over-discriminates (gaps compress in target)
        slope > 1  => source under-discriminates (gaps amplify in target)

    Mapped to [0,1] via 1/(1+|slope-1|).
    Weighted by TrueSkill sigma_t/(sigma_s+sigma_t) for directionality.
    """
    s_mean_s = _mean_s_transitive(ks_s.policy_s_transitive)
    s_mean_t = _mean_s_transitive(ks_t.policy_s_transitive)
    if s_mean_s and s_mean_t:
        arr_s, arr_t, common = _diff_vectors(s_mean_s, s_mean_t)
    else:
        arr_s, arr_t, common = _log_ratio_vectors(ks_s.policy_stat_mus, ks_t.policy_stat_mus)

    if arr_s is None or np.std(arr_s) < 1e-10:
        return 0.5

    # OLS through origin (log-ratio vectors are antisymmetric so intercept=0)
    slope = float(np.dot(arr_s, arr_t) / np.dot(arr_s, arr_s))
    rho_raw = 1.0 / (1.0 + abs(slope - 1.0))

    ts_sigmas_s = [ks_s.policy_sigmas[p] for p in common if p in ks_s.policy_sigmas]
    ts_sigmas_t = [ks_t.policy_sigmas[p] for p in common if p in ks_t.policy_sigmas]
    mean_sigma_s = float(np.mean(ts_sigmas_s)) if ts_sigmas_s else 1.0
    mean_sigma_t = float(np.mean(ts_sigmas_t)) if ts_sigmas_t else 1.0
    denom = mean_sigma_s + mean_sigma_t
    direction_weight = mean_sigma_t / denom if denom > 1e-10 else 0.5

    return float(rho_raw * direction_weight)


def compute_transfer_overhead(ks_s: KnowledgeSpace, ks_t: KnowledgeSpace) -> float:
    """omega(K_s -> K_t): mean positive log-ratio of relative standing degradation.

    Each policy's MAE is normalised by its engine's mean (within-engine comparison),
    removing the absolute scale difference between engines:

        relative(p, engine) = MAE_p / mean_MAE_engine

    For each policy, the change in log-relative-standing going source -> target:

        delta_log(p) = log(relative_t(p) / relative_s(p))
                     = log(MAE_p_t/mean_t) - log(MAE_p_s/mean_s)

    Only positive values (degradation) contribute; improvements are clipped to 0.
    The mean over all policies gives a continuous overhead score. A policy that
    is average in both environments contributes 0. A policy that was 3x better
    than average in source but average in target contributes log(3) ~ 1.1.

    Directional: degradation from source to target only. omega(A->B) != omega(B->A).
    """
    common = set(ks_s.policy_maes) & set(ks_t.policy_maes)
    if not common:
        return 0.0

    mean_s = float(np.mean(list(ks_s.policy_maes.values())))
    mean_t = float(np.mean(list(ks_t.policy_maes.values())))
    if mean_s < 1e-6 or mean_t < 1e-6:
        return 0.0

    total = 0.0
    for p in common:
        mae_s = ks_s.policy_maes[p]
        mae_t = ks_t.policy_maes[p]
        if mae_s < 1e-6 or mae_t < 1e-6:
            continue
        rel_s = mae_s / mean_s
        rel_t = mae_t / mean_t
        log_deg = np.log(rel_t / rel_s)
        total += max(0.0, float(log_deg))  # only degradation counts

    return float(total / len(common))


def compute_transfer_overhead_decomposed(ks_s: KnowledgeSpace, ks_t: KnowledgeSpace) -> Dict[str, float]:
    """Decompose omega into two distinct failure modes.

    omega_reversal: fraction of policy pairs (i,j) where MAE ordering flips
        source -> target. Measures rank instability regardless of magnitude.
        Range [0, 1]; 0 = perfect rank preservation, 0.5 = random.

    omega_magnitude: mean log-relative-standing degradation among policies
        that actually got worse. Measures how severe degradation is when it occurs.
        omega_magnitude > 0 always (only degrading policies counted).

    omega_total: combined continuous measure (existing formula).
    """
    common = sorted(set(ks_s.policy_maes) & set(ks_t.policy_maes))
    if not common:
        return {"omega_reversal": 0.0, "omega_magnitude": 0.0, "omega_total": 0.0}

    mean_s = float(np.mean(list(ks_s.policy_maes.values())))
    mean_t = float(np.mean(list(ks_t.policy_maes.values())))
    if mean_s < 1e-6 or mean_t < 1e-6:
        return {"omega_reversal": 0.0, "omega_magnitude": 0.0, "omega_total": 0.0}

    # omega_reversal: count pairwise ordering flips
    n_pairs = 0
    n_reversals = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            pi, pj = common[i], common[j]
            mae_s_i = ks_s.policy_maes[pi]
            mae_s_j = ks_s.policy_maes[pj]
            if abs(mae_s_i - mae_s_j) < 1e-6:
                continue  # tied in source -- skip
            mae_t_i = ks_t.policy_maes[pi]
            mae_t_j = ks_t.policy_maes[pj]
            n_pairs += 1
            if (mae_s_i < mae_s_j) != (mae_t_i < mae_t_j):  # lower MAE = better
                n_reversals += 1
    omega_reversal = n_reversals / n_pairs if n_pairs > 0 else 0.0

    # omega_magnitude + omega_total
    degrading = []
    total = 0.0
    for p in common:
        mae_s = ks_s.policy_maes[p]
        mae_t = ks_t.policy_maes[p]
        if mae_s < 1e-6 or mae_t < 1e-6:
            continue
        rel_s = mae_s / mean_s
        rel_t = mae_t / mean_t
        log_deg = float(np.log(rel_t / rel_s))
        total += max(0.0, log_deg)
        if log_deg > 0.0:
            degrading.append(log_deg)

    omega_magnitude = float(np.mean(degrading)) if degrading else 0.0
    omega_total = float(total / len(common))

    return {
        "omega_reversal": omega_reversal,
        "omega_magnitude": omega_magnitude,
        "omega_total": omega_total,
    }


def compute_transfer_delta(
    ks_s: KnowledgeSpace,
    ks_t: KnowledgeSpace,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    all_spaces: Optional[Dict[str, KnowledgeSpace]] = None,
) -> Dict[str, float]:
    """Delta(K_s -> K_t) = alpha*sigma + gamma*rho - delta*omega.

    beta is fixed at 0 -- mu (source quality) is removed because absolute
    performance is engine-local and cannot be compared across databases.
    The beta/all_spaces parameters are kept for API compatibility but have no effect.

    Returns dict with component breakdown and total.
    """
    sigma = compute_structural_compatibility(ks_s, ks_t)
    mu = 0.0  # removed: absolute performance is not cross-engine comparable
    rho = compute_mapping_efficiency(ks_s, ks_t)
    omega_breakdown = compute_transfer_overhead_decomposed(ks_s, ks_t)
    omega = omega_breakdown["omega_total"]
    total = alpha * sigma + gamma * rho - delta * omega

    # Enhanced metrics (trajectory sigma + gamut coverage rho) when data available
    from .trajectory_sigma import compute_trajectory_sigma
    from .gamut_coverage import compute_gamut_coverage_rho

    traj_sigma = compute_trajectory_sigma(ks_s, ks_t)
    gamut_rho = compute_gamut_coverage_rho(ks_s, ks_t)

    # trajectory_coverage_delta uses improved components where available, falls back to base
    eff_sigma = traj_sigma if traj_sigma is not None else sigma
    eff_rho = gamut_rho if gamut_rho is not None else rho
    enhanced_total = alpha * eff_sigma + gamma * eff_rho - delta * omega

    return {
        "delta": float(total),
        "structural_compatibility": sigma,
        "source_quality": mu,
        "mapping_efficiency": rho,
        "transfer_overhead": omega,
        "omega_reversal": omega_breakdown["omega_reversal"],
        "omega_magnitude": omega_breakdown["omega_magnitude"],
        "trajectory_sigma": float(traj_sigma) if traj_sigma is not None else None,
        "gamut_coverage_rho": float(gamut_rho) if gamut_rho is not None else None,
        "trajectory_coverage_delta": float(enhanced_total),
    }


def compute_all_transfer_deltas(
    spaces: Dict[str, KnowledgeSpace],
) -> Dict[str, Dict]:
    """Compute transfer_delta for all ordered pairs (A->B).

    Passes the full spaces dict so mu normalization uses the global scale.
    Returns {"sA->sB": result_dict}.
    """
    results = {}
    study_types = list(spaces.keys())
    for st_s in study_types:
        for st_t in study_types:
            if st_s == st_t:
                continue
            key = st_s + "->" + st_t
            results[key] = compute_transfer_delta(
                spaces[st_s], spaces[st_t], all_spaces=spaces
            )
    return results


def verify_asymmetry_conjecture(transfer_deltas: Dict[str, Dict]) -> Dict:
    """Design Conjecture 2.1: Delta(A->B) != Delta(B->A) for all pairs."""
    pairs_checked = 0
    asymmetric_count = 0
    details = []

    study_types = set()
    for key in transfer_deltas:
        a, b = key.split("->")
        study_types.add(a)
        study_types.add(b)
    study_types = sorted(study_types)

    for i, st_a in enumerate(study_types):
        for st_b in study_types[i + 1:]:
            ab = st_a + "->" + st_b
            ba = st_b + "->" + st_a
            if ab not in transfer_deltas or ba not in transfer_deltas:
                continue
            d_ab = transfer_deltas[ab]["delta"]
            d_ba = transfer_deltas[ba]["delta"]
            pairs_checked += 1
            is_asymmetric = abs(d_ab - d_ba) > 0.01
            if is_asymmetric:
                asymmetric_count += 1
            details.append({
                "pair": st_a + "<->" + st_b,
                "delta_ab": d_ab,
                "delta_ba": d_ba,
                "asymmetry": abs(d_ab - d_ba),
                "conjecture_holds": is_asymmetric,
            })

    overall = "SUPPORTED" if asymmetric_count == pairs_checked else "PARTIALLY_SUPPORTED"
    return {
        "conjecture": "Design Conjecture 2.1 (Asymmetric Transfer Principle)",
        "result": overall,
        "pairs_checked": pairs_checked,
        "asymmetric_pairs": asymmetric_count,
        "details": details,
    }


def verify_hybrid_superiority_conjecture(
    transfer_deltas: Dict[str, Dict],
    spaces: Dict[str, KnowledgeSpace],
) -> Dict:
    """Design Conjecture 2.2: For K_T and K_E sources, K_H is the best target."""
    from .knowledge_spaces import KnowledgeSpaceType

    results = []
    for source_st, source_ks in spaces.items():
        if source_ks.space_type not in (KnowledgeSpaceType.K_T, KnowledgeSpaceType.K_E):
            continue
        targets = {
            st: transfer_deltas.get(source_st + "->" + st, {}).get("delta", float("-inf"))
            for st in spaces
            if st != source_st
        }
        if not targets:
            continue
        best_target = max(targets, key=lambda k: targets[k])
        best_ks = spaces[best_target]
        conjecture_holds = best_ks.space_type in (
            KnowledgeSpaceType.K_H,
            KnowledgeSpaceType.K_H_OVERLAP,
        )
        results.append({
            "source": source_st,
            "source_type": source_ks.space_type.value,
            "best_target": best_target,
            "best_target_type": best_ks.space_type.value,
            "best_delta": float(targets[best_target]),
            "conjecture_holds": conjecture_holds,
        })

    overall = all(r["conjecture_holds"] for r in results) if results else False
    return {
        "conjecture": "Design Conjecture 2.2 (Hybrid Superiority Principle)",
        "result": "SUPPORTED" if overall else "NOT_SUPPORTED",
        "details": results,
    }
