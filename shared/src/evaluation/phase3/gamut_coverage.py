"""
Gamut-based mapping efficiency rho -- improved #4.

Since all engines share the same target color set, engine_exp_maes[e] is the
mean best_color_distance achieved by all policies on color e. This is a direct
measure of whether the engine can *reach* that color.

rho(s->t) = fraction of colors where both source AND target can reach it,
            divided by the fraction where target alone can reach it.
            (Coverage of target-reachable colors by source.)

This is direction-weighted by sigma_t / (sigma_s + sigma_t):
transfers from a confident source to an uncertain target score higher.

Returns None when engine_exp_maes is not populated (falls back to OLS rho).
"""

import numpy as np
from typing import Optional

from .knowledge_spaces import KnowledgeSpace

# MAE threshold below which a color is considered "reachable" by an engine.
# 30 corresponds to a visible but moderate color difference in absolute units.
_REACHABILITY_THRESHOLD = 30.0


def compute_gamut_coverage_rho(
    ks_s: KnowledgeSpace,
    ks_t: KnowledgeSpace,
    threshold: float = _REACHABILITY_THRESHOLD,
) -> Optional[float]:
    """Fraction of target-reachable colors that source can also reach.

    engine_reachable(engine, color_e) = engine_exp_maes[e] < threshold

    coverage(s->t) = |reachable_s & reachable_t| / |reachable_t|

    If reachable_t is empty (target cannot reach any color below threshold),
    returns 0.5 (uninformative default).

    Direction-weighted by sigma_t / (sigma_s + sigma_t) for asymmetry.
    Returns None if either space lacks engine_exp_maes data.
    """
    if not ks_s.engine_exp_maes or not ks_t.engine_exp_maes:
        return None

    n = min(len(ks_s.engine_exp_maes), len(ks_t.engine_exp_maes))
    if n < 2:
        return None

    arr_s = np.array(ks_s.engine_exp_maes[:n], dtype=float)
    arr_t = np.array(ks_t.engine_exp_maes[:n], dtype=float)

    reachable_s = set(int(e) for e in range(n) if arr_s[e] < threshold)
    reachable_t = set(int(e) for e in range(n) if arr_t[e] < threshold)

    if not reachable_t:
        return 0.5  # target can't reach any color -- uninformative

    coverage = len(reachable_s & reachable_t) / len(reachable_t)

    # Direction weight via TrueSkill sigma (same logic as OLS rho)
    all_sigma_s = list(ks_s.policy_sigmas.values())
    all_sigma_t = list(ks_t.policy_sigmas.values())
    mean_sigma_s = float(np.mean(all_sigma_s)) if all_sigma_s else 1.0
    mean_sigma_t = float(np.mean(all_sigma_t)) if all_sigma_t else 1.0
    denom = mean_sigma_s + mean_sigma_t
    direction_weight = mean_sigma_t / denom if denom > 1e-10 else 0.5

    return float(coverage * direction_weight)
