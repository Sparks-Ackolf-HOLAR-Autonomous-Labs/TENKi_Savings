"""
Trajectory-based structural compatibility sigma -- improved #3.

Since all engines use the same target color set (same seed), per-experiment
comparisons are valid across engines. For each target color (experiment), rank
all policies by their best_color_distance. The Spearman correlation of these
per-color rank vectors between source and target measures whether the same
policies dominate on the *same specific colors* -- not just in aggregate.

This is strictly more informative than point-estimate sigma:
- Two spaces can have identical aggregate rankings but swap ranks color-by-color.
- High trajectory sigma = robust, color-level rank preservation.
- Low trajectory sigma = fragile, aggregate rank hides per-color instability.

Returns None when policy_exp_maes is not populated (falls back to point-estimate).
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Optional

from .knowledge_spaces import KnowledgeSpace


def compute_trajectory_sigma(
    ks_s: KnowledgeSpace,
    ks_t: KnowledgeSpace,
) -> Optional[float]:
    """Per-color rank Spearman correlation, averaged over all target colors.

    For each experiment e (= one shared target color):
        rank all policies in source by MAE_e
        rank all policies in target by MAE_e
        compute Spearman correlation of the two rank vectors

    Mean over all experiments gives trajectory sigma.

    Reliability weighting: 1 / (1 + mean_ts_sigma_s) -- same as point-estimate sigma.
    Directional: reliability depends only on source sigma.

    Returns None if either space lacks policy_exp_maes data.
    """
    if not ks_s.policy_exp_maes or not ks_t.policy_exp_maes:
        return None

    common_policies = sorted(set(ks_s.policy_exp_maes) & set(ks_t.policy_exp_maes))
    if len(common_policies) < 2:
        return None

    # Align experiment count to the minimum available
    n_exp_s = len(next(iter(ks_s.policy_exp_maes.values())))
    n_exp_t = len(next(iter(ks_t.policy_exp_maes.values())))
    n_exp = min(n_exp_s, n_exp_t)
    if n_exp < 2:
        return None

    # Build MAE matrices: shape (n_exp, n_policies)
    mae_s = np.array(
        [[ks_s.policy_exp_maes[p][e] for p in common_policies] for e in range(n_exp)],
        dtype=float,
    )
    mae_t = np.array(
        [[ks_t.policy_exp_maes[p][e] for p in common_policies] for e in range(n_exp)],
        dtype=float,
    )

    # Per-experiment Spearman correlation
    rhos = []
    for e in range(n_exp):
        row_s = mae_s[e]
        row_t = mae_t[e]
        # Skip experiments where any value is NaN or all values are identical
        if np.any(np.isnan(row_s)) or np.any(np.isnan(row_t)):
            continue
        if np.std(row_s) < 1e-10 or np.std(row_t) < 1e-10:
            continue
        rho, _ = spearmanr(row_s, row_t)
        if not np.isnan(rho):
            rhos.append(float(rho))

    if not rhos:
        return None

    mean_rho = float(np.mean(rhos))

    # Reliability weight from source TrueSkill sigma
    ts_sigmas = [ks_s.policy_sigmas[p] for p in common_policies if p in ks_s.policy_sigmas]
    mean_sigma_s = float(np.mean(ts_sigmas)) if ts_sigmas else 1.0
    reliability_s = 1.0 / (1.0 + mean_sigma_s)

    return mean_rho * reliability_s
