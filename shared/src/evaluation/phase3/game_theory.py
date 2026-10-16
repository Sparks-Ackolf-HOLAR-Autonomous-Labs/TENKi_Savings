"""
Game-Theoretic Formulation -- Section 10.6

Implements:
    - Payoff matrix Pi[i,j] = transfer_delta(K_i -> K_j)
    - Nash mixed strategy over knowledge spaces for optimal query routing
      via zero-sum LP formulation
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, List

from .knowledge_spaces import KnowledgeSpace
from .transfer_equation import compute_transfer_delta


def build_payoff_matrix(spaces: Dict[str, KnowledgeSpace]) -> Dict:
    """Pi[i,j] = transfer_delta(K_i -> K_j). Diagonal = 0.

    Returns dict with keys:
        study_types: list of study type strings (row/column labels)
        matrix:      n x n list-of-lists (JSON-serializable)
    """
    study_types = sorted(spaces.keys())
    n = len(study_types)
    matrix = np.zeros((n, n))

    for i, st_i in enumerate(study_types):
        for j, st_j in enumerate(study_types):
            if i != j:
                result = compute_transfer_delta(spaces[st_i], spaces[st_j])
                matrix[i, j] = result["delta"]

    return {"study_types": study_types, "matrix": matrix.tolist()}


def compute_nash_routing(payoff_info: Dict) -> Dict:
    """Find Nash mixed strategy over knowledge spaces for optimal query routing.

    Uses zero-sum LP formulation:
        max_{p >= 0, sum(p)=1} min_j (p^T Pi_{:,j})

    Returns mixed strategy (probability over study types).
    """
    study_types = payoff_info["study_types"]
    Pi = np.array(payoff_info["matrix"])
    n = len(study_types)

    if n < 2:
        return {
            "strategy": {st: 1.0 for st in study_types},
            "game_value": 0.0,
            "interpretation": "Trivial: only one knowledge space available.",
            "method": "trivial",
        }

    # Normalize payoff matrix to [0,1] for numerical stability
    Pi_min = Pi.min()
    Pi_max = Pi.max()
    Pi_range = Pi_max - Pi_min
    if Pi_range > 1e-10:
        Pi_norm = (Pi - Pi_min) / Pi_range
    else:
        Pi_norm = Pi.copy()

    # LP formulation:
    #   Variables: x = [p_0, ..., p_{n-1}, v]
    #   Minimise: -v  (we want max v)
    #   Inequality: -Pi_norm^T @ p + v <= 0  (per column j)
    #   Equality: sum(p_i) = 1
    #   Bounds: p_i >= 0, v unbounded
    c = np.zeros(n + 1)
    c[-1] = -1.0  # minimise -v

    A_ub = np.hstack([-Pi_norm.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    A_eq = np.ones((1, n + 1))
    A_eq[0, -1] = 0.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n + [(None, None)]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if res.success:
        p = np.clip(res.x[:n], 0.0, None)
        p_sum = p.sum()
        p = p / p_sum if p_sum > 1e-10 else np.ones(n) / n
        game_value = float(res.x[-1]) * Pi_range + Pi_min
        strategy = {st: float(p[i]) for i, st in enumerate(study_types)}
        method = "linprog_zero_sum"
    else:
        # Fallback: uniform distribution
        strategy = {st: 1.0 / n for st in study_types}
        game_value = float(np.mean(Pi))
        method = "uniform_fallback"

    return {
        "strategy": strategy,
        "game_value": game_value,
        "interpretation": (
            "Probability distribution over knowledge spaces for optimal query routing"
        ),
        "method": method,
    }
