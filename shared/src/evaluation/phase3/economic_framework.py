"""
Economic Transfer Framework -- Section 11.5

Implements:
    - E_transfer(s1->s2) = (Benefit_target - Cost_transfer) / Investment
    - U_total = w_perf * P - w_risk * R - w_cost * C (risk-adjusted utility)
    - TRL-based knowledge space routing table (TRL 1-9)
"""

from typing import Optional
from .knowledge_spaces import KnowledgeSpaceType


def compute_economic_transfer(
    benefit_target: float,
    cost_transfer: float,
    investment: float,
) -> float:
    """E_transfer(s1->s2) = (Benefit_target - Cost_transfer) / Investment. Section 11.5"""
    if investment <= 0:
        if benefit_target > cost_transfer:
            return float("inf")
        return float("-inf")
    return (benefit_target - cost_transfer) / investment


def compute_risk_adjusted_utility(
    performance: float,
    risk: float,
    cost: float,
    w_perf: float = 1.0,
    w_risk: float = 0.3,
    w_cost: float = 0.2,
) -> float:
    """U_total = w_perf * P - w_risk * R - w_cost * C. Section 11.5"""
    return w_perf * performance - w_risk * risk - w_cost * cost


def trl_knowledge_dominance(trl_level: int) -> str:
    """Return dominant knowledge space type for a given TRL level. Section 11.5

    TRL 1-3: theoretical validation (K_T)
    TRL 4-6: empirical + pilot balance (K_E + K_H)
    TRL 7-9: manufacturing hybrid (K_H)
    """
    if trl_level <= 3:
        return KnowledgeSpaceType.K_T.value
    elif trl_level <= 6:
        return "K_E_plus_K_H"
    else:
        return KnowledgeSpaceType.K_H.value


def compute_trl_routing_table() -> list:
    """Build full TRL 1-9 routing table with dominant knowledge space per level."""
    return [
        {"trl": trl, "dominant_space": trl_knowledge_dominance(trl)}
        for trl in range(1, 10)
    ]
