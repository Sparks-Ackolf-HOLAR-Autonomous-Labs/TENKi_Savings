"""
Phase 3 Knowledge Space Transfer + Lab-to-Fab Evaluation Module.

Public exports:
    KnowledgeSpace, KnowledgeSpaceType, STUDY_TYPE_TO_KS
    load_knowledge_spaces, build_performance_vectors
    compute_transfer_delta, compute_all_transfer_deltas
    verify_asymmetry_conjecture, verify_hybrid_superiority_conjecture
    build_payoff_matrix, compute_nash_routing
    RobbinsMonroChecker
    ScaleParametrizedSpace, ProcessStateVectorTransfer, SCALE_LADDER
    compute_economic_transfer, compute_risk_adjusted_utility
    compute_trl_routing_table
    TransferValidator
    DigitalTwinTransfer
"""

from .knowledge_spaces import (
    KnowledgeSpace,
    KnowledgeSpaceType,
    STUDY_TYPE_TO_KS,
    load_knowledge_spaces,
    build_performance_vectors,
    _ks_type_for,
)
from .transfer_equation import (
    compute_transfer_delta,
    compute_all_transfer_deltas,
    verify_asymmetry_conjecture,
    verify_hybrid_superiority_conjecture,
)
from .game_theory import build_payoff_matrix, compute_nash_routing
from .convergence_checker import RobbinsMonroChecker
from .scale_transfer import ScaleParametrizedSpace, ProcessStateVectorTransfer, SCALE_LADDER
from .economic_framework import (
    compute_economic_transfer,
    compute_risk_adjusted_utility,
    compute_trl_routing_table,
)
from .validator import TransferValidator
from .digital_twin import DigitalTwinTransfer

__all__ = [
    "KnowledgeSpace",
    "KnowledgeSpaceType",
    "STUDY_TYPE_TO_KS",
    "load_knowledge_spaces",
    "build_performance_vectors",
    "compute_transfer_delta",
    "compute_all_transfer_deltas",
    "verify_asymmetry_conjecture",
    "verify_hybrid_superiority_conjecture",
    "build_payoff_matrix",
    "compute_nash_routing",
    "RobbinsMonroChecker",
    "ScaleParametrizedSpace",
    "ProcessStateVectorTransfer",
    "SCALE_LADDER",
    "compute_economic_transfer",
    "compute_risk_adjusted_utility",
    "compute_trl_routing_table",
    "TransferValidator",
    "DigitalTwinTransfer",
]
