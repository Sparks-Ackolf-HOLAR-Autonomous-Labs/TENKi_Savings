"""
Scale-Parametrized Knowledge Spaces + Process State Vector Transfer -- Sections 11.1-2

Implements:
    - ScaleParametrizedSpace: K(s) at a given physical scale
    - ProcessStateVectorTransfer: BC_industrial = M_scale @ BC_lab + b_scale + eps_scaling
    - run_scale_analysis: lab -> pilot -> manufacturing ladder
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .knowledge_spaces import KnowledgeSpaceType

# Make project root importable for LabToFabTransfer
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.lab_to_fab_transfer import LabToFabTransfer


SCALE_LADDER = {
    "lab":           1.0,
    "pilot":         1e3,
    "manufacturing": 1e6,
}


def _dimensionality_at_scale(scale: float) -> int:
    """Section 11.2 process vector dimensionality estimates by scale."""
    if scale < 1e-3:
        return 2
    elif scale < 1.0:
        return 7
    elif scale < 1e3:
        return 75
    else:
        return 1000


@dataclass
class ScaleParametrizedSpace:
    """K(s) -- knowledge space at a given physical scale. Section 11.1"""
    space_type: KnowledgeSpaceType
    scale: float                               # s in R+
    stage_name: str                            # 'lab', 'pilot', 'manufacturing'
    performance_vectors: Dict[str, np.ndarray]  # {policy_name: 4D vec}
    dimensionality: int = field(init=False)

    def __post_init__(self):
        self.dimensionality = _dimensionality_at_scale(self.scale)


class ProcessStateVectorTransfer:
    """Implements Section 11.2: BC_industrial = M_scale @ BC_lab + b_scale + eps_scaling."""

    def __init__(self):
        self._ltf = LabToFabTransfer()

    def transfer_vectors(
        self,
        source_space: ScaleParametrizedSpace,
        target_scale: float,
        noise_level: float = 0.01,
        rng_seed: int = 42,
    ) -> ScaleParametrizedSpace:
        """Apply physics scaling matrix to all policy performance vectors."""
        scale_ratio = (
            target_scale / source_space.scale
            if source_space.scale > 0
            else 1.0
        )
        rng = np.random.default_rng(rng_seed)

        # Identify target stage name
        target_name = str(target_scale)
        for stage_key, stage_val in SCALE_LADDER.items():
            if abs(stage_val - target_scale) < 1e-3:
                target_name = stage_key
                break

        scaled_vecs = {}
        for policy, vec in source_space.performance_vectors.items():
            bc_scaled = self._ltf.transform_bc_vectors(vec, scale_ratio)
            eps_nonlinear = rng.normal(0, noise_level, size=bc_scaled.shape)
            scaled_vecs[policy] = bc_scaled + eps_nonlinear

        return ScaleParametrizedSpace(
            space_type=source_space.space_type,
            scale=target_scale,
            stage_name=target_name,
            performance_vectors=scaled_vecs,
        )

    def run_scale_analysis(
        self,
        lab_space: ScaleParametrizedSpace,
    ) -> List[Dict]:
        """Run transfer across full lab -> pilot -> manufacturing ladder."""
        results = []
        current = lab_space
        stages = [
            ("pilot", SCALE_LADDER["pilot"]),
            ("manufacturing", SCALE_LADDER["manufacturing"]),
        ]

        for stage_name, target_scale in stages:
            transferred = self.transfer_vectors(current, target_scale)

            # Compute norm changes per policy
            policy_norms = {}
            for policy in current.performance_vectors:
                orig_norm = float(np.linalg.norm(current.performance_vectors[policy]))
                new_norm = float(np.linalg.norm(transferred.performance_vectors[policy]))
                denom = max(orig_norm, 1e-8)
                policy_norms[policy] = {
                    "original_norm": orig_norm,
                    "scaled_norm": new_norm,
                    "norm_change_pct": 100.0 * (new_norm - orig_norm) / denom,
                }

            results.append({
                "stage": stage_name,
                "source_scale": current.scale,
                "target_scale": target_scale,
                "scale_ratio": target_scale / current.scale,
                "target_dimensionality": transferred.dimensionality,
                "policy_norms": policy_norms,
            })
            current = transferred

        return results
