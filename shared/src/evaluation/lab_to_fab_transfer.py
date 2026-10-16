"""
Lab-to-Fab Transfer Framework for PEGKi

Implements scale-parametrized knowledge transfer from laboratory to manufacturing scale,
as described in TECHNICAL_REFERENCE.md Section 11.

Key concepts:
- Scale-parametrized knowledge spaces K(s) where s is the physical scale
- Physics-based scaling matrix M_physics using dimensional analysis exponents
- Process state vector transformation: BC_industrial = M_scale @ BC_lab + b_scale
- Control policy transfer matrices for lab->pilot and pilot->fab stages
- Transfer success criteria based on structural similarity (Section 11.6)
"""

import numpy as np
from typing import Optional, Dict, Tuple


class LabToFabTransfer:
    """
    Scale-parametrized knowledge transfer from lab to manufacturing scale (Section 11).

    Maps Blade-Chest vectors and control policies across physical scales using
    physics-based dimensional analysis (Section 11.3) and control transfer matrices
    (Section 11.4).
    """

    # Physics-based scaling exponents from dimensional analysis (Section 11.3)
    # Keys cycle across vector dimensions in order
    PHYSICS_EXPONENTS = {
        'mass':     2.0 / 3.0,   # Mass transfer: surface area scaling
        'heat':     2.0 / 3.0,   # Heat transfer: surface-to-volume ratio
        'mix':     -1.0 / 3.0,   # Mixing time
        'rxn':      0.0,          # Reaction kinetics: intrinsic rate independence
        'flow':     1.0,          # Fluid mechanics: linear velocity scaling
        'pressure': 2.0,          # Pressure drop: velocity squared
        'power':    3.0,          # Power consumption: cubic velocity
    }

    # Control policy transfer matrix: lab -> pilot (reduced aggressiveness, Section 11.4)
    M_CONTROL_LAB_TO_PILOT = np.array([
        [0.8, 0.2, 0.1],
        [0.1, 0.9, 0.2],
        [0.0, 0.1, 0.7],
    ])

    # Control policy transfer matrix: pilot -> manufacturing (highly conservative, Section 11.4)
    M_CONTROL_PILOT_TO_FAB = np.array([
        [0.6, 0.1, 0.0],
        [0.2, 0.8, 0.1],
        [0.1, 0.2, 0.9],
    ])

    # Scale ranges (Section 11.1)
    SCALE_MICRO_MIN = 1e-6   # Lab scale lower bound (microliters)
    SCALE_MICRO_MAX = 1e-3   # Lab scale upper bound (milliliters)
    SCALE_MACRO_MIN = 1e0    # Manufacturing scale lower bound (liters)
    SCALE_MACRO_MAX = 1e6    # Manufacturing scale upper bound (megaliters)

    def physics_scale_matrix(self, scale_ratio: float, vector_dim: int) -> np.ndarray:
        """
        Build the physics-based diagonal scaling matrix M_physics(s1->s2).

        M_physics = diag(scale_ratio^alpha_1, scale_ratio^alpha_2, ..., scale_ratio^alpha_n)

        Exponents cycle through PHYSICS_EXPONENTS keys across the vector dimension
        (Section 11.3).

        Args:
            scale_ratio: s2/s1 — ratio of target to source scale
            vector_dim:  dimensionality of the BC vector

        Returns:
            Diagonal numpy array of shape (vector_dim, vector_dim)
        """
        exponent_values = list(self.PHYSICS_EXPONENTS.values())
        n_exponents = len(exponent_values)
        diag_entries = np.array([
            scale_ratio ** exponent_values[i % n_exponents]
            for i in range(vector_dim)
        ])
        return np.diag(diag_entries)

    def transform_bc_vectors(self, bc_lab: np.ndarray, scale_ratio: float,
                              b_scale: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform Blade-Chest vectors from lab to industrial scale (Section 11.2).

        BC_industrial = M_scale @ BC_lab + b_scale + epsilon_scaling

        The scaling noise epsilon_scaling is omitted (zero by default) — it represents
        stochastic non-ideal effects that are not modeled deterministically.

        Args:
            bc_lab:      BC vector at lab scale, shape (d,)
            scale_ratio: Physical scale ratio s_macro / s_micro (e.g. 1000 for mL->L)
            b_scale:     Optional additive offset for scale-dependent bias, shape (d,).
                         Defaults to zeros if not provided.

        Returns:
            BC vector at industrial scale, shape (d,)
        """
        d = bc_lab.shape[0]
        M = self.physics_scale_matrix(scale_ratio, d)
        b = b_scale if b_scale is not None else np.zeros(d)
        return M @ bc_lab + b

    def transfer_success(self, bc_source: np.ndarray, bc_target: np.ndarray,
                          scale_ratio: float, epsilon: float = 0.1) -> bool:
        """
        Check structural similarity transfer success criterion (Section 11.6).

        Success if: ||BC_target - M_transfer @ BC_source|| < epsilon

        Args:
            bc_source:   BC vector at source scale
            bc_target:   Observed BC vector at target scale
            scale_ratio: Physical scale ratio used for the transfer
            epsilon:     Similarity threshold (default 0.1)

        Returns:
            True if the transfer meets the structural similarity criterion
        """
        d = bc_source.shape[0]
        M = self.physics_scale_matrix(scale_ratio, d)
        predicted = M @ bc_source
        residual = np.linalg.norm(bc_target - predicted)
        return bool(residual < epsilon)

    def control_transfer(self, control_vector: np.ndarray,
                          stage: str = 'lab_to_pilot') -> np.ndarray:
        """
        Apply control policy transfer matrix to a 3-element control vector (Section 11.4).

        Supported stages:
          'lab_to_pilot'   — lab to pilot scale (M_CONTROL_LAB_TO_PILOT)
          'pilot_to_fab'   — pilot to manufacturing (M_CONTROL_PILOT_TO_FAB)
          'lab_to_fab'     — lab directly to manufacturing (two-stage composition)

        Args:
            control_vector: 3-element control policy vector (e.g. [proportional, integral, derivative])
            stage:          Transfer stage name

        Returns:
            Transformed 3-element control vector
        """
        if control_vector.shape[0] != 3:
            raise ValueError(f"control_vector must have 3 elements, got {control_vector.shape[0]}")

        if stage == 'lab_to_pilot':
            return self.M_CONTROL_LAB_TO_PILOT @ control_vector
        elif stage == 'pilot_to_fab':
            return self.M_CONTROL_PILOT_TO_FAB @ control_vector
        elif stage == 'lab_to_fab':
            # Compose both transfer matrices: pilot->fab applied after lab->pilot
            M_combined = self.M_CONTROL_PILOT_TO_FAB @ self.M_CONTROL_LAB_TO_PILOT
            return M_combined @ control_vector
        else:
            raise ValueError(f"Unknown stage '{stage}'. Use 'lab_to_pilot', 'pilot_to_fab', or 'lab_to_fab'.")

    def scale_analysis(self, bc_lab: np.ndarray,
                        scale_ratios: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        """
        Analyse BC vector transformation across standard scale stages.

        Returns a dict mapping stage name to {'bc': transformed_vector, 'norm_change': ratio}.
        """
        if scale_ratios is None:
            scale_ratios = {
                'lab':         1.0,
                'pilot':       1e3,       # 1L (mL -> L)
                'manufacturing': 1e6,     # 1kL (mL -> kL)
            }

        results = {}
        lab_norm = float(np.linalg.norm(bc_lab)) + 1e-9
        for stage_name, ratio in scale_ratios.items():
            bc_scaled = self.transform_bc_vectors(bc_lab, scale_ratio=ratio)
            results[stage_name] = {
                'bc': bc_scaled,
                'scale_ratio': ratio,
                'norm': float(np.linalg.norm(bc_scaled)),
                'norm_change': float(np.linalg.norm(bc_scaled)) / lab_norm,
            }
        return results
