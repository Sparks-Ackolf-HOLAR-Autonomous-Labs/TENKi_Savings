"""
Digital Twin Transfer -- Section 11.7

Implements adaptive BC correction:
    BC_adaptive(t) = BC_base + K_feedback(t) * [BC_predicted(t) - BC_measured(t)]

Features:
    - Adaptive feedback gain based on prediction accuracy
    - Full update history tracking
    - Synthetic simulation for validation
"""

import numpy as np
from typing import List, Dict, Optional


class DigitalTwinTransfer:
    """
    Implements BC_adaptive(t) = BC_base + K_feedback(t) * [BC_predicted(t) - BC_measured(t)].

    Section 11.7
    """

    def __init__(self, feedback_gain: float = 0.5):
        self.feedback_gain = float(feedback_gain)
        self.bc_base: Optional[np.ndarray] = None
        self.update_history: List[Dict] = []
        self._step = 0

    def initialize(self, bc_source: np.ndarray):
        """Set BC_base from the source knowledge space."""
        self.bc_base = bc_source.astype(float).copy()
        self.update_history = []
        self._step = 0

    def update(self, bc_predicted: np.ndarray, bc_measured: np.ndarray) -> np.ndarray:
        """One adaptive update step. Adapts gain based on prediction accuracy."""
        if self.bc_base is None:
            raise RuntimeError("Call initialize() before update().")

        self._step += 1
        error = bc_predicted - bc_measured
        correction = self.feedback_gain * error
        bc_new = self.bc_base + correction

        # Prediction accuracy: 1 - mean(|error|) / (mean(|measured|) + eps)
        mean_abs_error = float(np.mean(np.abs(error)))
        mean_abs_measured = float(np.mean(np.abs(bc_measured)))
        accuracy = 1.0 - mean_abs_error / (mean_abs_measured + 1e-6)
        accuracy = float(np.clip(accuracy, 0.0, 1.0))

        self.update_history.append({
            "step": self._step,
            "feedback_gain": float(self.feedback_gain),
            "error_norm": float(np.linalg.norm(error)),
            "correction_norm": float(np.linalg.norm(correction)),
            "prediction_accuracy": accuracy,
            "bc_base_norm": float(np.linalg.norm(bc_new)),
        })

        # Adapt gain: increase when accurate, decrease when inaccurate
        new_gain = self.feedback_gain * (0.9 + 0.2 * accuracy)
        self.feedback_gain = float(np.clip(new_gain, 0.1, 2.0))
        self.bc_base = bc_new
        return bc_new.copy()

    def replay_history(self) -> List[Dict]:
        """Return full update history."""
        return self.update_history

    def simulate(
        self,
        bc_source: np.ndarray,
        n_steps: int = 5,
        noise_std: float = 0.05,
    ) -> List[Dict]:
        """Simulate n_steps of digital twin updates with synthetic measurements."""
        self.initialize(bc_source)
        rng = np.random.default_rng(0)

        for _ in range(n_steps):
            # Predicted: current base + small drift
            bc_predicted = self.bc_base * (1.0 + rng.normal(0, 0.02, size=self.bc_base.shape))
            # Measured: true source + noise
            bc_measured = bc_source + rng.normal(0, noise_std, size=bc_source.shape)
            self.update(bc_predicted, bc_measured)

        return self.replay_history()
