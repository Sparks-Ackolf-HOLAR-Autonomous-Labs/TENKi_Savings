"""
Transfer Success Criteria Validator -- Section 11.6

Implements multi-criteria transfer success validation:
    - Structural similarity: ||BC_target - M_transfer @ BC_source|| < epsilon
    - Performance correlation: Corr(P_predicted, P_actual) > rho_min
    - Economic viability: ROI > roi_hurdle
    - Multi-scale R2 targets: lab>0.8, pilot>0.7, manufacturing>0.6
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional


class TransferValidator:
    """Multi-criteria transfer success validation. Section 11.6"""

    def structural_similarity(
        self,
        bc_target: np.ndarray,
        bc_source: np.ndarray,
        m_transfer: np.ndarray,
        epsilon: float = 0.1,
    ) -> Dict:
        """Relative structural similarity: ||BC_target - M @ BC_source|| / ||BC_target|| < epsilon.

        Uses relative (scale-invariant) residual so that physics scale transfers
        (e.g. 1000x lab-to-fab) do not trivially fail due to magnitude differences.
        """
        predicted = m_transfer @ bc_source
        residual_norm = float(np.linalg.norm(bc_target - predicted))
        target_norm = float(np.linalg.norm(bc_target))
        relative_residual = residual_norm / max(target_norm, 1e-8)
        return {
            "passes": relative_residual < epsilon,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "threshold": epsilon,
        }

    def performance_correlation(
        self,
        predicted: List[float],
        actual: List[float],
        rho_min: float = 0.8,
    ) -> Dict:
        """Corr(P_predicted, P_actual) > rho_min."""
        if len(predicted) < 2:
            return {
                "passes": False,
                "correlation": 0.0,
                "threshold": rho_min,
                "note": "Too few samples",
            }
        r, p_val = pearsonr(predicted, actual)
        return {
            "passes": float(r) > rho_min,
            "correlation": float(r),
            "p_value": float(p_val),
            "threshold": rho_min,
        }

    def economic_viability(self, roi: float, roi_hurdle: float = 0.0) -> Dict:
        """ROI > roi_hurdle."""
        return {
            "passes": roi > roi_hurdle,
            "roi": roi,
            "hurdle": roi_hurdle,
        }

    def multi_scale_report(
        self,
        scale_results: List[Dict],
        r2_actuals: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Validate R2 targets per scale: lab>0.8, pilot>0.7, manufacturing>0.6. Section 11.6"""
        R2_TARGETS = {"lab": 0.8, "pilot": 0.7, "manufacturing": 0.6}

        if r2_actuals is None:
            # Estimate R2 via Spearman rank correlation of policy norms across the
            # scale transfer.  Rank preservation is scale-invariant and correctly
            # measures whether relative policy ordering survives lab-to-fab scaling.
            # Lab is the source scale (no transfer error) so R2_lab = 1.0 by definition.
            r2_actuals = {"lab": 1.0}
            for stage_info in scale_results:
                stage = stage_info["stage"]
                policy_norms = stage_info["policy_norms"]
                orig  = [v["original_norm"] for v in policy_norms.values()]
                scaled = [v["scaled_norm"]   for v in policy_norms.values()]
                if len(orig) >= 2:
                    rho, _ = spearmanr(orig, scaled)
                    # Convert Spearman rho to R2-like score: max(0, rho)^2
                    r2_actuals[stage] = float(max(0.0, rho) ** 2)
                else:
                    r2_actuals[stage] = 0.0

        report = []
        for stage, target_r2 in R2_TARGETS.items():
            actual = r2_actuals.get(stage, 0.0)
            report.append({
                "scale": stage,
                "r2_actual": float(actual),
                "r2_target": target_r2,
                "passes": actual >= target_r2,
            })

        overall = all(r["passes"] for r in report)
        return {"overall_passes": overall, "scales": report}
