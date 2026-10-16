"""
Statistical Distance Metrics for Mode Comparison

Based on: PEGKí Design Modification (2025-11-03), Section 2

Implements statistical distance measures for comparing rating distributions
across different acquisition modes, environments, or experimental conditions.
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.stats import norm
from scipy.special import rel_entr
from .trueskill_engine import Rating


class StatisticalDistances:
    """
    Statistical distance measures for comparing skill distributions across modes.

    For Gaussian skill posteriors s^(m) ~ N(μ_m, σ_m²), computes:
    - Probability of superiority
    - KL divergence
    - Hellinger distance
    - Wasserstein-2 distance
    - Jensen-Shannon divergence (for predictive distributions)
    """

    @staticmethod
    def probability_of_superiority(
        mu_1: float, sigma_1: float,
        mu_2: float, sigma_2: float
    ) -> float:
        """
        Compute P(s^(1) > s^(2)) for two Gaussian skill distributions.

        Formula: P(s^(1) > s^(2)) = Φ((μ_1 - μ_2) / √(σ_1² + σ_2²))

        Args:
            mu_1, sigma_1: Mean and std dev of distribution 1
            mu_2, sigma_2: Mean and std dev of distribution 2

        Returns:
            Probability that skill 1 exceeds skill 2 (0 to 1)
        """
        if sigma_1 <= 0 or sigma_2 <= 0:
            raise ValueError("Standard deviations must be positive")

        mean_diff = mu_1 - mu_2
        combined_var = sigma_1**2 + sigma_2**2
        prob = norm.cdf(mean_diff / np.sqrt(combined_var))

        return prob

    @staticmethod
    def kl_divergence(
        mu_1: float, sigma_1: float,
        mu_2: float, sigma_2: float
    ) -> float:
        """
        Compute KL divergence D_KL(N(μ_1,σ_1²) || N(μ_2,σ_2²)).

        Formula:
        D_KL(1||2) = ln(σ_2/σ_1) + (σ_1² + (μ_1-μ_2)²)/(2σ_2²) - 1/2

        Args:
            mu_1, sigma_1: Parameters of distribution 1 (P)
            mu_2, sigma_2: Parameters of distribution 2 (Q)

        Returns:
            KL divergence in nats (non-negative, asymmetric)
        """
        if sigma_1 <= 0 or sigma_2 <= 0:
            raise ValueError("Standard deviations must be positive")

        log_ratio = np.log(sigma_2 / sigma_1)
        variance_term = (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2)
        kl = log_ratio + variance_term - 0.5

        return max(0.0, kl)  # Ensure non-negative due to numerical errors

    @staticmethod
    def hellinger_distance(
        mu_1: float, sigma_1: float,
        mu_2: float, sigma_2: float
    ) -> float:
        """
        Compute Hellinger distance H between two Gaussian distributions.

        Formula:
        H² = 1 - √(2σ_1σ_2/(σ_1²+σ_2²)) * exp(-(μ_1-μ_2)²/(4(σ_1²+σ_2²)))

        Args:
            mu_1, sigma_1: Parameters of distribution 1
            mu_2, sigma_2: Parameters of distribution 2

        Returns:
            Hellinger distance (0 to 1, symmetric)
        """
        if sigma_1 <= 0 or sigma_2 <= 0:
            raise ValueError("Standard deviations must be positive")

        # Compute components
        var_sum = sigma_1**2 + sigma_2**2
        var_prod = sigma_1 * sigma_2
        mean_diff_sq = (mu_1 - mu_2)**2

        # Hellinger coefficient (BC - Bhattacharyya coefficient)
        bc = np.sqrt(2 * var_prod / var_sum) * np.exp(-mean_diff_sq / (4 * var_sum))

        # Hellinger distance
        h_squared = 1.0 - bc
        h = np.sqrt(max(0.0, h_squared))

        return h

    @staticmethod
    def wasserstein_2_distance(
        mu_1: float, sigma_1: float,
        mu_2: float, sigma_2: float
    ) -> float:
        """
        Compute Wasserstein-2 (W₂) distance between two Gaussian distributions.

        Formula: W_2 = √((μ_1-μ_2)² + (σ_1-σ_2)²)

        Args:
            mu_1, sigma_1: Parameters of distribution 1
            mu_2, sigma_2: Parameters of distribution 2

        Returns:
            W₂ distance (non-negative, symmetric)
        """
        if sigma_1 < 0 or sigma_2 < 0:
            raise ValueError("Standard deviations must be non-negative")

        mean_diff_sq = (mu_1 - mu_2)**2
        std_diff_sq = (sigma_1 - sigma_2)**2
        w2 = np.sqrt(mean_diff_sq + std_diff_sq)

        return w2

    @staticmethod
    def jensen_shannon_divergence(p: float, q: float) -> float:
        """
        Compute Jensen-Shannon divergence between two Bernoulli distributions.

        Used for comparing predictive win probabilities across modes.

        Formula: JS(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m) where m = (p+q)/2

        Args:
            p: Win probability in mode 1 (0 to 1)
            q: Win probability in mode 2 (0 to 1)

        Returns:
            JS divergence in nats (0 to ln(2) ≈ 0.693, symmetric)
        """
        # Clamp to valid probability range
        p = np.clip(p, 1e-10, 1 - 1e-10)
        q = np.clip(q, 1e-10, 1 - 1e-10)

        # Mixture distribution
        m = (p + q) / 2.0

        # KL divergences
        kl_pm = p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))
        kl_qm = q * np.log(q / m) + (1 - q) * np.log((1 - q) / (1 - m))

        js = 0.5 * kl_pm + 0.5 * kl_qm

        return js

    @staticmethod
    def predictive_distribution_distance(
        ratings_1: Dict[str, Rating],
        ratings_2: Dict[str, Rating],
        opponent_ids: List[str],
        rating_system
    ) -> float:
        """
        Compute predictive distribution distance between two modes.

        Measures how differently the same entity would perform against
        a fixed set of opponents under two different modes.

        Formula: D_pred = (1/|J|) Σ_j JS(Bern(p_j^(1)) || Bern(p_j^(2)))

        Args:
            ratings_1: Ratings under mode 1 {entity_id: Rating}
            ratings_2: Ratings under mode 2 {entity_id: Rating}
            opponent_ids: List of opponent entity IDs
            rating_system: Rating system with predict_outcome method

        Returns:
            Average JS divergence across opponent set
        """
        if not opponent_ids:
            return 0.0

        js_divergences = []

        for opp_id in opponent_ids:
            if opp_id not in ratings_1 or opp_id not in ratings_2:
                continue

            # Compute win probabilities under each mode
            # Using TrueSkill formula: P(i > j) = Φ((μ_i - μ_j) / √(σ_i² + σ_j² + 2β²))
            beta = 100.0  # Performance noise

            for entity_id in ratings_1.keys():
                if entity_id == opp_id or entity_id not in ratings_2:
                    continue

                # Mode 1 prediction
                r1_entity = ratings_1[entity_id]
                r1_opp = ratings_1[opp_id]
                mean_diff_1 = r1_entity.mu - r1_opp.mu
                combined_sigma_1 = np.sqrt(r1_entity.sigma**2 + r1_opp.sigma**2 + 2*beta**2)
                p1 = norm.cdf(mean_diff_1 / combined_sigma_1)

                # Mode 2 prediction
                r2_entity = ratings_2[entity_id]
                r2_opp = ratings_2[opp_id]
                mean_diff_2 = r2_entity.mu - r2_opp.mu
                combined_sigma_2 = np.sqrt(r2_entity.sigma**2 + r2_opp.sigma**2 + 2*beta**2)
                p2 = norm.cdf(mean_diff_2 / combined_sigma_2)

                # JS divergence
                js = StatisticalDistances.jensen_shannon_divergence(p1, p2)
                js_divergences.append(js)

        if not js_divergences:
            return 0.0

        return np.mean(js_divergences)


class ModeComparisonResult:
    """Container for mode comparison statistics."""

    def __init__(self, entity_id: str, mode_1_name: str, mode_2_name: str):
        self.entity_id = entity_id
        self.mode_1_name = mode_1_name
        self.mode_2_name = mode_2_name

        # Skill distribution parameters
        self.mu_1: float = 0.0
        self.sigma_1: float = 0.0
        self.mu_2: float = 0.0
        self.sigma_2: float = 0.0

        # Distance metrics
        self.prob_superiority: float = 0.0  # P(mode_1 > mode_2)
        self.kl_divergence: float = 0.0
        self.hellinger_distance: float = 0.0
        self.wasserstein_2: float = 0.0
        self.predictive_distance: float = 0.0

        # Statistical significance
        self.is_significant: bool = False
        self.effect_size: str = ""  # "negligible", "small", "medium", "large"

    def compute_effect_size(self):
        """
        Compute effect size category based on Wasserstein distance.

        Cohen's d-like interpretation for W₂:
        - < 100: negligible
        - 100-200: small
        - 200-400: medium
        - > 400: large
        """
        w2 = self.wasserstein_2

        if w2 < 100:
            self.effect_size = "negligible"
        elif w2 < 200:
            self.effect_size = "small"
        elif w2 < 400:
            self.effect_size = "medium"
        else:
            self.effect_size = "large"

    def assess_significance(self, threshold: float = 0.05):
        """
        Assess statistical significance using probability of superiority.

        If P(mode_1 > mode_2) is far from 0.5, modes are significantly different.
        """
        deviation_from_neutral = abs(self.prob_superiority - 0.5)
        self.is_significant = deviation_from_neutral > (0.5 - threshold)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'mode_1': self.mode_1_name,
            'mode_2': self.mode_2_name,
            'distributions': {
                'mode_1': {'mu': self.mu_1, 'sigma': self.sigma_1},
                'mode_2': {'mu': self.mu_2, 'sigma': self.sigma_2}
            },
            'distances': {
                'probability_of_superiority': self.prob_superiority,
                'kl_divergence': self.kl_divergence,
                'hellinger_distance': self.hellinger_distance,
                'wasserstein_2': self.wasserstein_2,
                'predictive_distance': self.predictive_distance
            },
            'significance': {
                'is_significant': self.is_significant,
                'effect_size': self.effect_size
            }
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Mode Comparison: {self.entity_id}\n"
            f"  {self.mode_1_name}: μ={self.mu_1:.1f}, σ={self.sigma_1:.1f}\n"
            f"  {self.mode_2_name}: μ={self.mu_2:.1f}, σ={self.sigma_2:.1f}\n"
            f"  P({self.mode_1_name} > {self.mode_2_name}) = {self.prob_superiority:.3f}\n"
            f"  Wasserstein-2 Distance: {self.wasserstein_2:.2f}\n"
            f"  Hellinger Distance: {self.hellinger_distance:.3f}\n"
            f"  Effect Size: {self.effect_size}\n"
            f"  Significant: {self.is_significant}"
        )
