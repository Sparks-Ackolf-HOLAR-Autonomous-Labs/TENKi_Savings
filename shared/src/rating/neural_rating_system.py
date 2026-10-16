import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import norm
from .trueskill_engine import TrueSkillEngine, Rating
from .blade_chest import BladeChest
from .prior_information import PriorInformationEngine
from ..utils.simulation_seed import get_rng, SeedScope

class BayesianNeuralRatingSystem:
    """
    Bayesian Neural Network rating system that computes TrueSkill priors based on 
    Blade-Chest information and prior knowledge without self-loops.
    """
    
    def __init__(self, agents: List[Any]):
        # Initialize base rating systems
        # Increased beta and tau values for more rating separation and visibility
        self.trueskill_engine = TrueSkillEngine(
            agent_beta=80.0, agent_tau=4.0, policy_beta=150.0, policy_tau=6.0,
            prior_beta=120.0, prior_tau=5.0, agent_policy_beta=180.0, agent_policy_tau=15.0,
            agent_policy_prior_beta=220.0, agent_policy_prior_tau=18.0
        )
        self.blade_chest_engine = BladeChest(self.trueskill_engine)
        self.prior_engine = PriorInformationEngine()

        # Deterministic RNG for reproducible results
        self.rng = get_rng(SeedScope.RATING_SYSTEM)

        # Bayesian Neural Network parameters
        self.layer_size = 12  # Larger hidden layer for Bayesian approach
        
        # Weight distributions (mean and variance for Bayesian weights)
        # Prior Network: Blade-Chest + Prior Info -> TrueSkill Priors (no self-loops)
        self.W_bc_to_hidden_mean = self.rng.normal(0, 0.2, (6, self.layer_size))  # [agent_blade_norm, policy_blade_norm, agent_chest_norm, policy_chest_norm, prior_confidence, prior_alignment] -> hidden
        self.W_bc_to_hidden_var = np.ones((6, self.layer_size)) * 0.01  # Initial uncertainty

        self.W_prior_to_hidden_mean = self.rng.normal(0, 0.15, (5, self.layer_size))  # [prior_mu, prior_sigma, prior_info_gain, prior_bias_strength, prior_type_encoding] -> hidden
        self.W_prior_to_hidden_var = np.ones((5, self.layer_size)) * 0.008
        
        # Hidden to TrueSkill prior parameters (mean and variance of rating)
        self.W_hidden_to_mu_mean = self.rng.normal(0, 0.1, (self.layer_size, 1))  # hidden -> prior_mu
        self.W_hidden_to_mu_var = np.ones((self.layer_size, 1)) * 0.005

        self.W_hidden_to_sigma_mean = self.rng.normal(0, 0.1, (self.layer_size, 1))  # hidden -> prior_sigma
        self.W_hidden_to_sigma_var = np.ones((self.layer_size, 1)) * 0.005
        
        # Bias distributions
        self.bias_hidden_mean = np.zeros(self.layer_size)
        self.bias_hidden_var = np.ones(self.layer_size) * 0.01
        
        self.bias_mu_mean = 0.0
        self.bias_mu_var = 0.01
        
        self.bias_sigma_mean = 0.0
        self.bias_sigma_var = 0.01
        
        # Bayesian learning parameters
        self.learning_rate = 0.005  # Lower for Bayesian approach
        self.prior_precision = 100.0  # Regularization strength
        self.likelihood_precision = 50.0  # Data fit strength
        
        # Sample count for Monte Carlo estimation
        self.n_samples = 10
        
        # Prior belief parameters
        self.default_prior_mu = 1200.0
        self.default_prior_sigma = 400.0
        self.prior_trust_factor = 0.3  # How much to trust prior info vs. data
        
        self.agents_map = {agent.agent_id: agent for agent in agents}
        
    def activation(self, x):
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, 0.01 * x)
    
    def activation_derivative(self, x):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1.0, 0.01)
    
    def sample_weights(self):
        """Sample weights from their posterior distributions"""
        W_bc_hidden = self.rng.normal(self.W_bc_to_hidden_mean, np.sqrt(self.W_bc_to_hidden_var))
        W_prior_hidden = self.rng.normal(self.W_prior_to_hidden_mean, np.sqrt(self.W_prior_to_hidden_var))
        W_hidden_mu = self.rng.normal(self.W_hidden_to_mu_mean, np.sqrt(self.W_hidden_to_mu_var)).flatten()
        W_hidden_sigma = self.rng.normal(self.W_hidden_to_sigma_mean, np.sqrt(self.W_hidden_to_sigma_var)).flatten()

        bias_hidden = self.rng.normal(self.bias_hidden_mean, np.sqrt(self.bias_hidden_var))
        bias_mu = self.rng.normal(self.bias_mu_mean, np.sqrt(self.bias_mu_var))
        bias_sigma = self.rng.normal(self.bias_sigma_mean, np.sqrt(self.bias_sigma_var))
        
        return {
            'W_bc_hidden': W_bc_hidden,
            'W_prior_hidden': W_prior_hidden, 
            'W_hidden_mu': W_hidden_mu,
            'W_hidden_sigma': W_hidden_sigma,
            'bias_hidden': bias_hidden,
            'bias_mu': bias_mu,
            'bias_sigma': bias_sigma
        }
    
    def compute_trueskill_prior(self, agent_id: str, policy_id: str, objective: str = 'color_distance'):
        """
        Compute Bayesian prior for TrueSkill rating based on Blade-Chest and prior information.
        No self-loops: Only uses Blade-Chest and prior info to influence TrueSkill.
        """
        # Extract Blade-Chest features (not TrueSkill features to avoid self-loops)
        agent_blade = self.blade_chest_engine.get_blade_vector(objective, agent_id, 'agent')
        policy_blade = self.blade_chest_engine.get_blade_vector(objective, policy_id, 'policy')
        agent_chest = self.blade_chest_engine.get_chest_vector(objective, agent_id, 'agent')
        policy_chest = self.blade_chest_engine.get_chest_vector(objective, policy_id, 'policy')
        
        # Normalize blade-chest features
        agent_blade_norm = np.linalg.norm(agent_blade) / 5.0
        policy_blade_norm = np.linalg.norm(policy_blade) / 5.0
        agent_chest_norm = np.linalg.norm(agent_chest) / 5.0
        policy_chest_norm = np.linalg.norm(policy_chest) / 5.0
        
        # Get prior information features
        # Use the first available prior or default
        available_priors = self.prior_engine.get_all_prior_ids()
        selected_prior = available_priors[hash(agent_id + policy_id) % len(available_priors)]
        
        prior_rating = self.trueskill_engine.get_rating(selected_prior, 'prior')
        # Use neutral 0.5 for features that lack real data; avoids hash-based noise
        prior_confidence = 0.5
        prior_alignment = 0.5

        # Encode prior features
        prior_mu_norm = prior_rating.mu / 1200.0
        prior_sigma_norm = prior_rating.sigma / 400.0
        prior_info_gain_norm = prior_rating.information_gain / 100.0
        prior_bias_strength = prior_confidence
        prior_type_encoding = 0.0  # no meaningful type encoding available

        # Create feature vectors (no TrueSkill features to avoid self-loops)
        bc_features = np.array([
            agent_blade_norm, policy_blade_norm, agent_chest_norm,
            policy_chest_norm, prior_confidence, prior_alignment
        ])

        prior_features = np.array([
            prior_mu_norm, prior_sigma_norm, prior_info_gain_norm,
            prior_bias_strength, prior_type_encoding
        ])
        
        # Monte Carlo estimation over weight samples
        mu_samples = []
        sigma_samples = []
        
        for _ in range(self.n_samples):
            weights = self.sample_weights()
            
            # Forward pass through Bayesian network
            hidden_from_bc = np.dot(bc_features, weights['W_bc_hidden'])
            hidden_from_prior = np.dot(prior_features, weights['W_prior_hidden'])
            hidden_input = hidden_from_bc + hidden_from_prior + weights['bias_hidden']
            hidden_output = self.activation(hidden_input)
            
            # Compute prior parameters
            prior_mu_raw = np.dot(hidden_output, weights['W_hidden_mu']) + weights['bias_mu']
            prior_sigma_raw = np.dot(hidden_output, weights['W_hidden_sigma']) + weights['bias_sigma']
            
            # Transform to valid ranges without aggressive bounds
            prior_mu = self.default_prior_mu + prior_mu_raw * 200.0  # Allow ±200 from default
            
            # Clamp prior_sigma_raw to prevent overflow
            prior_sigma_raw_clamped = np.clip(prior_sigma_raw, -3.0, 3.0)
            prior_sigma = self.default_prior_sigma * np.exp(prior_sigma_raw_clamped * 0.5)  # Log-normal for sigma
            prior_sigma = np.clip(prior_sigma, 50.0, 800.0)  # Reasonable bounds
            
            mu_samples.append(prior_mu)
            sigma_samples.append(prior_sigma)
        
        # Compute posterior estimates with enhanced numerical stability
        if not mu_samples or not sigma_samples:
            return self.default_prior_mu, self.default_prior_sigma
        
        mu_samples_array = np.array(mu_samples)
        sigma_samples_array = np.array(sigma_samples)
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(mu_samples_array)) or np.any(~np.isfinite(sigma_samples_array)):
            return self.default_prior_mu, self.default_prior_sigma
        
        # Robust statistics for mu estimation
        estimated_mu = np.median(mu_samples_array)  # Use median for robustness
        estimated_mu = np.clip(estimated_mu, 100.0, 2500.0)  # Reasonable bounds
        
        # Enhanced sigma estimation with bounds checking
        sigma_samples_clamped = np.clip(sigma_samples_array, 50.0, 800.0)
        if len(sigma_samples_clamped) == 0:
            return estimated_mu, self.default_prior_sigma
            
        # Use percentile-based approach for robustness
        estimated_sigma = np.percentile(sigma_samples_clamped, 75)  # Use 75th percentile
        estimated_sigma = np.clip(estimated_sigma, 50.0, 800.0)
        
        return estimated_mu, estimated_sigma
    
    def _calculate_prior_alignment(self, agent_id: str, policy_id: str, prior_id: str) -> float:
        """Calculate alignment between agent/policy and prior information"""
        # Simple heuristic based on IDs
        agent_hash = hash(agent_id) % 1000
        policy_hash = hash(policy_id) % 1000
        prior_hash = hash(prior_id) % 1000
        
        # Calculate similarity (0-1 range)
        alignment = 1.0 - abs(agent_hash - prior_hash) / 1000.0
        alignment += 1.0 - abs(policy_hash - prior_hash) / 1000.0
        return alignment / 2.0
    
    def update_bayesian_weights(self, bc_features, prior_features, target_mu, target_sigma, predicted_mu, predicted_sigma):
        """Update Bayesian weight distributions using variational inference"""
        
        # Calculate prediction errors
        mu_error = target_mu - predicted_mu
        sigma_error = target_sigma - predicted_sigma
        
        # Compute gradients for each weight distribution
        # For simplicity, we'll update only the means and keep variances fixed or slowly decaying
        
        # Hidden layer gradients (averaged over samples) with numerical stability
        hidden_mu_grad = np.clip(mu_error * self.likelihood_precision / max(self.n_samples, 1), -100.0, 100.0)
        hidden_sigma_grad = np.clip(sigma_error * self.likelihood_precision / max(self.n_samples, 1), -100.0, 100.0)
        
        # Update weight means using gradients with clipping
        # Blade-chest to hidden updates
        bc_to_hidden_grad = np.outer(bc_features, np.ones(self.layer_size)) * hidden_mu_grad
        bc_to_hidden_grad = np.clip(bc_to_hidden_grad, -10.0, 10.0)  # Clip gradients
        self.W_bc_to_hidden_mean += self.learning_rate * bc_to_hidden_grad
        
        # Prior to hidden updates  
        prior_to_hidden_grad = np.outer(prior_features, np.ones(self.layer_size)) * hidden_mu_grad
        prior_to_hidden_grad = np.clip(prior_to_hidden_grad, -10.0, 10.0)  # Clip gradients
        self.W_prior_to_hidden_mean += self.learning_rate * prior_to_hidden_grad
        
        # Hidden to output updates
        hidden_avg = np.ones(self.layer_size) * 0.5  # Approximate hidden activation
        mu_update = hidden_avg.reshape(-1, 1) * hidden_mu_grad
        sigma_update = hidden_avg.reshape(-1, 1) * hidden_sigma_grad
        self.W_hidden_to_mu_mean += self.learning_rate * np.clip(mu_update, -5.0, 5.0)
        self.W_hidden_to_sigma_mean += self.learning_rate * np.clip(sigma_update, -5.0, 5.0)
        
        # Update biases
        bias_hidden_update = hidden_mu_grad * np.ones(self.layer_size)
        self.bias_hidden_mean += self.learning_rate * np.clip(bias_hidden_update, -5.0, 5.0)
        self.bias_mu_mean += self.learning_rate * np.clip(hidden_mu_grad, -5.0, 5.0)
        self.bias_sigma_mean += self.learning_rate * np.clip(hidden_sigma_grad, -5.0, 5.0)
        
        # Decay weight variances slightly to represent increasing confidence
        decay_factor = 0.99
        self.W_bc_to_hidden_var *= decay_factor
        self.W_prior_to_hidden_var *= decay_factor
        self.W_hidden_to_mu_var *= decay_factor
        self.W_hidden_to_sigma_var *= decay_factor
        self.bias_hidden_var *= decay_factor
        self.bias_mu_var *= decay_factor
        self.bias_sigma_var *= decay_factor
        
        # Ensure minimum variance
        min_var = 1e-6
        self.W_bc_to_hidden_var = np.maximum(self.W_bc_to_hidden_var, min_var)
        self.W_prior_to_hidden_var = np.maximum(self.W_prior_to_hidden_var, min_var)
        self.W_hidden_to_mu_var = np.maximum(self.W_hidden_to_mu_var, min_var)
        self.W_hidden_to_sigma_var = np.maximum(self.W_hidden_to_sigma_var, min_var)
        self.bias_hidden_var = np.maximum(self.bias_hidden_var, min_var)
        self.bias_mu_var = max(self.bias_mu_var, min_var)
        self.bias_sigma_var = max(self.bias_sigma_var, min_var)
    
    def extract_bc_prior_features(self, agent_id: str, policy_id: str, objective: str):
        """Extract Blade-Chest and prior features (no TrueSkill to avoid self-loops)"""
        
        # Blade-Chest features
        agent_blade = self.blade_chest_engine.get_blade_vector(objective, agent_id, 'agent')
        policy_blade = self.blade_chest_engine.get_blade_vector(objective, policy_id, 'policy')
        agent_chest = self.blade_chest_engine.get_chest_vector(objective, agent_id, 'agent')
        policy_chest = self.blade_chest_engine.get_chest_vector(objective, policy_id, 'policy')
        
        # Normalize blade-chest features
        agent_blade_norm = np.linalg.norm(agent_blade) / 5.0
        policy_blade_norm = np.linalg.norm(policy_blade) / 5.0
        agent_chest_norm = np.linalg.norm(agent_chest) / 5.0
        policy_chest_norm = np.linalg.norm(policy_chest) / 5.0
        
        # Select and get prior information
        available_priors = self.prior_engine.get_all_prior_ids()
        selected_prior = available_priors[hash(agent_id + policy_id) % len(available_priors)]
        
        prior_rating = self.trueskill_engine.get_rating(selected_prior, 'prior')
        prior_confidence = 0.5
        prior_alignment = 0.5  # neutral; avoids hash-based noise

        bc_features = np.array([
            agent_blade_norm, policy_blade_norm, agent_chest_norm,
            policy_chest_norm, prior_confidence, prior_alignment
        ])

        prior_features = np.array([
            prior_rating.mu / 1200.0,
            prior_rating.sigma / 400.0,
            prior_rating.information_gain / 100.0,
            prior_confidence,
            0.0,  # no meaningful type encoding available
        ])
        
        return bc_features, prior_features
    
    def update_ratings(self, objective_performances, pairwise_outcomes_map):
        """Update rating systems with Bayesian neural network approach"""
        
        # First, update Blade-Chest systems (these provide input to our Bayesian NN)
        for objective_name, performances in objective_performances.items():
            agent_policy_pairs = [(agent_id, policy_id) for agent_id, policy_id, _ in performances]
            pairwise_outcomes = pairwise_outcomes_map.get(objective_name, [])
            prior_ids = self.prior_engine.get_all_prior_ids()
            if pairwise_outcomes:
                self.blade_chest_engine.update_embeddings(objective_name, agent_policy_pairs, pairwise_outcomes, prior_ids)
        
        # Update prior information ratings
        self.update_prior_information_ratings(objective_performances)
        
        # Now update TrueSkill using Bayesian priors computed from Blade-Chest + Prior info
        primary_objective_performances = objective_performances.get('combined_score', [])
        if not primary_objective_performances:
            primary_objective_performances = objective_performances.get('color_distance', [])
        
        if primary_objective_performances:
            # For each agent-policy pair, update BNN weights from BC+prior features.
            # BC captures intransitivity only — it does NOT produce TrueSkill priors.
            # TrueSkill ratings are updated separately below via standard EP (update_ratings).
            for agent_id, policy_id, actual_score in primary_objective_performances:

                # Compute Bayesian prior from Blade-Chest and prior information.
                # Used only to drive BNN weight updates — NOT injected into TrueSkill ratings.
                prior_mu, prior_sigma = self.compute_trueskill_prior(agent_id, policy_id, 'color_distance')

                # Update Bayesian network weights based on prediction accuracy
                bc_features, prior_features = self.extract_bc_prior_features(agent_id, policy_id, 'color_distance')
                
                # Target values based on actual performance.
                # actual_score is a z-score (higher = better), typically in [-3, +3].
                # Map to rating space: 0 -> default, +1 sigma -> +200 pts.
                target_mu = self.default_prior_mu + actual_score * 200.0
                target_sigma = max(50.0, self.default_prior_sigma - abs(actual_score) * 100.0)
                
                # Update Bayesian weights
                self.update_bayesian_weights(bc_features, prior_features, target_mu, target_sigma, prior_mu, prior_sigma)
            
            # Run standard TrueSkill EP update from actual performance outcomes.
            # TrueSkillEngine sorts ascending (lower = better), but obj_perfs uses
            # higher = better convention.  Negate scores to align.
            negated = [(a, p, -s) for a, p, s in primary_objective_performances]
            self.trueskill_engine.update_ratings(negated)
    
    def predict_outcome(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """Predict outcome using Bayesian neural network with prior-influenced TrueSkill"""
        
        # Get Bayesian prior-influenced ratings
        prior_mu_a, prior_sigma_a = self.compute_trueskill_prior(agent_id_a, policy_id_a, objective)
        prior_mu_b, prior_sigma_b = self.compute_trueskill_prior(agent_id_b, policy_id_b, objective)
        
        # Get current TrueSkill ratings
        ts_rating_a_agent = self.trueskill_engine.get_rating(agent_id_a, 'agent')
        ts_rating_a_policy = self.trueskill_engine.get_rating(policy_id_a, 'policy')
        ts_rating_b_agent = self.trueskill_engine.get_rating(agent_id_b, 'agent')
        ts_rating_b_policy = self.trueskill_engine.get_rating(policy_id_b, 'policy')
        
        # Use Bayesian priors directly (no mixing with TrueSkill to avoid circularity)
        combined_mu_a = prior_mu_a
        combined_mu_b = prior_mu_b
        combined_sigma_a = max(50.0, min(800.0, prior_sigma_a))
        combined_sigma_b = max(50.0, min(800.0, prior_sigma_b))
        
        # Add Blade-Chest intransitivity prediction
        blade_chest_score = self.blade_chest_engine.predict_outcome(
            objective, agent_id_a, policy_id_a, agent_id_b, policy_id_b,
            uncertainty_a=combined_sigma_a, uncertainty_b=combined_sigma_b,
            info_gain_a=ts_rating_a_agent.information_gain + ts_rating_a_policy.information_gain,
            info_gain_b=ts_rating_b_agent.information_gain + ts_rating_b_policy.information_gain
        )
        
        # Combine predictions
        rating_diff = combined_mu_a - combined_mu_b
        uncertainty_sum = combined_sigma_a**2 + combined_sigma_b**2
        
        # TrueSkill-style probability
        c = np.sqrt(uncertainty_sum + 2 * (100.0**2))  # Beta parameter
        trueskill_prob = norm.cdf(rating_diff / c)
        
        # Blend TrueSkill and Blade-Chest predictions
        # Clamp blade_chest_score to prevent overflow
        blade_chest_score_clamped = np.clip(blade_chest_score, -500, 500)
        blade_chest_prob = 1 / (1 + np.exp(-blade_chest_score_clamped))
        
        # Weight combination: more weight to TrueSkill for established ratings, more to Blade-Chest for uncertain ones
        uncertainty_factor = (combined_sigma_a + combined_sigma_b) / (2 * self.default_prior_sigma)
        trueskill_weight = max(0.3, 1.0 - uncertainty_factor)
        bladechest_weight = 1.0 - trueskill_weight
        
        final_prob = trueskill_weight * trueskill_prob + bladechest_weight * blade_chest_prob
        return final_prob
    
    def get_rating(self, entity_id: str, rating_type: str) -> Rating:
        """Get rating from TrueSkill system (base system)"""
        return self.trueskill_engine.get_rating(entity_id, rating_type)
    
    def update_prior_information_ratings(self, objective_performances):
        """Update prior information ratings"""
        for objective_name, performances in objective_performances.items():
            if not performances:
                continue
                
            # Simulate performance for each prior information type
            prior_performances = []
            
            for prior_id in self.prior_engine.get_all_prior_ids():
                # Calculate average hypothetical performance for this prior
                total_score = 0.0
                count = 0
                
                for agent_id, policy_id, actual_score in performances:
                    # Simulate what this prior would have scored
                    base_score = actual_score
                    
                    # Add prior-specific bias
                    if 'color_distance' in objective_name.lower():
                        bias_map = {
                            'balanced_colors': 0.95,
                            'high_saturation': 0.85,
                            'low_saturation': 0.88,
                            'warm_colors': 0.92,
                            'cool_colors': 0.91
                        }
                        bias = bias_map.get(prior_id, 0.90)
                    else:
                        bias = 0.90
                    
                    prior_score = base_score * bias + self.rng.normal(0, 3)
                    total_score += prior_score
                    count += 1
                
                if count > 0:
                    avg_score = total_score / count
                    prior_performances.append((prior_id, avg_score))
            
            # Update TrueSkill ratings for prior information sources
            if prior_performances:
                self.trueskill_engine.update_ratings_batch(prior_performances, 'prior')
    
    def get_bayesian_weights_summary(self):
        """Get summary of Bayesian neural network weights for debugging"""
        return {
            'bc_to_hidden_mean_norm': np.linalg.norm(self.W_bc_to_hidden_mean),
            'bc_to_hidden_var_mean': np.mean(self.W_bc_to_hidden_var),
            'prior_to_hidden_mean_norm': np.linalg.norm(self.W_prior_to_hidden_mean),
            'prior_to_hidden_var_mean': np.mean(self.W_prior_to_hidden_var),
            'hidden_to_mu_mean_norm': np.linalg.norm(self.W_hidden_to_mu_mean),
            'hidden_to_sigma_mean_norm': np.linalg.norm(self.W_hidden_to_sigma_mean),
            'learning_rate': self.learning_rate,
            'prior_trust_factor': self.prior_trust_factor,
            'n_samples': self.n_samples
        }


# Backward compatibility alias
class NeuralRatingSystem(BayesianNeuralRatingSystem):
    """Backward compatibility alias for BayesianNeuralRatingSystem"""
    
    def __init__(self, agents: List[Any]):
        super().__init__(agents)
    
    def get_neural_weights_summary(self):
        """Backward compatibility method"""
        return self.get_bayesian_weights_summary()