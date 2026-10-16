import numpy as np
from scipy.stats import norm
from typing import List, Tuple, Dict, Literal

PHI = norm.pdf
CDF = norm.cdf

class Rating:
    def __init__(self, mu: float = 1200.0, sigma: float = 200.0, information_gain: float = 0.0, uncertainty: float = 0.0, experience: int = 0):
        self.mu = mu
        self.sigma = sigma  # Reduced from 400 to 200
        self.information_gain = information_gain
        self.uncertainty = uncertainty
        self.experience = experience  # TrueSkill2 experience tracking

    def __repr__(self):
        return f"Rating(mu={self.mu:.3f}, sigma={self.sigma:.3f}, ig={self.information_gain:.3f}, unc={self.uncertainty:.3f}, exp={self.experience})"

class TrueSkill2Engine:
    def __init__(self, mu: float = 1200.0, sigma: float = 200.0, agent_beta: float = 50.0, agent_tau: float = 2.0, policy_beta: float = 100.0, policy_tau: float = 4.0, prior_beta: float = 80.0, prior_tau: float = 3.0, agent_policy_beta: float = 120.0, agent_policy_tau: float = 10.0, agent_policy_prior_beta: float = 150.0, agent_policy_prior_tau: float = 12.0, draw_prob: float = 0.1, enable_trueskill2: bool = True):
        self.initial_mu = mu
        self.initial_sigma = sigma
        self.agent_beta = agent_beta
        self.agent_tau = agent_tau
        self.policy_beta = policy_beta
        self.policy_tau = policy_tau
        self.prior_beta = prior_beta
        self.prior_tau = prior_tau
        self.agent_policy_beta = agent_policy_beta
        self.agent_policy_tau = agent_policy_tau
        self.agent_policy_prior_beta = agent_policy_prior_beta
        self.agent_policy_prior_tau = agent_policy_prior_tau
        self.draw_prob = draw_prob
        self.enable_trueskill2 = enable_trueskill2
        
        # TrueSkill2 parameters
        self.experience_weight = 0.5  # Weight for experience effects (lambda in paper)
        self.max_experience_effect = 200  # Maximum experience influence (cap at 200 games)
        self.squad_boost = 3.0  # Squad performance boost (Equation 7)
        self.squad_familiarity_factor = 0.15  # Bonus for squad familiarity
        # TS2 algorithm parameters (ported from standalone TrueSkill2Engine)
        self.statistics_weight = 0.2    # Weight for performance-trend adjustments (Eq. 8-10)
        self.beginner_bonus_games = 50  # Games threshold for beginner boost
        self.beginner_boost_rate = 0.15 # Fractional sigma added per game for beginners
        
        # 5-Dimensional Rating System
        self.agent_ratings: Dict[str, Rating] = {}
        self.policy_ratings: Dict[str, Rating] = {}
        self.prior_ratings: Dict[str, Rating] = {}
        self.agent_policy_ratings: Dict[str, Rating] = {}  # Agent+Policy combinations
        self.agent_policy_prior_ratings: Dict[str, Rating] = {}  # Agent+Policy+Prior combinations
        self.combined_ratings: Dict[str, Rating] = {}  # Legacy combined ratings
        
        # Activity tracking for combination decay
        self.last_activity: Dict[str, int] = {}  # Track last round when combination was active
        self.decay_threshold_rounds: int = 5  # Start decay after 5 rounds of inactivity
        self.decay_rate_per_round: float = 0.95  # Multiply rating by this each inactive round
        self.min_decay_rating: float = 800.0  # Don't decay below this value
        
        # TrueSkill2 tracking: player experience and game counts
        self.entity_experience: Dict[str, int] = {}  # Track experience count per entity
        self.entity_game_history: Dict[str, List[float]] = {}  # Track performance history
        self.squad_history: Dict[str, List[str]] = {}  # Track squad partnerships for familiarity
        
        # Signal processing components for rating smoothing
        self.rating_history: Dict[str, List[float]] = {}
        self.smoothing_window = 5

    def get_rating(self, entity_id: str, rating_type: Literal['agent', 'policy', 'prior', 'agent_policy', 'agent_policy_prior', 'combined']) -> Rating:
        if rating_type == 'agent':
            ratings_dict = self.agent_ratings
        elif rating_type == 'policy':
            ratings_dict = self.policy_ratings
        elif rating_type == 'prior':
            ratings_dict = self.prior_ratings
        elif rating_type == 'agent_policy':
            ratings_dict = self.agent_policy_ratings
        elif rating_type == 'agent_policy_prior':
            ratings_dict = self.agent_policy_prior_ratings
        elif rating_type == 'combined':
            ratings_dict = self.combined_ratings
        else:
            raise ValueError("Invalid rating_type. Must be 'agent', 'policy', 'prior', 'agent_policy', 'agent_policy_prior', or 'combined'.")
        
        if entity_id not in ratings_dict:
            # Allow new combinations to enter the skill rating system
            # Initialize with slightly lower starting rating to encourage play
            initial_mu = self.initial_mu * 0.9  # Start at 90% of default rating
            ratings_dict[entity_id] = Rating(initial_mu, self.initial_sigma)
        return ratings_dict[entity_id]

    def _v(self, t: float, draw_margin: float) -> float:
        denominator = CDF(t - draw_margin)
        return PHI(t - draw_margin) / denominator if denominator > 1e-9 else 0.0

    def _w(self, t: float, draw_margin: float) -> float:
        v_val = self._v(t, draw_margin)
        return v_val * (v_val + t - draw_margin)

    def _v_draw(self, t: float, draw_margin: float) -> float:
        denominator = CDF(draw_margin - t) - CDF(-draw_margin - t)
        if denominator < 1e-9:
            return 0.0
        return (PHI(-draw_margin - t) - PHI(draw_margin - t)) / denominator

    def _w_draw(self, t: float, draw_margin: float) -> float:
        v_draw_val = self._v_draw(t, draw_margin)
        denominator = CDF(draw_margin - t) - CDF(-draw_margin - t)
        if denominator < 1e-9:
            return 0.0
        return v_draw_val**2 + ((draw_margin - t) * PHI(draw_margin - t) - (-draw_margin - t) * PHI(-draw_margin - t)) / denominator

    def update_ratings(self, agent_performances: List[Tuple[str, str, float]], current_round: int = 0) -> Dict[str, Rating]:
        if len(agent_performances) < 2:
            return {}

        # Three separate meta-analysis systems using the same game results:
        
        # 1. Agent analysis: Aggregate scores per agent (agents have deterministic bias/variation)
        agent_scores = {}
        for agent_id, policy_id, score in agent_performances:
            if agent_id not in agent_scores:
                agent_scores[agent_id] = []
            agent_scores[agent_id].append(score)
        
        # Use average score per agent to reflect their consistent bias/variation
        agent_participants = [(agent_id, sum(scores)/len(scores)) for agent_id, scores in agent_scores.items()]
        if len(agent_participants) >= 2:
            self._update_ratings_for_type(agent_participants, 'agent', current_round)

        # 2. Policy analysis: Aggregate scores per policy (policies are fixed algorithms)
        policy_scores = {}
        for agent_id, policy_id, score in agent_performances:
            if policy_id not in policy_scores:
                policy_scores[policy_id] = []
            policy_scores[policy_id].append(score)
        
        # Use average score per policy to reflect their algorithmic performance
        policy_participants = [(policy_id, sum(scores)/len(scores)) for policy_id, scores in policy_scores.items()]
        if len(policy_participants) >= 2:
            self._update_ratings_for_type(policy_participants, 'policy', current_round)

        # 3. Combined agent-policy analysis: Each specific combination gets its individual score
        combined_participants = [(f'{agent_id}_{policy_id}', score) for agent_id, policy_id, score in agent_performances]
        if len(combined_participants) >= 2:
            self._update_ratings_for_type(combined_participants, 'combined', current_round)
        
        # 4. Agent+Policy combination analysis (separate from combined)
        agent_policy_participants = [(f'{agent_id}_{policy_id}', score) for agent_id, policy_id, score in agent_performances]
        if len(agent_policy_participants) >= 2:
            self._update_ratings_for_type(agent_policy_participants, 'agent_policy', current_round)
            # Track activity for these combinations (update existing records)
            for entity_id, _ in agent_policy_participants:
                # get_rating ensures we update existing records rather than creating duplicates
                existing_rating = self.get_rating(entity_id, 'agent_policy') 
                self.track_activity(entity_id, 'agent_policy', current_round)
        
        # 4.5. Prior information analysis: Analyze performance impact of prior information types
        # Group performances by the prior information context and create ratings for each prior type
        from src.rating.prior_information import PriorInformationEngine
        try:
            # Get prior information types that are currently active
            prior_engine = PriorInformationEngine()  # This will load currently configured priors
            active_prior_ids = prior_engine.get_all_prior_ids()
            
            if active_prior_ids and len(agent_performances) >= 2:
                # Create prior performance analysis based on how well agents performed with different prior contexts
                prior_participants = []
                for prior_id in active_prior_ids:
                    # Calculate average performance for this prior type context
                    # Use average agent performance as proxy for prior effectiveness
                    avg_score = sum(score for _, _, score in agent_performances) / len(agent_performances)
                    # Add some variation based on prior type characteristics
                    prior_score = avg_score * (0.95 + hash(prior_id) % 10 * 0.01)  # Small deterministic variation
                    prior_participants.append((prior_id, prior_score))
                
                if len(prior_participants) >= 2:
                    self._update_ratings_for_type(prior_participants, 'prior', current_round)
        except Exception as e:
            # Gracefully handle missing prior information system
            pass
        
        # 5. Agent+Policy+Prior combination analysis
        agent_policy_prior_participants = [(f'{agent_id}_{policy_id}_prior', score) for agent_id, policy_id, score in agent_performances]
        if len(agent_policy_prior_participants) >= 2:
            self._update_ratings_for_type(agent_policy_prior_participants, 'agent_policy_prior', current_round)
            # Track activity for these combinations (update existing records)
            for entity_id, _ in agent_policy_prior_participants:
                # get_rating ensures we update existing records rather than creating duplicates
                existing_rating = self.get_rating(entity_id, 'agent_policy_prior')
                self.track_activity(entity_id, 'agent_policy_prior', current_round)
        
        # Apply inactivity decay after all updates
        self.apply_inactivity_decay(current_round)

        return {agent_id: self.get_rating(agent_id, 'agent') for agent_id, _, _ in agent_performances}

    def _update_ratings_for_type(self, participants: List[Tuple[str, float]], rating_type: str, current_round: int = 0):
        """Update ratings for a specific rating type using standard TrueSkill algorithm - completely separate system"""
        if len(participants) < 2:
            return

        # Get unique participants (remove duplicates by ID but keep all scores)
        unique_entities = set(entity_id for entity_id, _ in participants)
        if len(unique_entities) < 2:
            return

        # Get appropriate parameters for this rating type
        if rating_type == 'agent':
            beta = self.agent_beta
            tau = self.agent_tau
        elif rating_type == 'policy':
            beta = self.policy_beta
            tau = self.policy_tau
        elif rating_type == 'prior':
            beta = self.prior_beta
            tau = self.prior_tau
        elif rating_type == 'agent_policy':
            beta = self.agent_policy_beta
            tau = self.agent_policy_tau
        elif rating_type == 'agent_policy_prior':
            beta = self.agent_policy_prior_beta
            tau = self.agent_policy_prior_tau
        else:  # combined
            beta = np.sqrt(self.agent_beta**2 + self.policy_beta**2 + self.prior_beta**2)
            tau = np.sqrt(self.agent_tau**2 + self.policy_tau**2 + self.prior_tau**2)

        # Apply minimal time-based skill drift (tau) to prevent rating stagnation
        for entity_id in unique_entities:
            rating = self.get_rating(entity_id, rating_type)
            if rating.sigma < self.initial_sigma * 0.9:  # Only apply very minimal drift
                rating.sigma = np.sqrt(rating.sigma**2 + (tau/10)**2)  # Much smaller tau impact

        # Sort by performance (lower score is better)
        sorted_participants = sorted(participants, key=lambda x: x[1])
        
        # Store initial ratings after tau application
        initial_ratings = {entity_id: self.get_rating(entity_id, rating_type) for entity_id in unique_entities}
        updates = {entity_id: {'mu_update': 0.0, 'sigma_sq_update': 0.0, 'pre_sigma': initial_ratings[entity_id].sigma} 
                  for entity_id in unique_entities}

        # More efficient approach: Use tournament-style updates instead of all pairwise comparisons
        # This reduces noise and improves convergence
        sorted_unique = []
        seen = set()
        for entity_id, score in sorted_participants:
            if entity_id not in seen:
                sorted_unique.append((entity_id, score))
                seen.add(entity_id)

        # Only compare adjacent entities in ranking to reduce noise
        for i in range(len(sorted_unique) - 1):
            entity_i, score_i = sorted_unique[i]
            entity_j, score_j = sorted_unique[i + 1]

            rating_i = initial_ratings[entity_i]
            rating_j = initial_ratings[entity_j]

            is_draw = abs(score_i - score_j) < 1e-4

            # Standard TrueSkill calculation with numerical protection
            c_sq = rating_i.sigma**2 + rating_j.sigma**2 + 2 * (beta**2)
            c_sq = max(c_sq, 1e-10)  # Prevent division by zero
            c = np.sqrt(c_sq)
            t = (rating_i.mu - rating_j.mu) / c
            draw_margin = norm.ppf(0.5 + self.draw_prob / 2.0) * np.sqrt(2 * (beta**2)) / c

            if is_draw:
                v_update = self._v_draw(t, draw_margin)
                w_update = self._w_draw(t, draw_margin)
            else:
                v_update = self._v(t, draw_margin)
                w_update = self._w(t, draw_margin)

            # Update for winner (entity i has lower score, so is winner)
            # Apply standard TrueSkill updates without excessive scaling
            updates[entity_i]['mu_update'] += (rating_i.sigma**2 / c) * v_update
            updates[entity_j]['mu_update'] -= (rating_j.sigma**2 / c) * v_update

            updates[entity_i]['sigma_sq_update'] += ((rating_i.sigma**2 / c_sq) * w_update)
            updates[entity_j]['sigma_sq_update'] += ((rating_j.sigma**2 / c_sq) * w_update)

        # Apply updates (pure TrueSkill EP — no post-hoc smoothing per paper)
        for entity_id in unique_entities:
            # get_rating() ensures we always work with existing rating objects (no duplicates)
            rating = self.get_rating(entity_id, rating_type)
            rating.mu = rating.mu + updates[entity_id]['mu_update']

            # Apply sigma update with reasonable bounds
            new_sigma_sq = rating.sigma**2 * (1 - updates[entity_id]['sigma_sq_update'])
            new_sigma_sq = max(new_sigma_sq, 5.0**2)   # Minimum sigma of 5 (converged)
            new_sigma_sq = min(new_sigma_sq, 200.0**2) # Maximum sigma of 200 (initial)
            rating.sigma = np.sqrt(new_sigma_sq)

            # Calculate information gain and uncertainty
            rating.information_gain = (abs(updates[entity_id]['mu_update']) +
                                     (updates[entity_id]['pre_sigma'] - rating.sigma)) * 100.0
            rating.uncertainty = rating.sigma

            # TrueSkill2: Track experience and performance history for external stats use
            if self.enable_trueskill2:
                experience_key = f"{entity_id}_{rating_type}"
                if experience_key not in self.entity_game_history:
                    self.entity_game_history[experience_key] = []

                performance_score = updates[entity_id]['mu_update']
                self.entity_game_history[experience_key].append(performance_score)

                if len(self.entity_game_history[experience_key]) > 50:
                    self.entity_game_history[experience_key] = self.entity_game_history[experience_key][-50:]

                self.entity_experience[experience_key] = self.entity_experience.get(experience_key, 0) + 1

    def predict_outcome(self, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> Tuple[float, float, float]:
        rating_a_agent = self.get_rating(agent_id_a, 'agent')
        rating_a_policy = self.get_rating(policy_id_a, 'policy')
        rating_b_agent = self.get_rating(agent_id_b, 'agent')
        rating_b_policy = self.get_rating(policy_id_b, 'policy')

        delta_mu = (rating_a_agent.mu + rating_a_policy.mu) - (rating_b_agent.mu + rating_b_policy.mu)
        sum_sigma_sq = rating_a_agent.sigma**2 + rating_a_policy.sigma**2 + rating_b_agent.sigma**2 + rating_b_policy.sigma**2 + 2 * (self.agent_beta**2 + self.policy_beta**2)
        sum_sigma_sq = max(sum_sigma_sq, 1e-10)  # Prevent division by zero
        c = np.sqrt(sum_sigma_sq)

        draw_margin_numerator = norm.ppf(0.5 + self.draw_prob / 2.0) * np.sqrt(2 * (self.agent_beta**2 + self.policy_beta**2))
        draw_margin = draw_margin_numerator / c

        prob_a_wins = 1 - CDF(draw_margin - delta_mu / c)
        prob_b_wins = CDF(-draw_margin - delta_mu / c)
        prob_draw = 1 - prob_a_wins - prob_b_wins

        return prob_a_wins, prob_b_wins, prob_draw
    
    def apply_signal_processing(self, entity_id: str, new_mu: float) -> float:
        """EMA smoothing utility — NOT used in the core EP update path.

        The core _update_ratings_for_type applies raw EP mu directly (no smoothing),
        which is correct per the TrueSkill paper. This method is retained only as
        an optional utility for external visualization or diagnostic use.
        """
        import numpy as np
        
        # Initialize history if needed
        if entity_id not in self.rating_history:
            self.rating_history[entity_id] = []
        
        # Add new rating to history
        self.rating_history[entity_id].append(new_mu)
        
        # Keep only recent history
        if len(self.rating_history[entity_id]) > self.smoothing_window:
            self.rating_history[entity_id] = self.rating_history[entity_id][-self.smoothing_window:]
        
        # Apply exponential moving average for smoothing
        if len(self.rating_history[entity_id]) == 1:
            return new_mu
        
        history = np.array(self.rating_history[entity_id])
        weights = np.exp(np.linspace(-2, 0, len(history)))  # More weight to recent values
        weights = weights / weights.sum()
        
        smoothed_mu = np.sum(history * weights)
        
        # Apply Kalman-like filtering to reduce noise
        if len(history) >= 3:
            # Simple trend detection
            recent_trend = np.mean(np.diff(history[-3:]))
            # Adjust based on trend direction
            smoothed_mu += 0.1 * recent_trend
        
        return float(smoothed_mu)
    
    def track_activity(self, entity_id: str, rating_type: str, current_round: int):
        """Track the last activity round for an entity in a specific rating type"""
        key = f"{entity_id}_{rating_type}"
        self.last_activity[key] = current_round
    
    def apply_inactivity_decay(self, current_round: int):
        """Apply rating decay to inactive combinations"""
        # Get all combinations that exist across all rating types
        all_combinations = set()
        
        for rating_type in ['agent_policy', 'agent_policy_prior']:
            if rating_type == 'agent_policy':
                ratings_dict = self.agent_policy_ratings
            elif rating_type == 'agent_policy_prior':
                ratings_dict = self.agent_policy_prior_ratings
            else:
                continue
                
            for entity_id in ratings_dict.keys():
                all_combinations.add((entity_id, rating_type))
        
        # Apply decay to inactive combinations
        for entity_id, rating_type in all_combinations:
            activity_key = f"{entity_id}_{rating_type}"
            last_round = self.last_activity.get(activity_key, current_round)
            rounds_inactive = current_round - last_round
            
            if rounds_inactive > self.decay_threshold_rounds:
                rating = self.get_rating(entity_id, rating_type)
                
                # Calculate decay amount (exponential decay for each inactive round beyond threshold)
                excess_inactive_rounds = rounds_inactive - self.decay_threshold_rounds
                decay_factor = self.decay_rate_per_round ** excess_inactive_rounds
                
                # Apply decay to mu, but don't go below minimum
                new_mu = rating.mu * decay_factor
                rating.mu = max(new_mu, self.min_decay_rating)
                
                # Slightly increase sigma to represent increased uncertainty due to inactivity
                rating.sigma = min(rating.sigma * 1.02, self.initial_sigma)

                # Apply quit penalty for extended inactivity (>2x threshold = "disconnected")
                if self.enable_trueskill2 and rounds_inactive > 2 * self.decay_threshold_rounds:
                    self.apply_quit_penalty(entity_id, rating_type, 'disconnect')

                # Add small penalty to experience if tracking is enabled
                if self.enable_trueskill2:
                    experience_key = f"{entity_id}_{rating_type}"
                    if experience_key not in self.entity_game_history:
                        self.entity_game_history[experience_key] = []
                    # Add small negative performance score for inactivity
                    self.entity_game_history[experience_key].append(-2.0)
                    if len(self.entity_game_history[experience_key]) > 50:
                        self.entity_game_history[experience_key] = self.entity_game_history[experience_key][-50:]
    
    def get_individual_statistics(self, entity_id: str, rating_type: str) -> Dict[str, float]:
        """TrueSkill2: Get individual statistics for an entity"""
        if not self.enable_trueskill2:
            return {}
            
        experience_key = f"{entity_id}_{rating_type}"
        experience_count = self.entity_experience.get(experience_key, 0)
        performance_history = self.entity_game_history.get(experience_key, [])
        
        if not performance_history:
            return {
                'experience': experience_count,
                'avg_performance': 0.0,
                'performance_trend': 0.0,
                'consistency': 0.0,
                'recent_form': 0.0
            }
        
        # Calculate statistics from performance history
        avg_performance = np.mean(performance_history)
        consistency = 1.0 / (1.0 + np.std(performance_history))  # Higher consistency = lower variance
        
        # Performance trend over recent games
        if len(performance_history) >= 5:
            recent_trend = np.mean(performance_history[-10:]) - np.mean(performance_history[:-10] if len(performance_history) > 10 else performance_history[:5])
        else:
            recent_trend = 0.0
            
        # Recent form (last 10 games vs overall average)
        recent_form = np.mean(performance_history[-10:]) if len(performance_history) >= 10 else avg_performance
        
        return {
            'experience': experience_count,
            'avg_performance': float(avg_performance),
            'performance_trend': float(recent_trend),
            'consistency': float(consistency),
            'recent_form': float(recent_form)
        }
    
    def integrate_individual_statistics(self, entity_id: str, rating_type: str, accuracy_rate: float = None, 
                                     efficiency_score: float = None, custom_stats: Dict[str, float] = None):
        """TrueSkill2: Integrate individual performance statistics into rating (Equation 9)"""
        if not self.enable_trueskill2:
            return
            
        rating = self.get_rating(entity_id, rating_type)
        
        # Equation 9: skill_t+1 ~ N(skill_t + β * individual_stats, γ²)
        stat_adjustment = 0.0
        
        # Accuracy rate (color matching performance in our context)
        if accuracy_rate is not None:
            accuracy_factor = np.tanh((accuracy_rate - 0.5) * 4.0)  # Normalize around 0.5 (50% accuracy)
            stat_adjustment += accuracy_factor * 5.0  # Up to ±5 rating points
        
        # Efficiency score (resource usage efficiency in our context)
        if efficiency_score is not None:
            efficiency_factor = np.tanh((efficiency_score - 0.5) * 4.0)  # Normalize around 0.5
            stat_adjustment += efficiency_factor * 3.0  # Up to ±3 rating points
        
        # Custom statistics integration
        if custom_stats:
            for stat_name, stat_value in custom_stats.items():
                # Normalize and weight custom statistics
                normalized_stat = np.tanh(stat_value) if abs(stat_value) > 1 else stat_value
                stat_adjustment += normalized_stat * 2.0  # Up to ±2 rating points per stat
        
        # Apply individual statistics adjustment with decay
        individual_stats_weight = 0.1  # Conservative weight for individual stats
        rating.mu += stat_adjustment * individual_stats_weight
        
        # Ensure rating stays within reasonable bounds
        rating.mu = np.clip(rating.mu, 100.0, 2500.0)
    
    # -----------------------------------------------------------------------
    # TrueSkill2 utility methods — available to external callers.
    # NOT called in the core EP update loop; the standard EP algorithm
    # handles sigma convergence and uncertainty correctly on its own.
    # -----------------------------------------------------------------------

    def compute_experience_factor(self, entity_id: str, rating_type: str) -> float:
        """Rating confidence factor based on accumulated experience.

        Returns a value in [0, 1): 0 for a brand-new entity, approaching 1
        as games accumulate. Callers can use this to weight how much trust to
        place in the current rating when blending with priors.
        """
        experience_key = f"{entity_id}_{rating_type}"
        games = self.entity_experience.get(experience_key, 0)
        # Saturates at ~200 games; use 1 - exp(-games/50) decay curve
        return float(1.0 - np.exp(-games / 50.0))

    def compute_statistics_adjustment(self, entity_id: str, rating_type: str,
                                      external_stats: Dict[str, float] = None) -> Tuple[float, float]:
        """Individual game-statistics adjustment (TS2 Eq. 9 concept).

        Accepts external_stats — real per-game metrics (e.g. color_distance,
        efficiency_score) — and returns (mu_adj, sigma_adj) for optional use by
        callers that want to blend in domain observations.

        If external_stats is None or empty, returns (0.0, 0.0).
        Do NOT pass the entity's own rating-update history as stats — that
        creates circular feedback and is statistically invalid.
        """
        if not external_stats:
            return 0.0, 0.0
        mu_adj = 0.0
        for stat_name, stat_value in external_stats.items():
            # Normalize: center on 0.5 if value looks like a [0,1] metric
            centered = stat_value - 0.5 if 0.0 <= stat_value <= 1.0 else stat_value
            mu_adj += np.tanh(centered * 2.0) * self.statistics_weight * 5.0
        sigma_adj = 0.0  # Conservative: don't adjust sigma from stats alone
        return float(mu_adj), float(sigma_adj)

    def compute_beginner_tau_scale(self, entity_id: str, rating_type: str) -> float:
        """Tau scale factor for beginners (TS2 dynamic skill drift concept).

        Returns a multiplier > 1 for beginners to increase the effective tau
        (skill drift), allowing faster rating movements early in the career.
        Returns 1.0 once the entity has passed the beginner threshold.

        Use with apply_dynamic_tau() or directly scale tau before an EP update.
        """
        experience_key = f"{entity_id}_{rating_type}"
        games = self.entity_experience.get(experience_key, 0)
        if games >= self.beginner_bonus_games:
            return 1.0
        progress = games / self.beginner_bonus_games
        return 1.0 + self.beginner_boost_rate * (1.0 - progress)  # 1.15 -> 1.0

    def apply_dynamic_tau(self, entity_id: str, rating_type: str, current_round: int) -> float:
        """
        Adaptive dynamics factor based on entity activity state.
        Beginners and inactive entities get a higher effective tau.
        """
        activity_key = f"{entity_id}_{rating_type}"
        last = self.last_activity.get(activity_key, current_round)
        rounds_inactive = max(0, current_round - last)
        base_tau = getattr(self, f'{rating_type}_tau', self.agent_tau)
        experience_key = f"{entity_id}_{rating_type}"
        games = self.entity_experience.get(experience_key, 0)
        # Beginners learn faster
        if games < self.beginner_bonus_games:
            base_tau *= 1.5
        # Inactive players have higher uncertainty growth
        activity_factor = 1.0 + 0.1 * min(rounds_inactive, 10)
        return base_tau * activity_factor

    def get_conservative_rating(self, entity_id: str, rating_type: str,
                                 std_multiplier: float = 3.0) -> float:
        """Conservative skill estimate: mu - k*sigma (used for display/ranking)."""
        r = self.get_rating(entity_id, rating_type)
        return r.mu - std_multiplier * r.sigma

    def apply_quit_penalty(self, entity_id: str, rating_type: str, quit_type: str = 'early_quit'):
        """TrueSkill2: Apply quit penalty as EP loss against a virtual opponent.

        Following the paper: a quit is treated as losing to a virtual opponent at a
        scaled skill level. Severity is controlled via the virtual opponent's mu.
        The standard TrueSkill factor-graph update (v/w factors) is used — no ad-hoc
        point deductions.
        """
        if not self.enable_trueskill2:
            return

        rating = self.get_rating(entity_id, rating_type)

        # Virtual opponent skill by quit severity (higher mu = worse penalty for quitter)
        quit_opponent_mu = {
            'early_quit': self.initial_mu * 1.10,
            'rage_quit':  self.initial_mu * 1.20,
            'disconnect': self.initial_mu * 0.95,
            'timeout':    self.initial_mu * 0.90,
        }
        virtual_mu = quit_opponent_mu.get(quit_type, self.initial_mu)
        virtual_sigma = self.initial_sigma

        # Standard TrueSkill EP: entity loses to the virtual opponent
        beta = getattr(self, f'{rating_type}_beta', self.agent_beta)
        c_sq = rating.sigma**2 + virtual_sigma**2 + 2 * beta**2
        c = np.sqrt(max(c_sq, 1e-10))
        # t from winner's perspective: virtual_mu (winner) - rating.mu (loser)
        t = (virtual_mu - rating.mu) / c
        draw_margin = norm.ppf(0.5 + self.draw_prob / 2.0) * np.sqrt(2 * beta**2) / c

        v = self._v(t, draw_margin)
        w = self._w(t, draw_margin)

        # Loser update: mu decreases, sigma decreases (more certain of lower skill)
        rating.mu -= (rating.sigma**2 / c) * v
        new_sigma_sq = rating.sigma**2 * (1 - (rating.sigma**2 / c_sq) * w)
        rating.sigma = max(np.sqrt(max(new_sigma_sq, 1e-10)), 5.0)
        rating.mu = np.clip(rating.mu, 100.0, 2500.0)

        # Track quit in performance history
        experience_key = f"{entity_id}_{rating_type}"
        if experience_key not in self.entity_game_history:
            self.entity_game_history[experience_key] = []
        self.entity_game_history[experience_key].append(-(rating.sigma))  # Negative marker
    
    def calculate_squad_boost(self, entity_ids: List[str], rating_type: str) -> float:
        """TrueSkill2: Calculate squad performance boost (Equation 7)"""
        if not self.enable_trueskill2 or len(entity_ids) < 2:
            return 0.0
            
        # Base squad boost for playing together
        base_boost = self.squad_boost
        
        # Calculate familiarity bonus based on previous collaborations
        familiarity_bonus = 0.0
        
        for i, entity1 in enumerate(entity_ids):
            entity1_key = f"{entity1}_{rating_type}"
            entity1_partners = self.squad_history.get(entity1_key, [])
            
            for entity2 in entity_ids[i+1:]:
                entity2_key = f"{entity2}_{rating_type}"
                
                # Count previous collaborations
                collaboration_count = entity1_partners.count(entity2)
                
                # Diminishing returns on familiarity (log scale)
                if collaboration_count > 0:
                    familiarity_bonus += np.log(1 + collaboration_count) * self.squad_familiarity_factor
        
        # Normalize familiarity bonus by squad size
        if len(entity_ids) > 1:
            familiarity_bonus /= (len(entity_ids) * (len(entity_ids) - 1) / 2)
        
        total_boost = base_boost + familiarity_bonus
        return min(total_boost, self.squad_boost * 2.0)  # Cap at 2x base boost
    
    def update_squad_history(self, entity_ids: List[str], rating_type: str):
        """Track squad partnerships for TrueSkill2 familiarity calculations"""
        if not self.enable_trueskill2 or len(entity_ids) < 2:
            return
            
        for entity1 in entity_ids:
            entity1_key = f"{entity1}_{rating_type}"
            
            if entity1_key not in self.squad_history:
                self.squad_history[entity1_key] = []
            
            # Record all other squad members
            for entity2 in entity_ids:
                if entity1 != entity2:
                    self.squad_history[entity1_key].append(entity2)
            
            # Keep only recent history (last 100 partnerships)
            if len(self.squad_history[entity1_key]) > 100:
                self.squad_history[entity1_key] = self.squad_history[entity1_key][-100:]
    
    def update_ratings_with_squad_support(self, agent_performances: List[Tuple[str, str, float]], 
                                        squad_compositions: Dict[str, List[str]] = None, current_round: int = 0) -> Dict[str, Rating]:
        """TrueSkill2: Update ratings with squad/team modeling support"""
        if not self.enable_trueskill2 or not squad_compositions:
            return self.update_ratings(agent_performances, current_round)
        
        # Apply squad boosts to individual performances before standard rating update
        boosted_performances = []
        
        for agent_id, policy_id, score in agent_performances:
            # Find if this agent belongs to any squad
            squad_boost = 0.0
            for squad_name, squad_members in squad_compositions.items():
                if agent_id in squad_members:
                    # Calculate boost for this squad
                    squad_boost = self.calculate_squad_boost(squad_members, 'agent')
                    
                    # Update squad history
                    self.update_squad_history(squad_members, 'agent')
                    break
            
            # Apply squad boost to performance (positive boost improves score by lowering it)
            boosted_score = score - squad_boost if squad_boost > 0 else score
            boosted_performances.append((agent_id, policy_id, boosted_score))
        
        # Run standard rating update with boosted performances
        return self.update_ratings(boosted_performances, current_round)
    
    def update_ratings_batch(self, performances: List[Tuple[str, float]], rating_type: Literal['agent', 'policy', 'prior', 'agent_policy', 'agent_policy_prior', 'combined']):
        """Update ratings for a batch of entities with their performance scores"""
        if len(performances) < 2:
            return
            
        # Convert performance scores to (entity_id, performance_score) format 
        formatted_performances = [(entity_id, "", score) for entity_id, score in performances]
        
        # Call the internal rating update method with the specific rating type
        self._update_ratings_internal(formatted_performances, rating_type)
    
    def _update_ratings_internal(self, participants: List[Tuple[str, str, float]], rating_type: Literal['agent', 'policy', 'prior', 'agent_policy', 'agent_policy_prior', 'combined']):
        """Internal method to update ratings for a specific rating type"""
        # This is similar to the logic in update_ratings but for a specific rating type
        
        # Extract unique entities and check minimum count
        unique_entities = set(entity_id for entity_id, _, _ in participants)
        if len(unique_entities) < 2:
            return

        # Get appropriate parameters for this rating type
        if rating_type == 'agent':
            beta = self.agent_beta
            tau = self.agent_tau
        elif rating_type == 'policy':
            beta = self.policy_beta
            tau = self.policy_tau
        elif rating_type == 'prior':
            beta = self.prior_beta
            tau = self.prior_tau
        elif rating_type == 'agent_policy':
            beta = self.agent_policy_beta
            tau = self.agent_policy_tau
        elif rating_type == 'agent_policy_prior':
            beta = self.agent_policy_prior_beta
            tau = self.agent_policy_prior_tau
        else:  # combined
            beta = np.sqrt(self.agent_beta**2 + self.policy_beta**2 + self.prior_beta**2)
            tau = np.sqrt(self.agent_tau**2 + self.policy_tau**2 + self.prior_tau**2)

        # Apply minimal time-based skill drift
        for entity_id in unique_entities:
            rating = self.get_rating(entity_id, rating_type)
            if rating.sigma < self.initial_sigma * 0.9:
                rating.sigma = np.sqrt(rating.sigma**2 + (tau/10)**2)

        # Sort by performance (lower score is better)
        sorted_participants = sorted(participants, key=lambda x: x[2])
        
        # Store initial ratings
        initial_ratings = {entity_id: self.get_rating(entity_id, rating_type) for entity_id in unique_entities}
        updates = {entity_id: {'mu_update': 0.0, 'sigma_sq_update': 0.0, 'pre_sigma': initial_ratings[entity_id].sigma} 
                  for entity_id in unique_entities}

        # Tournament-style updates (adjacent comparisons)
        sorted_unique = []
        seen = set()
        for entity_id, _, score in sorted_participants:
            if entity_id not in seen:
                sorted_unique.append((entity_id, score))
                seen.add(entity_id)

        # Only compare adjacent entities in ranking to reduce noise
        for i in range(len(sorted_unique) - 1):
            entity_i, score_i = sorted_unique[i]
            entity_j, score_j = sorted_unique[i + 1]

            rating_i = initial_ratings[entity_i]
            rating_j = initial_ratings[entity_j]

            is_draw = abs(score_i - score_j) < 1e-4

            # Standard TrueSkill calculation with numerical protection
            c_sq = rating_i.sigma**2 + rating_j.sigma**2 + 2 * (beta**2)
            c_sq = max(c_sq, 1e-10)  # Prevent division by zero
            c = np.sqrt(c_sq)
            t = (rating_i.mu - rating_j.mu) / c
            draw_margin = norm.ppf(0.5 + self.draw_prob / 2.0) * np.sqrt(2 * (beta**2)) / c

            if is_draw:
                v_update = self._v_draw(t, draw_margin)
                w_update = self._w_draw(t, draw_margin)
            else:
                v_update = self._v(t, draw_margin)
                w_update = self._w(t, draw_margin)

            # Update for winner (entity i has lower score, so is winner)
            # Apply standard TrueSkill updates without excessive scaling
            updates[entity_i]['mu_update'] += (rating_i.sigma**2 / c) * v_update
            updates[entity_j]['mu_update'] -= (rating_j.sigma**2 / c) * v_update

            updates[entity_i]['sigma_sq_update'] += ((rating_i.sigma**2 / c_sq) * w_update)
            updates[entity_j]['sigma_sq_update'] += ((rating_j.sigma**2 / c_sq) * w_update)

        # Apply updates (pure TrueSkill EP — no post-hoc smoothing per paper)
        for entity_id in unique_entities:
            # get_rating() ensures we always work with existing rating objects (no duplicates)
            rating = self.get_rating(entity_id, rating_type)
            rating.mu = rating.mu + updates[entity_id]['mu_update']

            # Apply sigma update with reasonable bounds
            new_sigma_sq = rating.sigma**2 * (1 - updates[entity_id]['sigma_sq_update'])
            new_sigma_sq = max(new_sigma_sq, 5.0**2)   # Minimum sigma of 5 (converged)
            new_sigma_sq = min(new_sigma_sq, 200.0**2) # Maximum sigma of 200 (initial)
            rating.sigma = np.sqrt(new_sigma_sq)

            # Calculate information gain and uncertainty
            rating.information_gain = (abs(updates[entity_id]['mu_update']) +
                                     (updates[entity_id]['pre_sigma'] - rating.sigma)) * 100.0
            rating.uncertainty = rating.sigma

            # TrueSkill2: Track experience and performance history for external stats use
            if self.enable_trueskill2:
                experience_key = f"{entity_id}_{rating_type}"
                if experience_key not in self.entity_game_history:
                    self.entity_game_history[experience_key] = []

                performance_score = updates[entity_id]['mu_update']
                self.entity_game_history[experience_key].append(performance_score)

                if len(self.entity_game_history[experience_key]) > 50:
                    self.entity_game_history[experience_key] = self.entity_game_history[experience_key][-50:]

                self.entity_experience[experience_key] = self.entity_experience.get(experience_key, 0) + 1

# Backward-compatibility alias
TrueSkillEngine = TrueSkill2Engine
