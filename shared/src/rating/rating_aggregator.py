from typing import Dict, List, Tuple, Any
from .trueskill_engine import TrueSkill2Engine, TrueSkillEngine, Rating
from .blade_chest import BladeChest
from .prior_information import PriorInformationEngine
from .neural_rating_system import BayesianNeuralRatingSystem
from .statistical_rating_system import StatisticalRatingSystem
from .icbt_model import ICBTModel
from .augmented_icbt_bc import AugmentedICBTBC
from .spinning_tops import SpinningTopsModel
import numpy as np

class CombinedRatingSystem:
    def __init__(self, agents: List[Any], use_statistical_model: bool = True,
                 use_extended_models: bool = False):
        """
        Initialize combined rating system.

        Args:
            agents: List of agent objects
            use_statistical_model: If True, uses StatisticalRatingSystem (hybrid antisymmetric).
                                  If False, uses legacy BayesianNeuralRatingSystem.
                                  Default: True (recommended)
            use_extended_models: If True, also initializes ICBTModel, AugmentedICBTBC,
                                  and SpinningTopsModel as additional scoring dimensions.
                                  Default: False (no breaking change).
        """
        # Create 5×3 matrix rating system: Party × Scoring
        # Party: Agent, Policy, Prior, Agent+Policy, Agent+Policy+Prior
        # Scoring: TS_only (ALL objectives), BC_only (EACH objective), TS+BC_combined (combined)

        self.use_statistical_model = use_statistical_model
        self.use_extended_models = use_extended_models

        # Initialize TrueSkill engines for each scoring method with corrected parameters
        # Increased beta and tau values for more rating separation and visibility
        self.ts_only_engine = TrueSkill2Engine(mu=1200.0, sigma=200.0, agent_beta=80.0, agent_tau=4.0, policy_beta=150.0, policy_tau=6.0, prior_beta=120.0, prior_tau=5.0, agent_policy_beta=180.0, agent_policy_tau=15.0, agent_policy_prior_beta=220.0, agent_policy_prior_tau=18.0)

        self.bc_only_trueskill_engine = TrueSkill2Engine(mu=1200.0, sigma=200.0, agent_beta=80.0, agent_tau=4.0, policy_beta=150.0, policy_tau=6.0, prior_beta=120.0, prior_tau=5.0, agent_policy_beta=180.0, agent_policy_tau=15.0, agent_policy_prior_beta=220.0, agent_policy_prior_tau=18.0)
        self.bc_only_engine = BladeChest(self.bc_only_trueskill_engine)

        # Third system: Either Statistical (recommended) or BNN (legacy)
        if use_statistical_model:
            # New hybrid antisymmetric scoring model (PEGKí Design Modification 2025-11-03)
            self.combined_engine = StatisticalRatingSystem(agents)
        else:
            # Legacy BNN approach
            self.ts_bc_bnn_trueskill_engine = TrueSkill2Engine(mu=1200.0, sigma=200.0, agent_beta=80.0, agent_tau=4.0, policy_beta=150.0, policy_tau=6.0, prior_beta=120.0, prior_tau=5.0, agent_policy_beta=180.0, agent_policy_tau=15.0, agent_policy_prior_beta=220.0, agent_policy_prior_tau=18.0)
            self.ts_bc_bnn_blade_chest_engine = BladeChest(self.ts_bc_bnn_trueskill_engine)
            self.combined_engine = BayesianNeuralRatingSystem(agents)
        
        # Matrix structure: rating_matrix[party][scoring] = engine
        # Scoring key: ts_bc_combined (replaces ts_bc_bnn for clarity)
        self.rating_matrix = {
            'agent': {
                'ts_only': self.ts_only_engine,
                'bc_only': self.bc_only_engine,
                'ts_bc_combined': self.combined_engine
            },
            'policy': {
                'ts_only': self.ts_only_engine,
                'bc_only': self.bc_only_engine,
                'ts_bc_combined': self.combined_engine
            },
            'prior': {
                'ts_only': self.ts_only_engine,
                'bc_only': self.bc_only_engine,
                'ts_bc_combined': self.combined_engine
            },
            'agent_policy': {
                'ts_only': self.ts_only_engine,
                'bc_only': self.bc_only_engine,
                'ts_bc_combined': self.combined_engine
            },
            'agent_policy_prior': {
                'ts_only': self.ts_only_engine,
                'bc_only': self.bc_only_engine,
                'ts_bc_combined': self.combined_engine
            }
        }

        # Backward compatibility alias
        for party in self.rating_matrix:
            self.rating_matrix[party]['ts_bc_bnn'] = self.combined_engine

        # Backward compat: statistical mode doesn't create ts_bc_bnn_blade_chest_engine,
        # so alias it to bc_only_engine so callers always have a valid BladeChest engine.
        if use_statistical_model:
            self.ts_bc_bnn_blade_chest_engine = self.bc_only_engine

        # Extended models (optional, no breaking change when use_extended_models=False)
        if use_extended_models:
            self.icbt_engine = ICBTModel()
            self.icbt_bc_engine = AugmentedICBTBC()
            self.spinning_tops_engine = SpinningTopsModel()
            # Add to rating matrix
            for party in self.rating_matrix:
                self.rating_matrix[party]['icbt'] = self.icbt_engine
                self.rating_matrix[party]['icbt_bc'] = self.icbt_bc_engine
                self.rating_matrix[party]['spinning_tops'] = self.spinning_tops_engine

        self.agents_map = {agent.agent_id: agent for agent in agents}

        # Initialize prior information engine
        self.prior_engine = PriorInformationEngine()

        # Training mode flags for separate system training
        self.training_mode = 'all'  # 'ts_only', 'bc_only', 'ts_bc_bnn', 'ts_bc_combined', 'all'
    
    def get_rating(self, entity_id: str, party: str, scoring: str = 'ts_only'):
        """Get rating for specific party-scoring combination from the 5×3 matrix"""
        if party not in self.rating_matrix:
            raise ValueError(f"Invalid party: {party}. Must be one of {list(self.rating_matrix.keys())}")
        if scoring not in self.rating_matrix[party]:
            raise ValueError(f"Invalid scoring: {scoring}. Must be one of {list(self.rating_matrix[party].keys())}")

        engine = self.rating_matrix[party][scoring]
        if scoring in ['ts_bc_bnn', 'ts_bc_combined', 'icbt', 'icbt_bc', 'spinning_tops']:
            return engine.get_rating(entity_id, party)
        elif scoring == 'bc_only':
            # BC does not produce individual rankings (mu/sigma); it captures pairwise
            # intransitivity via predict_outcome() only (Chen & Joachims 2016).
            # This returns the per-objective TrueSkill base that BC sits on top of,
            # which is a valid TS ranking trained per-objective alongside BC training.
            return engine.trueskill_engine.get_rating(entity_id, party)
        else:  # ts_only
            return engine.get_rating(entity_id, party)
    
    def get_available_party_categories(self):
        """Get all available party categories dynamically from rating matrix"""
        return list(self.rating_matrix.keys())
    
    def get_all_party_scoring_combinations(self):
        """Get all 15 party-scoring combinations"""
        combinations = []
        for party in self.rating_matrix.keys():
            for scoring in self.rating_matrix[party].keys():
                combinations.append((party, scoring))
        return combinations

    def set_training_mode(self, mode: str):
        """Set which rating systems should be updated: 'ts_only', 'bc_only', 'ts_bc_bnn', 'ts_bc_combined', or 'all'"""
        self.training_mode = mode

    def reset_for_new_experiment(self):
        """Reset per-experiment LR schedule in all BC engines while keeping learned vectors."""
        self.bc_only_engine.reset_for_new_experiment()
        if not self.use_statistical_model and hasattr(self, 'ts_bc_bnn_blade_chest_engine'):
            self.ts_bc_bnn_blade_chest_engine.reset_for_new_experiment()
    
    def update_ratings(self, objective_performances: Dict[str, List[Tuple[str, str, float]]], pairwise_outcomes_map: Dict[str, List[Tuple[str, str, float]]]):
        # Update TrueSkill-only system (global multi-objective rank)
        # TS trains on ALL objectives simultaneously for global ranking
        if self.training_mode in ['all', 'ts_only']:
            all_performances = []
            # Combine all objectives into one multi-objective performance list
            for objective_name, performances in objective_performances.items():
                all_performances.extend(performances)
            
            if all_performances:
                squad_comps: Dict[str, List[str]] = {}
                for agent_id, policy_id, _ in all_performances:
                    squad_comps.setdefault(policy_id, [])
                    if agent_id not in squad_comps[policy_id]:
                        squad_comps[policy_id].append(agent_id)
                self.ts_only_engine.update_ratings_with_squad_support(
                    all_performances, squad_compositions=squad_comps
                )

        # Update Blade-Chest-only system (local objective-specific strategy)
        # BC trains on EACH objective separately for local strategies
        if self.training_mode in ['all', 'bc_only']:
            # First update the underlying TrueSkill base with each objective separately
            for objective_name, performances in objective_performances.items():
                if performances:
                    bc_squad_comps: Dict[str, List[str]] = {}
                    for agent_id, policy_id, _ in performances:
                        bc_squad_comps.setdefault(policy_id, [])
                        if agent_id not in bc_squad_comps[policy_id]:
                            bc_squad_comps[policy_id].append(agent_id)
                    self.bc_only_trueskill_engine.update_ratings_with_squad_support(
                        performances, squad_compositions=bc_squad_comps
                    )
            
            # Then update Blade-Chest embeddings for objective-specific strategies
            for objective_name, performances in objective_performances.items():
                agent_policy_pairs = [(agent_id, policy_id) for agent_id, policy_id, _ in performances]
                pairwise_outcomes = pairwise_outcomes_map.get(objective_name, [])
                prior_ids = self.prior_engine.get_all_prior_ids()
                if pairwise_outcomes:
                    self.bc_only_engine.update_embeddings(objective_name, agent_policy_pairs, pairwise_outcomes, prior_ids)

            # Surface intransitivity signals found during this update.
            # Cycles are weak signals (especially early in training) but accumulate into
            # meaningful RPS-style meta-game patterns worth tracking.
            bc_intrans = self.bc_only_engine.get_intransitivity_summary()
            for obj, info in bc_intrans.items():
                cycles = info['current_cycles']
                if cycles:
                    cycle_strs = [' > '.join(c) + f' > {c[0]}' for c in cycles[:3]]
                    suffix = f'  ... +{len(cycles)-3} more' if len(cycles) > 3 else ''
                    print(f"  [BC] Intransitivity ({obj}): {len(cycles)} cycle(s) "
                          f"[cumulative: {info['total_count']}]  "
                          f"{'; '.join(cycle_strs)}{suffix}")

        # Update combined system (Statistical or BNN depending on configuration)
        if self.training_mode in ['all', 'ts_bc_bnn', 'ts_bc_combined']:
            if self.use_statistical_model:
                # Update StatisticalRatingSystem (hybrid antisymmetric model)
                self.combined_engine.update_ratings(objective_performances, pairwise_outcomes_map)
            else:
                # Legacy BNN update path
                # Update underlying TrueSkill base - use ALL objectives for global ranking
                all_performances = []
                for objective_name, performances in objective_performances.items():
                    all_performances.extend(performances)

                if all_performances:
                    bnn_squad_comps: Dict[str, List[str]] = {}
                    for agent_id, policy_id, _ in all_performances:
                        bnn_squad_comps.setdefault(policy_id, [])
                        if agent_id not in bnn_squad_comps[policy_id]:
                            bnn_squad_comps[policy_id].append(agent_id)
                    self.ts_bc_bnn_trueskill_engine.update_ratings_with_squad_support(
                        all_performances, squad_compositions=bnn_squad_comps
                    )

                # Update Blade-Chest embeddings for BNN input features - EACH objective separately
                for objective_name, performances in objective_performances.items():
                    agent_policy_pairs = [(agent_id, policy_id) for agent_id, policy_id, _ in performances]
                    pairwise_outcomes = pairwise_outcomes_map.get(objective_name, [])
                    prior_ids = self.prior_engine.get_all_prior_ids()
                    if pairwise_outcomes:
                        self.ts_bc_bnn_blade_chest_engine.update_embeddings(objective_name, agent_policy_pairs, pairwise_outcomes, prior_ids)

                # Update Bayesian Neural Network with combined information
                self.combined_engine.update_ratings(objective_performances, pairwise_outcomes_map)
        
        # Update extended models if enabled
        if self.use_extended_models:
            self.icbt_engine.update_ratings(objective_performances, pairwise_outcomes_map)
            self.icbt_bc_engine.update_ratings(objective_performances, pairwise_outcomes_map)
            self.spinning_tops_engine.update_ratings(objective_performances, pairwise_outcomes_map)

        # Update Prior Information ratings across all scoring methods
        if self.training_mode == 'all':
            self.update_prior_information_ratings(objective_performances)

    def get_combined_rating(self, objective: str, agent_id: str, policy_id: str, opponents: List[Tuple[str, str]]) -> float:
        """Get combined rating using the multi-dimensional combined system"""
        if self.use_statistical_model:
            # Use statistical model's hybrid score
            # Average score against all opponents
            scores = []
            for opponent_agent_id, opponent_policy_id in opponents:
                if opponent_agent_id != agent_id:
                    score = self.combined_engine.compute_hybrid_score(
                        objective, agent_id, policy_id, opponent_agent_id, opponent_policy_id
                    )
                    scores.append(score)
            return np.mean(scores) if scores else 0.0
        else:
            # Legacy BNN path
            agent_rating = self.ts_only_engine.get_rating(agent_id, 'agent')
            policy_rating = self.ts_only_engine.get_rating(policy_id, 'policy')
            trueskill_rating = agent_rating.mu + policy_rating.mu

            blade_chest_scores = []
            for opponent_agent_id, opponent_policy_id in opponents:
                if opponent_agent_id != agent_id:
                    score = self.ts_bc_bnn_blade_chest_engine.predict_outcome(objective, agent_id, policy_id, opponent_agent_id, opponent_policy_id)
                    blade_chest_scores.append(score)

            blade_chest_matchup_score = np.mean(blade_chest_scores) if blade_chest_scores else 0.0

            alpha = 0.8
            beta = 0.2

            final_rating = (alpha * trueskill_rating) + (beta * blade_chest_matchup_score)
            return final_rating

    def predict_outcome_trueskill_only(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """System 1: Pure TrueSkill predictions"""
        trueskill_win_prob, _, _ = self.ts_only_engine.predict_outcome(agent_id_a, policy_id_a, agent_id_b, policy_id_b)
        return trueskill_win_prob
    
    def predict_outcome_blade_chest_only(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """System 2: Pure Blade-Chest predictions (using its own TrueSkill base)"""
        trueskill_a = self.bc_only_engine.trueskill_engine.get_rating(agent_id_a, 'agent')
        trueskill_b = self.bc_only_engine.trueskill_engine.get_rating(agent_id_b, 'agent')
        blade_chest_score = self.bc_only_engine.predict_outcome(
            objective, 
            agent_id_a, policy_id_a, 
            agent_id_b, policy_id_b,
            trueskill_a.uncertainty, trueskill_a.information_gain,
            trueskill_b.uncertainty, trueskill_b.information_gain
        )
        # Use stable sigmoid to prevent overflow
        blade_chest_score_clamped = np.clip(blade_chest_score, -500, 500)
        return 1 / (1 + np.exp(-blade_chest_score_clamped))
    
    def predict_outcome_combined(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """System 3: TrueSkill + Blade-Chest combined predictions"""
        # Get TrueSkill prediction from combined system
        trueskill_win_prob, _, _ = self.ts_only_engine.predict_outcome(agent_id_a, policy_id_a, agent_id_b, policy_id_b)
        
        # Get Blade-Chest prediction from combined system
        trueskill_a = self.ts_only_engine.get_rating(agent_id_a, 'agent')
        trueskill_b = self.ts_only_engine.get_rating(agent_id_b, 'agent')
        blade_chest_score = self.ts_bc_bnn_blade_chest_engine.predict_outcome(
            objective, 
            agent_id_a, policy_id_a, 
            agent_id_b, policy_id_b,
            trueskill_a.uncertainty, trueskill_a.information_gain,
            trueskill_b.uncertainty, trueskill_b.information_gain
        )
        # Use stable sigmoid to prevent overflow
        blade_chest_score_clamped = np.clip(blade_chest_score, -500, 500)
        blade_chest_prob = 1 / (1 + np.exp(-blade_chest_score_clamped))
        
        # Combine predictions (70% TrueSkill, 30% Blade-Chest)
        return 0.7 * trueskill_win_prob + 0.3 * blade_chest_prob

    def predict_outcome_bayesian_neural(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """System 4: Combined system predictions (Statistical or BNN)"""
        return self.combined_engine.predict_outcome(objective, agent_id_a, policy_id_a, agent_id_b, policy_id_b)

    def predict_outcome_combined_system(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """Alias for combined system predictions (recommended name)"""
        return self.combined_engine.predict_outcome(objective, agent_id_a, policy_id_a, agent_id_b, policy_id_b)

    def predict_outcome(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str) -> float:
        """Legacy method - returns combined prediction for backward compatibility"""
        return self.predict_outcome_combined(objective, agent_id_a, policy_id_a, agent_id_b, policy_id_b)

    def get_ratings_trueskill_only(self, agent_id: str, policy_id: str) -> Dict[str, Rating]:
        """Get ratings from Dimension 1: TrueSkill-only"""
        return {
            'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
            'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
            'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
        }
    
    def get_ratings_blade_chest_only(self, agent_id: str, policy_id: str) -> Dict[str, Rating]:
        """Get ratings from Dimension 2: per-objective TrueSkill base co-trained with Blade-Chest.

        IMPORTANT: Blade-Chest (Chen & Joachims 2016) is an intransitivity model, not a ranking
        system.  It does not produce individual mu/sigma ratings; it only supports pairwise
        prediction via predict_outcome().  The values returned here come from the separate
        per-objective TrueSkill engine (bc_only_trueskill_engine) that BC uses internally for
        uncertainty estimates.  Use predict_outcome_blade_chest_only() for intransitivity scores.
        """
        return {
            'agent': self.bc_only_trueskill_engine.get_rating(agent_id, 'agent'),
            'policy': self.bc_only_trueskill_engine.get_rating(policy_id, 'policy'),
            'combined': self.bc_only_trueskill_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
        }
        
    def get_ratings_prior_only(self, agent_id: str, policy_id: str) -> Dict[str, Rating]: 
        """Get ratings from Dimension 3: Prior-Information-only"""
        return {
            'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
            'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
            'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
        }
    
    def get_ratings_combined(self, agent_id: str, policy_id: str) -> Dict[str, Rating]:
        """Get ratings from Dimension 5: TrueSkill+Blade-Chest+Prior combined"""
        return {
            'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
            'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
            'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
        }
    
    def get_ratings_bayesian_neural(self, agent_id: str, policy_id: str) -> Dict[str, Rating]:
        """Get ratings from combined system (Statistical or BNN)"""
        return {
            'agent': self.combined_engine.get_rating(agent_id, 'agent'),
            'policy': self.combined_engine.get_rating(policy_id, 'policy'),
            'combined': self.combined_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
        }

    def get_ratings_combined_system(self, agent_id: str, policy_id: str) -> Dict[str, Rating]:
        """Get ratings from combined system (recommended name)"""
        return self.get_ratings_bayesian_neural(agent_id, policy_id)
    
    def get_blade_chest_vectors_by_system(self, objective: str, system: str = 'combined') -> Dict[str, Dict[str, np.ndarray]]:
        """Get Blade-Chest vectors from specific system dimension"""
        if system == 'blade_chest_only':
            return self.bc_only_engine._get_objective_model(objective)
        elif system == 'prior_only':
            return self.bc_only_engine._get_objective_model(objective)
        elif system == 'combined':
            if self.use_statistical_model:
                return self.combined_engine.blade_chest_engine._get_objective_model(objective)
            else:
                return self.ts_bc_bnn_blade_chest_engine._get_objective_model(objective)
        else:  # trueskill_only
            return {}  # TrueSkill-only has no Blade-Chest vectors

    def get_blade_chest_vectors(self, objective: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Legacy method - returns combined system vectors"""
        if self.use_statistical_model:
            return self.combined_engine.blade_chest_engine._get_objective_model(objective)
        else:
            return self.ts_bc_bnn_blade_chest_engine._get_objective_model(objective)
    
    def update_prior_information_ratings(self, objective_performances: Dict[str, List[Tuple[str, str, float]]]):
        """Update ratings for different prior information sources based on their hypothetical performance"""
        # For this demo, we'll simulate how each prior would have performed
        # In practice, this would be integrated with actual game mechanics
        
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
                    # In practice, this would come from actual prior-guided actions
                    base_score = actual_score
                    
                    # Add prior-specific bias (some priors work better for certain objectives)
                    if 'color_distance' in objective_name.lower():
                        if prior_id == 'balanced_colors':
                            bias = 0.95  # Balanced approach works well for color matching
                        elif prior_id == 'high_saturation':
                            bias = 0.85  # High saturation can hurt color matching
                        else:
                            bias = 0.90  # Other priors are moderately effective
                    else:
                        bias = 0.90  # Default bias for other objectives
                    
                    prior_score = base_score * bias + np.random.normal(0, 5)  # Add some noise
                    total_score += prior_score
                    count += 1
                
                if count > 0:
                    avg_score = total_score / count
                    prior_performances.append((prior_id, avg_score))
            
            # Update TrueSkill ratings for prior information sources across all dimensions
            if prior_performances:
                # Dimension 1: TrueSkill-only system
                self.ts_only_engine.update_ratings_batch(prior_performances, 'prior')
                
                # Dimension 2: Blade-Chest system's TrueSkill base
                self.bc_only_trueskill_engine.update_ratings_batch(prior_performances, 'prior')
                
                # Dimension 3: Prior-Information system's TrueSkill base
                # Give enhanced prior performance scores for this dimension
                enhanced_prior_performances = [(pid, score * 1.2) for pid, score in prior_performances]
                self.ts_only_engine.update_ratings_batch(enhanced_prior_performances, 'prior')
                
                # Dimension 4: Agent+Policy combination system
                self.ts_only_engine.update_ratings_batch(prior_performances, 'prior')
                
                # Dimension 5: Combined system
                self.ts_only_engine.update_ratings_batch(prior_performances, 'prior')
    
    def get_prior_ratings(self) -> Dict[str, Dict[str, Rating]]:
        """Get ratings for all prior information sources from all five rating system dimensions"""
        prior_ids = self.prior_engine.get_all_prior_ids()
        return {
            'trueskill_only': {pid: self.ts_only_engine.get_rating(pid, 'prior') for pid in prior_ids},
            'blade_chest_only': {pid: self.bc_only_trueskill_engine.get_rating(pid, 'prior') for pid in prior_ids},
            'prior_only': {pid: self.ts_only_engine.get_rating(pid, 'prior') for pid in prior_ids},
            'agent_policy_only': {pid: self.ts_only_engine.get_rating(pid, 'prior') for pid in prior_ids},
            'combined': {pid: self.ts_only_engine.get_rating(pid, 'prior') for pid in prior_ids}
        }
    
    def get_ratings_agent_policy_only(self, agent_id: str, policy_id: str) -> Dict[str, Rating]: 
        """Get ratings from Dimension 4: Agent+Policy combination-only (separate from Agent+Policy+Prior)"""
        return {
            'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
            'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
            'agent_policy': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'agent_policy'),
            'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
        }
    
    def get_all_dimension_ratings(self, agent_id: str, policy_id: str) -> Dict[str, Dict[str, Rating]]:
        """Get ratings from all six rating system dimensions"""
        return {
            'trueskill_only': {
                'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
                'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
                'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            },
            # NOTE: BC does not produce rankings; these are from the per-objective TS base
            # that BC sits on. Use predict_outcome_blade_chest_only() for intransitivity scores.
            'blade_chest_only': {
                'agent': self.bc_only_trueskill_engine.get_rating(agent_id, 'agent'),
                'policy': self.bc_only_trueskill_engine.get_rating(policy_id, 'policy'),
                'combined': self.bc_only_trueskill_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            },
            'prior_only': {
                'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
                'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
                'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            },
            'agent_policy_only': {
                'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
                'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
                'agent_policy': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'agent_policy'),
                'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            },
            'combined': {
                'agent': self.ts_only_engine.get_rating(agent_id, 'agent'),
                'policy': self.ts_only_engine.get_rating(policy_id, 'policy'),
                'agent_policy': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'agent_policy'),
                'agent_policy_prior': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}_prior', 'agent_policy_prior'),
                'combined': self.ts_only_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            },
            'bayesian_neural': {
                'agent': self.combined_engine.get_rating(agent_id, 'agent'),
                'policy': self.combined_engine.get_rating(policy_id, 'policy'),
                'combined': self.combined_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            },
            'combined_system': {  # New alias for clarity
                'agent': self.combined_engine.get_rating(agent_id, 'agent'),
                'policy': self.combined_engine.get_rating(policy_id, 'policy'),
                'combined': self.combined_engine.get_rating(f'{agent_id}_{policy_id}', 'combined')
            }
        }
