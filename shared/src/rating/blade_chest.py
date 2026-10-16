import numpy as np
from typing import Dict, Tuple, List, Literal

from .trueskill_engine import TrueSkillEngine, Rating
from ..utils.simulation_seed import get_rng, SeedScope

class BladeChest:
    def __init__(self, trueskill_engine: TrueSkillEngine, num_dimensions: int = 2, initial_std_dev: float = 0.2, learning_rate: float = 3e-4, n_iterations: int = 30, lambda_param: float = 0.0005, trueskill_weight: float = 0.15):
        self.trueskill_engine = trueskill_engine
        self.num_dimensions = num_dimensions
        self.initial_std_dev = initial_std_dev
        self.learning_rate = learning_rate  # More conservative learning rate
        self.n_iterations = n_iterations  # Fewer iterations to reduce noise
        self.lambda_param = lambda_param  # Increased regularization
        self.trueskill_weight = trueskill_weight
        self.objective_models: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.update_count: Dict[str, int] = {}  # per-objective; prevents cross-objective LR coupling

        # Deterministic RNG for reproducible results
        self.rng = get_rng(SeedScope.RATING_SYSTEM)
        
        # Enhanced vector separation parameters
        self.vector_separation_factor = 0.8  # Minimum separation between blade and chest vectors
        self.rotation_strength = 0.3  # Strength of rotation to avoid axis alignment
        self.axis_avoidance_strength = 0.2  # Strength to avoid pure axis alignment
        
        # Pre-computed rotation matrices for better vector distribution
        self.rotation_matrices = self._generate_rotation_matrices()

    def _generate_rotation_matrices(self) -> List[np.ndarray]:
        """Generate rotation matrices to ensure better vector distribution and avoid axis alignment"""
        rotation_matrices = []
        
        if self.num_dimensions >= 2:
            # 2D rotations for the first two dimensions
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)  # 8 different rotation angles
            for angle in angles:
                if self.num_dimensions == 2:
                    rotation_matrix = np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]
                    ])
                else:  # 3D or higher
                    rotation_matrix = np.eye(self.num_dimensions)
                    rotation_matrix[0, 0] = np.cos(angle)
                    rotation_matrix[0, 1] = -np.sin(angle)
                    rotation_matrix[1, 0] = np.sin(angle)
                    rotation_matrix[1, 1] = np.cos(angle)
                
                rotation_matrices.append(rotation_matrix)
        
        if self.num_dimensions >= 3:
            # Additional 3D rotations around different axes
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            for angle in angles:
                # Rotation around z-axis (x-y plane)
                rotation_matrix = np.eye(self.num_dimensions)
                rotation_matrix[0, 0] = np.cos(angle)
                rotation_matrix[0, 2] = -np.sin(angle)
                rotation_matrix[2, 0] = np.sin(angle)
                rotation_matrix[2, 2] = np.cos(angle)
                rotation_matrices.append(rotation_matrix)
        
        return rotation_matrices

    def _apply_vector_separation_and_rotation(self, blade_vector: np.ndarray, chest_vector: np.ndarray, entity_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a deterministic rotation to break symmetry at initialisation.

        Only the rotation is kept.  The old separation-enforcement and
        axis-avoidance steps have been removed: both operated at the same scale
        as initial_std_dev (0.2), so they dominated the random draw and created
        a strong geometric prior that conflicted with gradient learning,
        especially on sparse early data where the prior could not be overcome.
        The rotation alone is sufficient to prevent all entities starting at the
        origin; the gradient updates then move vectors to wherever the data says.
        """
        import hashlib
        deterministic_hash = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
        rotation_idx = deterministic_hash % len(self.rotation_matrices)
        rotation_matrix = self.rotation_matrices[rotation_idx]

        if rotation_matrix.shape[0] == blade_vector.shape[0]:
            blade_vector = rotation_matrix @ blade_vector
            chest_vector = rotation_matrix @ chest_vector

        return blade_vector, chest_vector

    def _get_objective_model(self, objective: str) -> Dict[str, Dict[str, np.ndarray]]:
        if objective not in self.objective_models:
            self.objective_models[objective] = {
                'agent_blade_vectors': {},
                'agent_chest_vectors': {},
                'policy_blade_vectors': {},
                'policy_chest_vectors': {},
                'prior_blade_vectors': {},
                'prior_chest_vectors': {},
                # Intransitivity tracking (weak signal, useful for meta-game analysis)
                'intransitivity_cycles': [],       # cycles found in the most recent update
                'intransitivity_total_count': 0,   # cumulative cycle count across all updates
            }
        return self.objective_models[objective]

    def _initialize_vectors(self, objective: str, entity_id: str, vector_type: Literal['agent', 'policy', 'prior']):
        model = self._get_objective_model(objective)
        if entity_id not in model[f'{vector_type}_blade_vectors']:
            # Initialize blade and chest vectors with enhanced separation and distribution
            blade_vector = self.rng.normal(0, self.initial_std_dev, self.num_dimensions)
            chest_vector = self.rng.normal(0, self.initial_std_dev, self.num_dimensions)
            
            # Apply enhanced vector separation and rotation to prevent axis alignment
            blade_vector, chest_vector = self._apply_vector_separation_and_rotation(
                blade_vector, chest_vector, entity_id
            )
            
            model[f'{vector_type}_blade_vectors'][entity_id] = blade_vector
            model[f'{vector_type}_chest_vectors'][entity_id] = chest_vector

    def update_embeddings(self, objective: str, agent_policy_pairs: List[Tuple[str, str]], pairwise_outcomes: List[Tuple[str, str, str, str, float, float, float, float, float]], prior_ids: List[str] = None):
        model = self._get_objective_model(objective)

        for agent_id, policy_id in agent_policy_pairs:
            self._initialize_vectors(objective, agent_id, 'agent')
            self._initialize_vectors(objective, policy_id, 'policy')
        
        # Initialize vectors for prior information sources if provided
        if prior_ids:
            for prior_id in prior_ids:
                self._initialize_vectors(objective, prior_id, 'prior')

        if not pairwise_outcomes:
            return

        # Per-objective LR schedule: each objective decays independently so a
        # heavily-trained objective doesn't suppress learning in a sparse one.
        count = self.update_count.get(objective, 0) + 1
        self.update_count[objective] = count
        adaptive_lr = max(
            self.learning_rate * (0.99 ** count),
            self.learning_rate * 0.05,
        )
        adaptive_iterations = max(2, self.n_iterations - int(0.8 * np.log(1 + count)))
        
        # Vectorized processing for large outcome sets; small sets use the serial loop
        # (serial loop applies updates in-place sequentially which can help with sparse data)
        if len(pairwise_outcomes) > 5:
            self._update_embeddings_vectorized(model, pairwise_outcomes, adaptive_iterations, adaptive_lr)
        else:
            # Original method for small outcome sets
            for _ in range(adaptive_iterations):
                for winner_id, winner_policy_id, loser_id, loser_policy_id, margin, winner_uncertainty, winner_info_gain, loser_uncertainty, loser_info_gain in pairwise_outcomes:
                    winner_agent_blade = model['agent_blade_vectors'][winner_id]
                    winner_policy_blade = model['policy_blade_vectors'][winner_policy_id]
                    loser_agent_chest = model['agent_chest_vectors'][loser_id]
                    loser_policy_chest = model['policy_chest_vectors'][loser_policy_id]

                    loser_agent_blade = model['agent_blade_vectors'][loser_id]
                    loser_policy_blade = model['policy_blade_vectors'][loser_policy_id]
                    winner_agent_chest = model['agent_chest_vectors'][winner_id]
                    winner_policy_chest = model['policy_chest_vectors'][winner_policy_id]

                    # BC score (Chen & Joachims 2016): f(w,l) = b_w^T c_l - b_l^T c_w
                    # Extended to separate agent and policy vector spaces
                    agent_interaction = np.dot(winner_agent_blade, loser_agent_chest) - np.dot(loser_agent_blade, winner_agent_chest)
                    policy_interaction = np.dot(winner_policy_blade, loser_policy_chest) - np.dot(loser_policy_blade, winner_policy_chest)
                    f_raw = agent_interaction + policy_interaction

                    # Log-loss gradient: error = sigma(-f) = 1 - sigma(f)
                    p = 1.0 / (1.0 + np.exp(-np.clip(f_raw, -10.0, 10.0)))
                    error = 1.0 - p

                    # Standard BC log-loss gradients (grad_theta L = -error * d_f/d_theta + lambda*theta)
                    grad_winner_agent_blade  = -error * loser_agent_chest  + self.lambda_param * winner_agent_blade
                    grad_winner_policy_blade = -error * loser_policy_chest + self.lambda_param * winner_policy_blade
                    grad_loser_agent_blade   = +error * winner_agent_chest + self.lambda_param * loser_agent_blade
                    grad_loser_policy_blade  = +error * winner_policy_chest + self.lambda_param * loser_policy_blade
                    grad_winner_agent_chest  = +error * loser_agent_blade  + self.lambda_param * winner_agent_chest
                    grad_winner_policy_chest = +error * loser_policy_blade + self.lambda_param * winner_policy_chest
                    grad_loser_agent_chest   = -error * winner_agent_blade + self.lambda_param * loser_agent_chest
                    grad_loser_policy_chest  = -error * winner_policy_blade + self.lambda_param * loser_policy_chest

                    # Update vectors (theta -= lr * grad_L)
                    model['agent_blade_vectors'][winner_id]         -= adaptive_lr * grad_winner_agent_blade
                    model['policy_blade_vectors'][winner_policy_id] -= adaptive_lr * grad_winner_policy_blade
                    model['agent_blade_vectors'][loser_id]          -= adaptive_lr * grad_loser_agent_blade
                    model['policy_blade_vectors'][loser_policy_id]  -= adaptive_lr * grad_loser_policy_blade
                    model['agent_chest_vectors'][winner_id]         -= adaptive_lr * grad_winner_agent_chest
                    model['policy_chest_vectors'][winner_policy_id] -= adaptive_lr * grad_winner_policy_chest
                    model['agent_chest_vectors'][loser_id]          -= adaptive_lr * grad_loser_agent_chest
                    model['policy_chest_vectors'][loser_policy_id]  -= adaptive_lr * grad_loser_policy_chest

                    # Clip to prevent gradient explosion
                    for vid in [winner_id, loser_id]:
                        model['agent_blade_vectors'][vid] = np.clip(model['agent_blade_vectors'][vid], -5, 5)
                        model['agent_chest_vectors'][vid]  = np.clip(model['agent_chest_vectors'][vid],  -5, 5)
                    for pid in [winner_policy_id, loser_policy_id]:
                        model['policy_blade_vectors'][pid] = np.clip(model['policy_blade_vectors'][pid], -5, 5)
                        model['policy_chest_vectors'][pid]  = np.clip(model['policy_chest_vectors'][pid],  -5, 5)

        # Intransitivity detection is O(n^3) -- throttle to every 10 updates.
        # Early calls produce mostly spurious cycles (random vectors); meaningful
        # patterns only emerge once embeddings have received sufficient gradient
        # signal, so running every step wastes compute without improving accuracy.
        unique_pairs = list(dict.fromkeys(agent_policy_pairs))
        if len(unique_pairs) >= 3 and count % 10 == 1:
            cycles = self.detect_intransitivity(objective, unique_pairs)
            model['intransitivity_cycles'] = cycles
            model['intransitivity_total_count'] += len(cycles)
        elif len(unique_pairs) < 3:
            model['intransitivity_cycles'] = []

    def predict_outcome(self, objective: str, agent_id_a: str, policy_id_a: str, agent_id_b: str, policy_id_b: str, uncertainty_a: float = 0.0, info_gain_a: float = 0.0, uncertainty_b: float = 0.0, info_gain_b: float = 0.0) -> float:
        self._initialize_vectors(objective, agent_id_a, 'agent')
        self._initialize_vectors(objective, policy_id_a, 'policy')
        self._initialize_vectors(objective, agent_id_b, 'agent')
        self._initialize_vectors(objective, policy_id_b, 'policy')

        model = self._get_objective_model(objective)

        # BC score (Chen & Joachims 2016): f(a,b) = b_a^T c_b - b_b^T c_a
        agent_interaction = (np.dot(model['agent_blade_vectors'][agent_id_a], model['agent_chest_vectors'][agent_id_b]) -
                             np.dot(model['agent_blade_vectors'][agent_id_b], model['agent_chest_vectors'][agent_id_a]))
        policy_interaction = (np.dot(model['policy_blade_vectors'][policy_id_a], model['policy_chest_vectors'][policy_id_b]) -
                              np.dot(model['policy_blade_vectors'][policy_id_b], model['policy_chest_vectors'][policy_id_a]))

        return agent_interaction + policy_interaction

    def intransitivity_distance(self, objective: str, id_i: str, id_j: str,
                                type_i: str = 'agent', type_j: str = 'agent') -> float:
        """Intransitivity distance: ||blade_i - chest_j|| + ||chest_i - blade_j||"""
        self._initialize_vectors(objective, id_i, type_i)
        self._initialize_vectors(objective, id_j, type_j)
        model = self._get_objective_model(objective)
        blade_i = model[f'{type_i}_blade_vectors'][id_i]
        chest_i = model[f'{type_i}_chest_vectors'][id_i]
        blade_j = model[f'{type_j}_blade_vectors'][id_j]
        chest_j = model[f'{type_j}_chest_vectors'][id_j]
        return float(np.linalg.norm(blade_i - chest_j) + np.linalg.norm(chest_i - blade_j))

    def detect_intransitivity(self, objective: str, agent_policy_pairs: List[Tuple[str, str]]) -> List[List[str]]:
        cycles = []
        if len(agent_policy_pairs) < 3:
            return cycles

        for i in range(len(agent_policy_pairs)):
            for j in range(len(agent_policy_pairs)):
                if i == j: continue
                for k in range(len(agent_policy_pairs)):
                    if k == i or k == j: continue

                    a_id, a_policy = agent_policy_pairs[i]
                    b_id, b_policy = agent_policy_pairs[j]
                    c_id, c_policy = agent_policy_pairs[k]

                    if self.predict_outcome(objective, a_id, a_policy, b_id, b_policy) > 0 and \
                       self.predict_outcome(objective, b_id, b_policy, c_id, c_policy) > 0 and \
                       self.predict_outcome(objective, c_id, c_policy, a_id, a_policy) > 0:
                        cycle = sorted([a_id, b_id, c_id])
                        if cycle not in cycles:
                            cycles.append(cycle)

        return cycles

    def get_blade_vector(self, objective: str, entity_id: str, entity_type: Literal['agent', 'policy', 'prior']) -> np.ndarray:
        model = self._get_objective_model(objective)
        return model[f'{entity_type}_blade_vectors'].get(entity_id, np.zeros(self.num_dimensions))

    def get_chest_vector(self, objective: str, entity_id: str, entity_type: Literal['agent', 'policy', 'prior']) -> np.ndarray:
        model = self._get_objective_model(objective)
        return model[f'{entity_type}_chest_vectors'].get(entity_id, np.zeros(self.num_dimensions))
    
    def get_prior_blade_chest_vectors(self, objective: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Get blade and chest vectors for all prior entities"""
        model = self._get_objective_model(objective)
        
        blade_vectors = model.get('prior_blade_vectors', {})
        chest_vectors = model.get('prior_chest_vectors', {})
        
        return blade_vectors.copy(), chest_vectors.copy()

    def reset_objective_models(self):
        """Resets all objective-specific Blade-Chest models."""
        self.objective_models = {}
        self.update_count = {}

    def reset_for_new_experiment(self):
        """Reset per-objective LR schedules while keeping learned vectors."""
        self.update_count = {}
        # Reset per-update cycle list (but keep cumulative count across experiments)
        for model in self.objective_models.values():
            model['intransitivity_cycles'] = []

    def get_intransitivity_summary(self) -> Dict[str, Dict]:
        """
        Return intransitivity cycle summary per objective.

        Returns only objectives where at least one cycle has been detected.
        Each entry: {'current_cycles': [...], 'total_count': int}

        Note: early-training cycles are noisy (random vectors produce spurious cycles).
        Patterns that persist or grow across rounds are the meaningful signals.
        """
        summary = {}
        for obj, model in self.objective_models.items():
            cycles = model.get('intransitivity_cycles', [])
            total = model.get('intransitivity_total_count', 0)
            if cycles or total:
                summary[obj] = {
                    'current_cycles': cycles,
                    'total_count': total,
                }
        return summary
    
    def _update_embeddings_vectorized(self, model, pairwise_outcomes, adaptive_iterations, adaptive_lr):
        """
        True numpy-vectorized BC gradient update (replaces _update_embeddings_batch).

        Design:
          - Stack all entity vectors into [n_entities, d] matrices once per iteration.
          - Build integer index arrays (wa, wp, la, lp) once before the loop.
          - Gather [N, d] slices, compute all N scores and errors in one pass.
          - Accumulate gradients with np.add.at (scatter-add, handles repeated indices).
          - Apply theta = theta * (1 - lr*lam) - lr * data_grad in a single matrix op.
          - Write updated matrices back to model dicts after all iterations.

        Gradient signs (Chen & Joachims 2016, log-loss, e = 1 - sigma(f)):
          dBA[wa] += -e * CA_l    dBA[la] += +e * CA_w
          dCA[wa] += +e * BA_l    dCA[la] += -e * BA_w
          dBP[wp] += -e * CP_l    dBP[lp] += +e * CP_w
          dCP[wp] += +e * BP_l    dCP[lp] += -e * BP_w
        """
        agent_ids  = list(model['agent_blade_vectors'].keys())
        policy_ids = list(model['policy_blade_vectors'].keys())
        if not agent_ids or not policy_ids or not pairwise_outcomes:
            return

        agent_idx  = {a: i for i, a in enumerate(agent_ids)}
        policy_idx = {p: i for i, p in enumerate(policy_ids)}

        # Stack current vectors into [n_entities, d] matrices
        BA = np.array([model['agent_blade_vectors'][a]  for a in agent_ids],  dtype=float)
        CA = np.array([model['agent_chest_vectors'][a]  for a in agent_ids],  dtype=float)
        BP = np.array([model['policy_blade_vectors'][p] for p in policy_ids], dtype=float)
        CP = np.array([model['policy_chest_vectors'][p] for p in policy_ids], dtype=float)

        # Build index arrays — skip outcomes referencing uninitialised entities
        valid = [o for o in pairwise_outcomes
                 if o[0] in agent_idx and o[2] in agent_idx
                 and o[1] in policy_idx and o[3] in policy_idx]
        if not valid:
            return

        wa = np.array([agent_idx[o[0]]  for o in valid])
        wp = np.array([policy_idx[o[1]] for o in valid])
        la = np.array([agent_idx[o[2]]  for o in valid])
        lp = np.array([policy_idx[o[3]] for o in valid])

        lam   = self.lambda_param
        decay = 1.0 - adaptive_lr * lam   # L2-regularisation via weight decay

        # Margin weights [N, 1]: outcome[4] is the standardised margin.
        # Pairs with larger margin are more reliable signals; clip to [0.1, 3.0]
        # so near-ties barely contribute but no single pair dominates.
        margins = np.clip(
            np.array([o[4] for o in valid], dtype=float), 0.1, 3.0
        )[:, None]

        for _ in range(adaptive_iterations):
            # Gather [N, d] slices for all outcomes simultaneously
            BA_w = BA[wa]; CA_w = CA[wa]
            BA_l = BA[la]; CA_l = CA[la]
            BP_w = BP[wp]; CP_w = CP[wp]
            BP_l = BP[lp]; CP_l = CP[lp]

            # BC score f [N]: (b_w.c_l - b_l.c_w) for agents + policies
            f = ((BA_w * CA_l).sum(1) - (BA_l * CA_w).sum(1) +
                 (BP_w * CP_l).sum(1) - (BP_l * CP_w).sum(1))

            # Log-loss error weighted by margin: e = (1 - sigma(f)) * margin
            e = (1.0 - 1.0 / (1.0 + np.exp(-np.clip(f, -10.0, 10.0))))[:, None]
            ew = e * margins  # [N, 1]

            # Accumulate margin-weighted data-gradient via scatter-add
            dBA = np.zeros_like(BA); dCA = np.zeros_like(CA)
            dBP = np.zeros_like(BP); dCP = np.zeros_like(CP)

            np.add.at(dBA, wa, -ew * CA_l)
            np.add.at(dBA, la, +ew * CA_w)
            np.add.at(dCA, wa, +ew * BA_l)
            np.add.at(dCA, la, -ew * BA_w)
            np.add.at(dBP, wp, -ew * CP_l)
            np.add.at(dBP, lp, +ew * CP_w)
            np.add.at(dCP, wp, +ew * BP_l)
            np.add.at(dCP, lp, -ew * BP_w)

            # theta_new = theta * (1 - lr*lam) - lr * data_grad, then clip [-5, 5]
            BA = np.clip(BA * decay - adaptive_lr * dBA, -5.0, 5.0)
            CA = np.clip(CA * decay - adaptive_lr * dCA, -5.0, 5.0)
            BP = np.clip(BP * decay - adaptive_lr * dBP, -5.0, 5.0)
            CP = np.clip(CP * decay - adaptive_lr * dCP, -5.0, 5.0)

        # Write updated matrices back to model dicts
        for i, a in enumerate(agent_ids):
            model['agent_blade_vectors'][a] = BA[i]
            model['agent_chest_vectors'][a]  = CA[i]
        for i, p in enumerate(policy_ids):
            model['policy_blade_vectors'][p] = BP[i]
            model['policy_chest_vectors'][p]  = CP[i]