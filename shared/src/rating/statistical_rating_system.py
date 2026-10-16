"""
FFG/Hodge Rating Model - Geometrically Separated Transitive and Cyclic Components

Based on:
  Balduzzi et al. (2019) Open-ended Learning in Symmetric Zero-sum Games (arXiv 1901.08106)
  Hodge decomposition of antisymmetric win-fraction matrices.

Mathematical foundation:
  Win-fraction matrix per objective (tracked incrementally):
    W[i,j] = wins(i over j),  Total[i,j] = W[i,j] + W[j,i]
    A[i,j] = W[i,j]/Total[i,j] - 0.5  if Total > 0 else 0   (antisymmetric)

  Hodge decomposition (orthogonal by Hodge theorem):
    A[i,j] = (f[i] - f[j]) + C[i,j]
    f = lstsq(D, b) where D is the N(N-1)/2 x N incidence matrix,
                          b[e=(i,j)] = A[i,j] for i<j
    C[i,j] = A[i,j] - (f[i] - f[j])   (cyclic residual)
    Orthogonality: sum_{i<j} (f[i]-f[j]) * C[i,j] = 0 (exact, by lstsq property)

  Prediction:
    g(i,j) = tau * (f[i] - f[j]) + gamma * C[i,j]
    P(i > j) = sigmoid(g(i,j))

  Rating (TrueSkill-compatible):
    mu_i    = 1200.0 + f[i] * 200.0 / max(norm, 1e-9)
    sigma_i = cyclic_mag[i] * 200.0 / max(norm, 1e-9) + 50.0
    where norm = max(max(|f|), 1) and cyclic_mag[i] = mean_j |C[i,j]|
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .trueskill_engine import TrueSkillEngine, Rating
from .blade_chest import BladeChest
from .prior_information import PriorInformationEngine


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -500.0, 500.0))
    return 1.0 / (1.0 + np.exp(-x))


class StatisticalRatingSystem:
    """
    FFG/Hodge Rating Model implementing geometrically separated transitive and cyclic
    components via Hodge decomposition (Balduzzi et al. 2019, arXiv 1901.08106).

    Win-fraction matrix A is decomposed into:
      - Transitive component: A_trans[i,j] = f[i] - f[j]  (gradient flow)
      - Cyclic component:     C[i,j] = A[i,j] - (f[i] - f[j])  (curl)

    These are orthogonal by the Hodge theorem on graphs.

    Backward-compatible __init__ signature: (agents, alpha, beta, gamma_weight)
    where alpha = tau (transitive weight) and beta = gamma (cyclic weight).
    """

    def __init__(
        self,
        agents: List[Any],
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma_weight: float = 0.1,
    ):
        """
        Initialize the FFG/Hodge rating system.

        Args:
            agents: List of agent objects (kept for backward compat; agents_map built if non-empty)
            alpha: tau -- weight for transitive Hodge component (default 1.0)
            beta: gamma -- weight for cyclic Hodge component (default 1.0)
            gamma_weight: weight for context bias component (default 0.1)
        """
        self.alpha = alpha       # tau: transitive weight
        self.beta = beta         # gamma: cyclic weight
        self.gamma_weight = gamma_weight

        # Internal TrueSkill for fallback before enough win data accumulates
        self.trueskill_engine = TrueSkillEngine(
            mu=1200.0, sigma=200.0,
            agent_beta=80.0, agent_tau=4.0,
            policy_beta=150.0, policy_tau=6.0,
            prior_beta=120.0, prior_tau=5.0,
            agent_policy_beta=180.0, agent_policy_tau=15.0,
            agent_policy_prior_beta=220.0, agent_policy_prior_tau=18.0,
        )
        self.blade_chest_engine = BladeChest(self.trueskill_engine)
        self.prior_engine = PriorInformationEngine()

        # Win count tables: _wins[obj][entity_a][entity_b] = number of times a beat b
        self._wins: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Hodge decomposition results (recomputed after each update)
        # _transitive[obj] = {entity_key: f_value}
        self._transitive: Dict[str, Dict[str, float]] = {}
        # _cyclic[obj] = 2D numpy array (NxN), indexed by _entity_order[obj]
        self._cyclic: Dict[str, np.ndarray] = {}
        # _entity_order[obj] = sorted list of entity_keys included in decomp
        self._entity_order: Dict[str, List[str]] = {}

        # Context biases (kept for backward compat with existing callers)
        self.context_biases: Dict[str, Dict[str, float]] = {}
        self._context_bias_counts: Dict[str, Dict[str, int]] = {}

        if agents:
            self.agents_map = {a.agent_id: a for a in agents}
        else:
            self.agents_map = {}

    # ------------------------------------------------------------------
    # Entity key helpers
    # ------------------------------------------------------------------

    def _entity_key(self, agent_id: str, policy_id: str) -> str:
        """Return canonical entity key.  eval_agent -> policy_id only."""
        if agent_id in ('eval_agent', 'DUMMY_AGENT', ''):
            return policy_id
        return f"{agent_id}_{policy_id}"

    # ------------------------------------------------------------------
    # Win table update
    # ------------------------------------------------------------------

    def _record_win(self, obj: str, winner_key: str, loser_key: str) -> None:
        if obj not in self._wins:
            self._wins[obj] = {}
        wins_obj = self._wins[obj]
        if winner_key not in wins_obj:
            wins_obj[winner_key] = {}
        if loser_key not in wins_obj[winner_key]:
            wins_obj[winner_key][loser_key] = 0
        wins_obj[winner_key][loser_key] += 1

    # ------------------------------------------------------------------
    # Hodge decomposition
    # ------------------------------------------------------------------

    def _build_A_matrix(self, obj: str, keys: List[str]) -> np.ndarray:
        """Build antisymmetric win-fraction matrix A for the given entity list."""
        N = len(keys)
        idx = {k: i for i, k in enumerate(keys)}
        A = np.zeros((N, N), dtype=float)
        wins_obj = self._wins.get(obj, {})
        for ki in keys:
            for kj in keys:
                if ki == kj:
                    continue
                i, j = idx[ki], idx[kj]
                w_ij = wins_obj.get(ki, {}).get(kj, 0)
                w_ji = wins_obj.get(kj, {}).get(ki, 0)
                total = w_ij + w_ji
                if total > 0:
                    A[i, j] = w_ij / total - 0.5
        return A

    def _recompute_hodge(self, obj: str) -> None:
        """Recompute Hodge decomposition for a given objective."""
        wins_obj = self._wins.get(obj, {})
        # Collect all entity keys that have at least one matchup
        all_keys: set = set()
        for k, opponents in wins_obj.items():
            if opponents:
                all_keys.add(k)
                all_keys.update(opponents.keys())
        keys = sorted(all_keys)
        N = len(keys)
        if N < 2:
            return

        A = self._build_A_matrix(obj, keys)

        # Build incidence matrix D (shape: E x N) and b (shape: E,)
        # One row per ordered pair i<j
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        E = len(edges)
        D = np.zeros((E, N), dtype=float)
        b = np.zeros(E, dtype=float)
        for e_idx, (i, j) in enumerate(edges):
            D[e_idx, i] = 1.0
            D[e_idx, j] = -1.0
            b[e_idx] = A[i, j]

        # Solve: f = argmin ||D f - b||^2  (least-squares)
        f, _, _, _ = np.linalg.lstsq(D, b, rcond=None)
        f -= f.mean()  # Center (remove gauge freedom)

        # Cyclic residual C[i,j] = A[i,j] - (f[i] - f[j])
        idx_map = {k: i for i, k in enumerate(keys)}
        C = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j:
                    C[i, j] = A[i, j] - (f[i] - f[j])

        # Store results
        self._transitive[obj] = {keys[i]: float(f[i]) for i in range(N)}
        self._cyclic[obj] = C
        self._entity_order[obj] = keys

    # ------------------------------------------------------------------
    # Public rating update
    # ------------------------------------------------------------------

    def update_ratings(
        self,
        objective_performances: Dict[str, List[Tuple[str, str, float]]],
        pairwise_outcomes_map: Dict[str, List],
    ) -> None:
        """
        Update win counts from pairwise outcomes and recompute Hodge decomposition.

        Also updates internal TrueSkill engine (for fallback ratings) and BC engine.

        Args:
            objective_performances: {objective: [(agent_id, policy_id, score), ...]}
            pairwise_outcomes_map:  {objective: [(w_agent, w_policy, l_agent, l_policy,
                                                  margin, ...), ...]}
                                    (9-tuple or 5-tuple, only first 4 elements used here)
        """
        # --- Update internal TrueSkill fallback ---
        for obj, performances in objective_performances.items():
            if performances:
                ts_perfs = [(a, p, -s) for a, p, s in performances]
                self.trueskill_engine.update_ratings(ts_perfs)

        # --- Update Blade-Chest (backward compat) ---
        for obj, performances in objective_performances.items():
            pairs = [(a, p) for a, p, _ in performances]
            outcomes = pairwise_outcomes_map.get(obj, [])
            prior_ids = self.prior_engine.get_all_prior_ids()
            if outcomes:
                self.blade_chest_engine.update_embeddings(obj, pairs, outcomes, prior_ids)

        # --- Update win counts and recompute Hodge ---
        for obj, outcomes in pairwise_outcomes_map.items():
            for outcome in outcomes:
                w_agent, w_policy, l_agent, l_policy = outcome[0], outcome[1], outcome[2], outcome[3]
                winner_key = self._entity_key(w_agent, w_policy)
                loser_key = self._entity_key(l_agent, l_policy)
                self._record_win(obj, winner_key, loser_key)

        # Recompute Hodge for each objective that has data
        for obj in pairwise_outcomes_map:
            self._recompute_hodge(obj)

        # --- Update context biases (backward compat) ---
        self._update_context_biases(objective_performances)

        # --- Update prior information ratings (backward compat) ---
        self._update_prior_information_ratings(objective_performances)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _get_transitive_f(self, entity_key: str, obj: str) -> Optional[float]:
        """Return Hodge transitive f-value for entity in objective, or None if unknown."""
        return self._transitive.get(obj, {}).get(entity_key)

    def _get_cyclic_entry(self, entity_key_i: str, entity_key_j: str, obj: str) -> float:
        """Return C[i,j] cyclic entry, 0.0 if not available."""
        order = self._entity_order.get(obj)
        C = self._cyclic.get(obj)
        if order is None or C is None:
            return 0.0
        try:
            i = order.index(entity_key_i)
            j = order.index(entity_key_j)
        except ValueError:
            return 0.0
        return float(C[i, j])

    def _get_avg_cyclic_mag(self, entity_key: str, obj: str) -> float:
        """Return mean |C[i,j]| across j for entity_key in objective."""
        order = self._entity_order.get(obj)
        C = self._cyclic.get(obj)
        if order is None or C is None or entity_key not in order:
            return 0.0
        i = order.index(entity_key)
        row = C[i, :]
        mask = np.arange(len(order)) != i
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs(row[mask])))

    def compute_hybrid_score(
        self,
        objective: str,
        agent_id_i: str,
        policy_id_i: str,
        agent_id_j: str,
        policy_id_j: str,
    ) -> float:
        """
        Compute Hodge-based antisymmetric score g(i,j).

        g(i,j) = tau * (f[i] - f[j]) + gamma * C[i,j]

        Falls back to TrueSkill difference if Hodge data unavailable.
        """
        key_i = self._entity_key(agent_id_i, policy_id_i)
        key_j = self._entity_key(agent_id_j, policy_id_j)

        f_i = self._get_transitive_f(key_i, objective)
        f_j = self._get_transitive_f(key_j, objective)

        if f_i is None or f_j is None:
            # Fallback: use TrueSkill difference normalized to [-1, 1] scale
            r_i = self.trueskill_engine.get_rating(policy_id_i, 'policy')
            r_j = self.trueskill_engine.get_rating(policy_id_j, 'policy')
            ts_spread = max(self.trueskill_engine.policy_beta * np.sqrt(2), 1e-9)
            return self.alpha * (r_i.mu - r_j.mu) / ts_spread

        # Normalize f by max absolute value for scale-invariance
        all_f = list(self._transitive.get(objective, {}).values())
        norm = max(max(abs(v) for v in all_f) if all_f else 1.0, 1e-9)

        transitive_score = self.alpha * (f_i - f_j) / norm
        cyclic_score = self.beta * self._get_cyclic_entry(key_i, key_j, objective)

        # Context bias (backward compat)
        eid_i = self._entity_key(agent_id_i, policy_id_i)
        eid_j = self._entity_key(agent_id_j, policy_id_j)
        gamma_i = self.context_biases.get(objective, {}).get(eid_i, 0.0)
        gamma_j = self.context_biases.get(objective, {}).get(eid_j, 0.0)
        context_score = self.gamma_weight * (gamma_i - gamma_j)

        return transitive_score + cyclic_score + context_score

    def predict_outcome(
        self,
        objective: str,
        agent_id_a: str,
        policy_id_a: str,
        agent_id_b: str,
        policy_id_b: str,
    ) -> float:
        """Predict P(a > b) using Hodge decomposition. Returns probability in (0,1)."""
        g = self.compute_hybrid_score(objective, agent_id_a, policy_id_a, agent_id_b, policy_id_b)
        return _sigmoid(g)

    # ------------------------------------------------------------------
    # Ratings access
    # ------------------------------------------------------------------

    def get_rating(self, entity_id: str, rating_type: str) -> Rating:
        """
        Return Hodge-based Rating for entity_id.

        mu    = 1200 + f_avg * 200 / norm
        sigma = cyclic_mag_avg * 200 / norm + 50

        Falls back to TrueSkill if no Hodge data exists yet.
        """
        # Collect f values across all objectives
        f_values = []
        cyclic_mags = []
        for obj, trans_dict in self._transitive.items():
            if entity_id in trans_dict:
                f_values.append(trans_dict[entity_id])
                cyclic_mags.append(self._get_avg_cyclic_mag(entity_id, obj))

        if not f_values:
            # Fall back to TrueSkill
            return self.trueskill_engine.get_rating(entity_id, rating_type)

        f_avg = float(np.mean(f_values))
        cyclic_mag = float(np.mean(cyclic_mags))

        # Compute norm from all entities across all objectives
        all_f_vals = [
            v for trans_dict in self._transitive.values() for v in trans_dict.values()
        ]
        norm = max(max(abs(v) for v in all_f_vals) if all_f_vals else 1.0, 1e-9)
        norm = max(norm, 1.0)

        mu = 1200.0 + f_avg * 200.0 / norm
        sigma = cyclic_mag * 200.0 / norm + 50.0

        rating = Rating(mu=mu, sigma=sigma)
        return rating

    def get_hodge_components(self, objective: str) -> dict:
        """
        Return Hodge decomposition components for the given objective.

        Returns:
            {
              'transitive': {policy_key: f_value, ...},
              'cyclic': C_matrix (NxN numpy array),
              'policy_ids': [sorted list of entity keys],
            }
        """
        return {
            'transitive': dict(self._transitive.get(objective, {})),
            'cyclic': self._cyclic.get(objective, np.zeros((0, 0))).copy(),
            'policy_ids': list(self._entity_order.get(objective, [])),
        }

    def get_model_weights(self) -> dict:
        """Return current model hyperparameters."""
        return {
            'transitive_weight': self.alpha,
            'cyclic_weight': self.beta,
            'context_weight': self.gamma_weight,
        }

    def set_model_weights(
        self,
        alpha: float = None,
        beta: float = None,
        gamma_weight: float = None,
    ) -> None:
        """Update model hyperparameters."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma_weight is not None:
            self.gamma_weight = gamma_weight

    # ------------------------------------------------------------------
    # Context bias (backward compat helpers)
    # ------------------------------------------------------------------

    def _update_context_biases(
        self, objective_performances: Dict[str, List[Tuple[str, str, float]]]
    ) -> None:
        for obj, performances in objective_performances.items():
            if obj not in self.context_biases:
                self.context_biases[obj] = {}
            if obj not in self._context_bias_counts:
                self._context_bias_counts[obj] = {}
            obj_biases = self.context_biases[obj]
            obj_counts = self._context_bias_counts[obj]

            for agent_id, policy_id, score in performances:
                entity_id = self._entity_key(agent_id, policy_id)
                r_agent = self.trueskill_engine.get_rating(agent_id, 'agent')
                r_policy = self.trueskill_engine.get_rating(policy_id, 'policy')
                expected = (r_agent.mu + r_policy.mu - 1200.0) / 10.0 + 50.0
                residual = score - expected
                n = obj_counts.get(entity_id, 0)
                lr = 0.1 / (1.0 + n)
                obj_counts[entity_id] = n + 1
                current = obj_biases.get(entity_id, 0.0)
                new_bias = current * (1.0 - lr) + residual * lr
                obj_biases[entity_id] = float(np.clip(new_bias, -50.0, 50.0))

    def _update_prior_information_ratings(
        self, objective_performances: Dict[str, List[Tuple[str, str, float]]]
    ) -> None:
        for obj, performances in objective_performances.items():
            if not performances:
                continue
            prior_performances = []
            for prior_id in self.prior_engine.get_all_prior_ids():
                total_score = 0.0
                count = 0
                for agent_id, policy_id, actual_score in performances:
                    if 'color_distance' in obj.lower():
                        bias_map = {
                            'balanced_colors': 0.95,
                            'high_saturation': 0.85,
                            'low_saturation': 0.88,
                            'warm_colors': 0.92,
                            'cool_colors': 0.91,
                        }
                        bias = bias_map.get(prior_id, 0.90)
                    else:
                        bias = 0.90
                    prior_score = actual_score * bias + np.random.normal(0, 3)
                    total_score += prior_score
                    count += 1
                if count > 0:
                    prior_performances.append((prior_id, total_score / count))
            if prior_performances:
                self.trueskill_engine.update_ratings_batch(prior_performances, 'prior')

    # ------------------------------------------------------------------
    # Context bias public API (backward compat)
    # ------------------------------------------------------------------

    def get_context_bias(self, agent_id: str, policy_id: str, objective: str = None) -> float:
        entity_id = self._entity_key(agent_id, policy_id)
        if objective is not None:
            return self.context_biases.get(objective, {}).get(entity_id, 0.0)
        if not self.context_biases:
            return 0.0
        values = [d.get(entity_id, 0.0) for d in self.context_biases.values()]
        return float(np.mean(values))

    def set_context_bias(
        self, agent_id: str, policy_id: str, bias: float, objective: str = None
    ) -> None:
        entity_id = self._entity_key(agent_id, policy_id)
        clipped = float(np.clip(bias, -50.0, 50.0))
        if objective is not None:
            if objective not in self.context_biases:
                self.context_biases[objective] = {}
            self.context_biases[objective][entity_id] = clipped
        else:
            for d in self.context_biases.values():
                d[entity_id] = clipped


# Alias
FFGHodgeModel = StatisticalRatingSystem
