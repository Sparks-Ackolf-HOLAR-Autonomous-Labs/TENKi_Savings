"""
Spinning Tops Rating Model.

Based on: Czarnecki et al. (2020) Real World Games Look Like Spinning Tops (arXiv 2004.09468).

Hodge decomposition of the policy payoff matrix into:
  - Transitive axis (vertical): consistent skill ordering
  - Cyclic axis (radial): intransitive rock-paper-scissors cycles

Nash clustering: iteratively extract Nash equilibria from the cyclic subgame.
Each Nash cluster represents a group of policies with mutual intransitive dominance.

This model produces:
  - transitive_ratings: scalar per policy (skill hierarchy)
  - cyclic_matrix: antisymmetric matrix (cyclic interactions)
  - nash_clusters: list of policy groups (for Phase 3 game_theory.py)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linprog

try:
    from .trueskill_engine import Rating
    from .statistical_rating_system import StatisticalRatingSystem, _sigmoid
except ImportError:
    from collections import namedtuple
    Rating = namedtuple('Rating', ['mu', 'sigma'])
    from statistical_rating_system import StatisticalRatingSystem, _sigmoid

DUMMY_AGENT = 'eval_agent'


def _compute_nash_lp(A: np.ndarray) -> np.ndarray:
    """
    Compute a Nash equilibrium mixed strategy for zero-sum game with payoff matrix A.

    For row player: maximize v s.t. A^T x <= v*1, sum(x)=1, x>=0
    Equivalent LP: maximize v,  variables [x_1..x_N, v]
      - A^T x - v*1 <= 0    =>  [A^T | -1] [x;v] <= 0
      - sum(x) = 1           =>  [1..1 | 0] [x;v] = 1
      - x >= 0, v unconstrained

    Returns:
        x: Nash mixed strategy (np.array of length N, sums to 1)
    """
    N = A.shape[0]
    if N == 1:
        return np.array([1.0])
    if N == 2:
        # Analytical 2x2 solution
        a, b = A[0, 1], A[1, 0]
        denom = a + b
        if abs(denom) < 1e-10:
            return np.array([0.5, 0.5])
        p = b / denom
        p = float(np.clip(p, 0.0, 1.0))
        return np.array([p, 1.0 - p])

    # General case: LP
    # Variables: [x_0..x_{N-1}, v]  (total N+1)
    # Maximize v  =>  minimize -v  =>  c = [0..0, -1]
    c = np.zeros(N + 1)
    c[-1] = -1.0  # minimize -v

    # A^T x - v <= 0  =>  [A^T | -1] [x;v] <= 0
    A_ub = np.hstack([A.T, -np.ones((N, 1))])
    b_ub = np.zeros(N)

    # sum(x) = 1
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # x >= 0, v unbounded
    bounds = [(0.0, None)] * N + [(None, None)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if result.success:
            x = result.x[:N]
            x = np.maximum(x, 0.0)
            s = x.sum()
            if s > 1e-10:
                return x / s
    except Exception:
        pass

    # Fallback: uniform
    return np.ones(N) / N


class SpinningTopsModel:
    """
    Spinning Tops Rating Model (Czarnecki et al. 2020).

    Reuses Hodge decomposition math from FFGHodgeModel (StatisticalRatingSystem)
    and adds Nash clustering on the cyclic subgame.

    Interface is identical to other rating models in this package.
    """

    def __init__(self, nash_threshold: float = 0.05, recompute_every: int = 20):
        self.nash_threshold = nash_threshold
        self._recompute_every = recompute_every  # only recompute Hodge+Nash every N updates

        # Win count tables: _wins[obj][entity_a][entity_b] = count
        self._wins: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._total_updates: int = 0

        # Hodge decomposition results (shared across all objectives)
        self._transitive: Dict[str, Dict[str, float]] = {}  # {obj: {key: f}}
        self._cyclic: Dict[str, np.ndarray] = {}             # {obj: NxN array}
        self._entity_order: Dict[str, List[str]] = {}        # {obj: sorted keys}

        # Nash clusters (computed from cyclic, averaged across objectives)
        self._nash_clusters: List[List[str]] = []

    # ------------------------------------------------------------------
    # Entity key helper
    # ------------------------------------------------------------------

    def _entity_key(self, agent_id: str, policy_id: str) -> str:
        if agent_id in (DUMMY_AGENT, 'DUMMY_AGENT', ''):
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
    # Hodge decomposition (same algorithm as StatisticalRatingSystem)
    # ------------------------------------------------------------------

    def _build_A_matrix(self, obj: str, keys: List[str]) -> np.ndarray:
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
        wins_obj = self._wins.get(obj, {})
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

        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        E = len(edges)
        D = np.zeros((E, N), dtype=float)
        b = np.zeros(E, dtype=float)
        for e_idx, (i, j) in enumerate(edges):
            D[e_idx, i] = 1.0
            D[e_idx, j] = -1.0
            b[e_idx] = A[i, j]

        f, _, _, _ = np.linalg.lstsq(D, b, rcond=None)
        f -= f.mean()

        C = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j:
                    C[i, j] = A[i, j] - (f[i] - f[j])

        self._transitive[obj] = {keys[i]: float(f[i]) for i in range(N)}
        self._cyclic[obj] = C
        self._entity_order[obj] = keys

    # ------------------------------------------------------------------
    # Nash clustering
    # ------------------------------------------------------------------

    def _compute_nash_clusters(self) -> List[List[str]]:
        """
        Iteratively extract Nash equilibria from the cyclic subgame.

        Averages cyclic matrices across all objectives, then extracts
        clusters by finding Nash equilibria on the cyclic subgame.
        """
        # Collect all entity keys across all objectives
        all_keys: set = set()
        for keys in self._entity_order.values():
            all_keys.update(keys)
        all_keys_list = sorted(all_keys)
        N = len(all_keys_list)
        if N < 2:
            return [all_keys_list] if all_keys_list else []

        # Build averaged cyclic matrix
        key_idx = {k: i for i, k in enumerate(all_keys_list)}
        C_avg = np.zeros((N, N), dtype=float)
        count_per_obj = 0
        for obj, keys in self._entity_order.items():
            C_obj = self._cyclic.get(obj)
            if C_obj is None:
                continue
            count_per_obj += 1
            for local_i, ki in enumerate(keys):
                for local_j, kj in enumerate(keys):
                    if ki == kj:
                        continue
                    gi = key_idx.get(ki, -1)
                    gj = key_idx.get(kj, -1)
                    if gi >= 0 and gj >= 0:
                        C_avg[gi, gj] += C_obj[local_i, local_j]

        if count_per_obj > 0:
            C_avg /= count_per_obj

        # Iterative Nash extraction
        remaining = set(all_keys_list)
        clusters = []

        max_iters = N + 1  # safety bound
        iter_count = 0
        while remaining and iter_count < max_iters:
            iter_count += 1
            rem_list = sorted(remaining)
            n_rem = len(rem_list)
            rem_idx = {k: i for i, k in enumerate(rem_list)}

            # Extract cyclic submatrix for remaining entities
            C_sub = np.zeros((n_rem, n_rem), dtype=float)
            for ki in rem_list:
                for kj in rem_list:
                    if ki == kj:
                        continue
                    gi = key_idx.get(ki, -1)
                    gj = key_idx.get(kj, -1)
                    if gi >= 0 and gj >= 0:
                        C_sub[rem_idx[ki], rem_idx[kj]] = C_avg[gi, gj]

            # Scale to [0,1] for game matrix (0.5 + C / (2*max|C|))
            max_abs = np.max(np.abs(C_sub))
            if max_abs < 1e-9:
                # Cyclic subgame is trivially zero => all in one cluster
                clusters.append(rem_list)
                break

            A_game = 0.5 + C_sub / (2.0 * max_abs)

            # Compute Nash mixed strategy
            nash_x = _compute_nash_lp(A_game)

            # Entities with Nash weight > threshold form the cluster
            cluster = [rem_list[i] for i in range(n_rem) if nash_x[i] > self.nash_threshold]

            if not cluster:
                # Fallback: highest Nash weight
                best_i = int(np.argmax(nash_x))
                cluster = [rem_list[best_i]]

            clusters.append(cluster)
            remaining -= set(cluster)

        # Any remainder goes into final cluster
        if remaining:
            clusters.append(sorted(remaining))

        return clusters

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_ratings(
        self,
        objective_performances: Dict[str, List[Tuple[str, str, float]]],
        pairwise_outcomes_map: Dict[str, List],
    ) -> None:
        """Accumulate win counts; recompute Hodge+Nash only every recompute_every calls."""
        for obj, outcomes in pairwise_outcomes_map.items():
            for outcome in outcomes:
                w_agent, w_policy = outcome[0], outcome[1]
                l_agent, l_policy = outcome[2], outcome[3]
                winner_key = self._entity_key(w_agent, w_policy)
                loser_key = self._entity_key(l_agent, l_policy)
                self._record_win(obj, winner_key, loser_key)

        self._total_updates += 1
        if self._total_updates % self._recompute_every == 0:
            for obj in pairwise_outcomes_map:
                self._recompute_hodge(obj)
            self._nash_clusters = self._compute_nash_clusters()

    def predict_outcome(
        self,
        objective: str,
        agent_id_a: str,
        policy_id_a: str,
        agent_id_b: str,
        policy_id_b: str,
    ) -> float:
        """
        Predict P(a > b) using Hodge model (same as FFGHodgeModel).

        g(i,j) = f[i] - f[j] + C[i,j]
        """
        key_a = self._entity_key(agent_id_a, policy_id_a)
        key_b = self._entity_key(agent_id_b, policy_id_b)

        f_a = self._transitive.get(objective, {}).get(key_a)
        f_b = self._transitive.get(objective, {}).get(key_b)

        if f_a is None or f_b is None:
            return 0.5

        all_f = list(self._transitive.get(objective, {}).values())
        norm = max(max(abs(v) for v in all_f) if all_f else 1.0, 1.0)

        transitive = (f_a - f_b) / norm
        cyclic = self._get_cyclic_entry(key_a, key_b, objective)

        g = transitive + cyclic
        return _sigmoid(g)

    def _get_cyclic_entry(self, key_i: str, key_j: str, obj: str) -> float:
        order = self._entity_order.get(obj)
        C = self._cyclic.get(obj)
        if order is None or C is None:
            return 0.0
        try:
            i = order.index(key_i)
            j = order.index(key_j)
        except ValueError:
            return 0.0
        return float(C[i, j])

    def get_rating(self, entity_id: str, rating_type: str) -> Rating:
        """
        Return Hodge-based Rating.

        mu = 1200 + f_avg * 200 / norm
        sigma = cyclic_mag * 200 / norm + 50
        """
        f_values = []
        cyclic_mags = []
        for obj, trans_dict in self._transitive.items():
            if entity_id in trans_dict:
                f_values.append(trans_dict[entity_id])
                # Cyclic magnitude: mean |C[i,j]| across j
                order = self._entity_order.get(obj, [])
                C = self._cyclic.get(obj)
                if C is not None and entity_id in order:
                    i = order.index(entity_id)
                    mask = np.arange(len(order)) != i
                    if mask.any():
                        cyclic_mags.append(float(np.mean(np.abs(C[i, mask]))))

        if not f_values:
            return Rating(mu=1200.0, sigma=200.0)

        f_avg = float(np.mean(f_values))
        cyclic_mag = float(np.mean(cyclic_mags)) if cyclic_mags else 0.0

        all_f_vals = [
            v for trans_dict in self._transitive.values() for v in trans_dict.values()
        ]
        norm = max(max(abs(v) for v in all_f_vals) if all_f_vals else 1.0, 1.0)

        mu = 1200.0 + f_avg * 200.0 / norm
        sigma = cyclic_mag * 200.0 / norm + 50.0
        return Rating(mu=float(mu), sigma=float(sigma))

    def get_nash_clusters(self) -> List[List[str]]:
        """Return list of entity-key lists representing Nash clusters."""
        if not self._nash_clusters:
            # Return all entities in one cluster as fallback
            all_keys: set = set()
            for keys in self._entity_order.values():
                all_keys.update(keys)
            return [sorted(all_keys)] if all_keys else []
        return [list(c) for c in self._nash_clusters]

    def get_hodge_components(self, objective: str) -> dict:
        """Return Hodge decomposition components for the given objective."""
        return {
            'transitive': dict(self._transitive.get(objective, {})),
            'cyclic': self._cyclic.get(objective, np.zeros((0, 0))).copy(),
            'policy_ids': list(self._entity_order.get(objective, [])),
        }
