"""
Intransitive Clustered Bradley-Terry (ICBT) Rating Model.

Based on: Modelling intransitivity in pairwise comparisons (arXiv 2103.12094).

Model: K mixture clusters of Bradley-Terry models.
  Each cluster k: P_k(i > j) = sigmoid(log_pi[k][i] - log_pi[k][j])
  Cluster assignment soft: r_km = softmax([BT_loglik(k,m) for k in range(K)])

  E-step: r_km = softmax_k(log BT_k(winner > loser | matchup m))
  M-step: weighted Newton update for log_pi[k] per cluster
  K selected by BIC: BIC(K) = -2*LL + K*N*log(M)

Rating:
  mu_i    = 1200 + mean_k(r_k_mean * log_pi[k][i]) * scale
  sigma_i = std across clusters of log_pi[k][i] * scale + 50
  (captures how consistently i ranks across clusters = intransitivity measure)

Performance note:
  All EM inner loops are vectorized with NumPy (no Python loops over matchups).
  Complexity per _fit(): O(max_k * n_restarts * n_em_iter * (M + K * N)).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

try:
    from .trueskill_engine import Rating
except ImportError:
    from collections import namedtuple
    Rating = namedtuple('Rating', ['mu', 'sigma'])


DUMMY_AGENT = 'eval_agent'
_RATING_SCALE = 200.0


def _log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x)) for scalar use."""
    x = float(np.clip(x, -500.0, 500.0))
    if x >= 0:
        return -np.log1p(np.exp(-x))
    else:
        return x - np.log1p(np.exp(x))


def _log_sigmoid_vec(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x)) vectorized over array x."""
    x = np.clip(x, -500.0, 500.0)
    out = np.where(x >= 0,
                   -np.log1p(np.exp(-x)),
                   x - np.log1p(np.exp(x)))
    return out


def _softmax(log_probs: np.ndarray) -> np.ndarray:
    """Compute softmax of log_probs along last axis, numerically stable."""
    log_probs = np.asarray(log_probs, dtype=float)
    shifted = log_probs - log_probs.max()
    exp_vals = np.exp(shifted)
    return exp_vals / (exp_vals.sum() + 1e-300)


def _softmax_rows(log_r: np.ndarray) -> np.ndarray:
    """Softmax along axis=1 for a (M, K) matrix, numerically stable."""
    shifted = log_r - log_r.max(axis=1, keepdims=True)
    exp_r = np.exp(shifted)
    return exp_r / (exp_r.sum(axis=1, keepdims=True) + 1e-300)


def _matchups_to_arrays(matchups, idx):
    """Convert list of (ei, ej, i_won) to parallel numpy arrays."""
    M = len(matchups)
    i_arr = np.empty(M, dtype=np.int32)
    j_arr = np.empty(M, dtype=np.int32)
    won_arr = np.empty(M, dtype=bool)
    valid = np.ones(M, dtype=bool)
    for m, (ei, ej, i_won) in enumerate(matchups):
        ii = idx.get(ei, -1)
        jj = idx.get(ej, -1)
        if ii < 0 or jj < 0:
            valid[m] = False
            i_arr[m] = 0
            j_arr[m] = 0
            won_arr[m] = True
        else:
            i_arr[m] = ii
            j_arr[m] = jj
            won_arr[m] = i_won
    # Filter out invalid entries
    mask = valid
    return i_arr[mask], j_arr[mask], won_arr[mask]


class ICBTModel:
    """
    Intransitive Clustered Bradley-Terry (ICBT) Rating Model.

    K mixture clusters of Bradley-Terry models with EM training.
    K selected by BIC over K=1..max_k.

    All matchup history is accumulated globally so EM has sufficient data.
    EM is re-run only when the buffer grows by at least _refit_threshold new entries.
    """

    def __init__(
        self,
        max_k: int = 5,
        n_em_iter: int = 20,
        n_newton_iter: int = 5,
        lambda_reg: float = 0.01,
    ):
        self.max_k = max_k
        self.n_em_iter = n_em_iter
        self.n_newton_iter = n_newton_iter
        self.lambda_reg = lambda_reg

        # Accumulated matchup buffer: list of (entity_i, entity_j, i_won: bool)
        self._matchup_buffer: List[Tuple[str, str, bool]] = []
        self._last_fit_size: int = 0
        self._refit_threshold: int = 30   # was 10; larger = fewer expensive refits

        # EM solution
        self._best_k: int = 1
        self._log_pi: Dict[int, Dict[str, float]] = {}   # {cluster_k: {entity: float}}
        self._cluster_weights: List[float] = [1.0]       # mean responsibility per cluster

        # Cached entity list (sorted for stable ordering)
        self._entities: List[str] = []

    # ------------------------------------------------------------------
    # Entity key helper
    # ------------------------------------------------------------------

    def _entity_key(self, agent_id: str, policy_id: str) -> str:
        """Canonical entity key.  eval_agent -> policy_id only."""
        if agent_id in (DUMMY_AGENT, 'DUMMY_AGENT', ''):
            return policy_id
        return f"{agent_id}_{policy_id}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_ratings(
        self,
        objective_performances: Dict[str, List[Tuple[str, str, float]]],
        pairwise_outcomes_map: Dict[str, List],
    ) -> None:
        """
        Accumulate matchup data from pairwise outcomes and re-run EM if needed.
        """
        for obj, outcomes in pairwise_outcomes_map.items():
            for outcome in outcomes:
                w_agent, w_policy = outcome[0], outcome[1]
                l_agent, l_policy = outcome[2], outcome[3]
                winner_key = self._entity_key(w_agent, w_policy)
                loser_key = self._entity_key(l_agent, l_policy)
                self._matchup_buffer.append((winner_key, loser_key, True))

        # Re-run EM only when enough new data has arrived
        new_size = len(self._matchup_buffer)
        if new_size - self._last_fit_size >= self._refit_threshold:
            self._fit()
            self._last_fit_size = new_size

    def predict_outcome(
        self,
        objective: str,
        agent_id_a: str,
        policy_id_a: str,
        agent_id_b: str,
        policy_id_b: str,
    ) -> float:
        """
        Predict P(a > b) using mixture of BT models.

        P(a > b) = sigmoid(sum_k pi_k * (log_pi[k][a] - log_pi[k][b]))
        """
        key_a = self._entity_key(agent_id_a, policy_id_a)
        key_b = self._entity_key(agent_id_b, policy_id_b)

        if not self._log_pi:
            return 0.5

        score = 0.0
        for k in range(self._best_k):
            lp = self._log_pi.get(k, {})
            pi_k = self._cluster_weights[k] if k < len(self._cluster_weights) else 0.0
            la = lp.get(key_a, 0.0)
            lb = lp.get(key_b, 0.0)
            score += pi_k * (la - lb)

        x = float(np.clip(score, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-x))

    def get_rating(self, entity_id: str, rating_type: str) -> Rating:
        """
        Return Rating for entity_id.

        mu    = 1200 + weighted_mean_k(log_pi[k][entity]) * scale
        sigma = std_k(log_pi[k][entity]) * scale + 50
        """
        if not self._log_pi or not self._entities:
            return Rating(mu=1200.0, sigma=200.0)

        found_key = None
        if entity_id in self._entities:
            found_key = entity_id
        else:
            for ek in self._entities:
                if ek == entity_id or ek.endswith(f"_{entity_id}"):
                    found_key = ek
                    break

        if found_key is None:
            return Rating(mu=1200.0, sigma=200.0)

        lp_vals = np.array([
            self._log_pi.get(k, {}).get(found_key, 0.0)
            for k in range(self._best_k)
        ], dtype=float)
        weights = np.array(self._cluster_weights[:self._best_k], dtype=float)
        weights = weights / (weights.sum() + 1e-300)

        weighted_mean = float(np.dot(weights, lp_vals))
        unweighted_std = float(np.std(lp_vals)) if self._best_k > 1 else 0.0

        all_lp = [
            v for k in range(self._best_k)
            for v in self._log_pi.get(k, {}).values()
        ]
        if all_lp:
            lp_range = max(max(all_lp) - min(all_lp), 1e-9)
        else:
            lp_range = 1.0
        scale = min(_RATING_SCALE / lp_range, 50.0)

        mu = 1200.0 + weighted_mean * scale
        sigma = unweighted_std * scale + 50.0
        return Rating(mu=float(mu), sigma=float(sigma))

    def get_nash_clusters(self) -> List[List[str]]:
        """Not implemented in base ICBT (see SpinningTopsModel)."""
        return [list(self._entities)]

    # ------------------------------------------------------------------
    # EM fitting
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Run full EM: select K by BIC, store solution."""
        if len(self._matchup_buffer) < 2:
            return

        entity_set: set = set()
        for ei, ej, _ in self._matchup_buffer:
            entity_set.add(ei)
            entity_set.add(ej)
        entities = sorted(entity_set)
        N = len(entities)
        M = len(self._matchup_buffer)

        if N < 2:
            return

        idx = {e: i for i, e in enumerate(entities)}
        # Pre-convert matchups to arrays once (shared across all K, restarts, iters)
        i_arr, j_arr, won_arr = _matchups_to_arrays(self._matchup_buffer, idx)
        M_valid = len(i_arr)

        if M_valid < 2:
            return

        best_bic = float('inf')
        best_log_pi = None
        best_weights = None
        best_k_val = 1

        for K in range(1, min(self.max_k, N) + 1):
            try:
                log_pi, responsibilities, ll = self._em_one_k(
                    K, entities, idx, i_arr, j_arr, won_arr, N, M_valid
                )
            except Exception:
                continue

            bic = self._bic(K, N, M_valid, ll)
            if bic < best_bic:
                best_bic = bic
                best_log_pi = log_pi
                best_k_val = K
                if responsibilities is not None and len(responsibilities) > 0:
                    resp_arr = np.array(responsibilities)  # (M_valid, K)
                    best_weights = list(resp_arr.mean(axis=0))
                else:
                    best_weights = [1.0 / K] * K

        if best_log_pi is None:
            return

        self._best_k = best_k_val
        self._log_pi = best_log_pi
        self._cluster_weights = best_weights
        self._entities = entities

    def _em_one_k(
        self,
        K: int,
        entities: List[str],
        idx: Dict[str, int],
        i_arr: np.ndarray,
        j_arr: np.ndarray,
        won_arr: np.ndarray,
        N: int,
        M: int,
    ) -> Tuple[Dict[int, Dict[str, float]], List[np.ndarray], float]:
        """
        Run EM for K clusters (fully vectorized).

        E-step: log_r[m, k] = log_sigmoid(+/- diff_k[m]) depending on winner
        M-step: diagonal Newton via np.add.at (no Python loop over matchups)
        """
        rng = np.random.RandomState(42)
        best_ll = -float('inf')
        best_lp_arr = None

        for _restart in range(3):
            lp = rng.randn(K, N) * 0.01  # (K, N)

            for _iter in range(self.n_em_iter):
                # ---- E-step (vectorized) --------------------------------
                # diff[k, m] = lp[k, i_arr[m]] - lp[k, j_arr[m]]
                diff = lp[:, i_arr] - lp[:, j_arr]   # (K, M)
                # log_r[m, k]:  log_sigmoid(diff[k,m]) if won, else log_sigmoid(-diff[k,m])
                log_r = np.where(
                    won_arr[np.newaxis, :],            # broadcast (1, M)
                    _log_sigmoid_vec(diff),            # (K, M)
                    _log_sigmoid_vec(-diff),
                ).T                                    # (M, K)

                responsibilities = _softmax_rows(log_r)  # (M, K)

                # ---- M-step Newton (vectorized over matchups) -----------
                for k in range(K):
                    r_k = responsibilities[:, k]       # (M,)
                    diff_k = lp[k, i_arr] - lp[k, j_arr]  # (M,)
                    p = 1.0 / (1.0 + np.exp(-np.clip(diff_k, -500.0, 500.0)))  # (M,)

                    for _newton in range(self.n_newton_iter):
                        # Gradient contributions at i and j
                        # if won:  grad[i] += r*(1-p),  grad[j] -= r*(1-p)
                        # if lost: grad[i] -= r*p,       grad[j] += r*p
                        gi = np.where(won_arr, r_k * (1.0 - p), -r_k * p)  # (M,)
                        grad = np.zeros(N)
                        np.add.at(grad, i_arr, gi)
                        np.add.at(grad, j_arr, -gi)

                        # Hessian diagonal (always negative semi-definite)
                        h = r_k * p * (1.0 - p)       # (M,)
                        hess = np.full(N, -self.lambda_reg)
                        np.add.at(hess, i_arr, -h)
                        np.add.at(hess, j_arr, -h)

                        safe_hess = np.where(np.abs(hess) > 1e-8, hess, -1e-8)
                        delta = np.clip(-grad / safe_hess, -0.5, 0.5)
                        lp[k] += delta

                        # Recompute p for next Newton step
                        diff_k = lp[k, i_arr] - lp[k, j_arr]
                        p = 1.0 / (1.0 + np.exp(-np.clip(diff_k, -500.0, 500.0)))

            ll = self._compute_ll(K, lp, i_arr, j_arr, won_arr)
            if ll > best_ll:
                best_ll = ll
                best_lp_arr = lp.copy()

        if best_lp_arr is None:
            return {}, None, -float('inf')

        # Final E-step for responsibilities
        diff_final = best_lp_arr[:, i_arr] - best_lp_arr[:, j_arr]  # (K, M)
        log_r_final = np.where(
            won_arr[np.newaxis, :],
            _log_sigmoid_vec(diff_final),
            _log_sigmoid_vec(-diff_final),
        ).T                                                            # (M, K)
        final_resp = _softmax_rows(log_r_final)                       # (M, K)

        log_pi = {
            k: {entities[n]: float(best_lp_arr[k, n]) for n in range(N)}
            for k in range(K)
        }
        return log_pi, list(final_resp), best_ll

    def _compute_ll(
        self,
        K: int,
        lp: np.ndarray,
        i_arr: np.ndarray,
        j_arr: np.ndarray,
        won_arr: np.ndarray,
    ) -> float:
        """Compute total log-likelihood (vectorized)."""
        diff = lp[:, i_arr] - lp[:, j_arr]   # (K, M)
        diff = np.clip(diff, -500.0, 500.0)
        # Per-cluster likelihood: sigmoid(+/-diff)
        lik_k = np.where(
            won_arr[np.newaxis, :],
            1.0 / (1.0 + np.exp(-diff)),
            1.0 / (1.0 + np.exp(diff)),
        )                                      # (K, M)
        # Mixture: mean over clusters
        lik_mix = lik_k.mean(axis=0)           # (M,)
        return float(np.sum(np.log(np.maximum(lik_mix, 1e-300))))

    def _bic(self, K: int, N: int, M: int, log_likelihood: float) -> float:
        """BIC = -2*LL + K*N*log(M)."""
        n_params = K * N
        return -2.0 * log_likelihood + n_params * np.log(max(M, 2))
