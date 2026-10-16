"""
Augmented ICBT+BC Rating Model.

Extends ICBT with per-cluster Blade-Chest (BC) bilinear vectors.
Each cluster k has both BT skill parameters AND (blade_k, chest_k) vectors.

E-step assignment uses BOTH BT and BC log-likelihoods:
  log_r_km = BT_loglik(k, m) + BC_loglik(k, m)
  r_km = softmax_k(log_r_km)

BC log-likelihood for cluster k, matchup m (i beats j):
  BC_loglik_k(m) = log_sigmoid(blade_k[i] . chest_k[j] - blade_k[j] . chest_k[i])

M-step:
  - BT: same Newton update as ICBT, weighted by r_km (vectorized)
  - BC: soft gradient step (vectorized over matchups per cluster)

Performance note:
  All inner loops are vectorized with NumPy array operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from .trueskill_engine import Rating
    from .icbt_model import (
        ICBTModel, _log_sigmoid_vec, _softmax_rows,
        _matchups_to_arrays, DUMMY_AGENT, _RATING_SCALE,
    )
except ImportError:
    from collections import namedtuple
    Rating = namedtuple('Rating', ['mu', 'sigma'])
    from icbt_model import (
        ICBTModel, _log_sigmoid_vec, _softmax_rows,
        _matchups_to_arrays, DUMMY_AGENT, _RATING_SCALE,
    )


class AugmentedICBTBC(ICBTModel):
    """
    Augmented ICBT with per-cluster Blade-Chest bilinear vectors.

    Inherits EM orchestration from ICBTModel.  Overrides _em_one_k to include
    BC log-likelihoods in the E-step and adds BC gradient steps in M-step.
    """

    def __init__(
        self,
        max_k: int = 5,
        n_em_iter: int = 20,
        lambda_reg: float = 0.01,
        bc_lr: float = 1e-3,
        bc_dim: int = 2,
    ):
        super().__init__(
            max_k=max_k,
            n_em_iter=n_em_iter,
            lambda_reg=lambda_reg,
        )
        self.bc_lr = bc_lr
        self.bc_dim = bc_dim

        # Per-cluster BC matrices: blades[k] shape (N, bc_dim), chests[k] shape (N, bc_dim)
        # Populated lazily inside _em_one_k
        self._blades_arr: Dict[int, np.ndarray] = {}
        self._chests_arr: Dict[int, np.ndarray] = {}

        # Keep dict API for get_rating / predict_outcome (small N, not a bottleneck)
        self._blades: Dict[int, Dict[str, np.ndarray]] = {}
        self._chests: Dict[int, Dict[str, np.ndarray]] = {}

        self._bc_rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # BC helpers (vectorized array form)
    # ------------------------------------------------------------------

    def _init_bc_arrays(self, K: int, N: int) -> None:
        """Initialize per-cluster BC matrices (K, N, bc_dim)."""
        for k in range(K):
            if k not in self._blades_arr:
                self._blades_arr[k] = self._bc_rng.randn(N, self.bc_dim) * 0.1
                self._chests_arr[k] = self._bc_rng.randn(N, self.bc_dim) * 0.1

    def _bc_logit_vec(self, k: int, i_arr: np.ndarray, j_arr: np.ndarray) -> np.ndarray:
        """BC antisymmetric logit b[i].c[j] - b[j].c[i] for all matchups (vectorized)."""
        B = self._blades_arr[k]   # (N, D)
        C = self._chests_arr[k]   # (N, D)
        # dot(b[i], c[j]) = (B[i_arr] * C[j_arr]).sum(axis=1)
        bi_cj = (B[i_arr] * C[j_arr]).sum(axis=1)   # (M,)
        bj_ci = (B[j_arr] * C[i_arr]).sum(axis=1)   # (M,)
        return bi_cj - bj_ci

    def _bc_logit(self, k: int, entity_i: str, entity_j: str) -> float:
        """Scalar BC logit for predict_outcome (uses dict form)."""
        bi = self._blades.get(k, {}).get(entity_i, np.zeros(self.bc_dim))
        ci = self._chests.get(k, {}).get(entity_i, np.zeros(self.bc_dim))
        bj = self._blades.get(k, {}).get(entity_j, np.zeros(self.bc_dim))
        cj = self._chests.get(k, {}).get(entity_j, np.zeros(self.bc_dim))
        return float(np.dot(bi, cj) - np.dot(bj, ci))

    # ------------------------------------------------------------------
    # Overridden EM
    # ------------------------------------------------------------------

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
    ):
        """EM for K clusters with augmented BT+BC E-step (fully vectorized)."""
        rng = np.random.RandomState(42)
        self._init_bc_arrays(K, N)

        best_ll = -float('inf')
        best_lp_arr = None
        best_blades = None
        best_chests = None

        for _restart in range(3):
            lp = rng.randn(K, N) * 0.01    # (K, N)
            # Reinitialize BC arrays per restart
            blades = {k: rng.randn(N, self.bc_dim) * 0.1 for k in range(K)}
            chests = {k: rng.randn(N, self.bc_dim) * 0.1 for k in range(K)}

            for _iter in range(self.n_em_iter):
                # ---- E-step: BT + BC log-likelihoods -------------------
                bt_diff = lp[:, i_arr] - lp[:, j_arr]    # (K, M)

                log_r = np.zeros((K, M))
                for k in range(K):
                    B, C = blades[k], chests[k]
                    bc_logit = (B[i_arr] * C[j_arr]).sum(axis=1) - \
                               (B[j_arr] * C[i_arr]).sum(axis=1)   # (M,)
                    combined = bt_diff[k] + bc_logit
                    log_r[k] = np.where(
                        won_arr,
                        _log_sigmoid_vec(combined),
                        _log_sigmoid_vec(-combined),
                    )
                responsibilities = _softmax_rows(log_r.T)   # (M, K)

                # ---- M-step BT Newton (vectorized) ---------------------
                for k in range(K):
                    r_k = responsibilities[:, k]             # (M,)
                    diff_k = lp[k, i_arr] - lp[k, j_arr]   # (M,)
                    p = 1.0 / (1.0 + np.exp(-np.clip(diff_k, -500.0, 500.0)))

                    for _newton in range(self.n_newton_iter):
                        gi = np.where(won_arr, r_k * (1.0 - p), -r_k * p)
                        grad = np.zeros(N)
                        np.add.at(grad, i_arr, gi)
                        np.add.at(grad, j_arr, -gi)

                        h = r_k * p * (1.0 - p)
                        hess = np.full(N, -self.lambda_reg)
                        np.add.at(hess, i_arr, -h)
                        np.add.at(hess, j_arr, -h)

                        safe_hess = np.where(np.abs(hess) > 1e-8, hess, -1e-8)
                        lp[k] += np.clip(-grad / safe_hess, -0.5, 0.5)

                        diff_k = lp[k, i_arr] - lp[k, j_arr]
                        p = 1.0 / (1.0 + np.exp(-np.clip(diff_k, -500.0, 500.0)))

                # ---- M-step BC gradient (vectorized per cluster) -------
                for k in range(K):
                    r_k = responsibilities[:, k]   # (M,)
                    # Skip matchups with negligible responsibility
                    active = r_k > 0.05
                    if not active.any():
                        continue

                    B, C = blades[k], chests[k]
                    i_a = i_arr[active]
                    j_a = j_arr[active]
                    won_a = won_arr[active]
                    r_a = r_k[active]

                    bc_logit = ((B[i_a] * C[j_a]).sum(axis=1) -
                                (B[j_a] * C[i_a]).sum(axis=1))   # (M_a,)
                    # For "i_won" matchups the signed score is bc_logit; for losses, -bc_logit
                    signed = np.where(won_a, bc_logit, -bc_logit)
                    p_bc = 1.0 / (1.0 + np.exp(-np.clip(signed, -500.0, 500.0)))  # (M_a,)
                    omp = 1.0 - p_bc   # (M_a,)

                    step = (self.bc_lr * r_a)[:, np.newaxis]   # (M_a, 1)

                    # Gradient for winner side (i when won, j when lost)
                    # d blade_winner / d LL = omp * chest_loser
                    # d chest_loser  / d LL = omp * blade_winner
                    # Gradient for loser side
                    # d blade_loser  / d LL = -p * chest_winner
                    # d chest_winner / d LL = -p * blade_loser
                    win_idx = np.where(won_a, i_a, j_a)
                    los_idx = np.where(won_a, j_a, i_a)

                    dB_win = step * omp[:, np.newaxis] * C[los_idx]   # (M_a, D)
                    dC_los = step * omp[:, np.newaxis] * B[win_idx]
                    dB_los = -step * p_bc[:, np.newaxis] * C[win_idx]
                    dC_win = -step * p_bc[:, np.newaxis] * B[los_idx]

                    np.add.at(B, win_idx, dB_win)
                    np.add.at(C, los_idx, dC_los)
                    np.add.at(B, los_idx, dB_los)
                    np.add.at(C, win_idx, dC_win)

                    # Clip norms to <= 5
                    norms_B = np.linalg.norm(B, axis=1, keepdims=True)
                    norms_C = np.linalg.norm(C, axis=1, keepdims=True)
                    B[:] = np.where(norms_B > 5.0, B * (5.0 / norms_B), B)
                    C[:] = np.where(norms_C > 5.0, C * (5.0 / norms_C), C)
                    blades[k] = B
                    chests[k] = C

            ll = self._compute_ll_augmented(K, lp, blades, chests, i_arr, j_arr, won_arr)
            if ll > best_ll:
                best_ll = ll
                best_lp_arr = lp.copy()
                best_blades = {k: blades[k].copy() for k in range(K)}
                best_chests = {k: chests[k].copy() for k in range(K)}

        if best_lp_arr is None:
            return {}, None, -float('inf')

        # Store best BC arrays (array form for speed, dict form for predict/get_rating)
        self._blades_arr = best_blades
        self._chests_arr = best_chests
        self._blades = {
            k: {entities[n]: best_blades[k][n] for n in range(N)}
            for k in range(K)
        }
        self._chests = {
            k: {entities[n]: best_chests[k][n] for n in range(N)}
            for k in range(K)
        }

        # Final E-step
        log_r_final = np.zeros((K, M))
        for k in range(K):
            B, C = best_blades[k], best_chests[k]
            bc_logit = ((B[i_arr] * C[j_arr]).sum(axis=1) -
                        (B[j_arr] * C[i_arr]).sum(axis=1))
            combined = (best_lp_arr[k, i_arr] - best_lp_arr[k, j_arr]) + bc_logit
            log_r_final[k] = np.where(
                won_arr,
                _log_sigmoid_vec(combined),
                _log_sigmoid_vec(-combined),
            )
        final_resp = _softmax_rows(log_r_final.T)  # (M, K)

        log_pi = {
            k: {entities[n]: float(best_lp_arr[k, n]) for n in range(N)}
            for k in range(K)
        }
        return log_pi, list(final_resp), best_ll

    def _compute_ll_augmented(
        self,
        K: int,
        lp: np.ndarray,
        blades: Dict[int, np.ndarray],
        chests: Dict[int, np.ndarray],
        i_arr: np.ndarray,
        j_arr: np.ndarray,
        won_arr: np.ndarray,
    ) -> float:
        """Total LL using BT+BC likelihoods (vectorized)."""
        lik_k = np.zeros((K, len(i_arr)))
        for k in range(K):
            B, C = blades[k], chests[k]
            bt_diff = np.clip(lp[k, i_arr] - lp[k, j_arr], -500.0, 500.0)
            bc_logit = np.clip(
                (B[i_arr] * C[j_arr]).sum(axis=1) - (B[j_arr] * C[i_arr]).sum(axis=1),
                -500.0, 500.0,
            )
            bt_lik = np.where(won_arr,
                              1.0 / (1.0 + np.exp(-bt_diff)),
                              1.0 / (1.0 + np.exp(bt_diff)))
            bc_lik = np.where(won_arr,
                              1.0 / (1.0 + np.exp(-bc_logit)),
                              1.0 / (1.0 + np.exp(bc_logit)))
            lik_k[k] = bt_lik * bc_lik

        lik_mix = lik_k.mean(axis=0)
        return float(np.sum(np.log(np.maximum(lik_mix, 1e-300))))

    # ------------------------------------------------------------------
    # Prediction override: adds BC component
    # ------------------------------------------------------------------

    def predict_outcome(
        self,
        objective: str,
        agent_id_a: str,
        policy_id_a: str,
        agent_id_b: str,
        policy_id_b: str,
    ) -> float:
        """Predict P(a > b) using mixture of BT+BC models."""
        key_a = self._entity_key(agent_id_a, policy_id_a)
        key_b = self._entity_key(agent_id_b, policy_id_b)

        if not self._log_pi:
            return 0.5

        score = 0.0
        for k in range(self._best_k):
            pi_k = self._cluster_weights[k] if k < len(self._cluster_weights) else 0.0
            lp = self._log_pi.get(k, {})
            la = lp.get(key_a, 0.0)
            lb = lp.get(key_b, 0.0)
            bt_logit = la - lb
            bc_logit = self._bc_logit(k, key_a, key_b)
            score += pi_k * (bt_logit + bc_logit)

        x = float(np.clip(score, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-x))
