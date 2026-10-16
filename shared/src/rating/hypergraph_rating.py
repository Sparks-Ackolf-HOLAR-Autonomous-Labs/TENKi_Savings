"""
Hypergraph Rating Model - Multiway Comparison Events as First-Class Objects

Motivation (PEGKi hypergraph extension):
  Every rating system in this package consumes PAIRWISE comparisons: a round
  in which k policies were evaluated against the same target is shredded into
  C(k,2) independent edges of a comparison graph before any model sees it.
  That decomposition throws away the joint context (the C(k,2) edges from one
  round are treated as independent evidence although they share one noise
  realisation) and makes cross-channel pooling awkward (edges never span
  channels).

  Here the round itself is the atomic observation: a HYPEREDGE connecting all
  k participants, carrying their observed ranking. Two models are provided:

  1. PlackettLuceModel
       Plackett-Luce over ranking hyperedges, fitted with the MM algorithm
       (Hunter 2004, "MM algorithms for generalized Bradley-Terry models",
       Ann. Statist. 32(1), doi:10.1214/aos/1079120141; DOI-verified via
       Crossref 2026-09-25).
       For hyperedges of size 2 this is EXACTLY the Bradley-Terry model, so
       the pairwise comparison graph is recovered as the 2-uniform special
       case -- the extension is strict.

  2. PooledPlackettLuce
       Factorised Luce model over (channel, policy) entities:
           w[channel, policy] = exp(theta[policy]) * exp(delta[channel])
       fitted by block-coordinate MM. Hyperedges may span channels, which
       yields ONE combined policy ranking across all channels plus per-channel
       offsets -- the "combine all ratings together" perspective.

  Physics-informed relative priors:
    PlackettLuceModel.fit accepts prior_log_strengths: a RELATIVE prior over
    entity strengths (only differences matter) coming from outside the fitted
    data -- textbook physics orderings, or the fitted ranking of a different
    observation channel/system. It enters as an anchored MAP term
        w_i <- (W_i + lambda) / (D_i + lambda / pi_i),   pi_i = exp(theta_prior_i)
    the rank-data analogue of a PINN's physics-residual loss term: with no
    data the fit returns the prior; as evidence accumulates the likelihood
    dominates; lambda (prior_weight) is the trust knob. AdaptivePriorSelector
    LEARNS lambda from the event stream by prequential scoring, so a
    misspecified physics prior is discounted automatically -- the
    "adjusted/learned physics" part. With a flat prior (all-equal
    theta_prior) the update reduces exactly to the unanchored model.
    Theory and experiment map: shared/docs/PHYSICS_INFORMED_RANK_PRIORS.md.

  Diagnostics quantify what the hypergraph view adds:
    - per-hyperedge rank residual: Kendall tau between the observed ranking
      of an edge and the fitted consensus ordering restricted to that edge
      (the hypergraph analogue of the cyclic component);
    - clique-expansion Hodge comparison: the same edges shredded into pairs,
      decomposed with the identical lstsq machinery as
      statistical_rating_system.py, so pairwise cyclic energy can be compared
      against the hypergraph residual on the SAME underlying data.

Conventions:
  - Rankings are tuples ordered best -> worst.
  - Ties: exact ties inside one event carry no ordering signal; tied members
    are dropped from that event (mirrors the draw-drop rule of the pairwise
    builder in scripts/train_ratings_from_data.py). If fewer than 2 distinct
    members remain the event is skipped.
  - Any fitted ordering breaks ties by (value, name) -- the single canonical
    tie-break rule of this package.
  - ASCII only; deterministic given inputs (the MM algorithm has no RNG).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .trueskill_engine import Rating


# ------------------------------------------------------------------
# Hyperedge
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Hyperedge:
    """One multiway comparison event.

    ranking: entity keys ordered best -> worst (already tie-resolved).
    weight:  event weight (default 1.0).
    channel: observation channel the event came from (None = unspecified).
    """
    ranking: Tuple[str, ...]
    weight: float = 1.0
    channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if len(self.ranking) < 2:
            raise ValueError("Hyperedge needs at least 2 participants")
        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("Hyperedge participants must be distinct")
        if self.weight <= 0:
            raise ValueError("Hyperedge weight must be positive")


def hyperedge_from_scores(
    scores: Dict[str, float],
    higher_is_better: bool = True,
    weight: float = 1.0,
    channel: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Hyperedge]:
    """Build a Hyperedge from {entity: score} for one shared evaluation event.

    Exact ties are dropped (no ordering signal), mirroring the draw-drop rule
    of the pairwise builder. Returns None if fewer than 2 distinct scores
    remain.
    """
    # Drop all members that participate in an exact tie
    by_value: Dict[float, List[str]] = {}
    for name, val in scores.items():
        by_value.setdefault(float(val), []).append(name)
    untied = {n: v for v, names in by_value.items() for n in names if len(names) == 1}
    if len(untied) < 2:
        return None
    sign = -1.0 if higher_is_better else 1.0
    # (value, name): canonical tie-break rule (ties are gone, name is a
    # deterministic secondary key regardless)
    ordered = tuple(sorted(untied, key=lambda n: (untied[n] * sign, n)))
    return Hyperedge(
        ranking=ordered,
        weight=weight,
        channel=channel,
        metadata=metadata or {},
    )


# ------------------------------------------------------------------
# Plackett-Luce over hyperedges (MM algorithm)
# ------------------------------------------------------------------

class PlackettLuceModel:
    """Plackett-Luce strengths from ranking hyperedges via MM iteration.

    P(observed ranking r_1 > r_2 > ... > r_k) =
        prod_s  w[r_s] / (w[r_s] + w[r_{s+1}] + ... + w[r_k])

    MM update (Hunter 2004):
        W_i = total weight of stages won by i
        D_i = sum over stages whose remaining set contains i of
              weight / sum_{j in remaining set} w_j
        w_i <- (W_i + prior) / (D_i + prior / w_ref)

    The prior term (default 0.01) is a light MAP damping that keeps entities
    with sparse or disconnected appearances finite (fixed point near w_ref=1).
    For 2-entity edges the fixed point of the undamped update is
    w_a / w_b = wins_a / wins_b: exactly Bradley-Terry.
    """

    def __init__(self, prior_weight: float = 0.01):
        self.prior_weight = float(prior_weight)
        self.log_strengths: Dict[str, float] = {}
        self.std_errors: Dict[str, float] = {}
        self.appearances: Dict[str, int] = {}
        self.n_iterations: int = 0
        self.converged: bool = False
        self.prior_anchor: Optional[Dict[str, float]] = None
        self._edges: List[Hyperedge] = []

    # -- fitting -----------------------------------------------------

    def fit(
        self,
        hyperedges: Iterable[Hyperedge],
        max_iter: int = 1000,
        tol: float = 1e-9,
        prior_log_strengths: Optional[Dict[str, float]] = None,
    ) -> "PlackettLuceModel":
        """Fit strengths; optionally anchored to a relative prior.

        prior_log_strengths: {entity: theta_prior} from OUTSIDE the fitted
        data (physics knowledge, or another channel's fit). Only differences
        matter: the prior is mean-centred over the fitted entity set (gauge:
        geometric mean 1); entities absent from the dict anchor at 0 before
        centring. prior_weight is the trust lambda. When given, the fit
        initialises AT the prior (warm start) and skips the per-iteration
        gauge renormalisation -- the anchor pins the gauge instead.
        """
        edges = list(hyperedges)
        if not edges:
            raise ValueError("No hyperedges to fit")
        self._edges = edges

        names = sorted({n for e in edges for n in e.ranking})
        idx = {n: i for i, n in enumerate(names)}
        N = len(names)
        anchored = prior_log_strengths is not None
        if anchored:
            raw = np.array(
                [float(prior_log_strengths.get(n, 0.0)) for n in names],
                dtype=float,
            )
            raw = raw - raw.mean()
            pi = np.exp(raw)
            w = pi.copy()
            self.prior_anchor = {n: float(raw[idx[n]]) for n in names}
        else:
            pi = np.ones(N, dtype=float)
            w = np.ones(N, dtype=float)
            self.prior_anchor = None

        appearances = np.zeros(N, dtype=int)
        # Precompute stage structure: for each edge, list of (winner, members)
        stages: List[Tuple[int, np.ndarray, float]] = []
        for e in edges:
            ids = [idx[n] for n in e.ranking]
            for n in e.ranking:
                appearances[idx[n]] += 1
            # Last stage (single remaining entity) is deterministic: skip it
            for s in range(len(ids) - 1):
                members = np.array(ids[s:], dtype=int)
                stages.append((ids[s], members, e.weight))

        wins = np.zeros(N, dtype=float)
        for winner, _members, wt in stages:
            wins[winner] += wt

        prior = self.prior_weight
        self.converged = False
        it = 0
        for it in range(1, max_iter + 1):
            denom = np.zeros(N, dtype=float)
            for _winner, members, wt in stages:
                total = w[members].sum()
                if total <= 0:
                    continue
                denom[members] += wt / total
            w_new = (wins + prior) / (denom + prior / pi)  # anchored at pi
            if not anchored:
                # Gauge fix: geometric mean 1 (anchored fits are gauge-pinned
                # by the prior itself)
                w_new = w_new / np.exp(np.mean(np.log(np.maximum(w_new, 1e-300))))
            delta = float(np.max(np.abs(np.log(w_new) - np.log(np.maximum(w, 1e-300)))))
            w = w_new
            if delta < tol:
                self.converged = True
                break
        self.n_iterations = it

        theta = np.log(w)
        # Diagonal observed information for log-strengths:
        #   I_ii = sum over stages containing i of  weight * p_i * (1 - p_i)
        info = np.zeros(N, dtype=float)
        for _winner, members, wt in stages:
            p = w[members] / w[members].sum()
            info[members] += wt * p * (1.0 - p)
        se = 1.0 / np.sqrt(info + prior)

        self.log_strengths = {n: float(theta[idx[n]]) for n in names}
        self.std_errors = {n: float(se[idx[n]]) for n in names}
        self.appearances = {n: int(appearances[idx[n]]) for n in names}
        return self

    # -- inspection --------------------------------------------------

    def ranking(self) -> List[str]:
        """Entities ordered best -> worst; ties broken by (value, name)."""
        return sorted(self.log_strengths, key=lambda n: (-self.log_strengths[n], n))

    def log_likelihood(self, hyperedges: Optional[Iterable[Hyperedge]] = None) -> float:
        edges = list(hyperedges) if hyperedges is not None else self._edges
        ll = 0.0
        for e in edges:
            w = np.array([np.exp(self.log_strengths.get(n, 0.0)) for n in e.ranking])
            for s in range(len(w) - 1):
                ll += e.weight * float(np.log(w[s] / w[s:].sum()))
        return ll

    def predict_top(self, subset: Sequence[str]) -> Dict[str, float]:
        """P(entity ranks first among subset) under the fitted model."""
        w = np.array([np.exp(self.log_strengths.get(n, 0.0)) for n in subset])
        p = w / w.sum()
        return {n: float(p[i]) for i, n in enumerate(subset)}

    def get_rating(self, entity_id: str, rating_type: str = "policy") -> Rating:
        """TrueSkill-scale Rating (same convention as FFGHodgeModel.get_rating)."""
        if entity_id not in self.log_strengths:
            return Rating(mu=1200.0, sigma=200.0)
        vals = np.array(list(self.log_strengths.values()))
        norm = max(float(np.max(np.abs(vals))) if vals.size else 1.0, 1.0)
        mu = 1200.0 + self.log_strengths[entity_id] * 200.0 / norm
        sigma = self.std_errors[entity_id] * 200.0 / norm + 50.0
        return Rating(mu=mu, sigma=sigma)


# ------------------------------------------------------------------
# Adaptive trust in a relative (physics / donor-channel) prior
# ------------------------------------------------------------------

class AdaptivePriorSelector:
    """Learn how much to trust a relative prior from the event stream.

    The PINN analogy made adaptive: instead of fixing the physics-term weight,
    lambda is selected by PREQUENTIAL scoring (Dawid 1984 sense): each incoming
    event is scored under the model fitted to all PREVIOUS events at each
    candidate lambda, and the cumulative predictive log-score picks the trust
    level. A well-specified prior wins at high lambda (it predicts unseen
    events); a misspecified prior loses its score advantage and lambda decays
    toward the uninformed end of the grid. Before any event arrives the
    predictive model IS the prior.

    Deterministic given the event order; ties in score break toward the
    SMALLER lambda (conservative: trust physics only when it pays).

    refresh_every: refit the per-lambda scoring models only every m absorbed
    events (batched prequential scoring). m=1 is the fully online scheme;
    larger m scores each event against a model that lags at most m-1 events
    behind -- still strictly predictive (a model never sees the event it
    scores), at 1/m the fitting cost.
    """

    def __init__(
        self,
        prior_log_strengths: Dict[str, float],
        lambda_grid: Sequence[float] = (0.01, 0.5, 2.0, 8.0, 32.0),
        refresh_every: int = 1,
    ):
        self.prior = dict(prior_log_strengths)
        self.grid = tuple(float(l) for l in lambda_grid)
        if not self.grid:
            raise ValueError("lambda_grid must be non-empty")
        if refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        self.refresh_every = int(refresh_every)
        self.scores: Dict[float, float] = {l: 0.0 for l in self.grid}
        self.edges: List[Hyperedge] = []
        self._cached: Dict[float, PlackettLuceModel] = {}
        self._since_refresh = 0

    def _fit_at(self, lam: float) -> PlackettLuceModel:
        model = PlackettLuceModel(prior_weight=lam)
        if self.edges:
            model.fit(self.edges, prior_log_strengths=self.prior)
        else:
            # No data yet: the predictive model is the (centred) prior
            mean_ = float(np.mean(list(self.prior.values()))) if self.prior else 0.0
            model.log_strengths = {n: v - mean_ for n, v in self.prior.items()}
            model.prior_anchor = dict(model.log_strengths)
        return model

    def observe(self, edge: Hyperedge) -> None:
        """Score `edge` prequentially at every lambda, then absorb it."""
        if not self._cached or self._since_refresh >= self.refresh_every:
            self._cached = {lam: self._fit_at(lam) for lam in self.grid}
            self._since_refresh = 0
        for lam in self.grid:
            self.scores[lam] += self._cached[lam].log_likelihood([edge])
        self.edges.append(edge)
        self._since_refresh += 1

    def best_lambda(self) -> float:
        return max(self.grid, key=lambda l: (self.scores[l], -l))

    def fit_current(self) -> PlackettLuceModel:
        """Model fitted to all absorbed events at the current best lambda."""
        return self._fit_at(self.best_lambda())


# ------------------------------------------------------------------
# Pooled (cross-channel) Plackett-Luce
# ------------------------------------------------------------------

class PooledPlackettLuce:
    """Factorised Luce model over (channel, policy) participants.

        w[(channel, policy)] = exp(theta[policy] + delta[channel])

    Entities in hyperedges are "channel::policy" strings (build them with
    pooled_key()). Fitted by block-coordinate MM: holding channel offsets
    fixed, the policy update is a standard Hunter MM step with per-entity
    multipliers, and vice versa. Channel offsets are gauge-fixed to geometric
    mean 1, so theta is the single combined policy scale.
    """

    SEP = "::"

    @staticmethod
    def pooled_key(channel: str, policy: str) -> str:
        return f"{channel}{PooledPlackettLuce.SEP}{policy}"

    def __init__(self, prior_weight: float = 0.01):
        self.prior_weight = float(prior_weight)
        self.policy_log_strengths: Dict[str, float] = {}
        self.channel_offsets: Dict[str, float] = {}
        self.n_iterations: int = 0
        self.converged: bool = False

    def fit(
        self,
        hyperedges: Iterable[Hyperedge],
        max_iter: int = 2000,
        tol: float = 1e-9,
    ) -> "PooledPlackettLuce":
        edges = list(hyperedges)
        if not edges:
            raise ValueError("No hyperedges to fit")

        pairs: List[Tuple[str, str]] = []
        for e in edges:
            for key in e.ranking:
                if self.SEP not in key:
                    raise ValueError(
                        f"Pooled entities must be 'channel{self.SEP}policy', got {key!r}"
                    )
                ch, pol = key.split(self.SEP, 1)
                pairs.append((ch, pol))
        channels = sorted({c for c, _ in pairs})
        policies = sorted({p for _, p in pairs})
        cidx = {c: i for i, c in enumerate(channels)}
        pidx = {p: i for i, p in enumerate(policies)}

        # Stage structure with (channel_id, policy_id) members
        stages: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        pol_wins = np.zeros(len(policies), dtype=float)
        ch_wins = np.zeros(len(channels), dtype=float)
        for e in edges:
            ids = [(cidx[k.split(self.SEP, 1)[0]], pidx[k.split(self.SEP, 1)[1]])
                   for k in e.ranking]
            for s in range(len(ids) - 1):
                mem = ids[s:]
                stages.append((
                    s,
                    np.array([c for c, _ in mem], dtype=int),
                    np.array([p for _, p in mem], dtype=int),
                    e.weight,
                ))
                pol_wins[ids[s][1]] += e.weight
                ch_wins[ids[s][0]] += e.weight

        prior = self.prior_weight
        wp = np.ones(len(policies), dtype=float)
        wc = np.ones(len(channels), dtype=float)
        self.converged = False
        it = 0
        for it in range(1, max_iter + 1):
            # Policy block
            denom_p = np.zeros(len(policies), dtype=float)
            for _s, mc, mp, wt in stages:
                u = wp[mp] * wc[mc]
                total = u.sum()
                if total <= 0:
                    continue
                np.add.at(denom_p, mp, wt * wc[mc] / total)
            wp_new = (pol_wins + prior) / (denom_p + prior)
            wp_new = wp_new / np.exp(np.mean(np.log(np.maximum(wp_new, 1e-300))))

            # Channel block
            denom_c = np.zeros(len(channels), dtype=float)
            for _s, mc, mp, wt in stages:
                u = wp_new[mp] * wc[mc]
                total = u.sum()
                if total <= 0:
                    continue
                np.add.at(denom_c, mc, wt * wp_new[mp] / total)
            wc_new = (ch_wins + prior) / (denom_c + prior)
            wc_new = wc_new / np.exp(np.mean(np.log(np.maximum(wc_new, 1e-300))))

            delta = max(
                float(np.max(np.abs(np.log(wp_new) - np.log(np.maximum(wp, 1e-300))))),
                float(np.max(np.abs(np.log(wc_new) - np.log(np.maximum(wc, 1e-300))))),
            )
            wp, wc = wp_new, wc_new
            if delta < tol:
                self.converged = True
                break
        self.n_iterations = it

        self.policy_log_strengths = {p: float(np.log(wp[pidx[p]])) for p in policies}
        self.channel_offsets = {c: float(np.log(wc[cidx[c]])) for c in channels}
        return self

    def ranking(self) -> List[str]:
        return sorted(
            self.policy_log_strengths,
            key=lambda p: (-self.policy_log_strengths[p], p),
        )


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def _kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Kendall tau between two total orders over the same items."""
    common = [x for x in order_a if x in set(order_b)]
    n = len(common)
    if n < 2:
        return 1.0
    pos_b = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[common[i]] < pos_b[common[j]]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc)


def hyperedge_residuals(model: PlackettLuceModel, edges: Iterable[Hyperedge]) -> Dict[str, Any]:
    """Per-edge and per-entity agreement between observed and fitted orderings.

    tau_e = Kendall tau between the observed ranking of edge e and the fitted
    consensus restricted to e's participants. residual_e = (1 - tau_e) / 2 in
    [0, 1] is the hypergraph analogue of cyclic inconsistency; entity residual
    is the weighted mean over edges containing the entity.
    """
    fitted = model.ranking()
    per_edge: List[float] = []
    ent_sum: Dict[str, float] = {}
    ent_wt: Dict[str, float] = {}
    for e in edges:
        consensus = [n for n in fitted if n in set(e.ranking)]
        tau = _kendall_tau(list(e.ranking), consensus)
        res = (1.0 - tau) / 2.0
        per_edge.append(res)
        for n in e.ranking:
            ent_sum[n] = ent_sum.get(n, 0.0) + e.weight * res
            ent_wt[n] = ent_wt.get(n, 0.0) + e.weight
    entity_residual = {
        n: (ent_sum[n] / ent_wt[n]) if ent_wt[n] > 0 else 0.0 for n in ent_sum
    }
    return {
        "mean_edge_residual": float(np.mean(per_edge)) if per_edge else 0.0,
        "edge_residuals": per_edge,
        "entity_residuals": entity_residual,
    }


def clique_expansion_hodge(edges: Iterable[Hyperedge]) -> Dict[str, Any]:
    """Shred hyperedges into pairs and Hodge-decompose the win-fraction matrix.

    Identical machinery to StatisticalRatingSystem._recompute_hodge (incidence
    lstsq), applied to the clique expansion of the SAME events, so the
    pairwise cyclic-energy fraction ||C||^2 / ||A||^2 is directly comparable
    with the hypergraph residual from hyperedge_residuals().
    """
    wins: Dict[str, Dict[str, float]] = {}
    for e in edges:
        r = e.ranking
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                wins.setdefault(r[i], {}).setdefault(r[j], 0.0)
                wins[r[i]][r[j]] += e.weight
    names = sorted({n for a, d in wins.items() for n in [a, *d.keys()]})
    N = len(names)
    if N < 2:
        return {"cyclic_energy_fraction": 0.0, "transitive": {}, "ranking": names}
    idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((N, N), dtype=float)
    for a, d in wins.items():
        for b, w_ab in d.items():
            w_ba = wins.get(b, {}).get(a, 0.0)
            total = w_ab + w_ba
            if total > 0:
                A[idx[a], idx[b]] = w_ab / total - 0.5
                A[idx[b], idx[a]] = w_ba / total - 0.5
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    D = np.zeros((len(pairs), N), dtype=float)
    b_vec = np.zeros(len(pairs), dtype=float)
    for e_idx, (i, j) in enumerate(pairs):
        D[e_idx, i] = 1.0
        D[e_idx, j] = -1.0
        b_vec[e_idx] = A[i, j]
    f, _, _, _ = np.linalg.lstsq(D, b_vec, rcond=None)
    f -= f.mean()
    resid = b_vec - D @ f
    total_energy = float(np.sum(b_vec ** 2))
    cyclic_energy = float(np.sum(resid ** 2))
    transitive = {names[i]: float(f[i]) for i in range(N)}
    order = sorted(names, key=lambda n: (-transitive[n], n))
    return {
        "cyclic_energy_fraction": (cyclic_energy / total_energy) if total_energy > 0 else 0.0,
        "transitive": transitive,
        "ranking": order,
    }


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

def _self_test() -> int:
    print("hypergraph_rating self-test")
    rng = np.random.default_rng(42)
    failures = 0

    # 1. Synthetic recovery: 6 entities, known strengths, 400 3-way events
    true_theta = {f"p{i}": 1.5 - 0.6 * i for i in range(6)}
    names = list(true_theta)
    edges = []
    for _ in range(400):
        trio = list(rng.choice(names, size=3, replace=False))
        gumbel = {n: true_theta[n] + rng.gumbel() for n in trio}
        ranking = tuple(sorted(trio, key=lambda n: (-gumbel[n], n)))
        edges.append(Hyperedge(ranking=ranking))
    model = PlackettLuceModel().fit(edges)
    tau = _kendall_tau(model.ranking(), sorted(names, key=lambda n: -true_theta[n]))
    print(f"  [1] synthetic 3-way recovery tau = {tau:.3f} "
          f"(iters={model.n_iterations}, converged={model.converged})")
    if tau < 0.85:
        failures += 1
        print("  [1] FAIL: tau below 0.85")

    # 2. Bradley-Terry reduction: 2 entities, 30 vs 10 wins
    bt_edges = [Hyperedge(ranking=("a", "b"))] * 30 + [Hyperedge(ranking=("b", "a"))] * 10
    bt = PlackettLuceModel(prior_weight=1e-9).fit(bt_edges)
    ratio = np.exp(bt.log_strengths["a"] - bt.log_strengths["b"])
    print(f"  [2] BT reduction: w_a/w_b = {ratio:.4f} (expect 3.0)")
    if abs(ratio - 3.0) > 0.01:
        failures += 1
        print("  [2] FAIL: expected 3.0")

    # 3. Pairwise shred vs hyperedge fit agree on transitive synthetic data
    pair_edges = []
    for e in edges:
        r = e.ranking
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                pair_edges.append(Hyperedge(ranking=(r[i], r[j])))
    pair_model = PlackettLuceModel().fit(pair_edges)
    tau_hp = _kendall_tau(model.ranking(), pair_model.ranking())
    print(f"  [3] hyperedge vs shredded-pairs ranking tau = {tau_hp:.3f}")
    if tau_hp < 0.85:
        failures += 1
        print("  [3] FAIL")

    # 4. Pooled fit: two channels, one with +0.8 offset, same policy order
    pooled_edges = []
    for _ in range(300):
        ch = "chan_a" if rng.random() < 0.5 else "chan_b"
        off = 0.0 if ch == "chan_a" else 0.8
        trio = list(rng.choice(names, size=3, replace=False))
        gumbel = {n: true_theta[n] + off + rng.gumbel() for n in trio}
        ranking = tuple(
            PooledPlackettLuce.pooled_key(ch, n)
            for n in sorted(trio, key=lambda n: (-gumbel[n], n))
        )
        pooled_edges.append(Hyperedge(ranking=ranking, channel=ch))
    # Cross-channel bridge events so offsets are identified
    for _ in range(200):
        duo = list(rng.choice(names, size=2, replace=False))
        chs = ["chan_a", "chan_b"]
        rng.shuffle(chs)
        gumbel = {
            (c, n): true_theta[n] + (0.8 if c == "chan_b" else 0.0) + rng.gumbel()
            for c, n in zip(chs, duo)
        }
        ranking = tuple(
            PooledPlackettLuce.pooled_key(c, n)
            for c, n in sorted(gumbel, key=lambda k: (-gumbel[k], k[1]))
        )
        pooled_edges.append(Hyperedge(ranking=ranking))
    pooled = PooledPlackettLuce().fit(pooled_edges)
    tau_pool = _kendall_tau(pooled.ranking(), sorted(names, key=lambda n: -true_theta[n]))
    off_gap = pooled.channel_offsets["chan_b"] - pooled.channel_offsets["chan_a"]
    print(f"  [4] pooled recovery tau = {tau_pool:.3f}, "
          f"offset(chan_b - chan_a) = {off_gap:.3f} (expect approx +0.8)")
    if tau_pool < 0.85 or off_gap < 0.2:
        failures += 1
        print("  [4] FAIL")

    # 5. Diagnostics run and determinism
    diag = hyperedge_residuals(model, edges)
    hodge = clique_expansion_hodge(edges)
    model2 = PlackettLuceModel().fit(edges)
    same = all(
        abs(model.log_strengths[n] - model2.log_strengths[n]) < 1e-12 for n in names
    )
    print(f"  [5] mean edge residual = {diag['mean_edge_residual']:.4f}, "
          f"pairwise cyclic energy fraction = {hodge['cyclic_energy_fraction']:.4f}, "
          f"deterministic refit = {same}")
    if not same:
        failures += 1
        print("  [5] FAIL: refit not deterministic")

    # 6. Anchored prior: prior rules with sparse data, data rules with plenty
    prior = {"a": 1.0, "b": 0.0, "c": -1.0}          # prior says a > b > c
    contra = [Hyperedge(ranking=("c", "b", "a"))]     # truth is c > b > a
    sparse = PlackettLuceModel(prior_weight=8.0).fit(contra * 2,
                                                     prior_log_strengths=prior)
    plenty = PlackettLuceModel(prior_weight=0.5).fit(contra * 200,
                                                     prior_log_strengths=prior)
    ok6 = (sparse.ranking() == ["a", "b", "c"]
           and plenty.ranking() == ["c", "b", "a"])
    print(f"  [6] anchored prior: sparse-data ranking {sparse.ranking()}, "
          f"plenty-data ranking {plenty.ranking()}")
    if not ok6:
        failures += 1
        print("  [6] FAIL: prior/data dominance regimes wrong")

    # 7. Flat-prior reduction: anchored fit with all-zero prior matches the
    #    unanchored fit (centred log-strengths)
    flat = PlackettLuceModel().fit(edges,
                                   prior_log_strengths={n: 0.0 for n in names})
    base_c = np.array([model.log_strengths[n] for n in names])
    base_c -= base_c.mean()
    flat_c = np.array([flat.log_strengths[n] for n in names])
    flat_c -= flat_c.mean()
    max_dev = float(np.max(np.abs(base_c - flat_c)))
    ok7 = max_dev < 1e-3 and flat.ranking() == model.ranking()
    print(f"  [7] flat-prior reduction: max centred-theta deviation = {max_dev:.2e}, "
          f"same ranking = {flat.ranking() == model.ranking()}")
    if not ok7:
        failures += 1
        print("  [7] FAIL: flat anchor does not reduce to unanchored fit")

    # 8. Adaptive selector: trusts a correct prior, discounts a reversed one
    good = AdaptivePriorSelector(dict(true_theta))
    bad = AdaptivePriorSelector({n: -v for n, v in true_theta.items()})
    rng8 = np.random.default_rng(7)
    for _ in range(60):
        trio = list(rng8.choice(names, size=3, replace=False))
        gumbel = {n: true_theta[n] + rng8.gumbel() for n in trio}
        e = Hyperedge(ranking=tuple(sorted(trio, key=lambda n: (-gumbel[n], n))))
        good.observe(e)
        bad.observe(e)
    lam_good, lam_bad = good.best_lambda(), bad.best_lambda()
    ok8 = lam_good >= 2.0 and lam_bad <= 0.5
    print(f"  [8] adaptive trust: lambda(correct prior) = {lam_good}, "
          f"lambda(reversed prior) = {lam_bad}")
    if not ok8:
        failures += 1
        print("  [8] FAIL: expected high trust for correct, low for reversed")

    print(f"self-test {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
    return failures


if __name__ == "__main__":
    raise SystemExit(_self_test())
