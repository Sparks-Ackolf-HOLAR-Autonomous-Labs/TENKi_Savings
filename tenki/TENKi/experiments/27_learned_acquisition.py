"""
Experiment 27 -- Learned hyperedge acquisition: rule vs trained policy.

Addresses principled active hyperedge acquisition and the question
whether hyperedge acquisition is just a rule or can be TRAINED.

Exp 24 established that k-way ranking events beat pairwise queries at
matched budget (primary PASS on all 8 band-gap data sources), but its
hand-written epsilon-greedy minimal-spread rule never accelerated
learning over random k-way selection (secondary FAIL in every run:
20 seeds, and 50 seeds at k=3/4/6). This
experiment replaces the hand-written rule with

  (a) a PRINCIPLED UNTRAINED rule (kway_fisher): greedy subset
      construction maximising added expected Fisher information of the
      Plackett-Luce fit -- seed the subset with the entity of largest
      strength std-error, then repeatedly add the entity j maximising
      sum_{i in S} p_ij (1 - p_ij) (the observed-information kernel of
      Bradley-Terry; p from current fitted strengths); and

  (b) a TRAINED policy (kway_learned): a linear scoring function over
      per-entity features, trained by the cross-entropy method (CEM)
      and evaluated LEAVE-ONE-CHANNEL-OUT (LOCO) so the reported curves
      are always on a channel the policy never trained on.

The policy class deliberately CONTAINS the earlier baselines:
  features per entity (z-scored across entities at selection time)
    f_app   = -appearances(i)          (coverage: favour under-sampled)
    f_se    = std_error(theta_i)       (uncertainty: favour unresolved)
    f_gap   = -min |theta_i - theta_neighbour| in fitted order
                                       (closeness: the exp-24 min-spread rule)
    f_noise = standard normal draw     (exploration)
  score_i = w . f;  query the top-k entities by score.
  w = (0,0,1,0) is a min-spread rule, w = (1,0,0,0) is round-robin
  coverage, w = (0,0,0,1) is exactly random. Training searches this
  family, so WHAT the trained weights converge to is itself the answer
  to "what is the right acquisition rule".

Training objective: mean Kendall tau over all budget checkpoints (the
area under the tau-vs-budget curve), which rewards being fast AND good.
Training seeds and channels are disjoint from evaluation seeds and (via
LOCO) the evaluated channel.

Data: identical to exp 24 -- retrospective queries against the frozen
band-gap archive band-gap universe, one real archived record per participating
class per event, no invented observations. Reference ranking per
channel = classes by median gap, widest first.

Success criteria
----------------
  Primary  : LOCO kway_learned final tau >= kway_random final tau
             (training does no long-run harm on unseen channels).
  Secondary: kway_learned reaches 90% of kway_random's FINAL tau at a
             strictly smaller budget than kway_random itself (the speed
             criterion every hand-written rule has so far failed;
             the same shared target makes the comparison fair).
  Reported for kway_fisher and kway_active under the same criteria.

Outputs (default results/learned_acquisition/, override --output-dir)
---------------------------------------------------------------------
  learned_acquisition.json / .md / .png
  fold_<channel>.json   (per-fold checkpoint, written as each fold lands)

Run (from tenki/TENKi; PEGKI_ROOT when shared/ is not the sibling):
  uv run python experiments/27_learned_acquisition.py
Self-tests (policy-class containment + orientation): --self-test.
ASCII only; seeded and reproducible (--seed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pegki_root() -> Path:
    env = os.environ.get("PEGKI_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(_HERE) / ".." / ".." / ".." / "shared").resolve()


sys.path.insert(0, str(_pegki_root()))
from src.rating.hypergraph_rating import (  # noqa: E402
    PlackettLuceModel,
    hyperedge_from_scores,
    _kendall_tau,
)

# Mirrors exp 24 / classify_by_anion exactly.
ANION_PRIORITY = [
    ("halide", {"F", "Cl", "Br", "I"}),
    ("oxide", {"O"}),
    ("nitride", {"N"}),
    ("chalcogenide", {"S", "Se", "Te"}),
    ("carbide", {"C"}),
    ("boride", {"B"}),
    ("pnictide", {"P", "As", "Sb", "Bi"}),
]
MIN_GAP_EV = 0.01
FEATURES = ["coverage", "uncertainty", "closeness", "noise"]


def classify_formula(formula: str) -> str:
    species = set(re.findall(r"[A-Z][a-z]?", formula))
    for name, elems in ANION_PRIORITY:
        if species & elems:
            return name
    return "other"


def load_universe(path: Path, min_class_size: int):
    per_channel: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            formula = (rec.get("x") or {}).get("formula")
            gap = (rec.get("y") or {}).get("bandgap_eV")
            channel = rec.get("s")
            if not formula or gap is None or channel is None:
                continue
            gap = float(gap)
            if gap <= MIN_GAP_EV:
                continue
            cls = classify_formula(formula)
            per_channel.setdefault(channel, {}).setdefault(cls, []).append(gap)
    out = {}
    for channel, classes in per_channel.items():
        kept = {c: np.asarray(sorted(v), dtype=float)
                for c, v in classes.items() if len(v) >= min_class_size}
        if len(kept) >= 4:
            out[channel] = kept
    return out


def reference_ranking(classes: dict) -> list:
    med = {c: float(np.median(v)) for c, v in classes.items()}
    return sorted(med, key=lambda c: (-med[c], c))


def sample_event(classes: dict, subset: list, rng: np.random.Generator):
    scores = {c: float(classes[c][rng.integers(0, len(classes[c]))])
              for c in subset}
    return hyperedge_from_scores(scores, higher_is_better=True)


# ------------------------------------------------------------------
# Selection rules
# ------------------------------------------------------------------

def _zscore(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def entity_features(model: PlackettLuceModel, names: list,
                    counts: dict, rng: np.random.Generator) -> np.ndarray:
    """(n_entities, 4) z-scored feature matrix in `names` order."""
    n = len(names)
    f_app = _zscore(np.array([-float(counts.get(nm, 0)) for nm in names]))
    if model.log_strengths:
        se = np.array([float(model.std_errors.get(nm, 1.0)) for nm in names])
        theta = {nm: float(model.log_strengths.get(nm, 0.0)) for nm in names}
        ordered = sorted(names, key=lambda nm: (-theta[nm], nm))
        pos = {nm: i for i, nm in enumerate(ordered)}
        gap = np.empty(n)
        for i, nm in enumerate(names):
            p = pos[nm]
            g = []
            if p > 0:
                g.append(theta[ordered[p - 1]] - theta[nm])
            if p < n - 1:
                g.append(theta[nm] - theta[ordered[p + 1]])
            gap[i] = -min(g) if g else 0.0
        f_se, f_gap = _zscore(se), _zscore(gap)
    else:
        f_se = np.zeros(n)
        f_gap = np.zeros(n)
    f_noise = rng.standard_normal(n)
    return np.column_stack([f_app, f_se, f_gap, f_noise])


def pick_learned_subset(weights: np.ndarray, model: PlackettLuceModel,
                        names: list, counts: dict, k: int,
                        rng: np.random.Generator) -> list:
    feats = entity_features(model, names, counts, rng)
    scores = feats @ np.asarray(weights, dtype=float)
    order = sorted(range(len(names)), key=lambda i: (-scores[i], names[i]))
    return [names[i] for i in order[:k]]


def pick_active_subset(model: PlackettLuceModel, names: list, k: int,
                       rng: np.random.Generator, eps: float = 0.2) -> list:
    """Exp 24's epsilon-greedy minimal-spread window rule (unchanged)."""
    if rng.random() < eps or not model.log_strengths:
        return list(rng.choice(names, size=k, replace=False))
    theta = {n: float(model.log_strengths.get(n, 0.0)) for n in names}
    ordered = sorted(names, key=lambda n: (-theta[n], n))
    best_i, best_spread = 0, float("inf")
    for i in range(len(ordered) - k + 1):
        window = ordered[i:i + k]
        spread = theta[window[0]] - theta[window[-1]]
        if spread < best_spread:
            best_i, best_spread = i, spread
    return ordered[best_i:best_i + k]


def pick_fisher_subset(model: PlackettLuceModel, names: list, k: int,
                       rng: np.random.Generator, eps: float = 0.1) -> list:
    """Greedy expected-Fisher-information subset (untrained principled rule).

    Seed with the entity of largest strength std-error, then add the
    entity j maximising sum_{i in S} p_ij (1 - p_ij), the diagonal
    observed-information kernel of the Bradley-Terry/PL stage model.
    Epsilon-greedy random otherwise (lock-in guard, eps = 0.1).
    """
    if rng.random() < eps or not model.log_strengths:
        return list(rng.choice(names, size=k, replace=False))
    theta = np.array([float(model.log_strengths.get(n, 0.0)) for n in names])
    se = np.array([float(model.std_errors.get(n, 1.0)) for n in names])
    chosen = [int(np.argmax(se))]
    remaining = [i for i in range(len(names)) if i != chosen[0]]
    while len(chosen) < k:
        best_j, best_gain = None, -1.0
        for j in remaining:
            p = 1.0 / (1.0 + np.exp(-(theta[j] - theta[chosen])))
            gain = float(np.sum(p * (1.0 - p)))
            if gain > best_gain + 1e-15 or (
                    abs(gain - best_gain) <= 1e-15 and best_j is not None
                    and names[j] < names[best_j]):
                best_j, best_gain = j, gain
        chosen.append(best_j)
        remaining.remove(best_j)
    return [names[i] for i in chosen]


# ------------------------------------------------------------------
# Episode
# ------------------------------------------------------------------

def run_arm(classes: dict, arm: str, kway: int, budget: int,
            checkpoints: list, seed: int, weights=None):
    """-> list of tau values, one per checkpoint budget (as exp 24)."""
    rng = np.random.default_rng(seed)
    names = sorted(classes)
    ref = reference_ranking(classes)
    k = 2 if arm == "pair_random" else kway
    counts = {n: 0 for n in names}
    edges, taus, spent, ci = [], [], 0, 0
    model = PlackettLuceModel()
    while spent < budget and ci < len(checkpoints):
        if arm == "kway_active":
            subset = pick_active_subset(model, names, k, rng)
        elif arm == "kway_fisher":
            subset = pick_fisher_subset(model, names, k, rng)
        elif arm == "kway_learned":
            subset = pick_learned_subset(weights, model, names, counts,
                                         k, rng)
        else:
            subset = list(rng.choice(names, size=k, replace=False))
        e = sample_event(classes, subset, rng)
        spent += k
        for n in subset:
            counts[n] += 1
        if e is not None:
            edges.append(e)
        while ci < len(checkpoints) and spent >= checkpoints[ci]:
            if edges:
                model = PlackettLuceModel()
                model.fit(edges)
                taus.append(_kendall_tau(model.ranking(), ref))
            else:
                taus.append(0.0)
            ci += 1
    while len(taus) < len(checkpoints):
        taus.append(taus[-1] if taus else 0.0)
    return taus


# ------------------------------------------------------------------
# CEM training (leave-one-channel-out)
# ------------------------------------------------------------------

_POOL_UNIVERSE = None  # set once per worker by _pool_init


def _pool_init(universe_path: str, min_class_size: int):
    global _POOL_UNIVERSE
    _POOL_UNIVERSE = load_universe(Path(universe_path), min_class_size)


def _episode_auc(task):
    """(weights, channel, seed, kway, budget, step) -> mean tau over ckpts."""
    weights, channel, seed, kway, budget, step = task
    classes = _POOL_UNIVERSE[channel]
    checkpoints = list(range(step, budget + 1, step))
    taus = run_arm(classes, "kway_learned", kway, budget, checkpoints,
                   seed, weights=np.asarray(weights))
    return float(np.mean(taus))


def train_fold(held_out: str, channels: list, args, pool) -> dict:
    """CEM over the 4 feature weights; fitness = mean tau-AUC over
    sampled training channels x training seeds. Returns fold record."""
    train_channels = [c for c in channels if c != held_out]
    # crc32, not hash(): str hash() is salted per interpreter
    # (PYTHONHASHSEED), which made earlier runs irreproducible.
    rng = np.random.default_rng(
        args.seed + zlib.crc32(held_out.encode("ascii")) % 100000)
    dim = len(FEATURES)
    mu, sigma = np.zeros(dim), np.ones(dim)
    n_elite = max(2, args.cem_pop // 4)
    history = []
    for it in range(args.cem_iters):
        cand = mu + sigma * rng.standard_normal((args.cem_pop, dim))
        chans = list(rng.choice(
            train_channels,
            size=min(args.train_channels_per_iter, len(train_channels)),
            replace=False))
        seeds = [args.seed + 500000 + 1000 * it + s
                 for s in range(args.train_seeds)]
        tasks = [(tuple(c), ch, sd, args.kway, args.budget, args.budget_step)
                 for c in cand for ch in chans for sd in seeds]
        aucs = pool.map(_episode_auc, tasks)
        per_cand = np.asarray(aucs).reshape(
            args.cem_pop, len(chans) * len(seeds)).mean(axis=1)
        elite = cand[np.argsort(-per_cand)[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        history.append({"iter": it, "best_auc": float(per_cand.max()),
                        "mean_auc": float(per_cand.mean()),
                        "mu": mu.tolist()})
    return {"held_out": held_out, "weights": mu.tolist(),
            "feature_names": FEATURES, "history": history}


# ------------------------------------------------------------------
# Self-tests
# ------------------------------------------------------------------

def self_test() -> int:
    ok = True
    # Orientation (sign-convention rule: test both orientations).
    hi = hyperedge_from_scores({"a": 3.0, "b": 2.0, "c": 1.0},
                               higher_is_better=True)
    lo = hyperedge_from_scores({"a": 3.0, "b": 2.0, "c": 1.0},
                               higher_is_better=False)
    t = hi.ranking == ("a", "b", "c") and lo.ranking == ("c", "b", "a")
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] orientation")

    # Policy-class containment: coverage weights pick the least-queried
    # entities regardless of model state.
    rng = np.random.default_rng(0)
    names = ["a", "b", "c", "d", "e"]
    counts = {"a": 9, "b": 1, "c": 9, "d": 0, "e": 9}
    sub = pick_learned_subset(np.array([1.0, 0, 0, 0]),
                              PlackettLuceModel(), names, counts, 2, rng)
    t = set(sub) == {"b", "d"}
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] containment: coverage -> "
          f"least-queried {sorted(sub)}")

    # Closeness weights pick the tight pair of a fitted model.
    model = PlackettLuceModel()
    model.log_strengths = {"a": 2.0, "b": 0.55, "c": 0.5, "d": -2.0,
                           "e": -4.0}
    model.std_errors = {n: 1.0 for n in names}
    sub = pick_learned_subset(np.array([0, 0, 1.0, 0]), model, names,
                              {n: 1 for n in names}, 2, rng)
    t = set(sub) == {"b", "c"}
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] containment: closeness -> "
          f"tightest pair {sorted(sub)}")

    # Fisher greedy seeds at max std-error and prefers near-equal adds.
    model.std_errors = {"a": 0.1, "b": 2.0, "c": 0.1, "d": 0.1, "e": 0.1}
    sub = pick_fisher_subset(model, names, 2,
                             np.random.default_rng(1), eps=0.0)
    t = sub[0] == "b" and sub[1] == "c"
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] fisher: seed=max-se, "
          f"add=nearest-strength {sub}")

    # Noise-only weights reproduce a valid random subset.
    sub = pick_learned_subset(np.array([0, 0, 0, 1.0]), model, names,
                              {n: 1 for n in names}, 3, rng)
    t = len(set(sub)) == 3 and set(sub) <= set(names)
    ok &= t
    print(f"[{'PASS' if t else 'FAIL'}] containment: noise -> random "
          f"valid subset")
    print(f"[self-test] {'5/5 PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="TENKi exp 27: learned hyperedge acquisition")
    ap.add_argument("--universe",
                    default="output/bandgap_archive_build/bandgap_universe.jsonl")
    ap.add_argument("--kway", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--budget-step", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--min-class-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cem-pop", type=int, default=20)
    ap.add_argument("--cem-iters", type=int, default=8)
    ap.add_argument("--train-seeds", type=int, default=2)
    ap.add_argument("--train-channels-per-iter", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = min(10, cpu_count)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    universe = _pegki_root() / args.universe
    if not universe.exists():
        print(f"[error] universe not found: {universe}")
        return 1
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(_HERE).parent / "results" / "learned_acquisition")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_channel = load_universe(universe, args.min_class_size)
    channels = sorted(per_channel)
    print(f"[load] {len(channels)} channels")
    if not channels:
        return 1
    checkpoints = list(range(args.budget_step, args.budget + 1,
                             args.budget_step))

    n_workers = args.workers or min(10, os.cpu_count() or 1)
    baseline_arms = ["pair_random", "kway_random", "kway_active",
                     "kway_fisher"]
    arms = baseline_arms + ["kway_learned"]
    curves = {a: {} for a in arms}
    folds = {}

    with Pool(n_workers, initializer=_pool_init,
              initargs=(str(universe), args.min_class_size)) as pool:
        for held_out in channels:
            fold_file = out_dir / f"fold_{held_out}.json"
            if fold_file.exists():
                with open(fold_file, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                print(f"[skip] fold {held_out} already complete")
            else:
                print(f"[train] fold held_out={held_out} "
                      f"(CEM {args.cem_iters}x{args.cem_pop}, "
                      f"{n_workers} workers)")
                rec = train_fold(held_out, channels, args, pool)
                # Evaluate ALL arms on the held-out channel, eval seeds.
                classes = per_channel[held_out]
                w = np.asarray(rec["weights"])
                rec["curves"] = {}
                for arm in arms:
                    rows = [run_arm(classes, arm, args.kway, args.budget,
                                    checkpoints,
                                    seed=args.seed + 1000 * s,
                                    weights=w if arm == "kway_learned"
                                    else None)
                            for s in range(args.n_seeds)]
                    rec["curves"][arm] = np.asarray(rows).mean(
                        axis=0).tolist()
                with open(fold_file, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2)
                print(f"[done] fold {held_out}: weights "
                      + ", ".join(f"{n}={v:+.2f}" for n, v in
                                  zip(FEATURES, rec["weights"])))
            folds[held_out] = rec
            for arm in arms:
                curves[arm][held_out] = np.asarray(rec["curves"][arm])

    # Aggregate over channels (each fold's held-out curve).
    agg = {}
    for arm in arms:
        stack = np.stack([curves[arm][ch] for ch in channels])
        agg[arm] = {"mean": stack.mean(axis=0).tolist(),
                    "std": stack.std(axis=0).tolist()}
    final = {a: agg[a]["mean"][-1] for a in arms}

    def budget_to_reach(arm, target):
        for i, v in enumerate(agg[arm]["mean"]):
            if v >= target:
                return checkpoints[i]
        return checkpoints[-1]

    target = 0.9 * final["kway_random"]
    speed = {a: budget_to_reach(a, target) for a in arms}
    primary_pass = final["kway_learned"] >= final["kway_random"] - 1e-9
    secondary_pass = speed["kway_learned"] < speed["kway_random"]

    weight_matrix = np.array([folds[ch]["weights"] for ch in channels])
    results = {
        "config": {k: v for k, v in vars(args).items() if k != "self_test"},
        "checkpoints": checkpoints,
        "channels": channels,
        "aggregate": agg,
        "final_tau": final,
        "budget_to_90pct_of_random_final": speed,
        "learned_weights_per_fold": {ch: folds[ch]["weights"]
                                     for ch in channels},
        "learned_weights_mean": weight_matrix.mean(axis=0).tolist(),
        "learned_weights_std": weight_matrix.std(axis=0).tolist(),
        "feature_names": FEATURES,
        "criteria": {
            "primary_learned_ge_random_at_final_budget": bool(primary_pass),
            "secondary_learned_faster_to_90pct_of_random_final":
                bool(secondary_pass),
        },
    }
    with open(out_dir / "learned_acquisition.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                  gridspec_kw={"width_ratios": [3, 2]})
    styles = {"pair_random": ("tab:gray", "--"),
              "kway_random": ("tab:blue", "-"),
              "kway_active": ("tab:red", ":"),
              "kway_fisher": ("tab:orange", "-."),
              "kway_learned": ("tab:green", "-")}
    for arm in arms:
        m = np.asarray(agg[arm]["mean"])
        s = np.asarray(agg[arm]["std"])
        c, ls = styles[arm]
        ax.plot(checkpoints, m, ls, color=c, label=arm)
        ax.fill_between(checkpoints, m - s, m + s, color=c, alpha=0.10)
    ax.set_xlabel("cumulative evaluations")
    ax.set_ylabel("Kendall tau vs reference ranking")
    ax.set_title(f"Acquisition arms, leave-one-channel-out "
                 f"(k={args.kway}, {len(channels)} channels, "
                 f"{args.n_seeds} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    x = np.arange(len(FEATURES))
    ax2.bar(x, weight_matrix.mean(axis=0),
            yerr=weight_matrix.std(axis=0), color="tab:green",
            alpha=0.8, capsize=4)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xticks(x, FEATURES, rotation=20)
    ax2.set_ylabel("learned weight (mean +/- std over folds)")
    ax2.set_title("What the trained policy learned")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "learned_acquisition.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Experiment 27: learned hyperedge acquisition (rule vs trained)",
        "",
        f"Channels: {', '.join(channels)}",
        f"k = {args.kway}; budget = {args.budget}; {args.n_seeds} eval "
        f"seeds; CEM {args.cem_iters} iters x {args.cem_pop} pop; "
        f"leave-one-channel-out.",
        "",
        "| arm | final tau | budget to 90% of random-arm final |",
        "|---|---|---|",
    ]
    for arm in arms:
        lines.append(f"| {arm} | {final[arm]:+.4f} | {speed[arm]} |")
    lines += [
        "",
        "Learned weights (mean +/- std over folds): "
        + ", ".join(f"{n} {m:+.2f}+/-{s:.2f}" for n, m, s in zip(
            FEATURES, weight_matrix.mean(axis=0),
            weight_matrix.std(axis=0))),
        "",
        f"Primary (learned >= random at final budget): "
        f"**{'PASS' if primary_pass else 'FAIL'}**",
        f"Secondary (learned faster to 90% of random final: "
        f"{speed['kway_learned']} vs {speed['kway_random']}): "
        f"**{'PASS' if secondary_pass else 'FAIL'}**",
        "",
        "Every curve for kway_learned is on a channel the policy never",
        "trained on (LOCO). Training seeds disjoint from eval seeds.",
        "Real archived records only; no invented observations.",
    ]
    with open(out_dir / "learned_acquisition.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[save] {out_dir}")
    print("final tau: " + ", ".join(f"{a}={final[a]:+.4f}" for a in arms))
    print(f"primary {'PASS' if primary_pass else 'FAIL'}; "
          f"secondary {'PASS' if secondary_pass else 'FAIL'} "
          f"(budget to 90% of random final: learned "
          f"{speed['kway_learned']} vs random {speed['kway_random']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
