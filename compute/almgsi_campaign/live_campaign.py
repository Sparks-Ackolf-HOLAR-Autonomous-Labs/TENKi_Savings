"""On-demand live campaign: every query triggers a calculation on the DGX.

Receiver: MACE-MP-0 MD at 300 K (finite-temperature formation energy per
atom, the most expensive DGX source, about 45 s of GPU time per query).
Nothing is read from the precomputed tables in ``results/mace_md``: the
receiver values live in a separate cache, ``results_live/mace_md``, and a
candidate is computed the first time any arm asks for it. Each arm is charged
the measured wall time of every candidate it evaluates, so arms that share a
candidate are charged as if each had computed it.

Arms (k = 4 candidates per batch, fixed evaluation budget):
* ``surrogate``: active learning on composition only (no donor);
* ``fixed:mace_relax`` and ``learned:mace_relax``: the 0 K MACE ordering as
  the donor, with fixed or learned trust (definitions in campaign.py);
* ``learned:gpumd_nep89_md``: a donor that ranks poorly against the DFT
  levels, to test whether learned trust discounts it during a live run.

The replay counterpart (``--replay results_T/T300``) runs the same arms, seeds
and decision code over a completed table of the same receiver computed
earlier on CHPC; comparing the two checks whether a replay over a completed
table reproduces the decisions of a live run.

Usage: python live_campaign.py [--budget 24] [--k 4]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import campaign as C
import dgx_sources as D

HERE = Path(__file__).resolve().parent
LIVE = HERE / "results_live"
REFS = {"Al": "mp_134_Al", "Mg": "mp_153_Mg", "Si": "mp_149_Si"}


class LiveOracle:
    """Formation energy at the receiver, computed on first request."""

    def __init__(self, dtype: str = "float32"):
        D.RESULTS_DIR = LIVE          # redirect the source cache
        self.calc, self.load_s = D.get_mace(dtype)
        self.dtype = dtype
        self.index = C.load_index()
        self.log = []
        self.ref_e = {el: self._energy(cid)[0] for el, cid in REFS.items()}
        self.setup_cost = sum(self._rec(cid)["wall_s"] for cid in REFS.values())

    def _rec(self, cid: str) -> dict:
        p = D.result_path("mace_md", cid)
        if p.exists():
            return D.read_json(p)
        t0 = time.time()
        rec = D.mace_md_query(cid, self.index[cid]["kind"], self.calc, self.dtype)
        D.write_json(p, rec)
        self.log.append({"cid": cid, "computed_at": t0, "wall_s": rec["wall_s"], "status": rec["status"]})
        print(f"[live] computed {cid} {rec['status']} wall={rec['wall_s']:.1f}s", flush=True)
        return rec

    def _energy(self, cid: str):
        r = self._rec(cid)
        return (r["energy_per_atom"] if r["status"] == "ok" else None), r["wall_s"]

    def query(self, cid: str):
        e, cost = self._energy(cid)
        if e is None:
            return None, cost
        x = C.fractions(cid, None)
        ef = e - sum(xi * self.ref_e[el] for xi, el in zip(x, ("Al", "Mg", "Si")))
        return ef, cost


class ReplayOracle:
    """Same interface as LiveOracle, answered from a completed table of the same
    receiver (MACE-MP-0 MD at 300 K, computed earlier on CHPC for the
    temperature study). Cost = the recorded wall time of that calculation."""

    def __init__(self, live_values: Path):
        rows = json.loads(live_values.read_text(encoding="utf-8"))["mace_md"]
        self.rows = rows
        self.setup_cost = sum(float(rows[c].get("wall_s") or 0.0) for c in REFS.values())
        self.load_s = 0.0
        self.log = []

    def query(self, cid: str):
        r = self.rows.get(cid)
        if r is None:
            return None, 0.0
        cost = float(r.get("wall_s") or 0.0)
        v = r.get("value")
        if r.get("status") != "ok" or v is None:
            return None, cost
        return float(v), cost


def run_live_arm(arm, oracle, donors, pool, X, k, budget, seed):
    rng = np.random.default_rng(seed)
    jitter = {c: rng.random() * 1e-9 for c in pool}
    kind, _, dname = arm.partition(":")
    dval = {c: donors[dname][c]["value"] for c in pool if dname and c in donors[dname]}
    dscore = C.rank_score(dval, pool) if dname else None
    evald, failed, cost, steps = {}, set(), 0.0, []
    while len(evald) + len(failed) < min(budget, len(pool)):
        todo = [c for c in pool if c not in evald and c not in failed]
        sur = C.surrogate_scores(evald, pool, X)
        if dname:
            n = len(evald)
            if kind == "learned":
                t = C.tau(dval, evald, list(evald)) if n >= 3 else float("nan")
                w = 0.5 if not np.isfinite(t) else max(0.0, t) * n / (n + 8)
            else:
                w = 1.0
        else:
            w = 0.0
        s = {c: w * (dscore[c] if dscore else 0.5) + (1 - w) * sur.get(c, 0.5) + jitter[c] for c in todo}
        batch = sorted(todo, key=lambda c: -s[c])[:k]
        for c in batch:
            v, cst = oracle.query(c)
            cost += cst
            if v is None:
                failed.add(c)
            else:
                evald[c] = v
        steps.append({"n": len(evald), "cost_s": cost, "batch": batch, "w": w})
    return {"arm": arm, "k": k, "seed": seed, "evaluated": evald, "failed": sorted(failed), "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", default=None,
                    help="answer from <dir>/live_values.json instead of computing (e.g. results_T/T300)")
    args = ap.parse_args()
    LIVE.mkdir(exist_ok=True)
    tables = C.load_tables()
    donors = {d: tables[d] for d in ("mace_relax", "gpumd_nep89_md") if d in tables}
    idx = C.load_index()
    pool = sorted(c for c, r in idx.items()
                  if r["property"] == "formation_energy" and c in donors["mace_relax"])
    X = {c: C.fractions(c, idx) for c in pool}
    oracle = ReplayOracle(Path(args.replay) / "live_values.json") if args.replay else LiveOracle()
    tag = "replay_campaign" if args.replay else "live_campaign"
    arms = ["surrogate", "fixed:mace_relax", "learned:mace_relax", "learned:gpumd_nep89_md"]
    out = {"receiver": f"mace_md (replay over {args.replay})" if args.replay else "mace_md (computed on demand)", "budget": args.budget, "k": args.k,
           "seed": args.seed, "setup_cost_s": oracle.setup_cost, "model_load_s": oracle.load_s,
           "arms": {}}
    for arm in arms:
        r = run_live_arm(arm, oracle, donors, pool, X, args.k, args.budget, C.crc(f"{arm}|{args.k}|{args.seed}"))
        out["arms"][arm] = r
        (LIVE / f"{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        best = sorted(r["evaluated"].items(), key=lambda kv: kv[1])[:5]
        print(f"[live] {arm}: cost={r['steps'][-1]['cost_s']:.0f}s best={best}", flush=True)
    out["computed_log"] = oracle.log
    (LIVE / f"{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if not args.replay:
        (LIVE / "LIVE_DONE").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
