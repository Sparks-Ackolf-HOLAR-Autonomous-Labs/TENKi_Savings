"""
Even vs margin donor prior in TENKi exp 26 (paired campaign), matched seeds.

Even side: the committed exp 26 k=4 checkpoints for seeds 42 + 1000 s,
s = 0..19 (shared/output/bandgap_cluster_run2/paired_campaign_k4/checkpoints/;
their campaign curves are identical to the DGX 20-seed run that produced
tenki/results/paired_campaign/paired_campaign.json, and the aggregate below
is checked against that file). Margin side: checkpoints of the same run
with --prior-shape margin (this folder). Cold curves are shared (same
seeds, same event streams), so every difference is paired per seed.

Per (condition, arm, donor, receiver) cell and seed:
  early gain = mean(tau_cond[:5]) - mean(tau_cold[:5])   (exp 26 T_early)
  final gain = tau_cond[-1] - tau_cold[-1]               (exp 26 T_final)
Paired difference = margin - even. Bootstrap 95% CIs resample the 20
seeds (the pairing unit shared by all cells), 4000 draws, fixed seed.

If the frozen universe is reachable (PEGKI_ROOT/output/bandgap_archive_build/
bandgap_universe.jsonl, sha256[:16] checked), the report also lists the
per-channel class medians and the even vs margin priors toward the
experimental receiver, computed with exp 26's own loader and donor_prior.

Run from the package root: python tenki/results/paired_campaign_margin/
margin_vs_even.py  (numpy only). Writes margin_vs_even.md / .json here.
ASCII only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # package root (singular_obj)
EVEN_DIR = ROOT / "shared/output/bandgap_cluster_run2/paired_campaign_k4/checkpoints"
MARGIN_DIR = HERE / "checkpoints"
EVEN_SUMMARY = ROOT / "tenki/results/paired_campaign/paired_campaign.json"
SEEDS = [42 + 1000 * s for s in range(20)]
HALF = 5
CONDS = ["full", "selective", "combined"]
ARMS = ["pair", "kway"]
EXPT = "expt_zhuo"
N_BOOT = 4000


def load(ck_dir: Path):
    rows = {}
    for f in sorted(ck_dir.glob("*_s*.json")):
        row = json.loads(f.read_text(encoding="utf-8"))
        if row["seed"] in SEEDS:
            rows[(row["receiver"], row["seed"])] = row
    return rows


def gains(rows):
    """-> {(cond, arm, donor, rec): {seed: (early, final)}}"""
    out: dict = {}
    for (rec, seed), row in rows.items():
        for key, taus in row["curves"].items():
            parts = key.split("|")
            if parts[0] == "cold":
                continue
            cond, arm, donor = parts
            cold = row["curves"][f"cold|{arm}"]
            e = float(np.mean(taus[:HALF]) - np.mean(cold[:HALF]))
            f = float(taus[-1] - cold[-1])
            out.setdefault((cond, arm, donor, rec), {})[seed] = (e, f)
    return out


def boot(per_seed: np.ndarray, rng) -> list:
    """per_seed: (n_seeds,) values; CI of the mean over seeds."""
    n = len(per_seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = per_seed[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def block(cells, ge, gm, rng):
    """Summary over a list of cells for one (cond, arm) slice."""
    seeds = SEEDS
    # per-seed means over cells, (n_seeds, 2) for early/final
    ev = np.array([[np.mean([ge[c][s][j] for c in cells]) for j in (0, 1)]
                   for s in seeds])
    mv = np.array([[np.mean([gm[c][s][j] for c in cells]) for j in (0, 1)]
                   for s in seeds])
    d = mv - ev
    cell_e_even = [np.mean([ge[c][s][0] for s in seeds]) for c in cells]
    cell_e_marg = [np.mean([gm[c][s][0] for s in seeds]) for c in cells]
    cell_d = [m - e for m, e in zip(cell_e_marg, cell_e_even)]
    return {
        "n_cells": len(cells),
        "even_early": float(ev[:, 0].mean()),
        "margin_early": float(mv[:, 0].mean()),
        "diff_early": float(d[:, 0].mean()),
        "diff_early_ci": boot(d[:, 0], rng),
        "even_final": float(ev[:, 1].mean()),
        "margin_final": float(mv[:, 1].mean()),
        "diff_final": float(d[:, 1].mean()),
        "diff_final_ci": boot(d[:, 1], rng),
        "even_frac_pos_early": float(np.mean([v > 0 for v in cell_e_even])),
        "margin_frac_pos_early": float(np.mean([v > 0 for v in cell_e_marg])),
        "frac_cells_margin_better_early": float(np.mean([v > 0
                                                          for v in cell_d])),
    }


def prior_context():
    """Class medians (eV) and even/margin priors toward EXPT, or None."""
    root = Path(os.environ.get("PEGKI_ROOT", ROOT / "shared"))
    uni = root / "output/bandgap_archive_build/bandgap_universe.jsonl"
    if not uni.exists():
        return None
    h = hashlib.sha256(uni.read_bytes()).hexdigest()[:16]
    assert h == "f2158ecdc78e0229", h
    spec = importlib.util.spec_from_file_location(
        "e26", ROOT / "tenki/TENKi/experiments/26_paired_campaign.py")
    e26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e26)
    pc = e26.load_universe(uni, 30)
    med = {ch: {c: float(np.median(v)) for c, v in sorted(cl.items())}
           for ch, cl in sorted(pc.items())}
    pri = {d: {"even": e26.donor_prior(pc[EXPT], pc[d], "even"),
               "margin": e26.donor_prior(pc[EXPT], pc[d], "margin")}
           for d in sorted(pc) if d != EXPT}
    pri["__combined__"] = {"even": e26.combined_prior(EXPT, pc, "even"),
                           "margin": e26.combined_prior(EXPT, pc, "margin")}
    return {"universe_sha256_16": h, "class_medians_eV": med,
            f"priors_toward_{EXPT}": pri}


def main() -> int:
    rows_e, rows_m = load(EVEN_DIR), load(MARGIN_DIR)
    assert set(rows_e) == set(rows_m) and len(rows_e) == 160, \
        (len(rows_e), len(rows_m))
    for k in rows_e:   # cold streams must be identical (paired design)
        for arm in ARMS:
            assert rows_e[k]["curves"][f"cold|{arm}"] == \
                rows_m[k]["curves"][f"cold|{arm}"], k
    ge, gm = gains(rows_e), gains(rows_m)
    assert set(ge) == set(gm)

    # Reproduction check of the committed even summary
    committed = json.loads(EVEN_SUMMARY.read_text(encoding="utf-8"))["summary"]
    for cond in CONDS:
        for arm in ARMS:
            cells = [c for c in ge if c[0] == cond and c[1] == arm]
            val = float(np.mean([np.mean([ge[c][s][0] for s in SEEDS])
                                 for c in cells]))
            ref = committed[f"{cond}|{arm}"]["mean_T_early"]
            assert abs(val - ref) < 1e-12, (cond, arm, val, ref)

    rng = np.random.default_rng(42)
    res: dict = {"all_cells": {}, "expt_receiver": {}, "expt_cells": {}}
    for cond in CONDS:
        for arm in ARMS:
            cells = sorted(c for c in ge if c[0] == cond and c[1] == arm)
            res["all_cells"][f"{cond}|{arm}"] = block(cells, ge, gm, rng)
            ex = [c for c in cells if c[3] == EXPT]
            res["expt_receiver"][f"{cond}|{arm}"] = block(ex, ge, gm, rng)
            if cond != "combined":
                for c in ex:
                    res["expt_cells"][f"{cond}|{arm}|{c[2]}"] = block(
                        [c], ge, gm, rng)
    ctx = prior_context()
    if ctx is not None:
        res["prior_context"] = ctx
    res["config"] = {
        "even_checkpoints": str(EVEN_DIR.relative_to(ROOT)),
        "margin_checkpoints": str(MARGIN_DIR.relative_to(ROOT)),
        "seeds": SEEDS, "early_window": HALF, "n_boot": N_BOOT,
        "committed_even_summary_reproduced": True,
    }
    (HERE / "margin_vs_even.json").write_text(json.dumps(res, indent=2),
                                             encoding="utf-8")

    def ci(v):
        return f"[{v[0]:+.4f}, {v[1]:+.4f}]"

    def table(title, dct):
        out = [f"## {title}", "",
               "| slice | cells | early even | early margin | "
               "early diff (95% CI) | final even | final margin | "
               "final diff (95% CI) | cells>0 early even / margin | "
               "cells margin>even |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for key, b in dct.items():
            out.append(
                f"| {key.replace('|', ', ')} | {b['n_cells']} | "
                f"{b['even_early']:+.4f} | "
                f"{b['margin_early']:+.4f} | {b['diff_early']:+.4f} "
                f"{ci(b['diff_early_ci'])} | {b['even_final']:+.4f} | "
                f"{b['margin_final']:+.4f} | {b['diff_final']:+.4f} "
                f"{ci(b['diff_final_ci'])} | "
                f"{100 * b['even_frac_pos_early']:.0f}% / "
                f"{100 * b['margin_frac_pos_early']:.0f}% | "
                f"{100 * b['frac_cells_margin_better_early']:.0f}% |")
        return out + [""]

    lines = [
        "# Exp 26: even (ordinal) vs margin donor prior, matched seeds",
        "",
        "k=4, budget 240, budget step 24, 20 matched seeds (42 + 1000 s), "
        "lambda_full = 8, learned trust grid unchanged, universe sha256[:16] "
        "f2158ecdc78e0229. Even = committed run (evenly spaced ordinal "
        "prior). Margin = --prior-shape margin (donor class-median margins, "
        "z-scored over the shared classes and rescaled to the even prior's "
        "spread, so only the shape changes). Gains are Kendall tau vs the "
        "matched cold campaign; early = mean over the first 5 of 10 "
        "checkpoints (exp 26 T_early), final = last checkpoint. diff = "
        "margin - even, paired per seed; CIs are percentile bootstrap over "
        f"the 20 seeds ({N_BOOT} draws). 'cells>0' = fraction of "
        "donor->receiver cells with positive mean early gain.",
        "",
        "The even columns reproduce the committed "
        "tenki/results/paired_campaign/paired_campaign.json summary exactly "
        "(asserted by the script).",
        "",
    ]
    lines += table("All donor->receiver cells", res["all_cells"])
    lines += table(f"Experimental receiver ({EXPT}), all donors",
                   res["expt_receiver"])
    lines += table(f"DFT donor -> {EXPT}, per cell", res["expt_cells"])
    if ctx is not None:
        classes = sorted({c for m in ctx["class_medians_eV"].values()
                          for c in m})
        lines += ["## Class medians per channel (eV, records with gap > "
                  "0.01 eV, classes with >= 30 records)", "",
                  "| channel | " + " | ".join(classes) + " |",
                  "|---" * (len(classes) + 1) + "|"]
        for ch, m in ctx["class_medians_eV"].items():
            lines.append(f"| {ch} | " + " | ".join(
                f"{m[c]:.2f}" if c in m else "-" for c in classes) + " |")
        pri = ctx[f"priors_toward_{EXPT}"]
        rc = sorted(next(iter(pri.values()))["even"])
        lines += ["", f"## Donor priors toward {EXPT} (even / margin "
                  "log-strength anchors)", "",
                  "| donor | " + " | ".join(rc) + " |",
                  "|---" * (len(rc) + 1) + "|"]
        for d, pm in pri.items():
            lines.append(f"| {d} | " + " | ".join(
                f"{pm['even'][c]:+.2f} / {pm['margin'][c]:+.2f}"
                for c in rc) + " |")
        lines.append("")
    lines += ["Script: margin_vs_even.py; machine-readable: "
              "margin_vs_even.json. Each evaluation is one archived record "
              "(no invented observations)."]
    (HERE / "margin_vs_even.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
