"""
Experiment 29 -- Cost calibration of the paired-campaign logs (band-gap archive).

The recorded TENKi limit says campaign costs are counted in EVALUATIONS;
calibrating to real instrument-hours / CPU-hours per channel was planned,
not done. This script does that calibration over the measured exp 26
paired-campaign logs (the 50-seed k=4 CHPC run by default). It is an
ANALYSIS of committed campaign logs -- it runs no new campaigns and
queries no new observations.

Method
------
1. A PLANNING-GRADE per-channel unit-cost table (low, mid, high) in the
   receiver's NATIVE currency per evaluation: CPU-core-hours for the DFT
   channels, person+instrument-hours for the experimental channel. These
   are order-of-magnitude planning assumptions with stated provenance,
   NOT measured microdata; the user can override the whole table via
   --cost-json. CPU-core-hours and person/instrument-hours are
   deliberately never converted into each other; an optional dollars
   view is reported ONLY as a labelled illustration
   (--usd-per-cpu-hour, --usd-per-lab-hour).
2. Evaluations-saved per (donor -> receiver) cell, condition in
   {full, combined}, arm in {pair, kway}, recomputed from the per-seed
   checkpoint curves: per seed, take the reuse arm's tau at the END OF
   THE EARLY WINDOW (exp 26's own early-window definition, read from the
   aggregate config); find by linear interpolation the budget at which
   the matched-seed COLD arm reaches that tau (cold curve anchored at
   (budget=0, tau=0), matching exp 26's no-data cold value); if it never
   does, use the max budget as a conservative floor and flag the seed
   censored. evaluations_saved = cold budget - early-window budget.
   Mean, percentile-bootstrap CI over seeds, censoring fraction.
3. Real-cost translation: hours_saved = evaluations_saved *
   unit_cost(receiver), at low/mid/high. Donor-side cost, two framings:
   (a) archived reuse -- the donor ranking already exists in a public
   database, marginal cost ~ 0, hours_saved is pure leverage;
   (b) from-scratch break-even -- breakeven_records =
   hours_saved / unit_cost_mid(donor) donor records could be afforded;
   compared with the donor records the prior actually needs (the frozen
   universe required >= min_class_size records per class; the per-class
   counts actually present are reported).
4. Sensitivity: how many cells stay net-positive across the full
   (low..high) cost ranges, per condition x arm.

Outputs (default ../../results/cost_calibration/ = tenki/results/...)
---------------------------------------------------------------------
  cost_calibration.json / .md / .png

Run (from tenki/TENKi):
  uv run python experiments/29_cost_calibration.py
Self-test (synthetic miniature logs, hand-computed interpolation answer,
censoring path, cost-table completeness): --self-test.
ASCII only; seeded and reproducible (--seed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

COMBINED = "__combined__"
COST_CONDITIONS = ("full", "combined")
ARMS = ("pair", "kway")

# The 8 receivers of the exp 26 k=4 campaign run; the default cost table
# must cover all of them (checked in --self-test).
EXPECTED_RECEIVERS = [
    "expt_zhuo", "jarvis_mbj", "jarvis_optb88vdw", "mp_gga",
    "mp_gga_expsynth", "mp_r2scan", "snumat_gga", "snumat_hse",
]

# ---------------------------------------------------------------------------
# Planning-grade unit-cost table. These are ASSUMPTIONS (order-of-magnitude
# planning estimates with stated provenance), not measured microdata.
# User-adjustable via --cost-json (same schema).
# ---------------------------------------------------------------------------
_GGA_PROV = ("planning estimate: GGA-level static DFT relaxation+bands at "
             "high-throughput database scale, order 10-100 CPU-core-hours "
             "per material (typical MP/JARVIS/SNUMAT workflow scale); "
             "assumption, not measured microdata")
DEFAULT_COST_TABLE = {
    "mp_gga": {
        "low": 10.0, "mid": 40.0, "high": 120.0,
        "unit": "cpu_core_hours_per_evaluation", "provenance": _GGA_PROV},
    "mp_gga_expsynth": {
        "low": 10.0, "mid": 40.0, "high": 120.0,
        "unit": "cpu_core_hours_per_evaluation",
        "provenance": _GGA_PROV + " (same GGA workflow, experimentally "
        "synthesizable subset)"},
    "snumat_gga": {
        "low": 10.0, "mid": 40.0, "high": 120.0,
        "unit": "cpu_core_hours_per_evaluation", "provenance": _GGA_PROV},
    "jarvis_optb88vdw": {
        "low": 10.0, "mid": 40.0, "high": 120.0,
        "unit": "cpu_core_hours_per_evaluation",
        "provenance": _GGA_PROV + " (OptB88vdW is a GGA-class vdW "
        "functional)"},
    "mp_r2scan": {
        "low": 20.0, "mid": 80.0, "high": 240.0,
        "unit": "cpu_core_hours_per_evaluation",
        "provenance": "planning estimate: r2SCAN meta-GGA static workflow, "
        "roughly 2-3x GGA cost per material; assumption, not measured "
        "microdata"},
    "jarvis_mbj": {
        "low": 10.0, "mid": 50.0, "high": 150.0,
        "unit": "cpu_core_hours_per_evaluation",
        "provenance": "planning estimate: TBmBJ is a cheap potential-only "
        "correction on a converged density, near-GGA cost per material; "
        "assumption, not measured microdata"},
    "snumat_hse": {
        "low": 300.0, "mid": 2000.0, "high": 8000.0,
        "unit": "cpu_core_hours_per_evaluation",
        "provenance": "planning estimate: HSE06 hybrid-functional band "
        "structure, 30-100x GGA cost per material; assumption, not "
        "measured microdata"},
    "expt_zhuo": {
        "low": 8.0, "mid": 40.0, "high": 160.0,
        "unit": "person_instrument_hours_per_evaluation",
        "provenance": "planning estimate: experimental band gap = powder "
        "synthesis + diffuse-reflectance/UV-Vis measurement + analysis "
        "per sample; assumption, not measured microdata"},
}

# Anion classification: mirrors exp 26 / classify_by_anion exactly (used
# only to COUNT donor records per class in the frozen universe).
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


def classify_formula(formula: str) -> str:
    species = set(re.findall(r"[A-Z][a-z]?", formula))
    for name, elems in ANION_PRIORITY:
        if species & elems:
            return name
    return "other"


def _pegki_root() -> Path:
    env = os.environ.get("PEGKI_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(_HERE) / ".." / ".." / ".." / "shared").resolve()


def _default_input_dir() -> Path:
    return (Path(_HERE) / ".." / ".." / ".." / "shared" / "output" /
            "bandgap_cluster_run2" / "paired_campaign_k4").resolve()


def _default_output_dir() -> Path:
    return (Path(_HERE) / ".." / ".." / "results" /
            "cost_calibration").resolve()


# ---------------------------------------------------------------------------
# Loading (small local loader; deliberately duplicated with
# shared/scripts/omega_regret.py so both scripts stay independently runnable)
# ---------------------------------------------------------------------------
def load_aggregate(input_dir: Path) -> dict:
    with open(input_dir / "paired_campaign.json", "r",
              encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_rows(input_dir: Path) -> list:
    rows = []
    ckpt_dir = input_dir / "checkpoints"
    for path in sorted(ckpt_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def count_universe_classes(universe: Path, min_class_size: int):
    """{channel: {class: n_records}} for classes >= min_class_size.

    Returns None when the universe file is unavailable (the analysis then
    reports donor screen sizes as unknown instead of inventing them).
    """
    if not universe.exists():
        return None
    per: dict = {}
    with open(universe, "r", encoding="utf-8") as fh:
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
            if float(gap) <= MIN_GAP_EV:
                continue
            cls = classify_formula(formula)
            per.setdefault(channel, {}).setdefault(cls, 0)
            per[channel][cls] += 1
    return {ch: {c: n for c, n in sorted(classes.items())
                 if n >= min_class_size}
            for ch, classes in per.items()}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def budget_to_reach(cold_taus, budgets, target):
    """First budget at which the linearly interpolated cold curve reaches
    `target` tau. The cold curve is anchored at (budget=0, tau=0.0),
    matching exp 26's cold value when no event has been observed.

    -> (budget, censored). censored=True means the cold curve never
    reaches target within the run; budget is then the max budget
    (conservative floor).
    """
    xs = [0.0] + [float(b) for b in budgets]
    ys = [0.0] + [float(t) for t in cold_taus]
    eps = 1e-12
    for i in range(len(xs)):
        if ys[i] >= target - eps:
            if i == 0:
                return xs[0], False
            y0, y1 = ys[i - 1], ys[i]
            if y1 - y0 <= eps:
                return xs[i], False
            frac = (target - y0) / (y1 - y0)
            frac = min(max(frac, 0.0), 1.0)
            return xs[i - 1] + frac * (xs[i] - xs[i - 1]), False
    return xs[-1], True


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 n_boot: int = 2000):
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    means = [float(values[rng.integers(0, len(values), len(values))].mean())
             for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def compute_evaluations_saved(rows, checkpoints, early_checkpoints, seed):
    """Per-cell evaluations saved, recomputed from per-seed curves.

    -> {cond: {arm: {donor: {receiver: {mean, ci, censored_frac,
    n_seeds}}}}} for cond in COST_CONDITIONS.
    """
    early_idx = len(early_checkpoints) - 1
    early_budget = float(early_checkpoints[-1])
    per_cell: dict = {}
    for row in rows:
        rec = row["receiver"]
        curves = row["curves"]
        for key, taus in curves.items():
            parts = key.split("|")
            if parts[0] not in COST_CONDITIONS:
                continue
            cond, arm, donor = parts
            cold = curves.get(f"cold|{arm}")
            if cold is None or early_idx >= len(taus):
                continue
            target = float(taus[early_idx])
            b_cold, censored = budget_to_reach(cold, checkpoints, target)
            saved = b_cold - early_budget
            per_cell.setdefault((cond, arm, donor, rec),
                                []).append((saved, censored))
    rng = np.random.default_rng(seed)
    out: dict = {c: {a: {} for a in ARMS} for c in COST_CONDITIONS}
    for (cond, arm, donor, rec) in sorted(per_cell):
        vals = per_cell[(cond, arm, donor, rec)]
        saved = np.asarray([v[0] for v in vals], dtype=float)
        cens = np.asarray([v[1] for v in vals], dtype=float)
        out[cond][arm].setdefault(donor, {})[rec] = {
            "evals_saved_mean": float(saved.mean()),
            "evals_saved_ci": bootstrap_ci(saved, rng),
            "censored_frac": float(cens.mean()),
            "n_seeds": len(vals),
        }
    return out


def apply_costs(evals_saved, cost_table, class_counts, min_class_size,
                usd_per_cpu_hour, usd_per_lab_hour):
    """Translate evaluations saved into receiver-native hours saved plus
    donor-side break-even framing. Returns the cost-augmented cell dict.
    """
    out: dict = {}
    for cond, arms in evals_saved.items():
        out[cond] = {}
        for arm, donors in arms.items():
            out[cond][arm] = {}
            for donor, recs in donors.items():
                out[cond][arm][donor] = {}
                for rec, cell in recs.items():
                    rc = cost_table[rec]
                    m = cell["evals_saved_mean"]
                    lo_ci, hi_ci = cell["evals_saved_ci"]
                    lab = rc["unit"].startswith("person")
                    usd_rate = usd_per_lab_hour if lab else usd_per_cpu_hour
                    entry = dict(cell)
                    entry["receiver_unit"] = rc["unit"]
                    entry["hours_saved_low"] = m * rc["low"]
                    entry["hours_saved_mid"] = m * rc["mid"]
                    entry["hours_saved_high"] = m * rc["high"]
                    entry["hours_saved_mid_ci"] = [lo_ci * rc["mid"],
                                                   hi_ci * rc["mid"]]
                    entry["usd_illustration_mid"] = (
                        m * rc["mid"] * usd_rate)
                    # donor-side framings
                    entry["donor_framing_archived"] = (
                        "donor ranking already archived in a public "
                        "database; marginal donor cost ~ 0, hours_saved "
                        "is pure leverage")
                    if donor == COMBINED:
                        others = [c for c in cost_table if c != rec]
                        d_mid = float(np.mean(
                            [cost_table[c]["mid"] for c in others]))
                        d_high = float(np.mean(
                            [cost_table[c]["high"] for c in others]))
                        d_low = float(np.mean(
                            [cost_table[c]["low"] for c in others]))
                        entry["donor_cost_note"] = (
                            "combined prior pools all other channels; "
                            "donor unit cost taken as the MEAN unit cost "
                            "over the 7 donors (records would be spread "
                            "across donors)")
                        need = None
                        if class_counts is not None:
                            need = int(sum(
                                len(class_counts.get(c, {})) *
                                min_class_size for c in others))
                    else:
                        d_low = cost_table[donor]["low"]
                        d_mid = cost_table[donor]["mid"]
                        d_high = cost_table[donor]["high"]
                        need = None
                        if class_counts is not None:
                            need = int(len(class_counts.get(donor, {})) *
                                       min_class_size)
                    entry["breakeven_records_mid"] = (
                        entry["hours_saved_mid"] / d_mid if d_mid > 0
                        else None)
                    entry["breakeven_records_worst"] = (
                        entry["hours_saved_low"] / d_high if d_high > 0
                        else None)
                    entry["donor_records_needed"] = need
                    if need is not None:
                        entry["from_scratch_pays_mid"] = bool(
                            entry["breakeven_records_mid"] > need)
                        entry["from_scratch_pays_worst"] = bool(
                            entry["breakeven_records_worst"] > need)
                    else:
                        entry["from_scratch_pays_mid"] = None
                        entry["from_scratch_pays_worst"] = None
                    out[cond][arm][donor][rec] = entry
    return out


def sensitivity_table(cost_cells):
    """Per (condition, arm): cell counts that stay net-positive.

    Archived framing: donor cost ~ 0, so the sign of the net saving is
    the sign of evals_saved_mean at ANY receiver cost in (low..high) --
    cost-range invariant by construction. From-scratch framing: compared
    at mid costing and at worst-case costing (receiver low unit cost,
    donor high unit cost).
    """
    table = {}
    for cond, arms in cost_cells.items():
        for arm, donors in arms.items():
            cells = [c for d in donors.values() for c in d.values()]
            if not cells:
                continue
            known = [c for c in cells
                     if c["from_scratch_pays_mid"] is not None]
            table[f"{cond}|{arm}"] = {
                "n_cells": len(cells),
                "archived_net_positive": int(sum(
                    c["evals_saved_mean"] > 0 for c in cells)),
                "archived_ci_lower_positive": int(sum(
                    c["evals_saved_ci"][0] > 0 for c in cells)),
                "from_scratch_pays_mid": (int(sum(
                    c["from_scratch_pays_mid"] for c in known))
                    if known else None),
                "from_scratch_pays_worst_case": (int(sum(
                    c["from_scratch_pays_worst"] for c in known))
                    if known else None),
            }
    return table


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def make_plot(cost_cells, receivers, out_path):
    donors = sorted({d for d in cost_cells["full"]["pair"]})
    recs = [r for r in receivers]
    M = np.full((len(recs), len(donors)), np.nan)
    for j, d in enumerate(donors):
        for i, r in enumerate(recs):
            cell = cost_cells["full"]["pair"].get(d, {}).get(r)
            if cell:
                M[i, j] = cell["hours_saved_mid"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    finite = M[np.isfinite(M)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    norm = matplotlib.colors.SymLogNorm(linthresh=10.0, vmin=-vmax,
                                        vmax=vmax)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(donors)),
                  [d.replace("_", "\n") for d in donors], fontsize=8)
    ax.set_yticks(range(len(recs)),
                  [r.replace("_", " ") for r in recs], fontsize=8)
    ax.set_xlabel("donor A")
    ax.set_ylabel("receiver B")
    ax.set_title("Receiver-native hours saved (mid estimate), full reuse, "
                 "pair arm\nrows are in the RECEIVER's currency "
                 "(expt_zhuo: person+instrument-hours; others: "
                 "CPU-core-hours)")
    for i in range(len(recs)):
        for j in range(len(donors)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center",
                        fontsize=7)
    if "expt_zhuo" in recs:
        i = recs.index("expt_zhuo")
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(donors), 1.0,
                                   fill=False, edgecolor="black", lw=2.2))
        ax.text(len(donors) - 0.42, i,
                "instrument-hours\n(scarcest currency)", fontsize=7,
                va="center", ha="left")
    fig.colorbar(im, ax=ax, shrink=0.85,
                 label="hours saved (receiver-native, symlog)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(path, cfg_note, cost_table, cost_cells, sens,
                   class_counts, min_class_size, usd_cpu, usd_lab,
                   early_budget, max_budget):
    lines = [
        "# Experiment 29: cost calibration of the paired-campaign logs",
        "",
        "Analysis of the committed exp 26 paired-campaign logs "
        "(no new campaigns were run). " + cfg_note,
        "",
        "## Unit-cost table (planning-grade ASSUMPTIONS, not measured "
        "microdata)",
        "",
        "Costs are per evaluation, in the channel's NATIVE currency. "
        "CPU-core-hours and person/instrument-hours are deliberately NOT "
        "converted into each other or into dollars; the dollars column "
        "is a labelled illustration only "
        f"(USD {usd_cpu}/CPU-core-hour, USD {usd_lab}/lab-hour, "
        "user-adjustable). Override the whole table with --cost-json.",
        "",
        "| channel | low | mid | high | unit | provenance |",
        "|---|---|---|---|---|---|",
    ]
    for ch in sorted(cost_table):
        c = cost_table[ch]
        lines.append(f"| {ch} | {c['low']:g} | {c['mid']:g} | "
                     f"{c['high']:g} | {c['unit']} | {c['provenance']} |")
    lines += [
        "",
        "## How evaluations-saved is measured",
        "",
        f"Per (donor -> receiver) cell and seed: the reuse arm's tau at "
        f"the end of the early window (budget {early_budget:g} "
        "evaluations) is the target; linear interpolation on the "
        "matched-seed cold curve (anchored at budget 0, tau 0) gives the "
        "budget at which cold reaches that target. If cold never reaches "
        f"it, the max budget ({max_budget:g}) is used as a conservative "
        "floor and the seed is flagged censored. evaluations_saved = "
        f"cold budget - {early_budget:g}.",
        "",
        "Relation to exp 26's transfer matrices: exp 26 reports the MEAN "
        "tau gain over the whole early window (T_early), which credits a "
        "head start at the very first checkpoints. This script asks a "
        "harsher planning question: at the END of the early window, how "
        "long would the cold campaign take to reach the reuse arm's tau. "
        "A cell can be positive in exp 26 and negative here (e.g. "
        "combined -> expt_zhuo): the donated ranking helps immediately, "
        "but the cold campaign overtakes it before the early window "
        "closes. Both statements are true; they answer different budget "
        "questions, and the harsher one is the right basis for pricing "
        "hours.",
        "",
        "## Headline: expt_zhuo receiver (instrument-hours, the scarcest "
        "currency)",
        "",
        "| donor | evals saved (mean [95% CI]) | censored | hours saved "
        "mid [low..high] | USD (illustration) | breakeven donor records "
        "(mid) | donor records the prior needed |",
        "|---|---|---|---|---|---|---|",
    ]
    zh = {}
    for donor, recs in cost_cells["full"]["pair"].items():
        if "expt_zhuo" in recs:
            zh[donor] = recs["expt_zhuo"]
    comb = cost_cells.get("combined", {}).get("pair", {}).get(
        COMBINED, {}).get("expt_zhuo")
    if comb is not None:
        zh[COMBINED] = comb
    for donor in sorted(zh):
        c = zh[donor]
        need = c["donor_records_needed"]
        be = c["breakeven_records_mid"]
        lines.append(
            f"| {donor} | {c['evals_saved_mean']:+.1f} "
            f"[{c['evals_saved_ci'][0]:+.1f}, "
            f"{c['evals_saved_ci'][1]:+.1f}] | "
            f"{100 * c['censored_frac']:.0f}% | "
            f"{c['hours_saved_mid']:+.0f} [{c['hours_saved_low']:+.0f}.."
            f"{c['hours_saved_high']:+.0f}] h | "
            f"{c['usd_illustration_mid']:+.0f} | "
            f"{be:+.1f} | {need if need is not None else 'unknown'} |")
    lines += [
        "",
        "Reading the break-even column: under the FROM-SCRATCH framing, "
        "reuse pays if the hours saved on the receiver buy more donor "
        "records (at the donor's mid unit cost) than the prior actually "
        "needs. The prior needs only enough donor records to rank the "
        f"donor's material classes; the frozen universe required >= "
        f"{min_class_size} records per class, so 'records needed' = "
        f"n_donor_classes x {min_class_size}. Under the ARCHIVED-REUSE "
        "framing the donor ranking already exists in a public database "
        "(marginal cost ~ 0) and every positive hours-saved figure is "
        "pure leverage.",
        "",
        "## Sensitivity: cells net-positive across the (low..high) cost "
        "ranges",
        "",
        "Archived framing: the SIGN of the net saving does not depend on "
        "the unit cost at all (cost > 0 multiplies evaluations saved), "
        "so the count is stable across the whole low..high range by "
        "construction. From-scratch framing: 'worst case' prices the "
        "receiver at its LOW unit cost and the donor at its HIGH unit "
        "cost.",
        "",
        "| condition | arm | cells | net-positive (archived, any "
        "costing) | CI-lower > 0 | from-scratch pays (mid) | "
        "from-scratch pays (worst case) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(sens):
        s = sens[key]
        cond, arm = key.split("|")
        lines.append(
            f"| {cond} | {arm} | {s['n_cells']} | "
            f"{s['archived_net_positive']} | "
            f"{s['archived_ci_lower_positive']} | "
            f"{s['from_scratch_pays_mid']} | "
            f"{s['from_scratch_pays_worst_case']} |")
    lines += [
        "",
        "## Donor screen sizes actually present (frozen universe)",
        "",
    ]
    if class_counts is None:
        lines.append("Universe file unavailable; donor per-class counts "
                     "not reported (never invented).")
    else:
        lines += ["| channel | classes (>= min size) | records per class "
                  "| total |", "|---|---|---|---|"]
        for ch in sorted(class_counts):
            cc = class_counts[ch]
            detail = ", ".join(f"{k}: {v}" for k, v in
                               sorted(cc.items()))
            lines.append(f"| {ch} | {len(cc)} | {detail} | "
                         f"{sum(cc.values())} |")
    lines += [
        "",
        "## Full per-cell table: full reuse, pair arm (hours in the "
        "receiver's native currency)",
        "",
        "| donor | receiver | evals saved | censored | hours saved mid "
        "[low..high] | unit | breakeven recs (mid) | recs needed | pays "
        "from scratch (mid/worst) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for donor in sorted(cost_cells["full"]["pair"]):
        for rec in sorted(cost_cells["full"]["pair"][donor]):
            c = cost_cells["full"]["pair"][donor][rec]
            unit = ("lab-h" if c["receiver_unit"].startswith("person")
                    else "CPU-core-h")
            need = c["donor_records_needed"]
            lines.append(
                f"| {donor} | {rec} | {c['evals_saved_mean']:+.1f} | "
                f"{100 * c['censored_frac']:.0f}% | "
                f"{c['hours_saved_mid']:+.0f} "
                f"[{c['hours_saved_low']:+.0f}.."
                f"{c['hours_saved_high']:+.0f}] | {unit} | "
                f"{c['breakeven_records_mid']:+.1f} | "
                f"{need if need is not None else '?'} | "
                f"{c['from_scratch_pays_mid']}/"
                f"{c['from_scratch_pays_worst']} |")
    lines += [
        "",
        "Combined-prior cells, the k-way arm, bootstrap CIs and the USD "
        "illustration for every cell: cost_calibration.json. All "
        "evaluations-saved figures are recomputed from the committed "
        "per-seed campaign checkpoints; the only assumptions are the "
        "unit costs, which are labelled as such above.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    failures = 0

    # 1. interpolation, hand-computed: cold [0.2,0.4,0.6,0.8] at budgets
    # [10,20,30,40], anchored (0,0). target 0.5 lies between 0.4@20 and
    # 0.6@30 -> budget 25. target 0.0 -> budget 0. target 0.9 -> censored
    # at 40.
    budgets = [10, 20, 30, 40]
    cold = [0.2, 0.4, 0.6, 0.8]
    b, cens = budget_to_reach(cold, budgets, 0.5)
    ok = abs(b - 25.0) < 1e-9 and not cens
    print(f"[{'PASS' if ok else 'FAIL'}] interpolation: target 0.5 -> "
          f"budget {b:g} (expected 25), censored={cens}")
    failures += 0 if ok else 1
    b, cens = budget_to_reach(cold, budgets, -0.2)
    ok = b == 0.0 and not cens
    print(f"[{'PASS' if ok else 'FAIL'}] target below anchor -> budget "
          f"{b:g} (expected 0)")
    failures += 0 if ok else 1
    b, cens = budget_to_reach(cold, budgets, 0.9)
    ok = b == 40.0 and cens
    print(f"[{'PASS' if ok else 'FAIL'}] censoring: target 0.9 -> budget "
          f"{b:g} (expected 40), censored={cens}")
    failures += 0 if ok else 1

    # 2. miniature 2-seed end-to-end (synthetic logs in a temp dir; the
    # real analysis only ever reads committed campaign logs). Early
    # window = first 2 checkpoints -> early budget 20, early index 1.
    # Seed 0: full tau at ckpt idx 1 = 0.5 -> cold budget 25, saved 5.
    # Seed 1: full tau at ckpt idx 1 = 0.9 -> censored, saved 40-20=20.
    # Expect mean 12.5, censored_frac 0.5.
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "checkpoints").mkdir()
        agg = {"config": {"min_class_size": 30},
               "checkpoints": budgets,
               "early_window_checkpoints": [10, 20],
               "receivers": ["r1"]}
        with open(tdp / "paired_campaign.json", "w",
                  encoding="utf-8") as f:
            json.dump(agg, f)
        rows = [
            {"receiver": "r1", "seed": 0, "curves": {
                "cold|pair": cold,
                "full|pair|d1": [0.5, 0.5, 0.5, 0.5]}},
            {"receiver": "r1", "seed": 1, "curves": {
                "cold|pair": cold,
                "full|pair|d1": [0.9, 0.9, 0.9, 0.9]}},
        ]
        for r in rows:
            with open(tdp / "checkpoints" / f"r1_s{r['seed']}.json", "w",
                      encoding="utf-8") as f:
                json.dump(r, f)
        agg2 = load_aggregate(tdp)
        rows2 = load_checkpoint_rows(tdp)
        ev = compute_evaluations_saved(rows2, agg2["checkpoints"],
                                       agg2["early_window_checkpoints"],
                                       seed=0)
        cell = ev["full"]["pair"]["d1"]["r1"]
        ok = (abs(cell["evals_saved_mean"] - 12.5) < 1e-9 and
              abs(cell["censored_frac"] - 0.5) < 1e-9 and
              cell["n_seeds"] == 2)
        print(f"[{'PASS' if ok else 'FAIL'}] miniature end-to-end: mean "
              f"{cell['evals_saved_mean']:g} (expected 12.5), censored "
              f"{cell['censored_frac']:g} (expected 0.5)")
        failures += 0 if ok else 1

        # cost translation on the miniature with a synthetic table
        table = {"r1": {"low": 2.0, "mid": 4.0, "high": 8.0,
                        "unit": "cpu_core_hours_per_evaluation",
                        "provenance": "self-test"},
                 "d1": {"low": 1.0, "mid": 2.0, "high": 4.0,
                        "unit": "cpu_core_hours_per_evaluation",
                        "provenance": "self-test"}}
        cc = apply_costs(ev, table, None, 30, 0.05, 75.0)
        c = cc["full"]["pair"]["d1"]["r1"]
        ok = (abs(c["hours_saved_mid"] - 50.0) < 1e-9 and
              abs(c["breakeven_records_mid"] - 25.0) < 1e-9 and
              c["donor_records_needed"] is None)
        print(f"[{'PASS' if ok else 'FAIL'}] cost translation: mid hours "
              f"{c['hours_saved_mid']:g} (expected 50), breakeven "
              f"{c['breakeven_records_mid']:g} (expected 25), missing "
              "universe handled as unknown")
        failures += 0 if ok else 1

    # 3. default cost table completeness vs the campaign receiver list
    missing = [r for r in EXPECTED_RECEIVERS if r not in DEFAULT_COST_TABLE]
    extra_keys_ok = all(
        set(v) == {"low", "mid", "high", "unit", "provenance"} and
        0 < v["low"] <= v["mid"] <= v["high"]
        for v in DEFAULT_COST_TABLE.values())
    ok = not missing and extra_keys_ok
    print(f"[{'PASS' if ok else 'FAIL'}] cost-table completeness: "
          f"missing={missing}, ranges ordered={extra_keys_ok}")
    failures += 0 if ok else 1

    print(f"self-test: {'PASS' if failures == 0 else 'FAIL'}")
    return failures


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TENKi exp 29: real-cost calibration of the exp 26 "
                    "paired-campaign logs (analysis only)")
    ap.add_argument("--input-dir", default=None,
                    help="exp 26 output dir with paired_campaign.json and "
                         "checkpoints/ (default: the 50-seed k4 CHPC run)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--cost-json", default=None,
                    help="JSON file overriding the unit-cost table "
                         "(same schema as the built-in table)")
    ap.add_argument("--usd-per-cpu-hour", type=float, default=0.05)
    ap.add_argument("--usd-per-lab-hour", type=float, default=75.0)
    ap.add_argument("--seed", type=int, default=42,
                    help="bootstrap seed")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    input_dir = (Path(args.input_dir).resolve() if args.input_dir
                 else _default_input_dir())
    out_dir = (Path(args.output_dir).resolve() if args.output_dir
               else _default_output_dir())
    if not (input_dir / "paired_campaign.json").exists():
        print(f"[error] aggregate not found under {input_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_table = dict(DEFAULT_COST_TABLE)
    if args.cost_json:
        with open(args.cost_json, "r", encoding="utf-8") as f:
            cost_table.update(json.load(f))
        print(f"[load] cost table overridden from {args.cost_json}")

    agg = load_aggregate(input_dir)
    checkpoints = agg["checkpoints"]
    early = agg["early_window_checkpoints"]
    receivers = agg["receivers"]
    min_class_size = int(agg["config"].get("min_class_size", 30))
    missing_cost = [r for r in receivers if r not in cost_table]
    if missing_cost:
        print(f"[error] no unit cost for receivers: {missing_cost}")
        return 1

    rows = load_checkpoint_rows(input_dir)
    print(f"[load] {len(rows)} per-(receiver, seed) checkpoint files; "
          f"{len(receivers)} receivers; early window ends at budget "
          f"{early[-1]} of {checkpoints[-1]}")

    universe = (_pegki_root() / "output" / "bandgap_archive_build" /
                "bandgap_universe.jsonl")
    class_counts = count_universe_classes(universe, min_class_size)
    if class_counts is None:
        print(f"[warn] universe not found at {universe}; donor screen "
              "sizes reported as unknown")

    ev = compute_evaluations_saved(rows, checkpoints, early, args.seed)
    cost_cells = apply_costs(ev, cost_table, class_counts, min_class_size,
                             args.usd_per_cpu_hour, args.usd_per_lab_hour)
    sens = sensitivity_table(cost_cells)

    results = {
        "inputs": {
            "input_dir": str(input_dir),
            "n_checkpoint_files": len(rows),
            "checkpoints": checkpoints,
            "early_window_checkpoints": early,
            "receivers": receivers,
            "min_class_size": min_class_size,
            "bootstrap_seed": args.seed,
        },
        "cost_table": cost_table,
        "cost_table_note": (
            "planning-grade order-of-magnitude assumptions with stated "
            "provenance, user-adjustable via --cost-json; NOT "
            "measured microdata. Receiver-native currencies (CPU-core-"
            "hours vs person+instrument-hours) are never converted into "
            "each other; usd_illustration_mid is a labelled illustration "
            "only."),
        "usd_rates_illustration": {
            "usd_per_cpu_core_hour": args.usd_per_cpu_hour,
            "usd_per_lab_hour": args.usd_per_lab_hour,
        },
        "method": (
            "per seed: reuse tau at the early-window end (budget "
            f"{early[-1]}) -> linearly interpolated budget at which the "
            "matched-seed cold curve (anchored at budget 0, tau 0) "
            "reaches it; never -> max budget, censored; "
            "evaluations_saved = cold budget - early budget"),
        "cells": cost_cells,
        "sensitivity": sens,
        "donor_class_counts": class_counts,
    }
    with open(out_dir / "cost_calibration.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cfg_note = (f"Inputs: {len(rows)} (receiver, seed) checkpoint files "
                f"from {input_dir.name}; early window = budgets "
                f"{early[0]}..{early[-1]} of {checkpoints[-1]} "
                "evaluations; conditions full + combined, arms pair + "
                "kway; bootstrap CIs over seeds.")
    write_markdown(out_dir / "cost_calibration.md", cfg_note, cost_table,
                   cost_cells, sens, class_counts, min_class_size,
                   args.usd_per_cpu_hour, args.usd_per_lab_hour,
                   float(early[-1]), float(checkpoints[-1]))
    make_plot(cost_cells, receivers, out_dir / "cost_calibration.png")

    print(f"[save] {out_dir}")
    zh = cost_cells["full"]["pair"]
    saved = [(d, zh[d]["expt_zhuo"]) for d in sorted(zh)
             if "expt_zhuo" in zh[d]]
    for d, c in saved:
        print(f"  expt_zhuo <- {d}: {c['evals_saved_mean']:+.1f} evals "
              f"saved, {c['hours_saved_mid']:+.0f} instrument-h (mid), "
              f"censored {100 * c['censored_frac']:.0f}%")
    for key in sorted(sens):
        s = sens[key]
        print(f"  {key}: {s['archived_net_positive']}/{s['n_cells']} "
              f"cells net-positive (archived), "
              f"{s['from_scratch_pays_worst_case']} pay from scratch at "
              "worst-case costing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
