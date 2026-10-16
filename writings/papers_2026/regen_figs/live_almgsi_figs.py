#!/usr/bin/env python3
"""Figures for the computed Al-Mg-Si campaign in the three drafts.

Data: all numbers come from committed artifacts under
compute/almgsi_campaign/:
  results/analysis/source_agreement.json          Kendall tau vs QE PBE
  results/campaigns/qe_pbe_all/summary.json        PBE-receiver campaigns
  results/campaigns/qe_pbe_shapley/summary.json    Shapley over four donors
  results/campaigns/qe_hse06_all/summary.json      HSE06-receiver campaigns
  results/campaigns/qe_hse06_shapley/summary.json  Shapley over five donors
  results_T/analysis/temperature_study.json        temperature study
Three figures, one per paper:
  PEGKi  live_transfer.png   (a) ordering agreement of each computed source
         with QE PBE and with QE HSE06, labelled with its median cost per
         query; (b) within-
         engine versus between-engine tau across 300/453/803 K. 95% intervals
         (whiskers in a, bands in b) from 1,000 bootstrap resamples of the
         shared compounds with MD values redrawn within their standard
         errors, recomputed from results/live_values.json and
         results_T/T*/live_values.json; the recomputed point values are
         checked against the stored ones and printed.
  TENKi  live_campaign.png   (a, b) share of the 10 most stable PBE and
         HSE06 compounds found against receiver compute spent (batches of 4,
         mean and 5-95% band over 20 runs with MD-noise draws); (c) MD budget to find 8 of 10
         in each hot receiver, no donor against the same engine at 300 K with learned trust
         (mean and 5-95% whiskers over the same draws).
  ATEKi  live_attribution.png (a, b) Shapley credit of the donors to the PBE
         and HSE06 campaigns, fixed against learned trust; (c) Shapley credit of the
         donors of each hot MD campaign.
Size and fonts: paper_fig_style (true print size).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fig_style import (  # noqa: E402
    RSC_FULL, TMLR_FULL, IOP_FULL, MP_BLUE, JARVIS_GREEN, ACCENT_RED, INK, GREY, EXPT_GOLD,
    SNUMAT_PURPLE, apply_style, save,
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "compute" / "almgsi_campaign"
PAP = ROOT / "writings" / "papers_2026"

ENG = {"lammps_meam_md": "MEAM", "gpumd_nep89_md": "NEP89", "mace_md": "MACE MD",
       "mace_relax": "MACE 0 K", "qe_pbe": "PBE 0 K", "archive_mp": "MP archive"}
# Body names used in the TENKi and ATEKi figures (consistent across the
# ATEKi swarm figures): a model name plus what was run with it.
BODY = {"lammps_meam_md": "MEAM MD", "gpumd_nep89_md": "NEP89 MD",
        "mace_md": "MACE-MP-0 MD", "mace_relax": "MACE-MP-0 relaxation",
        "qe_pbe": "QE PBE", "archive_mp": "MP archive"}
COL = {"lammps_meam_md": MP_BLUE, "gpumd_nep89_md": JARVIS_GREEN, "mace_md": ACCENT_RED,
       "mace_relax": SNUMAT_PURPLE, "qe_pbe": INK, "archive_mp": EXPT_GOLD}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def short(src):
    e, _, t = src.partition("@")
    return f"{ENG.get(e, e)} {t}" if t and t != "0K" else ENG.get(e, e)


def body_t(src, sep=" "):
    """Body name plus temperature, e.g. 'NEP89 MD 453 K'."""
    e, _, t = src.partition("@")
    name = BODY.get(e, e)
    if t and t != "0K":
        return f"{name}{sep}{t[:-1]} K"
    return name


def _ok(r):
    v = r.get("value")
    return r.get("status") == "ok" and v is not None and np.isfinite(v)


def tables_0k():
    """Formation-energy tables of the 0 K sources and 300 K MD (as analyze_live.load_tables)."""
    idx = {r["cid"]: r for r in load(LIVE / "candidates_index.json")}
    live = load(LIVE / "results/live_values.json")
    out = {src: {c: (float(r["value"]), float(r.get("sem") or 0.0)) for c, r in rows.items()
                 if c in idx and idx[c]["property"] == "formation_energy" and _ok(r)}
           for src, rows in live.items()}
    out["archive_mp"] = {c: (float(r["mp_ef"]), 0.0) for c, r in idx.items()
                         if r.get("mp_ef") is not None and r["property"] == "formation_energy"}
    return out


def tables_t():
    """Formation-energy tables of each MD potential at 300, 453 and 803 K (as temperature_study)."""
    idx = {r["cid"]: r for r in load(LIVE / "candidates_index.json")}
    out = {}
    for T in (300, 453, 803):
        for eng, rows in load(LIVE / f"results_T/T{T}/live_values.json").items():
            if eng not in ("lammps_meam_md", "gpumd_nep89_md", "mace_md"):
                continue
            out[f"{eng}@{T}K"] = {c: (float(r["value"]), float(r.get("sem") or 0.0)) for c, r in rows.items()
                                  if not c.startswith("_") and c in idx and _ok(r)
                                  and r.get("property") == "formation_energy"}
    return out


def tau_interval(a, b, n_boot=1000, seed=7):
    """Kendall tau of two tables on their shared compounds and its 95% interval.

    Each replicate resamples the shared compounds with replacement and redraws
    every MD value within its standard error (values without one are kept)."""
    from scipy.stats import kendalltau
    cs = sorted(set(a) & set(b))
    x, sx = np.array([a[c][0] for c in cs]), np.array([a[c][1] for c in cs])
    y, sy = np.array([b[c][0] for c in cs]), np.array([b[c][1] for c in cs])
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(cs), len(cs))
        xv, yv = x + rng.normal(0.0, 1.0, len(cs)) * sx, y + rng.normal(0.0, 1.0, len(cs)) * sy
        reps.append(kendalltau(xv[i], yv[i]).statistic)
    return (float(kendalltau(x, y).statistic), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)), len(cs))


def fig_pegki():
    sa_all = load(LIVE / "results/analysis/source_agreement.json")
    ts = load(LIVE / "results_T/analysis/temperature_study.json")
    t0, tt = tables_0k(), tables_t()
    ivals = {}  # recomputed point, 95% interval, n; checked against the stored point values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(RSC_FULL, 2.85), gridspec_kw={"width_ratios": [1.0, 1.15]})
    order = ["mace_relax", "archive_mp", "lammps_meam_md", "gpumd_nep89_md", "qe_pbe"]
    names = {"mace_relax": "MACE-MP-0\nrelaxation", "archive_mp": "MP\narchive",
             "lammps_meam_md": "MEAM MD\n300 K", "gpumd_nep89_md": "NEP89 MD\n300 K", "qe_pbe": "QE PBE"}
    ys = np.arange(len(order))[::-1]
    h = 0.36
    eb = dict(ecolor=INK, elinewidth=0.7, capsize=1.5)

    def xerr(r, ci):
        return [[r["kendall_tau"] - ci[1]], [ci[2] - r["kendall_tau"]]]
    for y, s in zip(ys, order):
        r_h = sa_all["qe_hse06"][s]
        ci_h = ivals[f"{s}|qe_hse06"] = tau_interval(t0[s], t0["qe_hse06"])
        hi = ci_h[2]
        if s in sa_all["qe_pbe"]:
            r_p = sa_all["qe_pbe"][s]
            ci_p = ivals[f"{s}|qe_pbe"] = tau_interval(t0[s], t0["qe_pbe"])
            a1.barh(y + h / 2, r_p["kendall_tau"], color=COL[s], height=h, xerr=xerr(r_p, ci_p), error_kw=eb)
            hi = max(hi, ci_p[2])
        a1.barh(y - h / 2, r_h["kendall_tau"], color=COL[s], height=h, alpha=0.45, hatch="////", edgecolor=COL[s],
                lw=0, xerr=xerr(r_h, ci_h), error_kw=eb)
        cost = (r_h.get("cost") or {}).get("median_wall_s")
        lab = f"n={r_h['n']}, " + (f"{cost:.1f} s" if cost and cost < 100 else (f"{cost:.0f} s" if cost else "lookup"))
        a1.text(hi + 0.03 if hi >= 0 else 0.03, y, lab, va="center", ha="left", fontsize=6.5, color=INK)
    a1.set_yticks(ys, [names[s] for s in order])
    a1.axvline(0, color=GREY, lw=0.6)
    a1.set_xlim(-0.3, 1.3)
    a1.set_xlabel(r"Kendall $\tau$ with the DFT receiver")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=GREY, label="vs QE PBE"),
                       Patch(facecolor=GREY, alpha=0.45, hatch="////", edgecolor=GREY, label="vs QE HSE06")],
              fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.27), ncol=2)
    a1.set_title("(a) Ordering agreement with DFT", loc="left")

    tau = ts["tau_matrix"]["tau"]
    engines = ["lammps_meam_md", "gpumd_nep89_md", "mace_md"]
    temps = [300, 453, 803]
    x = np.arange(3)
    def band(k1, k2):
        ci = ivals[f"{k1}|{k2}"] = tau_interval(tt[k1], tt[k2])
        return ci[1], ci[2]
    for j, e in enumerate(engines):
        within = [1.0] + [tau.get(f"{e}@300K|{e}@{T}K", tau.get(f"{e}@{T}K|{e}@300K")) for T in temps[1:]]
        lo, hi = zip(*[(1.0, 1.0)] + [band(f"{e}@300K", f"{e}@{T}K") for T in temps[1:]])
        a2.fill_between(x, lo, hi, color=COL[e], alpha=0.18, lw=0)
        a2.plot(x, within, "-o", color=COL[e], ms=3.5, lw=1.2, label=f"{BODY[e]} vs its 300 K run")
    pairs = [("lammps_meam_md", "mace_md"), ("gpumd_nep89_md", "mace_md"), ("lammps_meam_md", "gpumd_nep89_md")]
    for (e1, e2), ls in zip(pairs, ["--", ":", "-."]):
        vals = [tau.get(f"{e1}@{T}K|{e2}@{T}K", tau.get(f"{e2}@{T}K|{e1}@{T}K")) for T in temps]
        lo, hi = zip(*[band(f"{e1}@{T}K", f"{e2}@{T}K") for T in temps])
        a2.fill_between(x, lo, hi, color=GREY, alpha=0.15, lw=0)
        a2.plot(x, vals, ls, color=GREY, lw=1.1, marker="s", ms=2.5, label=f"{BODY[e1]} vs {BODY[e2]}, same T")
    # recomputed point values must match the stored analysis outputs
    for k, (p, lo_, hi_, n) in ivals.items():
        a, b = k.split("|")
        ref = (sa_all[b][a]["kendall_tau"] if b in sa_all else
               tau.get(f"{a}|{b}", tau.get(f"{b}|{a}")))
        print(f"{k:45s} n={n:3d} tau={p:+.3f} stored={ref:+.3f} 95% [{lo_:+.3f}, {hi_:+.3f}]"
              + ("" if abs(p - ref) < 1e-9 else "  MISMATCH"))
    a2.set_xticks(x, ["300 K", "453 K\n(T6 ageing)", "803 K\n(solution\ntreatment)"])
    a2.axhline(0, color=GREY, lw=0.5)
    a2.set_ylim(-0.35, 1.05)
    a2.set_ylabel(r"Kendall $\tau$")
    a2.set_title("(b) Temperature shift versus potential shift", loc="left")
    a2.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.27), ncol=2, handlelength=2.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, PAP / "pegki_digital_discovery/figures/live_transfer.png")


def fig_tenki():
    ts = load(LIVE / "results_T/analysis/temperature_study.json")
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(IOP_FULL, 3.2), gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
    arms = [("surrogate", "No donor (composition surrogate)", GREY, "-"),
            ("fixed:mace_relax", "MACE-MP-0 relaxation donor, fixed trust", SNUMAT_PURPLE, "-"),
            ("fixed:qe_pbe", "PBE donor, fixed trust", INK, "-"),
            ("fixed:gpumd_nep89_md", "NEP89 MD donor, fixed trust", JARVIS_GREEN, "--"),
            ("learned:gpumd_nep89_md", "NEP89 MD donor, learned trust", JARVIS_GREEN, "-")]
    for ax, rec_name, tag in [(a0, "PBE", "qe_pbe_all"), (a1, "HSE06", "qe_hse06_all")]:
        s = load(LIVE / f"results/campaigns/{tag}/summary.json")
        for key, lab, c, ls in arms:
            b = s["recall_bands"].get(f"{key}|k4")
            if b is None:
                continue
            xf = np.array(b["cost_frac"]) * 100
            ax.plot(xf, b["mean"], ls, color=c, lw=1.2, label=lab)
            ax.fill_between(xf, b["p05"], b["p95"], color=c, alpha=0.12, lw=0)
        ax.axhline(0.8, color=INK, ls=":", lw=0.8)
        ax.text(62, 0.785, "eight of ten", ha="center", va="top", fontsize=7, color=INK)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"{rec_name} compute (% of pool)")
    a0.set_ylabel("Share of the 10 most stable found")
    a0.set_title("(a) PBE receiver", loc="left")
    a1.set_title("(b) HSE06 receiver", loc="left")
    a1.set_yticklabels([])
    h_, l_ = a1.get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0.02, -0.01))
    rec = ["lammps_meam_md@453K", "lammps_meam_md@803K", "gpumd_nep89_md@453K", "gpumd_nep89_md@803K",
           "mace_md@453K", "mace_md@803K"]
    r = ts["routing"]
    y = np.arange(len(rec))[::-1]
    # no donor and same potential at 300 K with learned trust (read from the arms, not from the
    # PEGKi pick); whiskers: 5-95% range over the 20 runs with MD-noise draws
    tk = ts["tenki"]
    cm = [tk[k]["arms"]["surrogate"]["cost_frac_to_8of10"] for k in rec]
    pm = [tk[k]["arms"]["learned:" + k.split("@")[0] + "@300K"]["cost_frac_to_8of10"] for k in rec]

    def err(ms):
        return [[100 * (m["mean"] - m["p05"]) for m in ms], [100 * (m["p95"] - m["mean"]) for m in ms]]
    eb = dict(ecolor=INK, elinewidth=0.6, capsize=1.5)
    a2.barh(y + 0.19, [100 * m["mean"] for m in cm], height=0.36, color=GREY, label="No donor",
            xerr=err(cm), error_kw=eb)
    a2.barh(y - 0.19, [100 * m["mean"] for m in pm], height=0.36, color=ACCENT_RED,
            label="Same potential at 300 K,\nlearned trust", xerr=err(pm), error_kw=eb)
    a2.set_yticks(y, [body_t(k, "\n") for k in rec], fontsize=7)
    a2.set_ylim(-0.6, len(rec) - 0.4)
    a2.set_xlabel("MD budget (%)")
    a2.set_xlim(0, 60)
    a2.set_title("(c) MD at 453 and 803 K", loc="right")
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h2, l2, fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.01),
               title="(c)", title_fontsize=7)
    fig.set_layout_engine(None)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.17, 1, 1))
    save(fig, PAP / "tenki_mlst/figures/live_campaign.png")


def fig_ateki():
    # Panels (a) and (b) of the earlier version (PBE and HSE06 games, two cost
    # accountings) moved to the main-text figure compute_credit.png
    # (regen_figs/ateki_compute_credit.py, with run intervals); this annex
    # figure keeps the molecular-dynamics receivers at temperature.
    # One small panel per hot receiver; bars = Shapley credit of each donor
    # role, dark = fixed trust, light = learned trust (as in compute_credit);
    # whiskers = 90% percentile-bootstrap interval of the mean over the 20
    # runs (MD values redrawn within their standard errors per run, file
    # coalitions_sem.json written by temperature_coalitions.py
    # --resample-sem); the learned-trust means equal temperature_study.json.
    from itertools import combinations
    from math import factorial
    ts = load(LIVE / "results_T/analysis/temperature_study.json")["ateki"]
    rec = ["lammps_meam_md@453K", "lammps_meam_md@803K", "gpumd_nep89_md@453K", "gpumd_nep89_md@803K",
           "mace_md@453K", "mace_md@803K"]
    roles = [("same potential\nat 300 K", lambda r, p: p == r.split("@")[0] + "@300K", ACCENT_RED),
             ("other two potentials\nat target T",
              lambda r, p: p.endswith("@" + r.split("@")[1]) and p != r, MP_BLUE),
             ("MACE-MP-0\nrelaxation", lambda r, p: p == "mace_relax@0K", SNUMAT_PURPLE),
             ("QE PBE", lambda r, p: p == "qe_pbe@0K", INK),
             ("QE HSE06", lambda r, p: p == "qe_hse06@0K", EXPT_GOLD)]

    def per_run_shapley(game):
        players = game["players"]
        vals = {frozenset(k.split("+")) if k else frozenset(): np.asarray(v)
                for k, v in game["per_run"].items()}
        n = len(players)
        out = {}
        for p in players:
            others = [q for q in players if q != p]
            s = 0.0
            for kk in range(n):
                w = factorial(kk) * factorial(n - kk - 1) / factorial(n)
                for c in combinations(others, kk):
                    s = s + w * (vals[frozenset(c) | {p}] - vals[frozenset(c)])
            out[p] = s
        return out  # player -> array over runs

    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 3, figsize=(TMLR_FULL, 4.2), sharex=True, sharey=True)
    maxdev = 0.0
    for ax, r in zip(axs.flat, rec):
        g = load(LIVE / "results_T/analysis/coalitions" / r.replace("@", "_") / "coalitions_sem.json")
        y = np.arange(len(roles))
        for j, (mode, light) in enumerate((("pooled_fixed", False), ("pooled_learned", True))):
            sh = per_run_shapley(g[mode])
            if mode == "pooled_learned":
                maxdev = max(maxdev, max(abs(sh[p].mean() - ts[r]["shapley"][p]) for p in sh))
            nrun = len(next(iter(sh.values())))
            idx = rng.integers(0, nrun, (4000, nrun))
            for i, (lab, f, c) in enumerate(roles):
                runs = sum(v for p, v in sh.items() if f(r, p))
                m = runs.mean()
                lo, hi = np.percentile(runs[idx].mean(1), [5, 95])
                yy = y[i] + (j - 0.5) * 0.4
                ax.barh(yy, m, height=0.37, color=c, alpha=0.42 if light else 1.0, lw=0)
                ax.plot([lo, hi], [yy, yy], color=INK, lw=0.8)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(y, [lab for lab, _, _ in roles], fontsize=7)
        ax.invert_yaxis() if ax is axs.flat[0] else None
        ax.grid(axis="y", visible=False)
        ax.set_title(f"({'abcdef'[rec.index(r)]}) {body_t(r)}", loc="left", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("Shapley credit\n(share of ten most stable)")
    print("F.1 learned-trust means vs temperature_study.json, max |diff|:", maxdev)
    from matplotlib.patches import Patch
    hdl = [Patch(facecolor="#555555", label="fixed trust"),
           Patch(facecolor="#555555", alpha=0.42, label="learned trust"),
           plt.Line2D([], [], color=INK, lw=0.8, label=f"90% interval over {nrun} runs")]
    fig.legend(handles=hdl, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    save(fig, PAP / "ateki_tmlr/figures/live_attribution.png")


if __name__ == "__main__":
    apply_style()
    # draw only the figures whose paper folder is present (the per-paper
    # public folders hold one paper each)
    # optional: python live_almgsi_figs.py ateki  (draw one paper only)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for folder, draw in (("pegki_digital_discovery", fig_pegki), ("tenki_mlst", fig_tenki),
                         ("ateki_tmlr", fig_ateki)):
        if only and not folder.startswith(only):
            continue
        if (PAP / folder / "figures").is_dir():
            draw()
