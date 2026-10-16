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
         engine versus between-engine tau across 300/453/803 K.
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


def fig_pegki():
    sa_all = load(LIVE / "results/analysis/source_agreement.json")
    ts = load(LIVE / "results_T/analysis/temperature_study.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(RSC_FULL, 2.85), gridspec_kw={"width_ratios": [1.0, 1.15]})
    order = ["mace_relax", "archive_mp", "lammps_meam_md", "gpumd_nep89_md", "qe_pbe"]
    names = {"mace_relax": "MACE-MP-0\nrelaxation", "archive_mp": "MP\narchive",
             "lammps_meam_md": "MEAM MD\n300 K", "gpumd_nep89_md": "NEP89 MD\n300 K", "qe_pbe": "QE PBE"}
    ys = np.arange(len(order))[::-1]
    h = 0.36
    for y, s in zip(ys, order):
        r_h = sa_all["qe_hse06"][s]
        if s in sa_all["qe_pbe"]:
            r_p = sa_all["qe_pbe"][s]
            a1.barh(y + h / 2, r_p["kendall_tau"], color=COL[s], height=h)
        a1.barh(y - h / 2, r_h["kendall_tau"], color=COL[s], height=h, alpha=0.45, hatch="////", edgecolor=COL[s], lw=0)
        cost = (r_h.get("cost") or {}).get("median_wall_s")
        lab = f"n={r_h['n']}, " + (f"{cost:.1f} s" if cost and cost < 100 else (f"{cost:.0f} s" if cost else "lookup"))
        x = max(r_h["kendall_tau"], sa_all["qe_pbe"].get(s, {}).get("kendall_tau", -1))
        a1.text(x + 0.03 if x >= 0 else 0.03, y, lab, va="center", ha="left", fontsize=6.5, color=INK)
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
    for j, e in enumerate(engines):
        within = [1.0] + [tau.get(f"{e}@300K|{e}@{T}K", tau.get(f"{e}@{T}K|{e}@300K")) for T in temps[1:]]
        a2.plot(x, within, "-o", color=COL[e], ms=3.5, lw=1.2, label=f"{BODY[e]} vs its 300 K run")
    pairs = [("lammps_meam_md", "mace_md"), ("gpumd_nep89_md", "mace_md"), ("lammps_meam_md", "gpumd_nep89_md")]
    for (e1, e2), ls in zip(pairs, ["--", ":", "-."]):
        vals = [tau.get(f"{e1}@{T}K|{e2}@{T}K", tau.get(f"{e2}@{T}K|{e1}@{T}K")) for T in temps]
        a2.plot(x, vals, ls, color=GREY, lw=1.1, marker="s", ms=2.5, label=f"{BODY[e1]} vs {BODY[e2]}, same T")
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
    ts = load(LIVE / "results_T/analysis/temperature_study.json")["ateki"]
    fig, a2 = plt.subplots(figsize=(TMLR_FULL * 0.75, 2.9))
    rec = ["lammps_meam_md@453K", "lammps_meam_md@803K", "gpumd_nep89_md@453K", "gpumd_nep89_md@803K",
           "mace_md@453K", "mace_md@803K"]
    roles = [("same potential at 300 K", lambda r, p: p == r.split("@")[0] + "@300K", ACCENT_RED),
             ("other two potentials at the target temperature",
              lambda r, p: p.endswith("@" + r.split("@")[1]) and p != r, MP_BLUE),
             ("MACE-MP-0 relaxation", lambda r, p: p == "mace_relax@0K", SNUMAT_PURPLE),
             ("QE PBE", lambda r, p: p == "qe_pbe@0K", INK),
             ("QE HSE06", lambda r, p: p == "qe_hse06@0K", EXPT_GOLD)]
    xr = np.arange(len(rec))
    w = 0.16
    for i, (lab, f, c) in enumerate(roles):
        vals = [sum(v for p, v in ts[r]["shapley"].items() if f(r, p)) for r in rec]
        a2.bar(xr + (i - 2) * w, vals, width=w, color=c, label=lab)
    a2.set_xticks(xr, [body_t(r, "\n") for r in rec], fontsize=7)
    a2.set_xlabel("Receiver (MD potential and temperature)")
    a2.set_ylabel("Shapley credit\n(share of ten most stable)")
    a2.axhline(0, color=INK, lw=0.6)
    a2.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    lo, hi = a2.get_ylim()
    a2.set_ylim(lo, hi + 0.16)
    save(fig, PAP / "ateki_tmlr/figures/live_attribution.png")


if __name__ == "__main__":
    apply_style()
    # draw only the figures whose paper folder is present (the per-paper
    # public folders hold one paper each)
    for folder, draw in (("pegki_digital_discovery", fig_pegki), ("tenki_mlst", fig_tenki),
                         ("ateki_tmlr", fig_ateki)):
        if (PAP / folder / "figures").is_dir():
            draw()
