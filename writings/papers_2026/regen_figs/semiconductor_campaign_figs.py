"""Figures for the computed semiconductor band-gap campaign.

Reads only compute/semiconductor_campaign/results/campaigns/*/summary.json.

* semiconductor_campaign.png (TENKi, full width): (a) Kendall tau of each
  donor against the HSE06 receiver and the measured receiver; (b) recall of
  the 10 widest HSE06 gaps (no-halide pool) against the HSE06 compute spent,
  mean with 5-95 % band over 20 seeds, k = 4; (c) the same for the measured
  receiver (one unit per compound looked up).
* semiconductor_attribution.png (ATEKi): Shapley credit of the donors in the
  HSE06 campaign, fixed against learned trust.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper_fig_style as st  # noqa: E402

st.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SO = HERE.parents[2]
CAMP = SO / "compute" / "semiconductor_campaign" / "results" / "campaigns"
NAMES = dict(st.SOURCE_NAMES, qe_pbe="QE PBE (computed)", qe_hse06="QE HSE06 (computed)",
             expt_zhuo="Measured gap")
COL = dict(st.SOURCE_COLOURS, qe_pbe=st.ACCENT_RED, qe_hse06="#E08E45")


def load(tag):
    return json.loads((CAMP / tag / "summary.json").read_text())


# Fig 8 (TENKi) names and colours: computed sources in dark ink tones, archives by database
FIG_NAMES = dict(st.SOURCE_NAMES, qe_pbe="Computed PBE", qe_hse06="Computed HSE06",
                 expt_zhuo="Measured (Zhuo)")
FIG_COL = dict(st.SOURCE_COLOURS, qe_pbe=st.INK, qe_hse06="#6B3A1F")
POOL_COL = st.ACCENT_RED
RAND_COL = "#B5B5B5"


def pct(x):
    """Budget share as printed in the text: one decimal below 10 %, whole percent above."""
    v = 100 * x
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def c8(s, arm):
    return s["arms"][f"{arm}|k4"]["cost_frac_to_8of10"]["mean"]


def curves(ax, s, arms, title, xlabel, hits=None):
    """arms: (arm, label, colour, linestyle); label gets the mean budget to 8 of 10 appended."""
    grid = np.linspace(0, 1, 201)
    for arm, lab, c, ls in arms:
        b = s["recall_bands"].get(f"{arm}|k4")
        if b is None:
            continue
        mean = np.array(b["mean"])
        ax.plot(grid, mean, color=c, ls=ls, lw=1.2, label=f"{lab} ({pct(c8(s, arm))})")
        ax.fill_between(grid, b["p05"], b["p95"], color=c, alpha=0.10, lw=0)
        if hits and arm in hits["arms"]:
            xs = np.array([v["median"] for v in hits["arms"][arm].values()])
            ys = np.interp(xs, grid, mean)
            ax.plot(xs, ys, ls="none", marker="o", ms=3.4, mew=0.9, mec=c, mfc="white", zorder=5)
    ax.axhline(0.8, color=st.INK, ls=":", lw=0.8)
    ax.text(0.99, 0.79, "eight of ten", ha="right", va="top", fontsize=7, color=st.INK)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ten widest gaps found (share)")
    ax.set_title(title, loc="left")


def fig_campaign(out):
    hse = load("qe_hse06_nohalide")
    exp = load("expt_zhuo_nohalide")
    hp = CAMP / "qe_hse06_nohalide" / "alloy_hits.json"
    hits = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else None
    fig, axs = plt.subplots(1, 3, figsize=(st.IOP_FULL, 4.25), gridspec_kw={"width_ratios": [1, 1, 1]},
                            constrained_layout=False)
    fig.subplots_adjust(left=0.175, right=0.975, top=0.94, bottom=0.55, wspace=0.36)
    ax = axs[0]
    donors = [d for d in ("qe_pbe", "qe_hse06", "mp_gga", "mp_r2scan", "jarvis_optb88vdw", "jarvis_mbj",
                          "snumat_gga", "snumat_hse", "expt_zhuo")
              if d in hse["agreement"] or d in exp["agreement"] or d == "qe_hse06"]
    y = np.arange(len(donors))
    th = [hse["agreement"].get(d, {}).get("tau", np.nan) for d in donors]
    te = [exp["agreement"].get(d, {}).get("tau", np.nan) for d in donors]
    cols = [FIG_COL.get(d, st.GREY) for d in donors]
    ax.barh(y - 0.19, th, 0.36, color=cols)
    ax.barh(y + 0.19, te, 0.36, color=cols, alpha=0.45)
    for yi, a, b in zip(y, th, te):
        if not np.isfinite(a):
            ax.text(0.02, yi - 0.19, "self", va="center", ha="left", fontsize=7, color=st.INK)
        if not np.isfinite(b):
            ax.text(0.02, yi + 0.19, "self", va="center", ha="left", fontsize=7, color=st.INK)
    ax.set_yticks(y, [FIG_NAMES.get(d, d) for d in donors])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("Kendall $\\tau$ with receiver")
    ax.set_title("(a) Agreement", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=st.INK, label="with computed HSE06"),
                       Patch(color=st.INK, alpha=0.45, label="with measured gap")],
              loc="upper right", bbox_to_anchor=(1.0, -0.31), ncol=1)
    curves(axs[1], hse, [("random", "Random choice", RAND_COL, ":"),
                         ("surrogate", "No donor (surrogate)", st.GREY, "--"),
                         ("fixed:qe_pbe", "Computed PBE, fixed", FIG_COL["qe_pbe"], "-"),
                         ("learned:qe_pbe", "Computed PBE, learned", FIG_COL["qe_pbe"], "--"),
                         ("fixed:mp_gga", "MP GGA, fixed", FIG_COL["mp_gga"], "-"),
                         ("learned:mp_gga", "MP GGA, learned", FIG_COL["mp_gga"], "--"),
                         ("pooled_fixed", "All donors, fixed", POOL_COL, "-"),
                         ("pooled_learned", "All donors, learned", POOL_COL, "--")],
           "(b) HSE06 receiver", "HSE06 compute spent\n(share)", hits=hits)
    curves(axs[2], exp, [("surrogate", "No donor", st.GREY, "--"),
                         ("fixed:qe_hse06", "Computed HSE06, fixed", FIG_COL["qe_hse06"], "-"),
                         ("fixed:qe_pbe", "Computed PBE, fixed", FIG_COL["qe_pbe"], (0, (4, 1.5))),
                         ("fixed:mp_gga", "MP GGA, fixed", FIG_COL["mp_gga"], (0, (1.5, 1))),
                         ("fixed:jarvis_mbj", "JARVIS TB-mBJ, fixed", FIG_COL["jarvis_mbj"], "-")],
           "(c) Measured gaps", "Compounds looked up\n(share)")
    axs[2].set_ylabel("")
    h1, l1 = axs[1].get_legend_handles_labels()
    if hits:
        from matplotlib.lines import Line2D
        h1.append(Line2D([], [], ls="none", marker="o", ms=3.4, mew=0.9, mec=st.INK, mfc="white"))
        l1.append("New MgCaO$_2$ and ZnBeO$_2$\norderings computed\n(median run)")
    for a, (h, l), bx in zip(axs[1:], [(h1, l1), axs[2].get_legend_handles_labels()], (0.36, 0.42)):
        a.legend(h, l, loc="upper center", bbox_to_anchor=(bx, -0.31), ncol=1, handlelength=2.0,
                 title="Budget to find eight of ten", title_fontsize=7)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white")   # true print size: exactly IOP_FULL wide
    plt.close(fig)
    print("wrote", out)


def fig_attribution(out):
    s = load("qe_hse06_nohalide")
    sh = s["shapley"]
    names = list(sh["pooled_fixed"]["shapley"])
    fig, ax = plt.subplots(figsize=(st.TMLR_FULL * 0.55, 2.2))
    y = np.arange(len(names))
    ax.barh(y - 0.19, [sh["pooled_fixed"]["shapley"][n] for n in names], 0.36,
            color=[COL.get(n, st.GREY) for n in names], label="fixed trust")
    ax.barh(y + 0.19, [sh["pooled_learned"]["shapley"][n] for n in names], 0.36,
            color=[COL.get(n, st.GREY) for n in names], alpha=0.45, label="learned trust")
    ax.axvline(0, color=st.INK, lw=0.6)
    from matplotlib.patches import Patch
    hl = [Patch(color=st.INK, label="fixed trust"), Patch(color=st.INK, alpha=0.45, label="learned trust")]
    ax.set_yticks(y, [NAMES.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel("Shapley credit (early recall)")
    ax.legend(handles=hl, loc="lower right")
    st.save(fig, out)


if __name__ == "__main__":
    P = HERE.parent
    fig_campaign(P / "tenki_mlst" / "figures" / "semiconductor_campaign.png")
    if (CAMP / "qe_hse06_nohalide" / "summary.json").exists() and "shapley" in load("qe_hse06_nohalide"):
        fig_attribution(P / "ateki_tmlr" / "figures" / "semiconductor_attribution.png")
