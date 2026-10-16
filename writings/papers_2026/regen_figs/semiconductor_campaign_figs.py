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


def curves(ax, s, arms, title, xlabel):
    grid = np.linspace(0, 1, 201)
    for arm, lab, c, ls in arms:
        b = s["recall_bands"].get(f"{arm}|k4")
        if b is None:
            continue
        ax.plot(grid, b["mean"], color=c, ls=ls, label=lab)
        ax.fill_between(grid, b["p05"], b["p95"], color=c, alpha=0.12, lw=0)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ten widest gaps found (share)")
    ax.set_title(title, loc="left")


def best_archive(s):
    a = {d: v for d, v in s["agreement"].items() if d not in ("qe_pbe", "qe_hse06", "expt_zhuo")}
    return max(a, key=lambda d: s["arms"][f"learned:{d}|k4"]["early_recall_area"]["mean"])


def fig_campaign(out):
    hse = load("qe_hse06_nohalide")
    exp = load("expt_zhuo_nohalide")
    fig, axs = plt.subplots(1, 3, figsize=(st.IOP_FULL, 3.0), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axs[0]
    donors = [d for d in ("qe_pbe", "mp_gga", "mp_r2scan", "jarvis_optb88vdw", "jarvis_mbj", "snumat_gga",
                          "snumat_hse", "qe_hse06", "expt_zhuo") if d in hse["agreement"] or d in exp["agreement"]]
    y = np.arange(len(donors))
    th = [hse["agreement"].get(d, {}).get("tau", np.nan) for d in donors]
    te = [exp["agreement"].get(d, {}).get("tau", np.nan) for d in donors]
    ax.barh(y - 0.19, th, 0.36, color=[COL.get(d, st.GREY) for d in donors], label="to HSE06")
    ax.barh(y + 0.19, te, 0.36, color=[COL.get(d, st.GREY) for d in donors], alpha=0.45, label="to measured")
    ax.set_yticks(y, [NAMES.get(d, d) for d in donors])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Kendall $\\tau$ with receiver")
    ax.set_title("(a) Agreement", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.34), ncol=2)
    ba = best_archive(hse)
    curves(axs[1], hse, [("surrogate", "No donor", st.GREY, "--"),
                         ("fixed:qe_pbe", "QE PBE, fixed", st.ACCENT_RED, "-"),
                         (f"learned:{ba}", f"{NAMES[ba]}, learned", COL[ba], "-"),
                         ("pooled_learned", "All donors, learned", st.INK, ":")],
           "(b) HSE06 receiver", "HSE06 compute spent\n(share)")
    ba2 = best_archive(exp)
    curves(axs[2], exp, [("surrogate", "No donor", st.GREY, "--"),
                         ("learned:qe_hse06", "QE HSE06, learned", "#E08E45", "-"),
                         ("learned:qe_pbe", "QE PBE, learned", st.ACCENT_RED, "-"),
                         (f"learned:{ba2}", f"{NAMES[ba2]}, learned", COL[ba2], "-")],
           "(c) Measured gaps", "Compounds looked up\n(share)")
    axs[2].set_ylabel("")
    axs[2].set_title("(c) Measured gaps", loc="left")
    for a in axs[1:]:
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=1, handlelength=1.6)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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
