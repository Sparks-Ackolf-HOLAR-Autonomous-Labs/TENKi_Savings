"""Summarise TENKi exp 30 (exclusion-list transfer) and grade the
registered predictions P1 and P2 (PREDICTION.md). ASCII only."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "exclusion_list_transfer.json", encoding="utf-8"))
rows = d["rows"]
CONDS = ["full_fixed", "full_learned", "excl_fixed", "excl_learned",
         "excl_acq", "top_fixed", "top_learned"]
LABEL = {"full_fixed": "full ordering, fixed trust",
         "full_learned": "full ordering, learned trust",
         "excl_fixed": "exclusion list, fixed trust",
         "excl_learned": "exclusion list, learned trust",
         "excl_acq": "exclusion list, not queried",
         "top_fixed": "top list, fixed trust",
         "top_learned": "top list, learned trust"}
GGA = ["mp_gga", "mp_gga_expsynth", "jarvis_optb88vdw", "snumat_gga"]
EXP = "expt_zhuo"


def agg(sel, key):
    v = np.asarray([r[key] for r in sel])
    if len(v) < 2:
        return float(v.mean()), [float(v.mean())] * 2
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return float(v.mean()), [float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))]


table = []
for arm in ["kway", "pair"]:
    for scope, filt in [("all receivers", lambda r: True),
                        ("experimental receiver, GGA-class donors",
                         lambda r: r["receiver"] == EXP
                         and r["donor"] in GGA),
                        ("computed receivers", lambda r: r["receiver"] != EXP)]:
        for c in CONDS:
            sel = [r for r in rows if r["arm"] == arm and r["condition"] == c
                   and filt(r)]
            row = {"arm": arm, "scope": scope, "condition": c, "n": len(sel)}
            for key in ["tau_early_delta", "tau_final_delta",
                        "top3_early_delta", "top3_final_delta"]:
                row[key], row[key + "_ci"] = agg(sel, key)
            table.append(row)


def get(arm, scope, c, key):
    return next(t[key] for t in table if t["arm"] == arm
                and t["scope"] == scope and t["condition"] == c)


sc = "experimental receiver, GGA-class donors"
p1 = {"excl_fixed": get("kway", sc, "excl_fixed", "tau_early_delta"),
      "full_fixed": get("kway", sc, "full_fixed", "tau_early_delta")}
p1["pass"] = p1["excl_fixed"] > p1["full_fixed"]
p2_vals = [get(a, "all receivers", "excl_acq", "tau_final_delta")
           for a in ["kway", "pair"]]
p2 = {"kway": p2_vals[0], "pair": p2_vals[1],
      "pass": all(v >= 0 for v in p2_vals)}
out = {"table": table, "P1": p1, "P2": p2,
       "donor_lists": d["donor_lists"],
       "reference_rankings": d["reference_rankings"]}
json.dump(out, open(HERE / "exclusion_summary.json", "w", encoding="utf-8"),
          indent=1)

L = ["# Exclusion-list transfer: summary", "",
     "Deltas are paired against the cold campaign at the same seed "
     "(Kendall tau; top-3 = share of the receiver's three widest-gap "
     "families in the fitted top three). Early = mean over the first half "
     "of the budget checkpoints. Brackets: 95% bootstrap interval over "
     "receiver-donor pairs.", "",
     f"P1 (experimental receiver, GGA-class donors, k-way): exclusion list "
     f"{p1['excl_fixed']:+.3f} vs full ordering {p1['full_fixed']:+.3f} "
     f"early tau delta at fixed trust: **{'PASS' if p1['pass'] else 'FAIL'}**",
     "", f"P2 (exclusion applied to acquisition, final tau delta over all "
     f"receivers): k-way {p2['kway']:+.3f}, pairs {p2['pair']:+.3f}: "
     f"**{'PASS' if p2['pass'] else 'FAIL'}**", ""]
for arm in ["kway", "pair"]:
    for scope in ["all receivers", sc, "computed receivers"]:
        L += [f"## {arm}, {scope}", "",
              "| condition | n | early tau delta | final tau delta | "
              "early top-3 delta | final top-3 delta |",
              "| --- | ---: | --- | --- | --- | --- |"]
        for t in table:
            if t["arm"] != arm or t["scope"] != scope:
                continue
            f = lambda k: (f"{t[k]:+.3f} [{t[k + '_ci'][0]:+.3f}, "
                           f"{t[k + '_ci'][1]:+.3f}]")
            L.append(f"| {LABEL[t['condition']]} | {t['n']} | "
                     f"{f('tau_early_delta')} | {f('tau_final_delta')} | "
                     f"{f('top3_early_delta')} | {f('top3_final_delta')} |")
        L.append("")
open(HERE / "exclusion_summary.md", "w", encoding="utf-8").write(
    "\n".join(L) + "\n")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "writings" / "papers_2026"))
from paper_fig_style import IOP_FULL, MP_BLUE, JARVIS_GREEN, ACCENT_RED, GREY, INK, apply_style, save  # noqa: E402
PAPER_FIG = Path(__file__).resolve().parents[3] / "writings" / "papers_2026" / "tenki_mlst" / "figures"
apply_style()
fig, axes = plt.subplots(1, 2, figsize=(IOP_FULL, 2.8), sharey=True)
for ax, scope, title in [(axes[0], "computed receivers",
                          "(a) computed-gap receivers"),
                         (axes[1], sc,
                          "(b) measured gaps, GGA donors")]:
    for j, arm in enumerate(["kway", "pair"]):
        ys = np.arange(len(CONDS)) + (0.18 if j else -0.18)
        T = [next(t for t in table if t["arm"] == arm and t["scope"] == scope
                  and t["condition"] == c) for c in CONDS]
        m = [t["tau_early_delta"] for t in T]
        lo = [t["tau_early_delta"] - t["tau_early_delta_ci"][0] for t in T]
        hi = [t["tau_early_delta_ci"][1] - t["tau_early_delta"] for t in T]
        ax.errorbar(m, ys, xerr=[lo, hi], fmt="o", ms=3,
                    color=MP_BLUE if j == 0 else GREY,
                    label="4-way queries" if j == 0 else "pairs", capsize=2)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("early tau change vs no donor")
axes[0].set_yticks(range(len(CONDS)))
axes[0].set_yticklabels([LABEL[c] for c in CONDS], fontsize=7)
axes[0].invert_yaxis()
axes[1].legend(frameon=False, fontsize=7)
fig.savefig(HERE / "fig_exclusion.png", dpi=200)
save(fig, PAPER_FIG / "exclusion_list.png")
print("\n".join(L[:8]))
