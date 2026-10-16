"""Compare the on-demand live campaign with its replay.

Both runs use the same arms, seed and decision code (live_campaign.py). The
live run computes MACE-MP-0 MD at 300 K on the DGX the first time a candidate
is asked for; the replay answers from the completed table of the same receiver
computed earlier on CHPC guest GPUs (results_T/T300). Differences come only
from MD sampling noise and hardware.

Writes results_live/live_vs_replay.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
L = json.loads((HERE / "results_live/live_campaign.json").read_text(encoding="utf-8"))
R = json.loads((HERE / "results_live/replay_campaign.json").read_text(encoding="utf-8"))
T = json.loads((HERE / "results_T/T300/live_values.json").read_text(encoding="utf-8"))["mace_md"]
idx = {r["cid"]: r for r in json.loads((HERE / "candidates_index.json").read_text(encoding="utf-8"))}
pool = [c for c in T if c in idx and idx[c]["property"] == "formation_energy" and T[c].get("status") == "ok"]
ref = sorted(pool, key=lambda c: T[c]["value"])
top10 = set(ref[:10])
out = {"reference_top10": ref[:10], "arms": {}}
for a in L["arms"]:
    l, r = L["arms"][a], R["arms"][a]
    le, re_ = set(l["evaluated"]), set(r["evaluated"])
    div = next((i for i, (x, y) in enumerate(zip(l["steps"], r["steps"])) if set(x["batch"]) != set(y["batch"])), None)
    out["arms"][a] = {"n_eval": len(le), "live_cost_s": l["steps"][-1]["cost_s"], "replay_cost_s": r["steps"][-1]["cost_s"],
                      "jaccard_evaluated": len(le & re_) / len(le | re_), "first_divergent_batch": div,
                      "n_batches": len(l["steps"]), "top10_found_live": len(le & top10),
                      "top10_found_replay": len(re_ & top10),
                      "final_w_live": l["steps"][-1]["w"], "final_w_replay": r["steps"][-1]["w"]}
lv = {}
for a in L["arms"]:
    lv.update(L["arms"][a]["evaluated"])
sh = sorted(set(lv) & set(pool))
x = np.array([lv[c] for c in sh])
y = np.array([T[c]["value"] for c in sh])
out["value_agreement"] = {"n": len(sh), "kendall_tau": float(kendalltau(x, y).statistic),
                          "mad_eV_atom": float(np.mean(np.abs(x - y))),
                          "median_sem_replay_eV_atom": float(np.median([T[c].get("sem") or 0.0 for c in sh]))}
log = L.get("computed_log", [])
out["live_computed"] = {"n": len(log), "total_wall_s": float(sum(e["wall_s"] for e in log))}
(HERE / "results_live/live_vs_replay.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
