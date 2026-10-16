#!/usr/bin/env python
"""Rerun the committed Al-Mg-Si campaigns of campaign.py with MD-noise draws.

Rebuilds each campaign's command from its committed summary.json (receiver,
donors, pool, cost model, k values, seeds, --shapley), adds the --extra
tables the temperature-dependent receivers need, and reruns it.

    python rerun_md_noise.py --check   # rerun with --no-md-noise into <tag>_check and
                                       # compare with the committed summary (must match)
    python rerun_md_noise.py           # rerun with MD noise into the committed tags
Run with an env that has ase (the campaign pool reads the candidate structures).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = HERE / "results" / "campaigns"


def extras(names):
    ex = []
    for n in names:
        if n == "mace_md_300":
            ex.append("results_T/T300/live_values.json:mace_md=mace_md_300")
        elif n.startswith("qha_"):
            ex.append(f"results/qha/qha_T{n[4:]}.json={n}")
        else:
            for pre, key in (("mace_md_", "mace_md"), ("meam_md_", "lammps_meam_md"), ("nep89_md_", "gpumd_nep89_md")):
                if n.startswith(pre) and n[len(pre):].isdigit():
                    ex.append(f"results_T/T{n[len(pre):]}/live_values.json:{key}={n}")
    return ex


def command(tag, check):
    s = json.loads((CAMP / tag / "summary.json").read_text(encoding="utf-8"))
    donors = list(s["agreement"])
    cmd = [sys.executable, str(HERE / "campaign.py"), "--receiver", s["receiver"], "--donors", ",".join(donors),
           "--k", ",".join(map(str, s["ks"])), "--seeds", str(s["seeds"]), "--pool", s["pool"],
           "--tag", tag + ("_check" if check else "")]
    if s.get("cost") == "combined":
        cmd += ["--cost", "combined"]
    if s["shapley"] or tag.endswith("_coalitions"):
        cmd.append("--shapley")
    for e in extras([s["receiver"]] + donors):
        cmd += ["--extra", e]
    if check:
        cmd.append("--no-md-noise")
    return cmd


def compare(tag):
    a = json.loads((CAMP / tag / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((CAMP / (tag + "_check") / "summary.json").read_text(encoding="utf-8"))
    bad = [k for k in a["arms"] if abs(a["arms"][k]["early_recall_area"]["mean"]
                                        - b["arms"].get(k, {}).get("early_recall_area", {}).get("mean", 9)) > 1e-12]
    for m, sh in a.get("shapley", {}).items():
        for p, v in sh["shapley"].items():
            if abs(v - b["shapley"][m]["shapley"][p]) > 1e-12:
                bad.append(f"shapley:{m}:{p}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tags", default="")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    tags = args.tags.split(",") if args.tags else sorted(
        p.name for p in CAMP.iterdir() if (p / "summary.json").exists() and not p.name.endswith("_check"))

    def go(tag):
        cmd = command(tag, args.check)
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if r.returncode:
            return tag, "FAIL " + r.stderr[-400:]
        if args.check:
            bad = compare(tag)
            shutil.rmtree(CAMP / (tag + "_check"))
            return tag, "reproduced" if not bad else f"MISMATCH {bad[:5]}"
        return tag, "done"

    with ThreadPoolExecutor(args.jobs) as ex:
        for tag, msg in ex.map(go, tags):
            print(f"[{tag}] {msg}", flush=True)


if __name__ == "__main__":
    main()
