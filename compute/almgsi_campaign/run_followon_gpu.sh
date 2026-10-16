#!/usr/bin/env bash
# Follow-on GPU chain, launched detached while the bulk mace_md chain runs.
#  1. wait for mace_md.DONE (the bulk mace_md chain is not interrupted) and
#     for any live_campaign.py process to exit (it shares the GPU)
#  2. mace_md clusters: move the 3 ps cluster JSONs to results/mace_md/_short/
#     and rerun the 13 cluster cids with 20 ps sampling -> mace_md_clusters.DONE
#  3. clean-timing reruns: mace_md queries whose run overlapped the GPU
#     contention window (GPUMD reruns on the shared GPU), and every GPUMD record
#     flagged settings.gpu_contended, are moved to <source>/_contended/ and
#     rerun alone on the GPU with identical settings -> gpu_reruns.DONE
# PYTHON must have ase and mace-torch; VENV_LIB (optional) is prepended to
# LD_LIBRARY_PATH for the reruns. Launch detached, e.g.:
#   nohup bash run_followon_gpu.sh > followon.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${ALMGSI_WORK:-$HERE/work}/logs"
PY="${PYTHON:-python3}"
cd "$HERE"
unset LIVE_GPU_CONTENDED
until [ -f "$LOGS/mace_md.DONE" ]; do sleep 30; done
# live_campaign.py (the on-demand mace_md receiver) shares the GPU;
# wait until it has exited so the reruns below measure uncontended wall_s,
# and extend the contention window to now.
while pgrep -f "live_campaign.py" > /dev/null; do sleep 30; done
date +%s > "$LOGS/gpu_contention_end.txt"
CL=$("$PY" -c "import json;print(','.join(d['cid'] for d in json.load(open('candidates_index.json',encoding='utf-8')) if d['kind'] in ('cluster','cluster_ref')))")

# --- step 2: mace_md clusters at 20 ps
mkdir -p results/mace_md/_short
for c in ${CL//,/ }; do
  [ -f "results/mace_md/$c.json" ] && mv "results/mace_md/$c.json" results/mace_md/_short/
done
"$PY" dgx_sources.py --source mace_md --cids "$CL" --sample-ps 20 > "$LOGS/mace_md_clusters.log" 2>&1
echo "exit=$? $(date -Is)" > "$LOGS/mace_md_clusters.DONE"

# --- step 3: move contended records aside
LOGS="$LOGS" "$PY" - <<'PYEOF'
import json, os, shutil
from pathlib import Path
logs = Path(os.environ["LOGS"])
t0 = float((logs / "gpu_contention_start.txt").read_text(encoding="utf-8").split()[0])
t1 = float((logs / "gpu_contention_end.txt").read_text(encoding="utf-8").split()[0])
res = Path("results")
moved = {}
for src in ("mace_md", "gpumd_nep89_md", "gpumd_nep89_relax"):
    d = res / src
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        end = p.stat().st_mtime
        start = end - (r.get("wall_s") or 0.0)
        flagged = bool(r.get("settings", {}).get("gpu_contended"))
        overlap = src == "mace_md" and start < t1 and end > t0
        if flagged or overlap:
            (d / "_contended").mkdir(exist_ok=True)
            shutil.move(str(p), str(d / "_contended" / p.name))
            moved.setdefault(src, []).append(r["cid"])
with open(logs / "contended_moved.json", "w", encoding="utf-8") as f:
    json.dump(moved, f, indent=1)
print({k: len(v) for k, v in moved.items()})
PYEOF
if [ -n "${VENV_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$VENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# clusters first with their long sampling, then everything else at the 3 ps default
"$PY" dgx_sources.py --source mace_md --cids "$CL" --sample-ps 20 >> "$LOGS/gpu_reruns.log" 2>&1
"$PY" dgx_sources.py --source mace_md --all >> "$LOGS/gpu_reruns.log" 2>&1
"$PY" dgx_sources.py --source gpumd_nep89_md --cids "$CL" --sample-ps 60 >> "$LOGS/gpu_reruns.log" 2>&1
"$PY" dgx_sources.py --source gpumd_nep89_md --all >> "$LOGS/gpu_reruns.log" 2>&1
"$PY" dgx_sources.py --source gpumd_nep89_relax --all >> "$LOGS/gpu_reruns.log" 2>&1
"$PY" aggregate_sources.py >> "$LOGS/gpu_reruns.log" 2>&1
echo "exit=$? $(date -Is)" > "$LOGS/gpu_reruns.DONE"
