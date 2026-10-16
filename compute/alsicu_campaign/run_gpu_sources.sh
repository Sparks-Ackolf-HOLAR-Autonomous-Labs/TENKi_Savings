#!/usr/bin/env bash
# Run the GPU-bound source(s) for the Al-Si-Cu pool (DGX Spark GB10).
# The 0 K MACE relaxation of all 168 candidates takes about 3.5 min, so it can
# run in the foreground. PYTHON must have ase and mace-torch. Detached launch, e.g.:
#   nohup bash run_gpu_sources.sh > gpu_chain.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${ALSICU_WORK:-$HERE/work}/logs"
PY="${PYTHON:-python3}"
mkdir -p "$LOGS"
cd "$HERE"
for SRC in ${GPU_SOURCES:-mace_relax}; do
  rm -f "$LOGS/$SRC.DONE"
  "$PY" dgx_sources.py --source "$SRC" --all > "$LOGS/$SRC.log" 2>&1
  echo "exit=$? $(date -Is)" > "$LOGS/$SRC.DONE"
done
