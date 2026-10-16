#!/usr/bin/env bash
# Run the GPU-bound sources one after another so they never compete for the
# GPU (on the DGX Spark GB10 a resident inference server also holds much of its memory).
# PYTHON must have ase, mace-torch and the GPUMD tooling used by dgx_sources.py.
# Launch detached, e.g.:
#   nohup bash run_gpu_sources.sh > gpu_chain.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${ALMGSI_WORK:-$HERE/work}/logs"
PY="${PYTHON:-python3}"
mkdir -p "$LOGS"
cd "$HERE"
for SRC in ${GPU_SOURCES:-mace_relax gpumd_nep89_md mace_md}; do
  rm -f "$LOGS/$SRC.DONE"
  "$PY" dgx_sources.py --source "$SRC" --all > "$LOGS/$SRC.log" 2>&1
  echo "exit=$? $(date -Is)" > "$LOGS/$SRC.DONE"
done
