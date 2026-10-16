#!/usr/bin/env bash
# QE PBE (then HSE06) for the observed Al-Mg-Si phases, run on the DGX Spark.
# QE_PREFIX must hold a QE 7.4 build (bin/pw.x, lib/); PYTHON must have ase.
# Launch detached, e.g.: QE_PREFIX=<prefix> nohup bash qe/run_dgx_qe.sh > logs/run.log 2>&1 &
set -u
R="${LIVE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
Q="${QE_PREFIX:?set QE_PREFIX to the QE 7.4 install prefix}"
export PATH="$Q/bin:$PATH" LD_LIBRARY_PATH="$Q/lib"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 LIVE_ROOT="$R" QE_WORK="$R/work" QE_NP="${QE_NP:-8}"
export LIVE_INDEX=candidates_index_observed.json CHUNK_GROUP=observed_dgx HSE_NPOOL="${HSE_NPOOL:-4}"
PY="${PYTHON:-python3}"
cd "$R"
for t in 0 1 2 3 4; do SKIP_HSE=1 "$PY" qe/qe_driver.py $t >> logs/pbe.log 2>&1; echo "pbe task $t exit=$? $(date -Is)" >> logs/status.log; done
echo "PBE_DONE $(date -Is)" >> logs/status.log
for t in 0 1 2 3 4; do "$PY" qe/qe_driver.py $t >> logs/hse.log 2>&1; echo "hse task $t exit=$? $(date -Is)" >> logs/status.log; done
echo "ALL_DONE $(date -Is)" >> logs/status.log
