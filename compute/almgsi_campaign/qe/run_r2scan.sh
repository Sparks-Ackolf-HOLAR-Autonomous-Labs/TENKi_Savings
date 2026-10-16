#!/usr/bin/env bash
# r2SCAN single points at the PBE geometries (third DFT level), run on the DGX Spark.
# QE_LIBXC_PREFIX must hold the libxc-enabled QE build (qe/bin/pw.x, bin/, lib/; see qe/R2SCAN.md).
# PYTHON must have ase. Launch detached, e.g.:
#   QE_LIBXC_PREFIX=<prefix> nohup bash qe/run_r2scan.sh > logs/run.log 2>&1 < /dev/null &
set -u
R="${LIVE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
E="${QE_LIBXC_PREFIX:?set QE_LIBXC_PREFIX to the libxc-enabled QE install prefix}"
export PATH="$E/qe/bin:$E/bin:$PATH" LD_LIBRARY_PATH="$E/lib"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 LIVE_ROOT="$R" QE_WORK="$R/work" QE_NP="${QE_NP:-12}"
PY="${PYTHON:-python3}"
cd "$R"
mkdir -p logs
rm -f logs/DONE
echo "START $(date -Is) np=$QE_NP" >> logs/status.log
"$PY" qe/r2scan_driver.py >> logs/r2scan.log 2>&1
echo "r2scan exit=$? $(date -Is)" >> logs/status.log
echo "DONE $(date -Is)" > logs/DONE
