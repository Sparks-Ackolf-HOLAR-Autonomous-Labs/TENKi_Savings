#!/usr/bin/env bash
# QE PBE vacancy cells; 8 MPI ranks.
# QE_PREFIX must hold a QE 7.4 build (bin/pw.x, lib/); PYTHON must have ase.
# Launch detached, e.g.: QE_PREFIX=<prefix> nohup bash qe/run_vac_qe.sh > logs/run.log 2>&1 &
set -u
V="${LIVE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
Q="${QE_PREFIX:?set QE_PREFIX to the QE 7.4 install prefix}"
export PATH="$Q/bin:$PATH" LD_LIBRARY_PATH="$Q/lib"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 LIVE_ROOT="$V" QE_WORK="$V/work" QE_NP="${QE_NP:-8}"
cd "$V"
"${PYTHON:-python3}" vacancy_study.py --source qe_pbe >> logs/qe.log 2>&1
echo "exit=$? $(date -Is)" > logs/QE_DONE
