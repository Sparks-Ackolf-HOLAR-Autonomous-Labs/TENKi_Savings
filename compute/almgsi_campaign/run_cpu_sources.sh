#!/usr/bin/env bash
# CPU source (LAMMPS MEAM), process pool of single-thread workers.
# PYTHON must have ase and the LAMMPS python module; VENV_LIB (optional) is
# prepended to LD_LIBRARY_PATH for the LAMMPS shared library.
# Launch detached, e.g.:
#   nohup bash run_cpu_sources.sh > /dev/null 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${ALMGSI_WORK:-$HERE/work}/logs"
PY="${PYTHON:-python3}"
if [ -n "${VENV_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$VENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export OMP_NUM_THREADS=1
mkdir -p "$LOGS"
cd "$HERE"
rm -f "$LOGS/lammps_meam_md.DONE"
"$PY" dgx_sources.py --source lammps_meam_md --all --workers "${WORKERS:-8}" > "$LOGS/lammps_meam_md.log" 2>&1
echo "exit=$? $(date -Is)" > "$LOGS/lammps_meam_md.DONE"
