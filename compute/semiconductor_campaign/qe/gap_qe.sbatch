#!/bin/bash
#SBATCH --job-name=semi_gap_qe
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.log
# Computed semiconductor campaign, QE PBE + HSE06 band gaps.
# Guest partition only: preemptible, no allocation charge. --requeue plus the
# driver's per-candidate resume make a preemption a delay, not a data loss.
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export LIVE_ROOT="${LIVE_ROOT:?}"
export QE_WORK="${LIVE_ROOT}/${WORK_SUBDIR:-work}"
cd "$LIVE_ROOT"
python3 qe/qe_gap_driver.py
# copy results home after every task (scratch is purge-subject)
mkdir -p "$HOME/live_semiconductor_results"
rsync -a "$LIVE_ROOT/results/" "$HOME/live_semiconductor_results/"
