#!/bin/bash
#SBATCH --job-name=live_almgsi_qe
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.log
# Live Al-Mg-Si campaign, QE PBE + HSE06. Guest partition only:
# preemptible, no allocation charge. --requeue + the driver's per-candidate
# resume make a preemption a delay, not a data loss.
# --ntasks / --mem / --array / CHUNK_GROUP are given on the sbatch command line.
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export LIVE_ROOT="${LIVE_ROOT:?}"
export QE_WORK="${LIVE_ROOT}/work"
cd "$LIVE_ROOT"
python3 qe/qe_driver.py
# copy results home after every task (scratch is purge-subject)
mkdir -p "$HOME/live_almgsi_results"
rsync -a "$LIVE_ROOT/results/" "$HOME/live_almgsi_results/"
