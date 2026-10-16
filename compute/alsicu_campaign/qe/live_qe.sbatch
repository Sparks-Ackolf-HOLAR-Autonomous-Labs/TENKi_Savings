#!/bin/bash
#SBATCH --job-name=live_alsicu_qe
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.log
# Computed Al-Si-Cu pool, QE PBE vc-relax. Guest partition only:
# preemptible, no allocation charge. --requeue + the driver's per-candidate
# resume make a preemption a delay, not a data loss.
# --array is given on the sbatch command line (see ../README.md).
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export LIVE_ROOT="${LIVE_ROOT:?}"
export QE_WORK="${LIVE_ROOT}/work"
cd "$LIVE_ROOT"
python3 qe/qe_driver.py
# copy results home after every task (scratch is purge-subject)
mkdir -p "$HOME/live_alsicu_results"
rsync -a "$LIVE_ROOT/results/" "$HOME/live_alsicu_results/"
