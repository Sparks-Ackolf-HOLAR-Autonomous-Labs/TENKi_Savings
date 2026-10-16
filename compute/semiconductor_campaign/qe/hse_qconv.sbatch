#!/bin/bash
#SBATCH --job-name=semi_hse_qconv
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# HSE06 q-grid convergence check for the semiconductor pool (q = k against q = k/2),
# at the experimental-structure starting geometry of each candidate.
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export LIVE_ROOT="${LIVE_ROOT:?}"
export QE_WORK="${LIVE_ROOT}/work_qconv"
cd "$LIVE_ROOT"
python3 qe/qe_gap_driver.py qconv arc_GaAs_mp-2534 arc_ZnO_mp-1986 arc_MgO_mp-1265
mkdir -p "$HOME/live_semiconductor_results"
rsync -a "$LIVE_ROOT/results/" "$HOME/live_semiconductor_results/"
