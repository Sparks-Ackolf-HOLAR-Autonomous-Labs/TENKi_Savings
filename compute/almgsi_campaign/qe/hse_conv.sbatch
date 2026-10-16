#!/bin/bash
#SBATCH --job-name=live_hse_conv
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export QE_WORK="${LIVE_ROOT}/work"
cd "$LIVE_ROOT"
python3 qe/hse_convergence.py
mkdir -p "$HOME/live_almgsi_results"; rsync -a "$LIVE_ROOT/results/" "$HOME/live_almgsi_results/"
