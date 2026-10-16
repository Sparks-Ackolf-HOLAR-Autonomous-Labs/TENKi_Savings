#!/bin/bash
#SBATCH --job-name=almgsi_gpaw_r2scan
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
# GPAW r2SCAN (and PBE) single points on CHPC guest nodes.
# Array task i runs XC_LIST[i] at ECUT_LIST[i] over the cells in GPAW_CIDS (colon-separated: sbatch --export splits on commas);
# results go to results/<OUTDIR prefix>_<xc>_<ecut>/ and are copied to $HOME.
set -uo pipefail
G=/scratch/general/vast/$USER/envs/gpaw_xc
export PATH=$G/bin:$PATH OMP_NUM_THREADS=1 LIVE_ROOT="${LIVE_ROOT:?}" GPAW_WORK="${LIVE_ROOT}/work_gpaw"
IFS=: read -ra XCS <<< "${XC_LIST:?}"; IFS=: read -ra ECS <<< "${ECUT_LIST:?}"
# one xc/ecut for all tasks when the lists have a single entry (production runs)
I=$SLURM_ARRAY_TASK_ID; [ ${#XCS[@]} -eq 1 ] && J=0 || J=$I
XC="${XCS[$J]}"; EC="${ECS[$J]}"
# CHUNK_FILE: line i+1 holds the comma-separated cells of task i (overrides GPAW_CIDS)
if [ -n "${CHUNK_FILE:-}" ]; then CIDS=$(sed -n "$((I + 1))p" "$LIVE_ROOT/$CHUNK_FILE"); else CIDS="${GPAW_CIDS//:/,}"; fi
OUT="${OUTDIR:-gpaw_conv}_${XC}_${EC}"
cd "$LIVE_ROOT"
mpirun -np "$SLURM_NTASKS" python gpaw/gpaw_r2scan.py --xc "$XC" --ecut "$EC" --outdir "$OUT" --cids "$CIDS"
mkdir -p "$HOME/live_almgsi_gpaw_results"
rsync -a "$LIVE_ROOT/results/$OUT" "$HOME/live_almgsi_gpaw_results/"
