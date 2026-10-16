#!/bin/bash
# MACE-MP-0 MD for the temperature study on CHPC guest GPUs.
# Array task t -> temperature TEMPS[t / NSHARD], shard t % NSHARD.
# Per-candidate JSON results are resumable, so a preempted task simply
# restarts with --requeue and skips finished candidates.
#SBATCH --job-name=almgsi_maceT
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --output=logs/maceT_%A_%a.out
set -u
TEMPS=(300 453 803)
NSHARD=4
T=${TEMPS[$((SLURM_ARRAY_TASK_ID / NSHARD))]}
S=$((SLURM_ARRAY_TASK_ID % NSHARD))
W=/scratch/general/vast/$USER/live_almgsi_T
HOMEOUT=$HOME/live_almgsi_T_results
module load python/3.12.4
# MACE_VENV: virtual environment with ase, mace-torch and torch (CUDA)
source "${MACE_VENV:-$HOME/live_almgsi_T_venv}/bin/activate"
export OMP_NUM_THREADS=4
cd "$W"
# some guest GPUs are allocated but not visible to CUDA ("No devices were found");
# put the task back in the queue instead of failing it
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA not available on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ] && scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 3
fi
# HELPER=1: work-stealing pass over the same shard in reverse order
CIDS=$(python shard_cids.py "$S" "$NSHARD" ${HELPER:+reverse})
echo "T=$T shard=$S host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date -Is)"
python dgx_sources.py --source mace_md --cids "$CIDS" --temperature "$T" \
  --host-lattice npt --results-dir "$W/results_T/T$T"
rc=$?
mkdir -p "$HOMEOUT/T$T"
rsync -a "$W/results_T/T$T/mace_md" "$HOMEOUT/T$T/"
echo "exit=$rc end=$(date -Is)"
