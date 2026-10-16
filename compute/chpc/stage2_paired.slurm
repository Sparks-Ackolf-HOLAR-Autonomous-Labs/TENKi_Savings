#!/usr/bin/env bash
#SBATCH --job-name=singobj_paired
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH -o %x-%j.out
#
# band-gap archive stage 2 on CHPC (guest partition; PREEMPTABLE + --requeue).
# CHPC-scale runs of the paired campaign protocol (exp 26) and the
# physics-informed prior curves (exp 25). Exp 26 checkpoints one JSON
# per (receiver, seed) inside its output dir, so a preempted requeue
# resumes instead of restarting; results are rsynced home after EVERY
# step, not just at the end.
# No-allocation alternative:
#   --account=<group> --partition=notchpeak-freecycle   (also preemptable)
#
# Expects a copy of the repository at $SCRATCH_PKG.

set -uo pipefail
SCRATCH_PKG="${SCRATCH_PKG:-/scratch/general/vast/$USER/singular_obj}"
VENVS="/scratch/general/vast/$USER/singobj_venvs"
RESULTS="$HOME/singular_obj_results"
TB="$SCRATCH_PKG/shared/output/bandgap_cluster_run2"
mkdir -p "$VENVS" "$RESULTS" "$TB"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv >/dev/null 2>&1 || { echo "FATAL: uv unavailable"; exit 1; }

cd "$SCRATCH_PKG"
UV_PROJECT_ENVIRONMENT="$VENVS/venv_shared" uv sync --project shared \
  --extra materials --extra viz
UV_PROJECT_ENVIRONMENT="$VENVS/venv_tenki" uv sync --project tenki/TENKi

save_home() { rsync -a "$TB/" "$RESULTS/bandgap_cluster_run2/"; }

# Frozen universe (stdlib build from bundled caches; byte-identical to
# manifest f2158ecd... -- build_stats.json records the check)
if [ ! -f "$SCRATCH_PKG/shared/output/bandgap_archive_build/bandgap_universe.jsonl" ]; then
  "$VENVS/venv_shared/bin/python" compute/bandgap_archive/build_inventory.py \
    && echo "OK universe" || { echo "FAIL universe"; exit 1; }
fi

run_tenki() {
  local name="$1"; shift
  ( cd "$SCRATCH_PKG/tenki/TENKi" && \
    UV_PROJECT_ENVIRONMENT="$VENVS/venv_tenki" PEGKI_ROOT="$SCRATCH_PKG/shared" \
    uv run --project . python "$@" ) && echo "OK $name" || echo "FAIL $name"
  save_home
}

# Exp 26 paired campaign at CHPC scale: 50 seeds, k=4 headline plus a
# k=6 arm-width check. Checkpoint dirs make both requeue-safe.
run_tenki paired_k4 experiments/26_paired_campaign.py \
  --n-seeds 50 --workers "$SLURM_CPUS_PER_TASK" \
  --output-dir "$TB/paired_campaign_k4"
run_tenki paired_k6 experiments/26_paired_campaign.py \
  --kway 6 --n-seeds 50 --workers "$SLURM_CPUS_PER_TASK" \
  --output-dir "$TB/paired_campaign_k6"

# Exp 25 physics-informed priors at 50 seeds (one shot; runs last so a
# preemption cannot cost the exp 26 checkpoints above)
run_tenki physics50 experiments/25_physics_informed_budget.py \
  --n-seeds 50 --output-dir "$TB/physics_prior_50seeds"

save_home
echo "DONE; results in $RESULTS/bandgap_cluster_run2"
