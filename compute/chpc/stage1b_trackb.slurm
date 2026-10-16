#!/usr/bin/env bash
#SBATCH --job-name=singobj_trackb1b
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH -o %x-%j.out
#
# Band-gap archive runs on a SLURM guest partition (preemptible):
# phase-2 rating rebuilds, phase-3 transfer scores, the hypergraph trials
# and experiment 24 at k = 3, 4, 6 with 50 seeds.
#   - results are rsynced to $HOME after every step
#   - the verification gate runs under the environment's Python (>= 3.13)
#   - phase-2 steps whose outputs already exist are skipped, so the job
#     can be resubmitted after a preemption
#   - 16 h walltime

set -uo pipefail
SCRATCH_PKG="${SCRATCH_PKG:-/scratch/general/vast/$USER/singular_obj}"
VENVS="/scratch/general/vast/$USER/singobj_venvs"
RESULTS="$HOME/singular_obj_results"
mkdir -p "$VENVS" "$RESULTS"
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv >/dev/null 2>&1 || { echo "FATAL: uv unavailable"; exit 1; }

cd "$SCRATCH_PKG"
UV_PROJECT_ENVIRONMENT="$VENVS/venv_shared" uv sync --project shared \
  --extra materials --extra viz
UV_PROJECT_ENVIRONMENT="$VENVS/venv_tenki" uv sync --project tenki/TENKi
UV_PROJECT_ENVIRONMENT="$VENVS/venv_ateki" uv sync --project ateki

# Verification gate -- under a modern interpreter this time
"$VENVS/venv_shared/bin/python" verify_package.py --venv-root "$VENVS"
echo "verify rc=$?"

TB_ABS="$SCRATCH_PKG/shared/output/bandgap_cluster_run1"
save() { rsync -a "$TB_ABS/" "$RESULTS/bandgap_cluster_run1/"; }

run_shared() {
  local name="$1"; shift
  UV_PROJECT_ENVIRONMENT="$VENVS/venv_shared" uv run \
    --project "$SCRATCH_PKG/shared" \
    python "$@" && echo "OK $name" || echo "FAIL $name"
  save
}
cd "$SCRATCH_PKG/shared"
TB=output/bandgap_cluster_run1
mkdir -p "$TB"

# Phase 2: skip anything the first job already finished
p2() {
  local name="$1" outdir="$2"; shift 2
  if [ -d "$TB/$outdir" ] && [ -n "$(find "$TB/$outdir" -name trained_ratings.json -print -quit)" ]; then
    echo "SKIP $name (phase-2 outputs exist)"
  else
    run_shared "$name" "$@" --output-root "$TB/$outdir"
  fi
}
p2 jarvis     jarvis                     scripts/jarvis_phase2.py --skip-fetch
p2 jarvis_pm  jarvis_prediction_methods  scripts/jarvis_prediction_methods_phase2.py --skip-fetch
p2 mp_mf      mp_multifidelity           scripts/mp_multifidelity_phase2.py --skip-fetch
p2 aflow      aflow_mp_jarvis            scripts/aflow_mp_jarvis_phase2.py --skip-fetch
p2 jarvis_lb  jarvis_leaderboard         scripts/jarvis_leaderboard_phase2.py --skip-fetch --n-rounds 50

for db in jarvis jarvis_prediction_methods mp_multifidelity aflow_mp_jarvis jarvis_leaderboard; do
  [ -d "$TB/$db" ] && run_shared "phase3_$db" scripts/run_phase3.py \
    --database-root "$TB/$db" --output-dir "$TB/${db}_phase3" --quiet
done

# Hypergraph layer: module self-test, archive build, both trials, experiment 24 k sweep
run_shared hg_selftest -m src.rating.hypergraph_rating
( cd "$SCRATCH_PKG" && "$VENVS/venv_shared/bin/python" \
    compute/bandgap_archive/build_inventory.py ) \
  && echo "OK universe" || echo "FAIL universe"
run_shared hg_colour  scripts/hypergraph_trial.py --output-dir "$TB/hypergraph_colour"
run_shared hg_trackb  scripts/hypergraph_trial_bandgap.py --output-dir "$TB/hypergraph_trackb"

cd "$SCRATCH_PKG/tenki/TENKi"
run_tenki() {
  local name="$1"; shift
  UV_PROJECT_ENVIRONMENT="$VENVS/venv_tenki" PEGKI_ROOT="$SCRATCH_PKG/shared" \
    uv run --project . python "$@" && echo "OK $name" || echo "FAIL $name"
  save
}
run_tenki tenki11 experiments/11_multi_domain_flip.py \
  --output-dir "$TB_ABS/tenki_multi_domain"
for K in 3 4 6; do
  run_tenki "tenki24_k$K" experiments/24_hyperedge_budget.py \
    --kway "$K" --n-seeds 50 \
    --output-dir "$TB_ABS/hyperedge_budget_k$K"
done

cd "$SCRATCH_PKG/ateki"
UV_PROJECT_ENVIRONMENT="$VENVS/venv_ateki" uv run --project . \
  python experiments/05_hypergraph_attribution.py \
  --output-dir "$TB_ABS/ateki_hypergraph" \
  && echo "OK ateki05" || echo "FAIL ateki05"
save
cp -f "$SCRATCH_PKG/VERIFICATION_RESULT.md" "$RESULTS/" 2>/dev/null
echo "DONE; results in $RESULTS/bandgap_cluster_run1"
