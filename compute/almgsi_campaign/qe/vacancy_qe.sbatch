#!/bin/bash
#SBATCH --job-name=almgsi_vac_qe
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
# QE PBE vacancy-solute cells for the Al-Mg-Si vacancy receiver.
# One cell per array task; VAC_CIDS (colon-separated; sbatch --export splits on commas) is indexed by the task id.
# DFT runs on CHPC guest partitions. Guest jobs are preemptible: --requeue plus the per-cell result
# check in vacancy_study.py make a preemption a restart of that cell only.
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export LIVE_ROOT="${LIVE_ROOT:?}"
export QE_WORK="${LIVE_ROOT}/work_vac"
IFS=: read -ra ALL <<< "${VAC_CIDS:?}"
CID="${ALL[$SLURM_ARRAY_TASK_ID]}"
cd "$LIVE_ROOT"
# a preempted attempt leaves work_vac/<cid>/pbe, which the resume check treats as done
[ -f "results/vacancy/qe_pbe/$CID.json" ] || rm -rf "$QE_WORK/$CID"
python3 vacancy_study.py --source qe_pbe --cids "$CID"
mkdir -p "$HOME/live_almgsi_vac_results"
rsync -a "$LIVE_ROOT/results/vacancy/" "$HOME/live_almgsi_vac_results/"
