#!/bin/bash
#SBATCH --job-name=almgsi_vacL_qe
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --qos=notchpeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.log
# QE PBE vacancy-solute cells at a larger supercell size (supercell size check).
# VAC_N = 3 (108 sites) or 4 (256 sites); one cell per array task, cells in
# VAC_CIDS (colon-separated) indexed by the task id. Settings are those of the
# 32-site cells (vacancy_qe.sbatch); only the cell size and core count change.
# Guest jobs are preemptible: --requeue restarts the interrupted cell only.
set -uo pipefail
module load openmpi/5.0.3 >/dev/null 2>&1
module load quantum-espresso/7.4 python/3.10.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1
export LIVE_ROOT="${LIVE_ROOT:?}" VAC_N="${VAC_N:?}"
export QE_WORK="${LIVE_ROOT}/work_vac_n${VAC_N}"
IFS=: read -ra ALL <<< "${VAC_CIDS:?}"
CID="${ALL[$SLURM_ARRAY_TASK_ID]}"
cd "$LIVE_ROOT"
[ -f "results/vacancy_n${VAC_N}/qe_pbe/$CID.json" ] || rm -rf "$QE_WORK/$CID"
python3 vacancy_study.py --source qe_pbe --cids "$CID"
mkdir -p "$HOME/live_almgsi_vac_results"
rsync -a "$LIVE_ROOT/results/vacancy_n${VAC_N}" "$HOME/live_almgsi_vac_results/"
