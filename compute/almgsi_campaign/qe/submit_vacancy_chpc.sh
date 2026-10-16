#!/bin/bash
# Stage and submit the remaining QE PBE vacancy cells to notchpeak-guest.
# Needs a live ControlMaster: python3 hpc/chpc_login.py notchpeak.chpc.utah.edu
# Usage: qe/submit_vacancy_chpc.sh vac_MgSiV,vac_SiSi,...
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CIDS="${1:?comma-separated cell ids}"
N=$(( $(tr ',' '\n' <<< "$CIDS" | wc -l) - 1 ))
CIDS_C="${CIDS//,/:}"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/live_almgsi_vac"
$SSHC "$H" "mkdir -p $R/logs $R/results/vacancy $R/qe $R/pseudos"
rsync -a -e "$SSHC" "$HERE/vacancy_study.py" "$HERE/candidates_vac" "$H:$R/"
rsync -a -e "$SSHC" "$HERE/qe/qe_driver.py" "$HERE/qe/vacancy_qe.sbatch" "$H:$R/qe/"
rsync -a -e "$SSHC" "$HERE/results/qe_pbe_al_lattice.json" "$H:$R/results/"
rsync -a -e "$SSHC" "$HERE/results/vacancy/qe_pbe" "$H:$R/results/vacancy/"
$SSHC "$H" "rsync -a /scratch/general/vast/\$USER/live_almgsi_obs/pseudos/ $R/pseudos/ && cd $R && \
  bash -lc 'sbatch --parsable --array=0-$N --export=ALL,LIVE_ROOT=$R,VAC_CIDS=$CIDS_C qe/vacancy_qe.sbatch'"
