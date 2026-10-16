#!/bin/bash
# Stage and submit QE PBE vacancy cells at a larger supercell size to notchpeak-guest.
# Needs a live ControlMaster: python3 hpc/chpc_login.py notchpeak.chpc.utah.edu
# Usage: [VAC_MEM=160G] qe/submit_vacancy_large_chpc.sh 3 vac_Al32,vac_V,...   (VAC_N, comma-separated cell ids)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:?VAC_N (3 or 4)}"; CIDS="${2:?comma-separated cell ids}"
LAST=$(( $(tr ',' '\n' <<< "$CIDS" | wc -l) - 1 ))
CIDS_C="${CIDS//,/:}"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/live_almgsi_vac"
$SSHC "$H" "mkdir -p $R/logs $R/results/vacancy_n$N $R/qe $R/pseudos"
rsync -a -e "$SSHC" "$HERE/vacancy_study.py" "$HERE/candidates_vac_n$N" "$H:$R/"
rsync -a -e "$SSHC" "$HERE/qe/qe_driver.py" "$HERE/qe/vacancy_qe_large.sbatch" "$H:$R/qe/"
rsync -a -e "$SSHC" "$HERE/results/qe_pbe_al_lattice.json" "$H:$R/results/"
$SSHC "$H" "cd $R && bash -lc 'sbatch --parsable --mem=${VAC_MEM:-88G} --array=0-$LAST --export=ALL,LIVE_ROOT=$R,VAC_N=$N,VAC_CIDS=$CIDS_C qe/vacancy_qe_large.sbatch'"
