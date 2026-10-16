#!/bin/bash
# Stage and submit the QE PBE migration saddles (migration_study.py) to notchpeak-guest.
# Needs a live ControlMaster. Fetch: qe/fetch_migration_chpc.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/live_almgsi_vac"
$SSHC "$H" "mkdir -p $R/logs $R/results/migration_n3 $R/qe"
rsync -a -e "$SSHC" "$HERE/vacancy_study.py" "$HERE/migration_study.py" "$HERE/candidates_vac_n3" "$HERE/candidates_mig_n3" "$H:$R/"
rsync -a -e "$SSHC" "$HERE/qe/qe_driver.py" "$HERE/qe/migration_qe.sbatch" "$H:$R/qe/"
rsync -a -e "$SSHC" "$HERE/results/qe_pbe_al_lattice.json" "$H:$R/results/"
$SSHC "$H" "cd $R && bash -lc 'sbatch --parsable --mem=${MIG_MEM:-120G} --array=0-2 --export=ALL,LIVE_ROOT=$R,VAC_N=3,VAC_CIDS=mig_Al:mig_Si:mig_Mg qe/migration_qe.sbatch'"
