#!/bin/bash
# Sync the pool, pseudopotentials and driver to CHPC scratch and submit the array.
# Needs a live ControlMaster: python3 hpc/chpc_login.py notchpeak.chpc.utah.edu
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/semiconductor_campaign"
$SSHC "$H" "mkdir -p $R/logs $R/results"
rsync -a -e "$SSHC" "$HERE/candidates" "$HERE/candidates_index.json" "$HERE/pseudos" "$HERE/qe" "$H:$R/"
N=$(python3 -c "import json;print(len(json.load(open('$HERE/qe/chunks.json')))-1)")
$SSHC "$H" "bash -lc 'cd $R && sbatch --parsable --array=0-$N ${ARRAY_EXTRA:-} --export=ALL,LIVE_ROOT=$R${EXPORT_EXTRA:-} qe/gap_qe.sbatch'"
