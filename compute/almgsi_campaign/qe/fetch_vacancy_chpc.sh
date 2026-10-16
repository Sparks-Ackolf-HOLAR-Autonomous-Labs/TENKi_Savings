#!/bin/bash
# Fetch QE PBE vacancy records from CHPC (home copy survives scratch purge).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
rsync -a -e "$SSHC" "$H:live_almgsi_vac_results/qe_pbe/" "$HERE/results/vacancy/qe_pbe/"
ls "$HERE/results/vacancy/qe_pbe" | grep -c json
