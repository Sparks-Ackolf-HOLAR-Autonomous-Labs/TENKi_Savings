#!/bin/bash
# Fetch QE results (PBE + HSE06) from CHPC scratch into results/.
# Needs a live ControlMaster: python3 hpc/chpc_login.py notchpeak.chpc.utah.edu
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/almgsi_campaign"
rsync -a -e "$SSHC" "$H:$R/results/qe_pbe" "$H:$R/results/qe_hse06" "$H:$R/results/qe_pbe_al_lattice.json" "$HERE/results/" 2>/dev/null || \
  rsync -a -e "$SSHC" "$H:$R/results/qe_pbe" "$HERE/results/"
for s in qe_pbe qe_hse06; do
  [ -d "$HERE/results/$s" ] && echo "$s: $(ls "$HERE/results/$s"/*.json 2>/dev/null | wc -l) json"
done
