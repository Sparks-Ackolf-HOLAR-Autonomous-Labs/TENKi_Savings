#!/bin/bash
# Fetch QE gap results (PBE + HSE06) from CHPC scratch into results/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/semiconductor_campaign"
mkdir -p "$HERE/results"
rsync -a -e "$SSHC" "$H:$R/results/" "$HERE/results/"
for s in qe_pbe qe_hse06; do
  [ -d "$HERE/results/$s" ] && echo "$s: $(ls "$HERE/results/$s"/*.json 2>/dev/null | grep -v failed | wc -l) json"
done
