#!/bin/bash
# Fetch QE PBE results for the Al-Si-Cu pool from CHPC into results/qe_pbe/.
# Needs a live ControlMaster (socket under ~/.ssh/cm/). Falls back to the
# home-directory copy written by each array task if scratch was purged.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/alsicu_campaign"
mkdir -p "$HERE/results"
rsync -a -e "$SSHC" "$H:$R/results/qe_pbe" "$HERE/results/" 2>/dev/null || \
  rsync -a -e "$SSHC" "$H:live_alsicu_results/qe_pbe" "$HERE/results/"
mkdir -p "$HERE/qe/logs"
rsync -a -e "$SSHC" "$H:$R/logs/" "$HERE/qe/logs/" 2>/dev/null || true
n=$(ls "$HERE/results/qe_pbe"/*.json 2>/dev/null | grep -v '\.[a-z0-9_]*\.json$' | wc -l)
nok=$(grep -l '"status": "ok"' "$HERE/results/qe_pbe"/*.json 2>/dev/null | wc -l)
echo "qe_pbe: $n json ($nok ok)"
$SSHC "$H" 'squeue -u $USER -n live_alsicu_qe -h -o "%i %T %M" | head -40' 2>/dev/null | sed "s/$CHPC_USERNAME/<user>/g" || true
