#!/bin/bash
# Fetch QE PBE migration saddles into results/migration_n3/qe_pbe, then run migration_study.py --analyse.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
rsync -a -e "$SSHC" "$H:live_almgsi_vac_results/migration_n3/" "$HERE/results/migration_n3/"
ls "$HERE/results/migration_n3/qe_pbe"
