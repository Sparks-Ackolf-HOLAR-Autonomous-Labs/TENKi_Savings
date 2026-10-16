#!/bin/bash
# Fetch GPAW results (convergence and pool runs) and larger-cell vacancy results from CHPC.
# Needs a live ControlMaster: python3 hpc/chpc_login.py notchpeak.chpc.utah.edu
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
S=/scratch/general/vast/$CHPC_USERNAME
# cutoff scans go to results/r2scan_conv/; the pool r2SCAN run becomes the source
# results/gpaw_r2scan/ and the PBE code check results/r2scan_conv/gpaw_pbe/
G="$S/live_almgsi_gpaw/results"
rsync -a -e "$SSHC" "$H:$G/gpaw_conv_*" "$HERE/results/r2scan_conv/" || true
mkdir -p "$HERE/results/gpaw_r2scan" "$HERE/results/r2scan_conv/gpaw_pbe"
rsync -a -e "$SSHC" "$H:$G/gpaw_pool_r2scan_700/" "$HERE/results/gpaw_r2scan/" || true
rsync -a -e "$SSHC" "$H:$G/gpaw_pool_pbe_700/" "$HERE/results/r2scan_conv/gpaw_pbe/" || true
for n in 3 4; do
  rsync -a -e "$SSHC" "$H:$S/live_almgsi_vac/results/vacancy_n$n/qe_pbe" "$HERE/results/vacancy_n$n/" 2>/dev/null \
    || rsync -a -e "$SSHC" "$H:live_almgsi_vac_results/vacancy_n$n/qe_pbe" "$HERE/results/vacancy_n$n/" || true
done
$SSHC "$H" 'squeue -u $USER -o "%i %j %T %M"' | grep -E "almgsi_gpaw|almgsi_vacL" || echo "no GPAW or vacancy jobs left"
