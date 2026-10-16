#!/bin/bash
# Fetch PBE/HSE06 for the observed Al-Mg-Si phases (CHPC array 15979565,
# notchpeak-guest) into results/qe_pbe_dgx and results/qe_hse06_dgx, the
# folders lab_sequence_study.py reads. Needs a live ControlMaster:
#   python3 hpc/chpc_login.py notchpeak.chpc.utah.edu
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
SSHC="ssh -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o BatchMode=yes"
R="/scratch/general/vast/$CHPC_USERNAME/live_almgsi_obs/results"
mkdir -p "$HERE/results/qe_pbe_dgx" "$HERE/results/qe_hse06_dgx"
rsync -a -e "$SSHC" "$H:$R/qe_pbe/" "$HERE/results/qe_pbe_dgx/"
rsync -a -e "$SSHC" --exclude 'mp_134_Al.*' --exclude 'mp_153_Mg.*' --exclude 'mp_149_Si.*' --exclude 'mp_1367_Mg2Si.*' \
  "$H:$R/qe_hse06/" "$HERE/results/qe_hse06_dgx/" || true
ls "$HERE/results/qe_hse06_dgx" | grep -c json || true
