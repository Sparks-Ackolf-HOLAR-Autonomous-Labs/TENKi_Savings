#!/usr/bin/env bash
# Temperature study (context shift): every MD source at 300, 453 and 803 K on
# the full candidate pool, cluster host cell from the source's own NPT fcc-Al
# lattice at each temperature.  PYTHON must have ase, mace-torch and the
# LAMMPS python module; VENV_LIB (optional) is prepended to LD_LIBRARY_PATH.
# Launch detached, e.g.:
#   nohup bash run_temperature_study.sh > /dev/null 2>&1 &
# Results: results_T/T<temp>/<source>/<cid>.json; markers in results_T/logs/.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
TEMPS="${TEMPS:-300 453 803}"
SHARDS="${MACE_SHARDS:-3}"
LOGS="$HERE/results_T/logs"
if [ -n "${VENV_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$VENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p "$LOGS"
cd "$HERE"

# MACE MD is CPU-bound in ASE (about 40 s per query), so it runs as parallel shards;
# shard 0 carries fcc Al first (defines the host cell) and all cluster cells last.
ALL=$("$PY" -c "import json;print(' '.join(d['cid'] for d in json.load(open('candidates_index.json'))))")
mace_shards() {
  "$PY" - "$SHARDS" <<'PY'
import json, sys
n = int(sys.argv[1])
idx = json.load(open("candidates_index.json"))
clu = [d["cid"] for d in idx if d["kind"] in ("cluster", "cluster_ref")]
rest = [d["cid"] for d in idx if d["kind"] not in ("cluster", "cluster_ref") and d["cid"] != "mp_134_Al"]
shards = [[] for _ in range(n)]
# shard 0 gets fewer formation candidates to balance its cluster tail
load = [len(clu) + 1] + [0] * (n - 1)
for c in rest:
    i = load.index(min(load)); shards[i].append(c); load[i] += 1
shards[0] = ["mp_134_Al"] + shards[0] + clu
for s in shards:
    print(",".join(s))
PY
}

for T in $TEMPS; do
  (
    OUT="$HERE/results_T/T$T"
    OMP_NUM_THREADS=1 "$PY" dgx_sources.py --source lammps_meam_md --all --workers 4 \
      --temperature "$T" --host-lattice npt --results-dir "$OUT" > "$LOGS/lammps_T$T.log" 2>&1
    echo "exit=$? $(date -Is)" > "$LOGS/lammps_T$T.DONE"
    "$PY" dgx_sources.py --source gpumd_nep89_md --all \
      --temperature "$T" --host-lattice npt --results-dir "$OUT" > "$LOGS/gpumd_T$T.log" 2>&1
    echo "exit=$? $(date -Is)" > "$LOGS/gpumd_T$T.DONE"
  ) &
done

i=0
for T in $TEMPS; do
  OUT="$HERE/results_T/T$T"
  s=0
  while read -r CIDS; do
    (
      OMP_NUM_THREADS=2 "$PY" dgx_sources.py --source mace_md --cids "$CIDS" \
        --temperature "$T" --host-lattice npt --results-dir "$OUT" > "$LOGS/mace_T${T}_s$s.log" 2>&1
      echo "exit=$? $(date -Is)" > "$LOGS/mace_T${T}_s$s.DONE"
    ) &
    s=$((s + 1))
  done < <(mace_shards)
done
wait
echo "done $(date -Is)" > "$LOGS/ALL.DONE"
