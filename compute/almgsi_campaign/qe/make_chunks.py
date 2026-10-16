"""Split the candidate pool into SLURM array chunks for qe_driver.py.

Formation-energy candidates: the three elemental references first, then
candidates sorted by site count (small cells first, so most PBE values land
early), CHUNK candidates per task. Cluster cells: one per task.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 4
idx = json.loads((HERE.parent / "candidates_index.json").read_text(encoding="utf-8"))
refs = ["mp_134_Al", "mp_153_Mg", "mp_149_Si"]
form = [r for r in idx if r["property"] == "formation_energy" and r["cid"] not in refs]
form.sort(key=lambda r: (r["nsites"], r["cid"]))
order = refs + [r["cid"] for r in form]
chunks = {"formation": [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)],
          "cluster": [[r["cid"]] for r in idx if r["property"] == "cluster_binding"]}
(HERE / "chunks.json").write_text(json.dumps(chunks, indent=1), encoding="utf-8")
print({k: len(v) for k, v in chunks.items()})
