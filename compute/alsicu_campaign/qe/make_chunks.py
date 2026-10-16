"""Split the Al-Si-Cu pool into SLURM array chunks for qe_driver.py.

Chunk 0 holds only the three elemental references (fcc Al, diamond Si,
fcc Cu); it doubles as the smoke test. The remaining candidates are sorted
by site count (small cells first) and split into chunks of CHUNK.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 6
idx = json.loads((HERE.parent / "candidates_index.json").read_text(encoding="utf-8"))
refs = ["mp_134_Al", "mp_149_Si", "mp_30_Cu"]
form = [r for r in idx if r["property"] == "formation_energy" and r["cid"] not in refs]
form.sort(key=lambda r: (r["nsites"], r["cid"]))
order = [r["cid"] for r in form]
chunks = {"formation": [refs] + [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]}
(HERE / "chunks.json").write_text(json.dumps(chunks, indent=1), encoding="utf-8")
print({k: len(v) for k, v in chunks.items()}, "candidates:", sum(len(c) for c in chunks["formation"]))
