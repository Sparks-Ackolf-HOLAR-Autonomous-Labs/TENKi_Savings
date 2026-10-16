"""Split the semiconductor pool into SLURM array chunks, balancing an HSE06
cost proxy (n_atoms * n_electrons^2) over the chunks."""
import json
import sys
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NCHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 36
idx = json.loads((ROOT / "candidates_index.json").read_text())
zv = {}


def zval(el):
    import re
    if el not in zv:
        txt = (ROOT / "pseudos" / f"{el}.upf").read_text(errors="replace")
        zv[el] = float(re.search(r'z_valence\s*=\s*"\s*([-\d.Ee+]+)', txt).group(1))
    return zv[el]


cost = {}
for r in idx:
    a = read(ROOT / "candidates" / f"{r['cid']}.xyz")
    ne = sum(zval(e) for e in a.get_chemical_symbols())
    cost[r["cid"]] = len(a) * ne ** 2
chunks = [[] for _ in range(NCHUNK)]
load = [0.0] * NCHUNK
for cid in sorted(cost, key=lambda c: -cost[c]):
    i = min(range(NCHUNK), key=lambda j: load[j])
    chunks[i].append(cid)
    load[i] += cost[cid]
(HERE / "chunks.json").write_text(json.dumps(chunks, indent=1))
print(NCHUNK, "chunks; max/min load", max(load) / min(load))
