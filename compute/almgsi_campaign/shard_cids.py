#!/usr/bin/env python
"""Print shard i of n of the candidate pool as a comma-separated cid list.

Shard 0 starts with fcc Al (mp_134_Al, whose NPT volume defines the cluster
host cell under --host-lattice npt) and ends with every cluster cell, so the
cluster queries never run before their host lattice exists. Shard 0 gets
fewer formation-energy candidates to balance that tail.

Usage: python shard_cids.py <i> <n> [reverse]

"reverse" gives a helper list for work stealing: the shard's formation-energy
candidates in reverse order, without fcc Al and the cluster cells (those stay
with shard 0, which computes the host lattice first).
"""
import json
import sys
from pathlib import Path

i, n = int(sys.argv[1]), int(sys.argv[2])
idx = json.loads((Path(__file__).resolve().parent / "candidates_index.json").read_text(encoding="utf-8"))
clu = [d["cid"] for d in idx if d["kind"] in ("cluster", "cluster_ref")]
rest = [d["cid"] for d in idx if d["kind"] not in ("cluster", "cluster_ref") and d["cid"] != "mp_134_Al"]
shards = [[] for _ in range(n)]
load = [len(clu) + 1] + [0] * (n - 1)
for c in rest:
    j = load.index(min(load))
    shards[j].append(c)
    load[j] += 1
shards[0] = ["mp_134_Al"] + shards[0] + clu
out = shards[i]
if len(sys.argv) > 3 and sys.argv[3] == "reverse":
    out = [c for c in reversed(out) if c != "mp_134_Al" and c not in clu]
print(",".join(out))
