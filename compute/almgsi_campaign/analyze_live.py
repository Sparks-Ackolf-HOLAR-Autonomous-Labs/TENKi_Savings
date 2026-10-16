"""Analysis of the live Al-Mg-Si calculations (stages 2-5).

Writes ``results/analysis/``:
* ``source_agreement.json/.md``: for every source against QE PBE and QE HSE06
  on the candidates both hold: Kendall tau, Spearman rho, mean absolute
  difference of formation energy, sign agreement (E_f < 0), coverage, and
  the measured cost per query (median wall seconds and cores).
* ``hull.json``: distance to the convex hull of each source's own formation
  energies (pymatgen PhaseDiagram), and which suggested orderings lie within
  25 meV/atom of the PBE hull.
* ``archive_vs_live.json``: the replay-over-archive view against the live
  view. On candidates MP holds: agreement of MP formation energies with live
  PBE. On the whole pool: how many of the live PBE 10 lowest candidates, and
  of the near-hull candidates, the archive cannot answer at all.
* ``cluster_binding.json``: solute-cluster binding energies per source.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "analysis"
SOURCES = ["mace_relax", "mace_md", "gpumd_nep89_md", "lammps_meam_md", "archive_mp", "qe_pbe", "qe_hse06"]
LABEL = {"mace_relax": "MACE-MP-0, 0 K relaxation", "mace_md": "MACE-MP-0 MD, 300 K",
         "gpumd_nep89_md": "GPUMD NEP89 MD, 300 K", "lammps_meam_md": "LAMMPS MEAM MD, 300 K",
         "archive_mp": "Materials Project archive", "qe_pbe": "QE PBE", "qe_hse06": "QE HSE06"}


def load():
    import campaign as C
    idx = C.load_index()
    tabs = C.load_tables()
    raw = json.loads((RES / "live_values.json").read_text(encoding="utf-8"))
    return idx, tabs, raw


def cost_stats(src: str) -> dict:
    d = RES / src
    if not d.exists():
        return {}
    walls, cores = [], set()
    for p in d.glob("*.json"):
        if p.name.startswith("_") or "attempt" in p.name:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "ok" and r.get("wall_s") is not None and not r["cid"].startswith("clu_"):
            walls.append(r["wall_s"])
            cores.add(r.get("device"))
    if not walls:
        return {}
    return {"median_wall_s": statistics.median(walls), "total_wall_s": sum(walls), "n": len(walls),
            "device": sorted(x for x in cores if x)}


def compare(a: dict, b: dict) -> dict:
    cs = sorted(set(a) & set(b))
    if len(cs) < 5:
        return {"n": len(cs)}
    x = np.array([a[c]["value"] for c in cs])
    y = np.array([b[c]["value"] for c in cs])
    return {"n": len(cs), "kendall_tau": float(kendalltau(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic),
            "mad_eV_atom": float(np.mean(np.abs(x - y))),
            "mean_diff_eV_atom": float(np.mean(x - y)),
            "sign_agreement": float(np.mean((x < 0) == (y < 0)))}


def hull_distances(tab: dict) -> dict:
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    from ase.io import read
    entries = []
    for cid, r in tab.items():
        s = read(HERE / "candidates" / f"{cid}.xyz").get_chemical_symbols()
        comp = Composition({el: s.count(el) for el in set(s)})
        entries.append(PDEntry(comp, r["value"] * comp.num_atoms, name=cid))
    for el in ("Al", "Mg", "Si"):
        entries.append(PDEntry(Composition(el), 0.0, name=f"ref_{el}"))
    pd = PhaseDiagram(entries)
    return {e.name: float(pd.get_e_above_hull(e)) for e in entries if not e.name.startswith("ref_")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx, tabs, raw = load()
    kinds = {c: r["kind"] for c, r in idx.items()}

    # 1. agreement of every source with the two DFT levels
    agree = {}
    for recv in ("qe_pbe", "qe_hse06"):
        if recv not in tabs:
            continue
        agree[recv] = {s: {**compare(tabs[s], tabs[recv]), "cost": cost_stats(s)}
                       for s in SOURCES if s in tabs and s != recv}
    (OUT / "source_agreement.json").write_text(json.dumps(agree, indent=1), encoding="utf-8")
    lines = ["| Source | vs | n | Kendall tau | MAD (eV/atom) | mean shift | sign agreement | median wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for recv, rows in agree.items():
        for s, r in rows.items():
            if "kendall_tau" not in r:
                continue
            c = r["cost"].get("median_wall_s")
            lines.append(f"| {LABEL[s]} | {LABEL[recv]} | {r['n']} | {r['kendall_tau']:+.3f} | {r['mad_eV_atom']:.3f} | "
                         f"{r['mean_diff_eV_atom']:+.3f} | {r['sign_agreement']:.2f} | "
                         f"{'-' if c is None else f'{c:.1f}'} |")
    (OUT / "source_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. convex hull per source
    hull = {s: hull_distances(tabs[s]) for s in SOURCES if s in tabs and s != "archive_mp" and len(tabs[s]) > 10}
    near = {}
    if "qe_pbe" in hull:
        near = {c: d for c, d in hull["qe_pbe"].items() if d <= 0.025}
    (OUT / "hull.json").write_text(json.dumps({"e_above_hull": hull, "pbe_within_25meV": near,
                                               "pbe_within_25meV_kinds": {c: kinds[c] for c in near}}, indent=1),
                                   encoding="utf-8")

    # 3. archive replay versus live
    av = {}
    if "qe_pbe" in tabs:
        import campaign as C
        # compounds only (binaries and ternaries), as in the campaigns
        pbe = {c: v for c, v in tabs["qe_pbe"].items() if (C.fractions(c, idx) > 0).sum() >= 2}
        top10 = sorted(pbe, key=lambda c: pbe[c]["value"])[:10]
        av = {"mp_vs_live_pbe": compare(tabs["archive_mp"], pbe),
              "n_pool_live": len(pbe),
              "n_pool_archive_answerable": sum(c in tabs["archive_mp"] for c in pbe),
              "live_pbe_top10": [{"cid": c, "kind": kinds[c], "E_f": pbe[c]["value"],
                                  "in_archive": c in tabs["archive_mp"]} for c in top10],
              "live_pbe_top10_not_in_archive": sum(c not in tabs["archive_mp"] for c in top10),
              "pbe_near_hull_not_in_archive": sorted(c for c in near if c not in tabs["archive_mp"])}
    (OUT / "archive_vs_live.json").write_text(json.dumps(av, indent=1), encoding="utf-8")

    # 4. cluster binding energies
    clu = {s: {c: {"E_b_eV": v.get("value"), "sem": v.get("sem"), "status": v.get("status")}
               for c, v in rows.items() if c.startswith("clu_") and idx[c]["kind"] == "cluster"}
           for s, rows in raw.items()}
    (OUT / "cluster_binding.json").write_text(json.dumps(clu, indent=1), encoding="utf-8")
    print((OUT / "source_agreement.md").read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in av.items() if k != "live_pbe_top10"}, indent=1))


if __name__ == "__main__":
    main()
