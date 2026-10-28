#!/usr/bin/env python3
"""
step10_rank_ensemble.py
=======================
Does the CORRECT topology appear in ANY of AlphaFold2's five ranked models?

Every ColabFold analysis so far used rank_001 only. That answers "what does
the model output", not "what does the model represent". Two very different
findings hide behind the same rank_001 result:

  ABSENCE      the correct topology never appears in any of the five models.
               The information is not there. Our claim stands as measured.

  MIS-RANKING  the correct topology appears at rank 3 or 4, but the confidence
               score puts the wrong one on top. The information IS there and
               is recoverable -- a different, more actionable claim, since it
               points at a fix rather than a wall.

KnotFold (PRX Life 2:043018, 2024) scores "incorrect in >=3 of 5 predictions",
a majority vote that never inspects the minority models. So this is not
settled by prior work.

Usage
-----
    python step10_rank_ensemble.py --cf-out cf_out
"""

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations


# --------------------------------------------------------------------------
# connectivity from coordinates (same rule used throughout the project)
# --------------------------------------------------------------------------

def parse_cys_sg(pdb_text):
    """SG atom of each cysteine, ordered along the chain -> C1..CN."""
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        line = line.ljust(80)
        if line[12:16].strip() == "SG" and line[17:20].strip() == "CYS":
            atoms.append((int(line[22:26]),
                          (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))))
    atoms.sort(key=lambda a: a[0])
    return [(i + 1, xyz) for i, (_, xyz) in enumerate(atoms)]


def mean_plddt(pdb_text):
    """ColabFold writes per-residue pLDDT into the B-factor column."""
    vals, seen = [], set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        line = line.ljust(80)
        if line[12:16].strip() != "CA":
            continue
        key = line[22:27]
        if key in seen:
            continue
        seen.add(key)
        try:
            vals.append(float(line[60:66]))
        except ValueError:
            pass
    return sum(vals) / len(vals) if vals else None


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def realized_pairing(pdb_text, cutoff=3.0):
    """Greedy shortest-first, each cysteine used once, cutoff in angstroms."""
    cys = parse_cys_sg(pdb_text)
    cand = sorted((dist(xi, xj), i, j)
                  for (i, xi), (j, xj) in combinations(cys, 2))
    used, bonds = set(), []
    for d, i, j in cand:
        if i in used or j in used or d > cutoff:
            continue
        bonds.append((i, j))
        used.update((i, j))
    return tuple(sorted(bonds))


def as_key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def fmt(t):
    return str([list(p) for p in t]) if t else "[]"


# --------------------------------------------------------------------------

RANK_RE = re.compile(r"_rank_(\d+)_")


def collect_ranked(cf_dir, job):
    """All ranked PDBs for one job, as {rank: path}. Searches subdirs too."""
    out = {}
    pats = [os.path.join(cf_dir, f"{job}_*rank_*.pdb"),
            os.path.join(cf_dir, "**", f"{job}_*rank_*.pdb")]
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            m = RANK_RE.search(os.path.basename(p))
            if m:
                out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def pick(meta, *names, default=None):
    """jobmap key names have drifted between scripts -- accept any of them."""
    for n in names:
        if n in meta and meta[n] is not None:
            return meta[n]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-out", default="cf_out")
    ap.add_argument("--jobmap", default="colabfold_jobmap.json")
    ap.add_argument("--source-json", default="metric_discrimination.json",
                    help="used only to recover accessions by sequence")
    ap.add_argument("--cutoff", type=float, default=3.0)
    ap.add_argument("--out", default="rank_ensemble_results.json")
    args = ap.parse_args()

    if not os.path.isdir(args.cf_out):
        print(f"Could not find directory {args.cf_out}")
        return
    if not os.path.exists(args.jobmap):
        print(f"Could not find {args.jobmap} -- it is written by "
              f"step4_colabfold_oracle.py --prepare")
        return

    jobmap = json.load(open(args.jobmap))

    # accession lookup, keyed by sequence
    acc = {}
    if os.path.exists(args.source_json):
        for r in json.load(open(args.source_json)):
            s = r.get("sequence") or r.get("mature_sequence")
            if s:
                acc[s] = r.get("_accession") or r.get("accession") or "?"

    records, missing, badmeta = [], [], []

    for job, meta in jobmap.items():
        ranked = collect_ranked(args.cf_out, job)
        if not ranked:
            missing.append(job)
            continue

        pairs = pick(meta, "intended_pairs", "true_pairs", "pairs",
                     "intended", "topology")
        if pairs is None:
            badmeta.append(job)
            continue
        seq = pick(meta, "sequence", "seq", default="")
        group = pick(meta, "label", "template", "group", default="?")

        true_t = as_key(pairs)
        preds, plddts = {}, {}
        for rank, path in ranked.items():
            txt = open(path).read()
            preds[rank] = realized_pairing(txt, args.cutoff)
            plddts[rank] = mean_plddt(txt)

        hit_rank = next((r for r in sorted(preds) if preds[r] == true_t), None)
        records.append({
            "job": job,
            "group": group,
            "accession": acc.get(seq, "?"),
            "sequence": seq,
            "true": [list(p) for p in true_t],
            "preds": {str(r): [list(p) for p in preds[r]] for r in preds},
            "plddt": {str(r): plddts[r] for r in plddts},
            "hit_rank": hit_rank,
            "n_models": len(preds),
            "n_distinct": len(set(preds.values())),
            "rank1_correct": preds.get(1) == true_t,
        })

    if badmeta:
        print(f"WARNING: no intended-pairs key in jobmap for {len(badmeta)} "
              f"job(s): {badmeta[:6]}")
        print(f"         keys present: {list(jobmap[badmeta[0]].keys())}\n")
    if not records:
        print("No ranked PDBs found. Expected files like "
              "pep0001_unrelaxed_rank_001_*.pdb inside the directory.")
        return
    if missing:
        print(f"WARNING: no structures for {len(missing)} job(s): "
              f"{missing[:6]}\n")

    # ---------------- per peptide ----------------
    print("=" * 78)
    print("PER PEPTIDE -- all five ranked models")
    print("=" * 78)

    by_group = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)

    for g in sorted(by_group, key=str):
        print(f"\n### {g}")
        for r in by_group[g]:
            flag = ("rank1 OK" if r["rank1_correct"]
                    else f"true at rank {r['hit_rank']}" if r["hit_rank"]
                    else "TRUE NEVER APPEARS")
            true_k = as_key(r["true"])
            print(f"\n  {r['accession']:<12} true={fmt(true_k)}   [{flag}]")
            for rank in sorted(r["preds"], key=int):
                p = tuple(map(tuple, r["preds"][rank]))
                mark = "  <-- true" if p == true_k else ""
                pl = r["plddt"].get(rank)
                pl_s = f"pLDDT {pl:5.1f}" if pl is not None else "pLDDT   n/a"
                print(f"      rank {rank}  {pl_s}   {fmt(p)}{mark}")
            print(f"      distinct topologies across the 5 models: "
                  f"{r['n_distinct']}")

    # ---------------- summary ----------------
    n = len(records)
    r1 = sum(1 for r in records if r["rank1_correct"])
    anywhere = sum(1 for r in records if r["hit_rank"] is not None)
    recovered = anywhere - r1                       # wrong at 1, right lower
    uniform = sum(1 for r in records if r["n_distinct"] == 1)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  peptides analysed                        : {n}")
    print(f"  true topology correct at rank_001        : {r1}/{n}")
    print(f"  true topology present in ANY of 5 models : {anywhere}/{n}")
    print(f"  RECOVERABLE (wrong at 1, right lower)    : {recovered}/{n}")
    print(f"  ensembles where all 5 models agree       : {uniform}/{n}")
    print()
    print(f"  distinct-topology counts across peptides : "
          f"{dict(sorted(Counter(r['n_distinct'] for r in records).items()))}")

    # does a correct lower-ranked model carry lower confidence?
    deltas = []
    for r in records:
        if r["hit_rank"] and r["hit_rank"] != 1:
            a, b = r["plddt"].get("1"), r["plddt"].get(str(r["hit_rank"]))
            if a is not None and b is not None:
                deltas.append(b - a)
    if deltas:
        print("\n  pLDDT of the correct lower-ranked model minus rank_001:")
        print(f"    mean {sum(deltas)/len(deltas):+.2f} over "
              f"{len(deltas)} case(s)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if recovered == 0 and anywhere == r1:
        print("  ABSENCE. Where rank_001 is wrong, the true topology appears in")
        print("  NO other ranked model. The ensemble carries no alternative, so")
        print("  the finding is not a ranking artefact -- it hardens the claim")
        print("  against the obvious reviewer question.")
    elif recovered > 0:
        print(f"  MIS-RANKING in {recovered} case(s). The true topology IS")
        print("  represented but is not ranked first. The information is")
        print("  recoverable, which changes the claim from 'the model cannot")
        print("  represent this' to 'the model cannot RANK this' -- a different")
        print("  and more actionable finding. Report the hit-rank distribution,")
        print("  and check whether MSA subsampling surfaces it more often.")
    if uniform == n:
        print("\n  Note: all five models agreed for EVERY peptide. The ensemble")
        print("  is degenerate here, so rank order carries no extra information.")

    print("\n  CAVEAT: only SP3 (P0CAP0 vs P85123) has experimental annotation")
    print("  on both sides. A 'true topology at rank 3' in SP1/SP2/SP4 may be")
    print("  the label being wrong rather than the model being right.")

    json.dump(records, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()