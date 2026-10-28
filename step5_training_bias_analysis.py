#!/usr/bin/env python3
"""
step5_training_bias_analysis.py
===============================
WHY do ESMFold and AlphaFold both default to the beta-defensin topology?

The hypothesis
--------------
The models learned a prior from their training data. If disulfide-rich peptides
with the beta pattern [1,5],[2,4],[3,6] massively outnumber the alternatives,
a model that just predicts the majority pattern for a given cysteine spacing
would look exactly like what we observed: 0/4 groups discriminated, everything
predicted as beta.

Why this needs care: ESMFold has TWO parts with DIFFERENT training data.

    ESM-2 language model  <- trained on UniRef SEQUENCES
    folding head          <- trained on PDB STRUCTURES

So there are two separate imbalances to measure, and they can disagree. A first
look at the curated defensin sets suggests they DO disagree:

    PDB defensins (ground_truth.json)   alpha 35 vs beta 31   (~balanced)
    UniProt set (1104 sequences)        beta 283 vs alpha 68  (~4:1 toward beta)

If that holds at scale, the bias is more likely inherited from the SEQUENCE
side (ESM-2) than the structure side -- which is a sharper and more interesting
claim than "the training data was skewed".

What this script does
---------------------
PART 1 (offline, instant): recount both of your existing datasets properly,
       broken down by cysteine count, so the comparison is apples-to-apples.

PART 2 (needs network): query RCSB PDB for ALL short disulfide-rich peptide
       structures -- not just ones whose text mentions "defensin" -- parse their
       SSBOND records, and count topology frequencies across the whole set.
       This is the real test of the structure-side imbalance, because the
       folding head trained on everything, not only on defensins.

Usage
-----
    python step5_training_bias_analysis.py                  # Part 1 only
    python step5_training_bias_analysis.py --query-pdb      # Parts 1 and 2
    python step5_training_bias_analysis.py --query-pdb --max-len 60 --limit 600
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

TOPO_NAMES = {
    ((1, 6), (2, 4), (3, 5)): "alpha-defensin",
    ((1, 5), (2, 4), (3, 6)): "beta-defensin",
    ((1, 4), (2, 5), (3, 6)): "cystine-knot / ICK",
    ((1, 6), (2, 5), (3, 4)): "theta-defensin",
    ((1, 8), (2, 5), (3, 6), (4, 7)): "CSab plant defensin",
    ((1, 5), (2, 6), (3, 7), (4, 8)): "ladder (4SS)",
}

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


def topo_key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def show(counter, total, indent="  ", top=10):
    for t, n in counter.most_common(top):
        name = TOPO_NAMES.get(t, "")
        pct = 100 * n / total if total else 0
        print(f"{indent}{n:>5}  ({pct:>5.1f}%)  {name:<22} {[list(p) for p in t]}")


# ==========================================================================
# PART 1 -- recount the datasets you already have
# ==========================================================================

def part1(args):
    print("=" * 78)
    print("PART 1 - TOPOLOGY BALANCE IN YOUR EXISTING DATASETS")
    print("=" * 78)

    summary = {}

    for label, path, src in [
        ("PDB structures (ground_truth.json)", args.pdb_json, "structure"),
        ("UniProt sequences (defensin_connectivity_dataset.json)",
         args.uniprot_json, "sequence"),
    ]:
        if not os.path.exists(path):
            print(f"\n{label}: FILE NOT FOUND ({path}) - skipping")
            continue
        recs = [r for r in json.load(open(path))
                if get_sequence(r) and r.get("pairs")]
        print(f"\n### {label}   [n={len(recs)}]")
        print("    (this is the training source for the "
              f"{'folding head' if src == 'structure' else 'ESM-2 language model'})")

        by_ncys = defaultdict(Counter)
        for r in recs:
            by_ncys[get_sequence(r).count("C")][topo_key(r["pairs"])] += 1

        for ncys in sorted(by_ncys):
            tot = sum(by_ncys[ncys].values())
            if tot < 5:
                continue
            print(f"\n  {ncys} cysteines  (n={tot})")
            show(by_ncys[ncys], tot, indent="    ")

        six = by_ncys.get(6, Counter())
        a = six[((1, 6), (2, 4), (3, 5))]
        b = six[((1, 5), (2, 4), (3, 6))]
        summary[label] = (a, b)

    print("\n" + "-" * 78)
    print("HEAD-TO-HEAD, 6-cysteine peptides only (alpha vs beta):")
    print("-" * 78)
    for label, (a, b) in summary.items():
        ratio = f"{b/a:.2f}" if a else "n/a"
        verdict = ("beta dominant" if a and b / a > 1.5 else
                   "alpha dominant" if b and a / b > 1.5 else "roughly balanced")
        print(f"  {label[:46]:<48} alpha={a:<5} beta={b:<5} "
              f"beta/alpha={ratio:<6} {verdict}")

    print("""
HOW TO READ THIS
  If the SEQUENCE set is beta-skewed while the STRUCTURE set is balanced, the
  bias more likely enters through ESM-2 (the language-model half) than through
  the folding head. That is a sharper claim than "training data was skewed",
  and it is testable: a structure predictor that does NOT use a pretrained
  protein language model should then be less biased.

  Caveat: these are YOUR curated subsets, not the models' actual training sets.
  Part 2 widens the structure side to all short disulfide-rich PDB entries,
  which is much closer to what the folding head actually saw.""")


# ==========================================================================
# PART 2 -- query the PDB for all short disulfide-rich peptides
# ==========================================================================

def parse_chain_residues(pdb_text, chain):
    seen, out = set(), []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        line = line.ljust(80)
        if line[12:16].strip() != "CA" or line[21] != chain:
            continue
        if line[16] not in (" ", "A"):
            continue
        resseq = line[22:27].strip()
        if resseq in seen:
            continue
        seen.add(resseq)
        out.append((resseq, line[17:20].strip()))
    return out


def parse_ssbond(pdb_text):
    bonds = []
    for line in pdb_text.splitlines():
        if not line.startswith("SSBOND"):
            continue
        line = line.ljust(80)
        bonds.append((line[15], (line[17:21] + line[21]).strip(),
                      line[29], (line[31:35] + line[35]).strip()))
    return bonds


def topology_from_pdb(pdb_text, chain):
    residues = parse_chain_residues(pdb_text, chain)
    seq = "".join(AA3TO1.get(rn, "X") for _, rn in residues)
    cys_index, k = {}, 0
    for resseq, rn in residues:
        if rn == "CYS":
            k += 1
            cys_index[resseq] = k
    pairs = []
    for c1, r1, c2, r2 in parse_ssbond(pdb_text):
        if c1 != chain or c2 != chain:
            continue
        if r1 in cys_index and r2 in cys_index:
            pairs.append(tuple(sorted((cys_index[r1], cys_index[r2]))))
    return seq, sorted(set(pairs)), len(cys_index)


def part2(args):
    import requests

    print("\n" + "=" * 78)
    print("PART 2 - TOPOLOGY BALANCE ACROSS *ALL* SHORT DISULFIDE-RICH PDB ENTRIES")
    print("=" * 78)
    print("(no text filter -- this is closer to what the folding head trained on)\n")

    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "less_or_equal", "value": args.max_len}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.disulfide_bond_count",
                "operator": "greater_or_equal", "value": 2}},
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": args.limit}},
    }
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                      json=query, timeout=90)
    r.raise_for_status()
    ids = [h["identifier"] for h in r.json().get("result_set", [])]
    print(f"found {len(ids)} candidate PDB entries (cap {args.limit})\n")

    counts, per_ncys, kept, skipped = Counter(), defaultdict(Counter), 0, 0
    seen_seqs = set()

    for n, pid in enumerate(ids, 1):
        try:
            txt = requests.get(f"https://files.rcsb.org/download/{pid}.pdb",
                               timeout=45).text
        except Exception:
            skipped += 1
            continue
        chains = sorted({l.ljust(80)[21] for l in txt.splitlines()
                         if l.startswith("ATOM")})
        for ch in chains:
            seq, pairs, ncys = topology_from_pdb(txt, ch)
            if not (10 <= len(seq) <= args.max_len):
                continue
            if len(pairs) < 2 or len(pairs) * 2 != ncys:
                continue
            if seq in seen_seqs:            # de-duplicate identical chains
                continue
            seen_seqs.add(seq)
            counts[topo_key(pairs)] += 1
            per_ncys[ncys][topo_key(pairs)] += 1
            kept += 1
        if n % 25 == 0:
            print(f"  processed {n}/{len(ids)} entries, kept {kept} chains")
        time.sleep(0.15)

    print(f"\ntotal unique chains kept: {kept}  (skipped {skipped} downloads)\n")

    for ncys in sorted(per_ncys):
        tot = sum(per_ncys[ncys].values())
        if tot < 5:
            continue
        print(f"{ncys} cysteines  (n={tot})")
        show(per_ncys[ncys], tot, indent="  ")
        print()

    six = per_ncys.get(6, Counter())
    a = six[((1, 6), (2, 4), (3, 5))]
    b = six[((1, 5), (2, 4), (3, 6))]
    print("-" * 78)
    print(f"6-cysteine PDB structures:  alpha={a}   beta={b}"
          + (f"   beta/alpha={b/a:.2f}" if a else ""))
    print("-" * 78)
    if a and b / a > 1.5:
        print("STRUCTURE SIDE IS BETA-SKEWED -> the folding head plausibly")
        print("  inherited the bias directly from PDB composition.")
    elif b and a / b > 1.5:
        print("STRUCTURE SIDE IS ALPHA-SKEWED -> the beta bias CANNOT come from")
        print("  PDB structure composition. It more likely enters via ESM-2,")
        print("  the sequence-trained half. That is the more interesting result.")
    else:
        print("STRUCTURE SIDE IS ROUGHLY BALANCED -> PDB composition does not")
        print("  explain the beta bias. Look to the sequence side (ESM-2), whose")
        print("  UniRef training data is far more beta-skewed.")

    json.dump({str(k): v for k, v in counts.items()},
              open("pdb_topology_counts.json", "w"), indent=2)
    print("\nwrote pdb_topology_counts.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-json", default="ground_truth.json")
    ap.add_argument("--uniprot-json",
                    default="defensin_connectivity_dataset.json")
    ap.add_argument("--query-pdb", action="store_true",
                    help="also run Part 2 (needs network; takes a while)")
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    part1(args)
    if args.query_pdb:
        part2(args)
    else:
        print("\n(Part 2 skipped. Re-run with --query-pdb to query the PDB.)")


if __name__ == "__main__":
    main()