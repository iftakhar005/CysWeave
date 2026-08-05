#!/usr/bin/env python3
"""
step0b_check_fidelity.py
========================
Step 0 Part A: the go/no-go check.

Reads ground_truth.json (from step0a), folds each sequence with ESMFold, infers
the disulfide connectivity from the predicted 3D structure, and compares it with
the experimentally known connectivity.

The single number this produces is: how often does the predicted structure
reproduce the KNOWN cysteine pairing?

    >= ~80%  -> the metric and the folder are trustworthy. Proceed.
    <  ~80%  -> ESMFold cannot be your oracle for disulfides. Switch predictor
                (AlphaFold2/3, Boltz) or add a dedicated cysteine-pairing
                predictor. This is itself a reportable finding.

Usage:
    pip install requests
    python step0b_check_fidelity.py                 # all entries
    python step0b_check_fidelity.py --limit 25      # first 25 only
    python step0b_check_fidelity.py --cutoff 3.5    # looser bonding cutoff

No GPU required: folding goes through the public ESMFold API.
"""

import argparse
import json
import math
import time
from itertools import combinations

import requests

ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"


# --------------------------------------------------------------------------
# Read connectivity out of a predicted structure
# --------------------------------------------------------------------------

def parse_cys_sg(pdb_text):
    """[(cys_index, resseq, (x,y,z))] for CYS SG atoms, ordered along the chain."""
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
    return [(i + 1, rs, xyz) for i, (rs, xyz) in enumerate(atoms)]


def _dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def realized_pairing(pdb_text, max_bond_dist=3.0):
    """Greedy nearest-neighbour matching on SG-SG distance.
    -> (bonds as set of frozensets, {bond: distance}, n_cys)"""
    cys = parse_cys_sg(pdb_text)
    cand = sorted((_dist(xi, xj), i, j)
                  for (i, _, xi), (j, _, xj) in combinations(cys, 2))
    used, bonds, dists = set(), set(), {}
    for d, i, j in cand:
        if i in used or j in used or d > max_bond_dist:
            continue
        key = frozenset((i, j))
        bonds.add(key)
        dists[key] = round(d, 2)
        used.update((i, j))
    return bonds, dists, len(cys)


def to_set(pairs):
    return {frozenset(p) for p in pairs}


def fmt(bonds):
    return sorted(tuple(sorted(b)) for b in bonds)


# --------------------------------------------------------------------------
# Folding
# --------------------------------------------------------------------------

def fold(sequence, retries=3, timeout=300):
    seq = "".join(sequence.split()).upper()
    for attempt in range(retries):
        try:
            r = requests.post(ESMFOLD_API, data=seq, timeout=timeout,
                              verify=False)
            r.raise_for_status()
            return r.text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ground_truth.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cutoff", type=float, default=3.0,
                    help="max SG-SG distance (A) counted as a disulfide bond")
    ap.add_argument("--out", default="step0_results.json")
    args = ap.parse_args()

    data = json.load(open(args.input))
    if args.limit:
        data = data[:args.limit]

    print(f"Checking {len(data)} peptides (bond cutoff {args.cutoff} A)\n")
    print(f"{'entry':<10}{'len':>4}{'cys':>4}  {'known':<26}{'predicted':<26}"
          f"{'exact':>6}")
    print("-" * 80)

    results, exact_n, partial_sum, folded_n = [], 0, 0.0, 0
    for rec in data:
        entry = f"{rec['pdb_id']}_{rec['chain']}"
        known = to_set(tuple(p) for p in rec["pairs"])
        try:
            pdb = fold(rec["sequence"])
        except Exception as e:
            print(f"{entry:<10} FOLD FAILED ({e})")
            continue

        pred, dists, ncys = realized_pairing(pdb, args.cutoff)
        folded_n += 1
        is_exact = (pred == known)
        exact_n += int(is_exact)
        frac = len(known & pred) / len(known) if known else 0.0
        partial_sum += frac

        print(f"{entry:<10}{len(rec['sequence']):>4}{ncys:>4}  "
              f"{str(fmt(known)):<26}{str(fmt(pred)):<26}{str(is_exact):>6}")

        results.append({
            "entry": entry, "sequence": rec["sequence"],
            "known": fmt(known), "predicted": fmt(pred),
            "exact": is_exact, "fraction_correct": round(frac, 3),
            "sg_distances": {str(sorted(k)): v for k, v in dists.items()},
        })
        time.sleep(1)                      # be polite to the API

    json.dump(results, open(args.out, "w"), indent=2)

    if not folded_n:
        print("\nNothing folded successfully - check your network / the API.")
        return

    pct = 100 * exact_n / folded_n
    print("\n" + "=" * 80)
    print(f"EXACT connectivity recovered: {exact_n}/{folded_n} = {pct:.0f}%")
    print(f"Mean fraction of bonds correct: {100 * partial_sum / folded_n:.0f}%")
    print("=" * 80)
    if pct >= 80:
        print("\n>= 80%: metric and folder look trustworthy. Proceed to Part B.")
    else:
        print("\n< 80%: ESMFold is NOT reliable for these peptides.")
        print("  - try a looser --cutoff (e.g. 3.5) and re-run")
        print("  - then try AlphaFold2/3 or Boltz, or a dedicated")
        print("    cysteine-pairing predictor")
        print("  - if nothing works, that unreliability is itself your finding")
    print(f"\nPer-entry detail written to {args.out}")
    print("Inspect sg_distances: values near 2.0 A mean the folder is confident;")
    print("values near the cutoff mean the prediction is soft.")


if __name__ == "__main__":
    main()