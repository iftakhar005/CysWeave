#!/usr/bin/env python3
"""
check_template_leakage.py
=========================
Checks whether the GENERATION TEMPLATES (from PDB, ground_truth.json) also
appear in the TRAINING SET (from UniProt, train.json).

Why this matters
----------------
The templates and the training data were curated from different sources and
never cross-checked. If a template — or a close homolog — is in the training
set, then masking 60% of it and asking the model to fill it back in is partly a
MEMORY test, not a design test: the model can recover the native residues it
memorised, which gives the native topology "for free".

That would mean the 70.9% -> 85% improvement measures memorisation rather than
improved generation, and the comparison would need re-running on templates that
are genuinely held out.

Usage:
    python check_template_leakage.py
    python check_template_leakage.py --threshold 0.8

Reports, for each template, the most similar training sequence and its identity.
"""

import argparse
import json
import os
from difflib import SequenceMatcher


def load(path):
    if not os.path.exists(path):
        print(f"  (missing: {path})")
        return None
    return json.load(open(path))


def identity(a, b):
    """Approximate sequence identity (alignment-free, good enough to flag leaks)."""
    return SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="ground_truth.json")
    ap.add_argument("--train", default="train.json")
    ap.add_argument("--test", default="test.json")
    ap.add_argument("--templates", nargs="*", default=[
        "7RC7_A", "2LG5_A", "4AB0_A", "1ICA_A", "2B68_A", "1HVZ_A", "1N4N_A"])
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="identity above this counts as leakage")
    args = ap.parse_args()

    gt = load(args.ground_truth)
    train = load(args.train)
    test = load(args.test)
    if gt is None or train is None:
        print("\nRun this in the folder holding ground_truth.json and train.json.")
        return

    by_id = {f"{r['pdb_id']}_{r['chain']}": r for r in gt}
    train_seqs = [r["mature_sequence"] for r in train]
    test_seqs = [r["mature_sequence"] for r in test] if test else []

    print(f"templates: {len(args.templates)}   train: {len(train_seqs)}   "
          f"test: {len(test_seqs)}\n")
    print(f"{'template':<9}{'len':>5}{'best train match':>19}{'in test?':>12}")
    print("-" * 60)

    leaked, exact = [], []
    for t in args.templates:
        if t not in by_id:
            print(f"{t:<9}  (not in ground_truth.json)")
            continue
        s = by_id[t]["sequence"]

        best_tr = max((identity(s, x), x) for x in train_seqs) if train_seqs else (0, "")
        best_te = max((identity(s, x), x) for x in test_seqs) if test_seqs else (0, "")

        if best_tr[0] >= 0.999:
            exact.append(t)
        elif best_tr[0] >= args.threshold:
            leaked.append((t, best_tr[0]))

        mark = "  <-- EXACT" if best_tr[0] >= 0.999 else (
               "  <-- LEAK" if best_tr[0] >= args.threshold else "")
        print(f"{t:<9}{len(s):>5}{100*best_tr[0]:>17.1f}%"
              f"{100*best_te[0]:>11.1f}%{mark}")

    print("-" * 60)
    print()
    if exact:
        print(f"EXACT MATCHES IN TRAINING SET: {', '.join(exact)}")
        print("  These templates ARE training examples. Masked reconstruction on")
        print("  them is partly memory recall, so the fidelity gain on these")
        print("  templates cannot be attributed to improved generation.")
    if leaked:
        print(f"CLOSE HOMOLOGS IN TRAINING SET (>={100*args.threshold:.0f}%): "
              f"{', '.join(t for t, _ in leaked)}")
        print("  Near-duplicates carry the same risk as exact matches.")
    if not exact and not leaked:
        print("NO LEAKAGE DETECTED. Templates are genuinely held out and the")
        print("baseline-vs-conditioned comparison stands as measured.")
    else:
        print()
        print("WHAT TO DO:")
        print("  1. Re-run the baseline and conditioned arms on templates whose")
        print("     best training identity is below the threshold.")
        print("  2. Or exclude template sequences (and their clusters) from")
        print("     train.json and retrain.")
        print("  3. Either way, report template-vs-training identity in the paper.")


if __name__ == "__main__":
    main()