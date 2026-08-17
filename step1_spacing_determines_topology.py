#!/usr/bin/env python3
"""
step1_spacing_determines_topology.py
====================================
THE DIAGNOSTIC THAT DECIDES THE PROJECT'S DIRECTION.

Question
--------
Is a peptide's disulfide topology determined by its CYSTEINE SPACING alone
(how many residues sit between consecutive cysteines), or does the intervening
sequence carry information about the pairing too?

Why it matters
--------------
In our setup cysteine positions are always held fixed, so the spacing is always
given to the model. If spacing determines topology, then the conditioning vector
is redundant by construction — the model can infer the answer without it, which
is exactly why three controls found the conditioning to be non-causal.

    SPACING DETERMINES TOPOLOGY  -> sequence-level conditioning CANNOT work here.
                                    The method must control spacing instead.
                                    (Our negative result is then explained, not
                                     just observed.)

    SPACING DOES NOT DETERMINE   -> there is room for the sequence to matter, so
                                    the failure was a training-setup problem and
                                    is fixable without changing the task.

Two independent tests
---------------------
1. COLLISION TEST (no ML, fully interpretable): do any two sequences share an
   identical spacing pattern but carry DIFFERENT topologies? Each collision is a
   direct counterexample to "spacing determines topology".

2. PREDICTABILITY TEST: how accurately can topology be predicted from spacing
   alone? Uses a nearest-neighbour rule with leave-one-out, so no dependencies
   beyond the standard library. High accuracy = spacing is near-deterministic.

Usage
-----
    python step1_spacing_determines_topology.py
    python step1_spacing_determines_topology.py --input ground_truth.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict


# --------------------------------------------------------------------------

def get_sequence(rec):
    """Datasets differ: UniProt records use mature_sequence, PDB use sequence."""
    return rec.get("mature_sequence") or rec.get("sequence")


def spacing_pattern(seq):
    """Residues BETWEEN consecutive cysteines.

    'ACYCRIPACIAGERRYGTCIYQGRLWAFCC' has cysteines at 1,3,8,18,28,29
    -> gaps [1, 4, 9, 9, 0]
    This is the classic 'cysteine framework' descriptor.
    """
    pos = [i for i, a in enumerate(seq) if a == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def topology_key(rec):
    return tuple(sorted(tuple(sorted(p)) for p in rec["pairs"]))


# --------------------------------------------------------------------------

def collision_test(records):
    """Same spacing, different topology = counterexample."""
    by_spacing = defaultdict(set)
    examples = defaultdict(list)
    for r in records:
        sp = spacing_pattern(get_sequence(r))
        tp = topology_key(r)
        by_spacing[sp].add(tp)
        examples[(sp, tp)].append(r)

    ambiguous = {sp: tps for sp, tps in by_spacing.items() if len(tps) > 1}
    total_patterns = len(by_spacing)

    print("=" * 74)
    print("TEST 1 — COLLISION TEST")
    print("=" * 74)
    print(f"distinct spacing patterns: {total_patterns}")
    print(f"patterns mapping to MORE THAN ONE topology: {len(ambiguous)}")
    if total_patterns:
        print(f"  -> {100*len(ambiguous)/total_patterns:.1f}% of spacing "
              f"patterns are ambiguous")
    print()

    if not ambiguous:
        print("NO COLLISIONS. Every spacing pattern in the dataset maps to")
        print("exactly one topology — spacing is fully determining here.")
    else:
        print("COLLISIONS FOUND — spacing does NOT fully determine topology.")
        print("Examples (same spacing, different pairing):\n")
        for k, (sp, tps) in enumerate(list(ambiguous.items())[:5]):
            print(f"  spacing {list(sp)}")
            for tp in tps:
                rec = examples[(sp, tp)][0]
                acc = rec.get("accession") or f"{rec.get('pdb_id')}_{rec.get('chain')}"
                print(f"     -> {[list(p) for p in tp]}   e.g. {acc}")
            print()
    return ambiguous, by_spacing


def predictability_test(records):
    """Leave-one-out nearest-neighbour: predict topology from spacing alone.

    Compares only within the same cysteine count (topologies are not comparable
    across different numbers of cysteines).
    """
    print("=" * 74)
    print("TEST 2 — PREDICTABILITY TEST (leave-one-out nearest neighbour)")
    print("=" * 74)

    by_ncys = defaultdict(list)
    for r in records:
        seq = get_sequence(r)
        by_ncys[seq.count("C")].append(
            (spacing_pattern(seq), topology_key(r), seq))

    print(f"{'n_cys':>6}{'n_seqs':>9}{'accuracy':>11}{'majority baseline':>20}")
    print("-" * 74)

    overall_correct = overall_n = 0
    for ncys in sorted(by_ncys):
        group = by_ncys[ncys]
        if len(group) < 10:
            continue
        correct = 0
        for i, (sp_i, tp_i, _) in enumerate(group):
            best, best_d = None, None
            for j, (sp_j, tp_j, _) in enumerate(group):
                if i == j:
                    continue
                d = sum(abs(a - b) for a, b in zip(sp_i, sp_j))
                if best_d is None or d < best_d:
                    best_d, best = d, tp_j
            correct += int(best == tp_i)
        acc = 100 * correct / len(group)
        maj = Counter(t for _, t, _ in group).most_common(1)[0][1]
        base = 100 * maj / len(group)
        overall_correct += correct
        overall_n += len(group)
        print(f"{ncys:>6}{len(group):>9}{acc:>10.1f}%{base:>19.1f}%")

    print("-" * 74)
    if overall_n:
        acc = 100 * overall_correct / overall_n
        print(f"{'ALL':>6}{overall_n:>9}{acc:>10.1f}%")
        print()
        print("Compare accuracy against the majority baseline. Accuracy far above")
        print("baseline means spacing carries most of the topology information.")
    return (100 * overall_correct / overall_n) if overall_n else 0.0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="defensin_connectivity_dataset.json")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        alt = "ground_truth.json"
        if os.path.exists(alt):
            print(f"({args.input} not found — using {alt})\n")
            args.input = alt
        else:
            print(f"Could not find {args.input}. Run this in the project folder.")
            return

    records = [r for r in json.load(open(args.input))
               if get_sequence(r) and r.get("pairs")]
    print(f"loaded {len(records)} sequences from {args.input}\n")

    ambiguous, by_spacing = collision_test(records)
    print()
    acc = predictability_test(records)

    print()
    print("=" * 74)
    print("WHAT THIS MEANS")
    print("=" * 74)
    frac_amb = len(ambiguous) / len(by_spacing) if by_spacing else 0
    if frac_amb < 0.05 and acc > 85:
        print("SPACING DETERMINES TOPOLOGY.")
        print("  Sequence-level conditioning cannot work while cysteine positions")
        print("  are fixed — the signal is redundant by construction. This EXPLAINS")
        print("  the null/scramble/swap results rather than merely restating them,")
        print("  and it says a working method must control SPACING, not sequence.")
    elif frac_amb > 0.15 or acc < 70:
        print("SPACING DOES NOT DETERMINE TOPOLOGY.")
        print("  The intervening sequence carries real information about pairing,")
        print("  so there IS room for sequence-level conditioning to work. The")
        print("  failure was a training-setup problem, not an impossibility.")
    else:
        print("PARTIALLY DETERMINING.")
        print("  Spacing carries most but not all of the signal. Focus the method")
        print("  on the ambiguous spacing patterns listed above — those are the")
        print("  cases where conditioning could actually change the outcome.")
    print("=" * 74)


if __name__ == "__main__":
    main()