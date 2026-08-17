#!/usr/bin/env python3
"""
step1b_annotation_circularity.py
================================
Is "spacing predicts topology" a fact about FOLDING, or about ANNOTATION?

The worry
---------
About 89% of our connectivity labels are ECO:0000250 — "by similarity". A
curator found a homolog with experimental data and copied its pairing across.
Similar sequences therefore receive the same annotated topology BY
CONSTRUCTION, and similar sequences have similar cysteine spacing.

So the step-1 result (spacing predicts topology at ~94.6%) may partly measure
how UniProt propagates annotations rather than how peptides actually fold. If
so, the mechanism we are using to explain the null/scramble/swap results is
built on sand.

The test
--------
Re-run the same analysis stratified by evidence class. Experimental labels
(ECO:0000269) were assigned from real structures, so they cannot be circular.

    high predictability on EXPERIMENTAL-only  -> the finding is real biology
    much lower on experimental than inferred  -> the finding is partly an
                                                 annotation artifact

Also checks whether the ambiguous (collision) spacing patterns rest on
experimental or inferred labels — a collision built from inferred labels may be
an annotation ERROR rather than genuine biological ambiguity, which matters
because those collisions are the test bed for everything downstream.

Usage
-----
    python step1b_annotation_circularity.py
"""

import argparse
import json
import os
from collections import Counter, defaultdict


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


def spacing_pattern(seq):
    pos = [i for i, a in enumerate(seq) if a == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def topology_key(rec):
    return tuple(sorted(tuple(sorted(p)) for p in rec["pairs"]))


def name_of(rec):
    return rec.get("accession") or f"{rec.get('pdb_id')}_{rec.get('chain')}"


def loo_accuracy(group):
    """Leave-one-out nearest-neighbour on spacing, within one cysteine count."""
    if len(group) < 10:
        return None, None, len(group)
    correct = 0
    for i, (sp_i, tp_i) in enumerate(group):
        best_d, best_tp = None, None
        for j, (sp_j, tp_j) in enumerate(group):
            if i == j:
                continue
            d = sum(abs(a - b) for a, b in zip(sp_i, sp_j))
            if best_d is None or d < best_d:
                best_d, best_tp = d, tp_j
        correct += int(best_tp == tp_i)
    acc = 100 * correct / len(group)
    maj = Counter(t for _, t in group).most_common(1)[0][1]
    return acc, 100 * maj / len(group), len(group)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="defensin_connectivity_dataset.json")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Could not find {args.input}. Run this in the project folder.")
        return

    records = [r for r in json.load(open(args.input))
               if get_sequence(r) and r.get("pairs")]
    for r in records:
        seq = get_sequence(r)
        r["_sp"] = spacing_pattern(seq)
        r["_tp"] = topology_key(r)
        r["_n"] = seq.count("C")
        r["_ev"] = r.get("evidence_class", "unknown")

    print(f"loaded {len(records)} sequences")
    print("evidence classes:", dict(Counter(r["_ev"] for r in records)), "\n")

    # ---------------- stratified predictability ----------------
    strata = {
        "experimental only": lambda r: r["_ev"] == "experimental",
        "experimental+mixed": lambda r: r["_ev"] in ("experimental", "mixed"),
        "inferred only": lambda r: r["_ev"] in ("inferred", "unknown"),
        "ALL": lambda r: True,
    }

    print("=" * 78)
    print("SPACING -> TOPOLOGY PREDICTABILITY, BY EVIDENCE CLASS")
    print("=" * 78)
    print(f"{'stratum':<22}{'n_cys':>6}{'n':>7}{'accuracy':>11}"
          f"{'baseline':>11}{'lift':>8}")
    print("-" * 78)

    summary = {}
    for label, keep in strata.items():
        subset = [r for r in records if keep(r)]
        by_n = defaultdict(list)
        for r in subset:
            by_n[r["_n"]].append((r["_sp"], r["_tp"]))
        shown = False
        for n in sorted(by_n):
            acc, base, cnt = loo_accuracy(by_n[n])
            if acc is None:
                continue
            shown = True
            print(f"{label if not shown or True else '':<22}{n:>6}{cnt:>7}"
                  f"{acc:>10.1f}%{base:>10.1f}%{acc-base:>+8.1f}")
            if n == 6:                       # 6-cys is the informative group
                summary[label] = (acc, base, cnt)
            label = ""                       # print stratum name once
        if not shown:
            print(f"{label:<22}{'—':>6}{len(subset):>7}"
                  f"{'too few to test':>30}")
        print("-" * 78)

    # ---------------- the verdict ----------------
    print("\n" + "=" * 78)
    print("IS THE FINDING CIRCULAR?")
    print("=" * 78)
    exp = summary.get("experimental only") or summary.get("experimental+mixed")
    inf = summary.get("inferred only")
    if exp and inf:
        e_lift = exp[0] - exp[1]
        i_lift = inf[0] - inf[1]
        print(f"  6-cysteine lift, experimental : {e_lift:+.1f} pts (n={exp[2]})")
        print(f"  6-cysteine lift, inferred     : {i_lift:+.1f} pts (n={inf[2]})")
        print()
        if e_lift >= 0.7 * i_lift and e_lift > 15:
            print("  NOT CIRCULAR. Spacing predicts topology about as well on")
            print("  experimentally annotated sequences as on inferred ones, so")
            print("  the finding reflects real structure, not annotation transfer.")
        elif e_lift < 0.4 * i_lift:
            print("  LARGELY CIRCULAR. Predictability collapses on experimental")
            print("  labels — step 1 was substantially measuring how UniProt")
            print("  propagates annotations by homology. The mechanism claim must")
            print("  be re-derived from experimental data only (PDB SSBOND), and")
            print("  the collision patterns must be re-checked.")
        else:
            print("  PARTLY CIRCULAR. Report the experimental-only numbers as the")
            print("  headline and the inferred ones separately.")
    else:
        print("  Not enough experimentally annotated sequences to compare.")
        print("  Fall back on the PDB set (ground_truth.json), whose pairings are")
        print("  all experimentally determined, and report that as the check.")

    # ---------------- collisions: real or annotation error? ----------------
    print("\n" + "=" * 78)
    print("ARE THE COLLISION PATTERNS EXPERIMENTALLY SUPPORTED?")
    print("=" * 78)
    by_sp = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_sp[r["_sp"]][r["_tp"]].append(r)
    ambiguous = {sp: t for sp, t in by_sp.items() if len(t) > 1}

    solid = 0
    for sp, tops in ambiguous.items():
        evs = {tp: Counter(x["_ev"] for x in recs) for tp, recs in tops.items()}
        n_exp_topos = sum(1 for c in evs.values()
                          if c["experimental"] + c["mixed"] > 0)
        ok = n_exp_topos >= 2
        solid += ok
        print(f"\n  spacing {list(sp)}   "
              f"{'EXPERIMENTALLY SUPPORTED' if ok else 'inferred labels only'}")
        for tp, recs in tops.items():
            c = evs[tp]
            names = ", ".join(name_of(x) for x in recs[:3])
            print(f"     {[list(p) for p in tp]}  n={len(recs)}  "
                  f"exp={c['experimental']} mixed={c['mixed']} "
                  f"inf={c['inferred'] + c['unknown']}   {names}")

    print("\n" + "-" * 78)
    print(f"collisions with >=2 experimentally supported topologies: "
          f"{solid}/{len(ambiguous)}")
    if solid == 0:
        print("  WARNING: no collision is backed by experimental labels on both")
        print("  sides. They may be annotation errors, not real ambiguity — do")
        print("  NOT build the next experiment on them without verification.")
    else:
        print("  These are genuine biological ambiguity and are a sound test bed.")


if __name__ == "__main__":
    main()