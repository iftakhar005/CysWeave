#!/usr/bin/env python3
"""
step2_build_ambiguity_set.py
============================
Builds the ONLY test bed on which our conditioning idea can be fairly judged.

Why
---
Step 1 showed cysteine spacing predicts topology for ~95% of 6-cysteine
defensins. On those sequences the scaffold already fixes the answer, so no
conditioning signal — ours or anyone's — could change the outcome. Our earlier
evaluation was ~95% made of exactly those cases, which is why null, scramble and
swap all found the conditioning inert.

The remaining ~5% is different. There, the same cysteine spacing occurs with
MORE THAN ONE topology in nature, so the pairing is genuinely underdetermined by
the scaffold and something else must decide it. If sequence-level conditioning
works anywhere, it works there.

This script finds those sequences and tells us whether there are enough of them
to run a focused experiment.

Three tiers of ambiguity
------------------------
  HARD  : the exact spacing pattern maps to >1 topology in the dataset.
  SOFT  : the nearest spacing neighbour (L1 distance) carries a DIFFERENT
          topology — spacing is not locally decisive.
  NEAR  : some sequence within L1 distance <= --radius carries a different
          topology, even if the closest one agrees.

Usage
-----
    python step2_build_ambiguity_set.py
    python step2_build_ambiguity_set.py --radius 3

Writes ambiguity_set.json (the focused subset, tier-labelled).
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


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else None


def name_of(rec):
    return rec.get("accession") or f"{rec.get('pdb_id')}_{rec.get('chain')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="defensin_connectivity_dataset.json")
    ap.add_argument("--radius", type=int, default=2,
                    help="L1 spacing distance counted as 'near'")
    ap.add_argument("--out", default="ambiguity_set.json")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Could not find {args.input}. Run this in the project folder.")
        return

    records = [r for r in json.load(open(args.input))
               if get_sequence(r) and r.get("pairs")]
    for r in records:
        r["_sp"] = spacing_pattern(get_sequence(r))
        r["_tp"] = topology_key(r)
        r["_n"] = get_sequence(r).count("C")
    print(f"loaded {len(records)} sequences\n")

    # ---------- HARD: exact spacing collisions ----------
    by_sp = defaultdict(set)
    for r in records:
        by_sp[r["_sp"]].add(r["_tp"])
    hard_patterns = {sp for sp, tps in by_sp.items() if len(tps) > 1}

    # ---------- SOFT / NEAR: neighbourhood ambiguity ----------
    by_n = defaultdict(list)
    for r in records:
        by_n[r["_n"]].append(r)

    for group in by_n.values():
        for r in group:
            nearest_d, nearest_tp = None, None
            near_conflict = False
            for o in group:
                if o is r:
                    continue
                d = l1(r["_sp"], o["_sp"])
                if d is None:
                    continue
                if nearest_d is None or d < nearest_d:
                    nearest_d, nearest_tp = d, o["_tp"]
                if d <= args.radius and o["_tp"] != r["_tp"]:
                    near_conflict = True
            r["_soft"] = (nearest_tp is not None and nearest_tp != r["_tp"])
            r["_near"] = near_conflict

    for r in records:
        if r["_sp"] in hard_patterns:
            r["_tier"] = "HARD"
        elif r["_soft"]:
            r["_tier"] = "SOFT"
        elif r["_near"]:
            r["_tier"] = "NEAR"
        else:
            r["_tier"] = "determined"

    # ---------- report ----------
    print("=" * 74)
    print("AMBIGUITY TIERS")
    print("=" * 74)
    tiers = Counter(r["_tier"] for r in records)
    for t in ("HARD", "SOFT", "NEAR", "determined"):
        c = tiers[t]
        print(f"  {t:<12}{c:>6}   ({100*c/len(records):>5.1f}%)")
    amb = [r for r in records if r["_tier"] != "determined"]
    print(f"\n  usable ambiguous pool: {len(amb)} sequences")

    print("\n--- by cysteine count ---")
    print(f"{'n_cys':>6}{'total':>8}{'ambiguous':>12}{'%':>8}")
    for n in sorted(by_n):
        g = by_n[n]
        a = sum(1 for r in g if r["_tier"] != "determined")
        print(f"{n:>6}{len(g):>8}{a:>12}{100*a/len(g):>7.1f}%")

    print("\n--- which topology pairs are confusable ---")
    pairs = Counter()
    for sp in hard_patterns:
        tps = sorted(by_sp[sp])
        for i in range(len(tps)):
            for j in range(i + 1, len(tps)):
                pairs[(tps[i], tps[j])] += 1
    for (a, b), c in pairs.most_common(10):
        print(f"  {[list(p) for p in a]}  vs  {[list(p) for p in b]}   x{c}")

    # ---------- write ----------
    out = []
    for r in amb:
        out.append({k: v for k, v in r.items() if not k.startswith("_")}
                   | {"tier": r["_tier"], "spacing": list(r["_sp"]),
                      "n_cys": r["_n"]})
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    n_amb = len(amb)
    if n_amb >= 150:
        print(f"{n_amb} ambiguous sequences — ENOUGH for a focused train/test.")
        print("  Next: train with conditioning on this subset (and evaluate on a")
        print("  held-out slice of it). Here the signal is NOT redundant, so if")
        print("  conditioning works at all, it must show up here.")
    elif n_amb >= 50:
        print(f"{n_amb} ambiguous sequences — enough to EVALUATE on, probably not")
        print("  to train on. Next: keep training on the full set but evaluate")
        print("  separately on this subset, and/or widen --radius.")
    else:
        print(f"only {n_amb} ambiguous sequences — too few for defensins alone.")
        print("  Next: widen --radius, or extend the dataset to related")
        print("  disulfide-rich families (conotoxins, knottins, cyclotides)")
        print("  where the same spacing supports different topologies.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()