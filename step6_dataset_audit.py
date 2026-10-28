#!/usr/bin/env python3
"""
step6_dataset_audit.py
======================
Integrity audit of the peptide dataset. NOT a rebalancing tool.

Why not rebalancing
-------------------
The dataset is now TEST material for a diagnostic about structure predictors,
not training material. The alpha/beta imbalance is not an artefact we
introduced -- it is the real composition of annotated defensin sequence data,
and it is part of the EVIDENCE for the mechanism (the 4.16:1 sequence-side
skew). Rebalancing would delete our own finding. What matters instead is
whether the data is CORRECT and whether our claims about it are accurate.

Three checks, in order of how much they could change a claim
-----------------------------------------------------------
CHECK 1 (highest stakes) - REDUNDANCY INSIDE THE DISCRIMINATION GROUPS.
    We plan to lead with "44 genuinely different sequences, one identical
    prediction". If most of those 44 are >90%-identical paralogs, "44 diverse
    sequences" overstates it and the wording must change. This measures the
    real diversity of each group and reports an effective (non-redundant) count.

CHECK 2 - CONTRADICTORY ANNOTATIONS.
    We already found P46156 vs P80389: 96.2% identical, different recorded
    topologies. Two peptides that similar cannot plausibly fold differently, so
    one label is wrong. This scans the whole dataset for the same pattern.

CHECK 3 - EXPERIMENTAL-ONLY RECOUNT.
    ~86% of labels are inferred-by-similarity. The headline topology counts
    should be reported for experimentally-backed entries too, so a reviewer can
    see the conclusion does not rest on propagated annotations.

Usage
-----
    python step6_dataset_audit.py
    python step6_dataset_audit.py --identity-threshold 0.90
    python step6_dataset_audit.py --groups metric_discrimination_full.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from difflib import SequenceMatcher

TOPO_NAMES = {
    ((1, 6), (2, 4), (3, 5)): "alpha-defensin",
    ((1, 5), (2, 4), (3, 6)): "beta-defensin",
    ((1, 4), (2, 5), (3, 6)): "cystine-knot",
    ((1, 6), (2, 5), (3, 4)): "theta-defensin",
    ((1, 8), (2, 5), (3, 6), (4, 7)): "CSab plant defensin",
}


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


def topo_key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def name_of(rec):
    return (rec.get("accession") or rec.get("_accession")
            or (f"{rec.get('pdb_id')}_{rec.get('chain')}"
                if rec.get("pdb_id") else "?"))


def ident(a, b):
    return SequenceMatcher(None, a, b).ratio()


def greedy_clusters(seqs, thr):
    """Cluster by identity, longest-first as representative. Returns cluster ids."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    reps, assign = [], [None] * len(seqs)
    for i in order:
        for cid, ri in enumerate(reps):
            if ident(seqs[i], seqs[ri]) >= thr:
                assign[i] = cid
                break
        else:
            assign[i] = len(reps)
            reps.append(i)
    return assign, len(reps)


# ==========================================================================
# CHECK 1 - redundancy inside the discrimination groups
# ==========================================================================

def check1(args):
    print("=" * 78)
    print("CHECK 1 - HOW DIVERSE ARE THE DISCRIMINATION GROUPS, REALLY?")
    print("=" * 78)
    if not os.path.exists(args.groups):
        print(f"  {args.groups} not found - skipping")
        return
    recs = json.load(open(args.groups))
    by_group = defaultdict(list)
    for r in recs:
        by_group[r.get("template", "?")].append(r)

    print(f"\n{'group':<8}{'n':>5}{'clusters':>10}{'mean id':>10}"
          f"{'max pair id':>13}   claim to use")
    print("-" * 78)
    for g in sorted(by_group):
        rows = by_group[g]
        seqs = [get_sequence(r) for r in rows]
        _, ncl = greedy_clusters(seqs, args.identity_threshold)
        pairs = [ident(seqs[i], seqs[j])
                 for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
        mean_id = sum(pairs) / len(pairs) if pairs else 0
        max_id = max(pairs) if pairs else 0
        if ncl >= 0.7 * len(rows):
            claim = f"'{len(rows)} diverse sequences' OK"
        elif ncl >= 0.3 * len(rows):
            claim = f"say '{len(rows)} seqs, {ncl} clusters'"
        else:
            claim = f"REWORD: only {ncl} independent"
        print(f"{g:<8}{len(rows):>5}{ncl:>10}{100*mean_id:>9.1f}%"
              f"{100*max_id:>12.1f}%   {claim}")

    print(f"\n  (clusters computed at {100*args.identity_threshold:.0f}% identity)")
    print("""
  HOW TO READ
    clusters ~= n      -> the group really is diverse; the strong claim holds.
    clusters << n      -> most members are near-duplicates. The finding still
                          stands (the model still failed to discriminate), but
                          describe it as "N sequences in C independent clusters"
                          rather than "N diverse sequences".""")


# ==========================================================================
# CHECK 2 - contradictory annotations
# ==========================================================================

def check2(args):
    print("\n" + "=" * 78)
    print("CHECK 2 - NEAR-IDENTICAL SEQUENCES WITH DIFFERENT RECORDED TOPOLOGIES")
    print("=" * 78)
    if not os.path.exists(args.dataset):
        print(f"  {args.dataset} not found - skipping")
        return
    recs = [r for r in json.load(open(args.dataset))
            if get_sequence(r) and r.get("pairs")]

    # only compare within the same cysteine count (else topologies aren't comparable)
    by_ncys = defaultdict(list)
    for r in recs:
        by_ncys[get_sequence(r).count("C")].append(r)

    suspects = []
    for ncys, rows in by_ncys.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if topo_key(a["pairs"]) == topo_key(b["pairs"]):
                    continue
                sa, sb = get_sequence(a), get_sequence(b)
                # cheap length prefilter before the expensive compare
                if abs(len(sa) - len(sb)) > 0.25 * max(len(sa), len(sb)):
                    continue
                v = ident(sa, sb)
                if v >= args.contradiction_threshold:
                    suspects.append((v, ncys, a, b))

    suspects.sort(key=lambda x: -x[0])
    if not suspects:
        print(f"\n  none found at >={100*args.contradiction_threshold:.0f}% identity."
              "\n  (the known P46156/P80389 case should appear if the threshold"
              "\n   is set at or below ~0.96)")
    else:
        print(f"\n  {len(suspects)} suspicious pair(s) at "
              f">={100*args.contradiction_threshold:.0f}% identity:\n")
        for v, ncys, a, b in suspects[:20]:
            print(f"  {100*v:5.1f}%  {ncys}-cys   {name_of(a)} vs {name_of(b)}")
            print(f"           {[list(p) for p in topo_key(a['pairs'])]}"
                  f"  ({a.get('evidence_class','?')})")
            print(f"           {[list(p) for p in topo_key(b['pairs'])]}"
                  f"  ({b.get('evidence_class','?')})")
            print()
        print("  Two sequences this similar are unlikely to fold differently.")
        print("  Where one side is 'inferred' and the other 'experimental',")
        print("  the inferred label is the more likely error. Flag these in the")
        print("  paper rather than silently dropping them.")


# ==========================================================================
# CHECK 3 - experimental-only recount
# ==========================================================================

def check3(args):
    print("\n" + "=" * 78)
    print("CHECK 3 - DO THE HEADLINE COUNTS SURVIVE ON EXPERIMENTAL LABELS ONLY?")
    print("=" * 78)
    if not os.path.exists(args.dataset):
        print(f"  {args.dataset} not found - skipping")
        return
    recs = [r for r in json.load(open(args.dataset))
            if get_sequence(r) and r.get("pairs")]

    ev = Counter(r.get("evidence_class", "unknown") for r in recs)
    print(f"\n  evidence classes: {dict(ev)}")

    strata = [
        ("ALL", lambda r: True),
        ("experimental+mixed", lambda r: r.get("evidence_class") in
         ("experimental", "mixed")),
        ("experimental only", lambda r: r.get("evidence_class") == "experimental"),
    ]

    for ncys in (6, 8):
        print(f"\n  --- {ncys} cysteines ---")
        print(f"  {'stratum':<22}{'n':>6}{'alpha':>8}{'beta':>8}"
              f"{'knot':>8}{'beta/alpha':>12}")
        for label, keep in strata:
            sub = [r for r in recs
                   if keep(r) and get_sequence(r).count("C") == ncys]
            if not sub:
                continue
            c = Counter(topo_key(r["pairs"]) for r in sub)
            a = c[((1, 6), (2, 4), (3, 5))]
            b = c[((1, 5), (2, 4), (3, 6))]
            k = c[((1, 4), (2, 5), (3, 6))]
            ratio = f"{b/a:.2f}" if a else "n/a"
            print(f"  {label:<22}{len(sub):>6}{a:>8}{b:>8}{k:>8}{ratio:>12}")

    print("""
  HOW TO READ
    If the beta skew persists on the experimental-only rows, the sequence-side
    imbalance is real and not an artefact of annotation propagation. If it
    vanishes, the 4.16 figure is partly a labelling effect and must be reported
    with that caveat.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="defensin_connectivity_dataset.json")
    ap.add_argument("--groups", default="metric_discrimination_full.json")
    ap.add_argument("--identity-threshold", type=float, default=0.90,
                    help="clustering threshold for CHECK 1")
    ap.add_argument("--contradiction-threshold", type=float, default=0.85,
                    help="flag different-topology pairs above this identity")
    args = ap.parse_args()

    check1(args)
    check2(args)
    check3(args)

    print("\n" + "=" * 78)
    print("REMINDER: this audit is about CORRECTNESS, not balance. Do not")
    print("rebalance the dataset -- the alpha/beta skew is the finding, not a bug.")
    print("=" * 78)


if __name__ == "__main__":
    main()