#!/usr/bin/env python3
"""
step7_completeness_vs_length.py
===============================
WHY did ESMFold produce complete predictions for defensins but not conotoxins?

The observation
---------------
Two runs, two very different failure modes:

    defensins   (65 peptides)  ->  65/65 predictions COMPLETE, all identical
    conotoxins  (85 peptides)  ->  15/85 complete, 55 partial, 15 EMPTY

"Complete" means the predicted structure placed every cysteine close enough to a
partner to form a full set of disulfides. Partial/empty means ESMFold produced
no confident disulfide geometry -- it did not predict a WRONG topology, it
failed to predict one at all.

That distinction matters, because a partial prediction cannot test
discrimination. Only complete predictions can.

The hypothesis
--------------
Length. Conotoxins are typically 10-30 residues; defensins 30-50. Single-
sequence folding models degrade badly on very short peptides, which would show
up exactly as an inability to close disulfide geometry.

If true, this is a clean, reportable BOUNDARY CONDITION: the connectivity-
fidelity method only applies above some peptide length, and below it the oracle
simply has nothing to say. That is a useful limitation to state, and it also
tells any future user of the metric where it can and cannot be trusted.

What this does
--------------
Pools every folded result you have, joins each peptide to its length, and
reports completeness as a function of length -- overall, per dataset, and
binned. Also checks two alternative explanations: cysteine density (more
cysteines packed into fewer residues may be harder) and cysteine count.

Usage
-----
    python step7_completeness_vs_length.py
    python step7_completeness_vs_length.py --folds partb_conotoxin.jsonl \\
        partb_discrimination_full.jsonl
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def classify(rec):
    """complete / partial / empty, based on bonds found vs bonds expected."""
    want = len(rec.get("intended", []))
    got = len(rec.get("realized", []))
    if want == 0:
        return None
    if got == 0:
        return "empty"
    if got >= want:
        return "complete"
    return "partial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None,
                    help="folded .jsonl files (default: all partb_*.jsonl)")
    ap.add_argument("--bin-width", type=int, default=5)
    args = ap.parse_args()

    files = args.folds or sorted(glob.glob("partb_*.jsonl"))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("No folded .jsonl files found.")
        return

    rows = []
    seen = set()
    for f in files:
        for r in load_jsonl(f):
            seq = r.get("sequence")
            if not seq or seq in seen:
                continue
            seen.add(seq)
            cls = classify(r)
            if cls is None:
                continue
            rows.append({
                "file": os.path.basename(f),
                "seq": seq,
                "len": len(seq),
                "ncys": seq.count("C"),
                "cls": cls,
            })

    if not rows:
        print("No usable rows (need 'sequence', 'intended', 'realized').")
        return

    print(f"pooled {len(rows)} unique peptides from {len(files)} file(s)\n")

    # ---------------- per file ----------------
    print("=" * 74)
    print("COMPLETENESS BY SOURCE FILE")
    print("=" * 74)
    print(f"{'file':<34}{'n':>5}{'complete':>10}{'partial':>9}{'empty':>7}"
          f"{'med len':>9}")
    print("-" * 74)
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    for f, rs in sorted(by_file.items()):
        n = len(rs)
        c = sum(1 for r in rs if r["cls"] == "complete")
        p = sum(1 for r in rs if r["cls"] == "partial")
        e = sum(1 for r in rs if r["cls"] == "empty")
        lens = sorted(r["len"] for r in rs)
        med = lens[len(lens) // 2]
        print(f"{f[:33]:<34}{n:>5}{100*c/n:>9.0f}%{100*p/n:>8.0f}%"
              f"{100*e/n:>6.0f}%{med:>9}")

    # ---------------- by length bin ----------------
    print("\n" + "=" * 74)
    print("COMPLETENESS BY PEPTIDE LENGTH  (the hypothesis under test)")
    print("=" * 74)
    w = args.bin_width
    bins = defaultdict(list)
    for r in rows:
        bins[(r["len"] // w) * w].append(r)
    print(f"{'length':<12}{'n':>5}{'complete':>11}{'partial':>9}{'empty':>7}   bar")
    print("-" * 74)
    for lo in sorted(bins):
        rs = bins[lo]
        n = len(rs)
        c = sum(1 for r in rs if r["cls"] == "complete")
        p = sum(1 for r in rs if r["cls"] == "partial")
        e = sum(1 for r in rs if r["cls"] == "empty")
        pct = 100 * c / n
        bar = "#" * int(pct / 5)
        print(f"{lo:>3}-{lo+w-1:<8}{n:>5}{pct:>10.0f}%{100*p/n:>8.0f}%"
              f"{100*e/n:>6.0f}%   {bar}")

    # ---------------- correlation-ish summary ----------------
    short = [r for r in rows if r["len"] < 30]
    long_ = [r for r in rows if r["len"] >= 30]
    def rate(rs):
        return 100 * sum(1 for r in rs if r["cls"] == "complete") / len(rs) if rs else 0
    print("\n" + "-" * 74)
    print(f"  peptides <30 residues : {len(short):>4}   complete {rate(short):>5.1f}%")
    print(f"  peptides >=30 residues: {len(long_):>4}   complete {rate(long_):>5.1f}%")

    # ---------------- alternative explanation: cysteine density ----------------
    print("\n" + "=" * 74)
    print("ALTERNATIVE EXPLANATION - CYSTEINE DENSITY (cys per residue)")
    print("=" * 74)
    dens = defaultdict(list)
    for r in rows:
        d = r["ncys"] / r["len"]
        key = round(d * 20) / 20          # 0.05 buckets
        dens[key].append(r)
    print(f"{'cys/res':<12}{'n':>5}{'complete':>11}")
    print("-" * 74)
    for k in sorted(dens):
        rs = dens[k]
        if len(rs) < 3:
            continue
        print(f"{k:<12.2f}{len(rs):>5}{rate(rs):>10.0f}%")

    # ---------------- and by cysteine count ----------------
    print("\n" + "=" * 74)
    print("BY CYSTEINE COUNT")
    print("=" * 74)
    byc = defaultdict(list)
    for r in rows:
        byc[r["ncys"]].append(r)
    print(f"{'n_cys':<12}{'n':>5}{'complete':>11}{'med len':>10}")
    print("-" * 74)
    for k in sorted(byc):
        rs = byc[k]
        if len(rs) < 3:
            continue
        lens = sorted(r["len"] for r in rs)
        print(f"{k:<12}{len(rs):>5}{rate(rs):>10.0f}%{lens[len(lens)//2]:>10}")

    print("""
HOW TO READ
  If completeness rises sharply with length and the cysteine-density and
  cysteine-count tables show no comparable trend, LENGTH is the driver. That
  gives a clean boundary condition to state in the paper: the connectivity
  metric is only interpretable above roughly the length where completeness
  becomes reliable, and below it ESMFold declines to predict rather than
  predicting wrongly.

  If density tracks better than length, the limit is about packing cysteines
  into a small structure, not raw length -- a different and also reportable
  claim.

  Either way: partial and empty predictions must be reported SEPARATELY from
  wrong ones. Pooling them (as a raw accuracy figure does) mixes "the model was
  confidently wrong" with "the model said nothing", which are not the same
  result.""")


if __name__ == "__main__":
    main()