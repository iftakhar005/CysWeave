#!/usr/bin/env python3
"""
split_results.py
================
Recovers clean per-arm results from a mixed fold output.

Folding generated_conditioned.json into partb_results.jsonl APPENDED the 70
conditioned rows onto the 140 baseline rows, so the script's printed summary
mixed both arms. No need to re-fold: the rows are all there and can be
separated by matching sequences against the generation files.

Usage:
    python split_results.py

Writes partb_baseline.jsonl and partb_conditioned.jsonl, and prints each arm's
fidelity plus the per-template breakdown.
"""

import json
import os
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def seqs_of(path):
    return {r["sequence"] for r in json.load(open(path))} if os.path.exists(path) else set()


def report(rows, label):
    if not rows:
        print(f"{label}: no rows")
        return
    m = sum(r["match"] for r in rows)
    n = len(rows)
    print(f"\n{'='*60}\n{label}: {m}/{n} = {100*m/n:.1f}%\n{'='*60}")
    per = {}
    for r in rows:
        a, b = per.get(r["template"], (0, 0))
        per[r["template"]] = (a + r["match"], b + 1)
    for t, (h, tot) in sorted(per.items()):
        print(f"   {t:<9} {h:>3}/{tot:<3} = {100*h/tot:5.1f}%")


def main():
    mixed = load_jsonl("partb_results.jsonl")
    print(f"loaded {len(mixed)} folded rows from partb_results.jsonl")

    cond_seqs = seqs_of("generated_conditioned.json")
    base_seqs = seqs_of("generated.json")
    print(f"generated_conditioned.json: {len(cond_seqs)} sequences")
    print(f"generated.json (baseline) : {len(base_seqs)} sequences")

    cond, base, other = [], [], []
    for r in mixed:
        s = r["sequence"]
        if s in cond_seqs:
            cond.append(r)
        elif s in base_seqs:
            base.append(r)
        else:
            other.append(r)

    for rows, path in ((cond, "partb_conditioned.jsonl"),
                       (base, "partb_baseline.jsonl")):
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    report(base, "BASELINE (unconditioned)")
    report(cond, "CONDITIONED")
    if other:
        print(f"\n{len(other)} rows matched neither generation file "
              f"(older run?) -- ignored")

    if cond and base:
        cb = 100 * sum(r["match"] for r in cond) / len(cond)
        bb = 100 * sum(r["match"] for r in base) / len(base)
        print(f"\ndelta (conditioned - baseline, both raw): {cb - bb:+.1f} points")
        print("NOTE: the headline baseline is the CONTROLLED 70.9% (sequences "
              "that kept the exact cysteine count), not the raw number above.")

    print("\nwrote partb_conditioned.jsonl and partb_baseline.jsonl")


if __name__ == "__main__":
    main()