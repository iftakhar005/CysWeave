#!/usr/bin/env python3
"""
step17_framework_drift.py
=========================
Did the generator preserve the cysteine framework it was asked for?

The baseline arm requested 80 six-cysteine, 40 eight-cysteine and 20
ten-cysteine peptides and returned counts from 6 to 11. A peptide that gains a
cysteine cannot satisfy the requested pairing under the connectivity metric --
it fails mechanically rather than losing the topology. Pooling those with real
topology failures deflates the baseline and inflates any arm compared against
it.

This splits every arm into PRESERVED (cysteine count equals twice the number of
requested bonds) and DRIFTED, and reports the success rate separately, so the
baseline can be restated on a like-for-like basis.

Usage
-----
  python step17_framework_drift.py
  python step17_framework_drift.py --files partb_baseline.jsonl partb_conditioned.jsonl
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


SEQ = ("sequence", "mature_sequence", "seq")
REQ = ("intended_pairs", "intended", "requested", "target_pairs")
ORA = ("realized", "got", "predicted", "realized_pairs", "observed")


def first(d, names):
    for n in names:
        v = d.get(n)
        if v:
            return v
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="*", default=None)
a = ap.parse_args()

files = a.files or sorted(glob.glob("partb_*.jsonl"))

print("=" * 90)
print("FRAMEWORK PRESERVATION BY ARM")
print("PRESERVED = cysteines in the sequence == 2 x requested bonds")
print("=" * 90)
print(f"  {'arm':<24} {'n':>5} {'presv':>6} {'drift':>6} "
      f"{'drift%':>7} {'hit|presv':>10} {'hit|drift':>10} {'pooled':>8}")

for f in files:
    if not os.path.exists(f):
        continue
    rs = [json.loads(l) for l in open(f) if l.strip()]
    P = D = ph = dh = 0
    comp_p = 0
    drift_by = Counter()
    for r in rs:
        seq, req, ora = first(r, SEQ), first(r, REQ), first(r, ORA)
        if not seq or not req:
            continue
        ncys = seq.upper().count("C")
        need = len(req) * 2
        hit = ora is not None and key(ora) == key(req)
        if ncys == need:
            P += 1
            ph += hit
            if ora is not None and len(ora) == ncys // 2:
                comp_p += 1
        else:
            D += 1
            dh += hit
            drift_by[ncys - need] += 1
    n = P + D
    if not n:
        continue
    arm = os.path.basename(f).replace("partb_", "").replace(".jsonl", "")
    pv = f"{100*ph/P:9.1f}%" if P else "       n/a"
    dv = f"{100*dh/D:9.1f}%" if D else "       n/a"
    print(f"  {arm:<24} {n:>5} {P:>6} {D:>6} {100*D/n:6.1f}% {pv} {dv} "
          f"{100*(ph+dh)/n:7.1f}%")
    if D:
        print(f"      extra cysteines gained: "
              f"{dict(sorted(drift_by.items()))}")

print("""
HOW TO READ
  hit|presv is the like-for-like number: the topology the generator was asked
  for, among peptides whose cysteine framework it actually kept.

  hit|drift should be near zero by construction -- a peptide with the wrong
  number of cysteines cannot match the requested pairing. If it is near zero,
  the pooled figure is arithmetically dragged down by the drift rate and the
  arm should be reported on the preserved subset, with the drift rate stated
  alongside as a separate failure mode.

  If drift% differs between two arms being compared, the comparison is not
  measuring what it appears to. That is the case here for baseline against
  conditioned, and it is a correction the report needs.
""")