#!/usr/bin/env python3
"""Find pairings where a cysteine appears in more than one bond (impossible)."""
import json, sys, glob
from collections import Counter

def get_seq(r): return r.get("mature_sequence") or r.get("sequence")

for path in (sys.argv[1:] or sorted(glob.glob("*_connectivity_dataset.json"))):
    try:
        recs = json.load(open(path))
    except Exception as e:
        print(f"{path}: cannot read ({e})"); continue
    bad = []
    for r in recs:
        pairs = r.get("pairs") or []
        flat = [x for p in pairs for x in p]
        ncys = get_seq(r).count("C") if get_seq(r) else 0
        dupes = [c for c, n in Counter(flat).items() if n > 1]
        missing = [c for c in range(1, ncys + 1) if c not in flat]
        if dupes or (ncys and len(flat) != len(set(flat))):
            bad.append((r.get("accession", "?"), pairs, dupes, missing))
    print(f"\n{path}")
    print(f"  records: {len(recs)}   MALFORMED: {len(bad)} "
          f"({100*len(bad)/len(recs):.2f}%)" if recs else "")
    for acc, pairs, d, m in bad[:8]:
        print(f"    {acc:<14} {pairs}   cys in 2 bonds: {d}   never bonded: {m}")
    if len(bad) > 8:
        print(f"    ... and {len(bad)-8} more")