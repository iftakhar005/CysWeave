#!/usr/bin/env python3
"""MSA depth per peptide vs whether the ensemble surfaced the true topology.

Tests KnotFold's mechanism (PRX Life 2:043018, 2024): they report connectivity
was correct for every peptide with MSA depth > 283, because conserved cysteines
carry no differential coevolution signal in deep alignments.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--cf-out", default="cf_out")
ap.add_argument("--input", default="rank_ensemble_results.json")
a = ap.parse_args()

def depth(job, cf):
    """Sequences in the job's a3m (first line is the query)."""
    pats = [os.path.join(cf, f"{job}.a3m"),
            os.path.join(cf, f"{job}_env", "*.a3m"),
            os.path.join(cf, "**", f"{job}*.a3m")]
    for p in pats:
        for f in glob.glob(p, recursive=True):
            n = 0
            with open(f, errors="ignore") as fh:
                for line in fh:
                    if line.startswith(">"):
                        n += 1
            if n:
                return n, os.path.basename(f)
    return None, None

recs = json.load(open(a.input))
key = lambda p: tuple(sorted(tuple(sorted(x)) for x in p))

print(f"{'group':<6} {'acc':<11} {'depth':>7}  {'support':>7}  {'r1':>5}  file")
print("-" * 78)
rows = []
for r in sorted(recs, key=lambda x: (str(x["group"]), x["accession"])):
    d, f = depth(r["job"], a.cf_out)
    sup = sum(1 for v in r["preds"].values() if key(v) == key(r["true"]))
    rows.append((r["group"], r["accession"], d, sup, len(r["preds"]),
                 r["rank1_correct"], r["n_distinct"]))
    ds = f"{d}" if d else "  n/a"
    print(f"{r['group']:<6} {r['accession']:<11} {ds:>7}  {sup}/{len(r['preds']):<5}"
          f"  {'OK' if r['rank1_correct'] else 'X':>5}  {f or '(not found)'}")

have = [x for x in rows if x[2]]
if have:
    hit = [x[2] for x in have if x[3] > 0]
    miss = [x[2] for x in have if x[3] == 0]
    div = [x[2] for x in have if x[6] > 1]
    uni = [x[2] for x in have if x[6] == 1]
    m = lambda v: f"{sum(v)/len(v):.0f} (n={len(v)}, range {min(v)}-{max(v)})" if v else "none"
    print("\nmean MSA depth")
    print(f"  true topology present in ensemble : {m(hit)}")
    print(f"  true topology absent              : {m(miss)}")
    print(f"  diverse ensemble (>1 topology)    : {m(div)}")
    print(f"  degenerate ensemble (all 5 agree) : {m(uni)}")
    print("\n  KnotFold's threshold was depth > 283 -> connectivity correct.")
else:
    print("\nNo .a3m files found. List what's in a job folder:")
    print(f"  dir {a.cf_out}")