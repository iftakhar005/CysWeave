#!/usr/bin/env python3
"""Ensemble-fraction readout vs rank_001, on rank_ensemble_results.json."""
import json, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="rank_ensemble_results.json")
a = ap.parse_args()

recs = json.load(open(a.input))
key = lambda p: tuple(sorted(tuple(sorted(x)) for x in p))

print("=" * 74)
print("ENSEMBLE-FRACTION READOUT  (support = models matching a topology / 5)")
print("=" * 74)

by_g, r1_hits, ens_hits = defaultdict(list), 0, 0
for r in recs:
    true_k = key(r["true"])
    preds = {k: key(v) for k, v in r["preds"].items()}
    n = len(preds)
    sup = sum(1 for v in preds.values() if v == true_k)
    # best-supported topology in the ensemble
    tally = defaultdict(int)
    for v in preds.values():
        tally[v] += 1
    top = max(tally.items(), key=lambda kv: kv[1])
    by_g[r["group"]].append((r["accession"], true_k, sup, n, top, r["rank1_correct"]))
    r1_hits += bool(r["rank1_correct"])
    ens_hits += (top[0] == true_k)

for g in sorted(by_g, key=str):
    print(f"\n### {g}")
    for acc, true_k, sup, n, top, r1 in by_g[g]:
        plur = "plurality OK " if top[0] == true_k else "plurality WRONG"
        print(f"  {acc:<11} support {sup}/{n}   {plur}   "
              f"rank1 {'OK ' if r1 else 'wrong'}   true={[list(p) for p in true_k]}")
    # within-group discrimination: do members with different truths get
    # different support profiles?
    truths = {t for _, t, _, _, _, _ in by_g[g]}
    if len(truths) > 1:
        ok = all(sup > max([s for a2, t2, s, _, _, _ in by_g[g] if t2 != t] or [-1])
                 or True for _, t, sup, _, _, _ in by_g[g])
        ranked = sorted(by_g[g], key=lambda x: -x[2])
        print(f"  -> support ordering: " +
              ", ".join(f"{acc}={sup}/{n}" for acc, _, sup, n, _, _ in ranked))

print("\n" + "=" * 74)
print(f"  rank_001 correct              : {r1_hits}/{len(recs)}")
print(f"  ensemble plurality correct    : {ens_hits}/{len(recs)}")
nonb = [r for r in recs if key(r["true"]) != ((1,5),(2,4),(3,6))]
nb_r1 = sum(1 for r in nonb if r["rank1_correct"])
nb_sup = sum(1 for r in nonb
             if sum(1 for v in r["preds"].values() if key(v) == key(r["true"])) > 0)
print(f"  non-beta truths               : {len(nonb)}")
print(f"    rank_001 correct            : {nb_r1}/{len(nonb)}")
print(f"    any ensemble support        : {nb_sup}/{len(nonb)}")
print("=" * 74)