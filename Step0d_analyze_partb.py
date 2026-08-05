#!/usr/bin/env python3
"""
step0d_analyze_partb.py
=======================
Step 0 Part B, analysis half. THE DECIDING EXPERIMENT.

Folds the baseline generations from step0c and answers one question:

    Without topology conditioning, how often does a redesigned defensin
    scaffold fold to the topology it was SUPPOSED to have?

Reading the result:
    LOW  match rate (say < 60%)  -> real gap. Conditioning has a clear job.
                                    The project has a finding. PROCEED.
    HIGH match rate (say > 85%)  -> little headroom. Conditioning would add
                                    little. RECONSIDER before building further.
    In between                   -> partial gap; report the number honestly and
                                    focus on the topologies that fail most.

Note the question is NOT "did a valid disulfide pattern form?" but "did the
INTENDED one form?" A peptide that folds cleanly into the wrong topology is a
failure for this purpose, and is exactly what conditioning is meant to fix.

Usage:
    python step0d_analyze_partb.py --limit 100

No GPU required: folding goes through the public ESMFold API.
"""

import argparse
import json
import time
from collections import Counter

# reuse the tested geometry code from Part A
try:                                     # filename case varies on Windows
    from step0b_check_fidelity import realized_pairing, to_set, fmt, fold
except ModuleNotFoundError:
    from Step0b_check_fidelity import realized_pairing, to_set, fmt, fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="generated.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cutoff", type=float, default=3.0)
    ap.add_argument("--out", default="partb_results.json")
    ap.add_argument("--skip-broken", action="store_true", default=True,
                    help="skip sequences that broke the fixed positions")
    args = ap.parse_args()

    data = json.load(open(args.input))
    if args.skip_broken:
        data = [d for d in data if d.get("constraints_respected", True)]
    if args.limit:
        data = data[:args.limit]

    print(f"Folding {len(data)} generated sequences "
          f"(bond cutoff {args.cutoff} A)\n")

    results = []
    match_n = folded_n = 0
    per_template = {}
    realized_counter = Counter()

    for k, rec in enumerate(data):
        intended = to_set(tuple(p) for p in rec["intended_pairs"])
        try:
            pdb = fold(rec["sequence"])
        except Exception as e:
            print(f"[{k:>3}] FOLD FAILED ({e})")
            continue

        pred, dists, ncys = realized_pairing(pdb, args.cutoff)
        folded_n += 1
        hit = (pred == intended)
        match_n += int(hit)
        realized_counter[tuple(fmt(pred))] += 1

        t = rec["template"]
        per_template.setdefault(t, [0, 0])
        per_template[t][0] += int(hit)
        per_template[t][1] += 1

        print(f"[{k:>3}] {t:<9} cys={ncys}  intended={fmt(intended)}  "
              f"got={fmt(pred)}  {'MATCH' if hit else 'miss'}")

        results.append({
            "template": t, "sequence": rec["sequence"],
            "intended": fmt(intended), "realized": fmt(pred),
            "match": hit, "n_cys": ncys,
            "sg_distances": {str(sorted(kk)): v for kk, v in dists.items()},
        })
        time.sleep(1)

    json.dump(results, open(args.out, "w"), indent=2)

    if not folded_n:
        print("\nNothing folded. Check the network / the ESMFold API.")
        return

    pct = 100 * match_n / folded_n
    print("\n" + "=" * 72)
    print(f"PART B RESULT: intended topology reached in "
          f"{match_n}/{folded_n} = {pct:.1f}% of unconditioned generations")
    print("=" * 72)

    print("\nPer template:")
    for t, (h, n) in sorted(per_template.items()):
        print(f"   {t:<9} {h:>3}/{n:<3} = {100*h/n:5.1f}%")

    print("\nWhat topologies actually formed (top 8):")
    for topo, c in realized_counter.most_common(8):
        print(f"   {list(topo)}  ->  {c}  ({100*c/folded_n:.0f}%)")

    print("\n" + "-" * 72)
    if pct < 60:
        print("VERDICT: clear gap. Unconditioned redesign frequently loses the")
        print("intended topology, so conditioning has real work to do. PROCEED.")
    elif pct > 85:
        print("VERDICT: little headroom. The base model mostly preserves the")
        print("intended topology already. RECONSIDER the framing before")
        print("committing -- see the backup plans in the report.")
    else:
        print("VERDICT: partial gap. Report the number honestly and target the")
        print("topologies with the lowest match rates (see per-template table).")
    print("-" * 72)
    print(f"\nPer-sequence detail in {args.out}")


if __name__ == "__main__":
    main()