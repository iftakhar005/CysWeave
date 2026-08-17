#!/usr/bin/env python3
"""
step2_metric_discrimination.py
==============================
FOUNDATIONAL TEST: does our metric actually measure folding, or only spacing?

The worry
---------
Every fidelity number we have comes from ESMFold's predicted structure. If
ESMFold infers disulfide connectivity mainly from the CYSTEINE SPACING pattern
(a strong statistical regularity it would have learned from the PDB), then:

  * generated peptides keep the template's spacing,
  * so ESMFold predicts the template's topology,
  * regardless of what the peptide would really do.

Under that reading our metric measures "does this sequence have spacing
consistent with topology X", not "does it fold to X" — which would explain the
null / scramble / swap results with no claim about the generator at all, and
would make any spacing-based method circular to evaluate.

The test
--------
Step 1 found spacing patterns that occur in nature with MORE THAN ONE topology.
Those are natural sequences, with curated annotations, identical spacing, and
genuinely different pairings. They are the perfect discriminator:

    ESMFold predicts each one's OWN annotated topology
        -> it uses sequence context. The metric is sound. Sequence-level
           information about pairing exists and is learnable.

    ESMFold gives the SAME topology to every sequence sharing a spacing
        -> it is reading spacing only. The metric cannot support any
           sequence-level claim, and that is a reportable limitation of the
           tooling the whole subfield relies on.

Usage
-----
  1) python step2_metric_discrimination.py            # builds the test set
  2) python step0d_colab_local_esmfold.py --input metric_discrimination.json \
         --force-cpu --arm conditioned --out partb_discrimination.jsonl
  3) python step2_metric_discrimination.py --analyze partb_discrimination.jsonl

Typically only a few dozen sequences, so folding is well under an hour on CPU.
"""

import argparse
import json
import os
from collections import defaultdict


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


def spacing_pattern(seq):
    pos = [i for i, a in enumerate(seq) if a == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def topology_key(rec):
    return tuple(sorted(tuple(sorted(p)) for p in rec["pairs"]))


def name_of(rec):
    return rec.get("accession") or f"{rec.get('pdb_id')}_{rec.get('chain')}"


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args):
    records = [r for r in json.load(open(args.input))
               if get_sequence(r) and r.get("pairs")]
    print(f"loaded {len(records)} sequences")

    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[spacing_pattern(get_sequence(r))][topology_key(r)].append(r)

    ambiguous = {sp: tops for sp, tops in groups.items() if len(tops) > 1}
    print(f"spacing patterns occurring with >1 topology: {len(ambiguous)}\n")

    out, n_groups = [], 0
    for sp, tops in sorted(ambiguous.items(), key=lambda kv: -len(kv[1])):
        n_groups += 1
        print(f"spacing {list(sp)}  ->  {len(tops)} topologies")
        for tp, recs in tops.items():
            take = recs[:args.per_topology]
            print(f"    {[list(p) for p in tp]}   n={len(recs)} "
                  f"(using {len(take)}): {', '.join(name_of(r) for r in take)}")
            for r in take:
                out.append({
                    # step0d expects these field names
                    "template": f"SP{n_groups}",
                    "template_sequence": get_sequence(r),
                    "sequence": get_sequence(r),
                    "intended_pairs": [list(p) for p in tp],  # its OWN annotation
                    "constraints_respected": True,
                    "n_cys": get_sequence(r).count("C"),
                    # bookkeeping for the analysis pass
                    "_spacing": list(sp),
                    "_accession": name_of(r),
                })
        print()

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {len(out)} sequences across {n_groups} spacing groups "
          f"to {args.out}")
    if len(out) < 6:
        print("\nWARNING: very few ambiguous sequences. Widen the pool by adding "
              "related disulfide-rich families, or treat this as a pilot.")
    print("\nNext: fold it, then re-run this script with --analyze")


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def analyze(args):
    built = {r["sequence"]: r for r in json.load(open(args.out))}
    rows = [json.loads(l) for l in open(args.analyze) if l.strip()]
    print(f"analysing {len(rows)} folded sequences\n")

    by_group = defaultdict(list)
    for r in rows:
        meta = built.get(r["sequence"], {})
        by_group[r["template"]].append((
            meta.get("_accession", "?"),
            tuple(map(tuple, r["intended"])),   # the true annotated topology
            tuple(map(tuple, r["realized"])),   # what ESMFold predicted
            r["match"],
        ))

    print("=" * 74)
    print("PER SPACING GROUP — same spacing, different true topologies")
    print("=" * 74)
    discriminating = collapsed = 0
    for g, items in sorted(by_group.items()):
        true_set = {t for _, t, _, _ in items}
        pred_set = {p for _, _, p, _ in items}
        correct = sum(1 for *_, m in items if m)
        print(f"\n{g}:  {len(items)} sequences, {len(true_set)} true topologies")
        for acc, t, p, m in items:
            print(f"   {acc:<12} true={[list(x) for x in t]}"
                  f"  pred={[list(x) for x in p]}  {'OK' if m else 'wrong'}")
        if len(pred_set) == 1 and len(true_set) > 1:
            collapsed += 1
            print("   -> ESMFold gave ALL of them the SAME topology (collapsed)")
        elif correct == len(items):
            discriminating += 1
            print("   -> ESMFold recovered each sequence's own topology")
        else:
            print(f"   -> partial: {correct}/{len(items)} correct")

    total = len(by_group)
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"groups where ESMFold discriminated : {discriminating}/{total}")
    print(f"groups where ESMFold collapsed     : {collapsed}/{total}")
    print()
    if total and discriminating / total >= 0.6:
        print("ESMFold USES SEQUENCE CONTEXT, not just spacing.")
        print("  The metric is sound for sequence-level claims. Sequence")
        print("  information about pairing exists and is in principle learnable,")
        print("  so a conditioning method targeted at these cases is worth trying.")
    elif total and collapsed / total >= 0.6:
        print("ESMFold IS READING SPACING, not sequence.")
        print("  Our metric cannot validate any sequence-level topology claim.")
        print("  This explains the null/scramble/swap results without reference")
        print("  to the generator at all, and is a significant limitation of the")
        print("  structure predictors this subfield depends on — worth reporting")
        print("  in its own right. Next step would be a second predictor")
        print("  (AlphaFold/Boltz) to see whether the limitation is universal.")
    else:
        print("MIXED. ESMFold discriminates in some groups but not others.")
        print("  Report the split, and restrict method claims to the groups")
        print("  where the metric demonstrably discriminates.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="defensin_connectivity_dataset.json")
    ap.add_argument("--out", default="metric_discrimination.json")
    ap.add_argument("--per-topology", type=int, default=3,
                    help="sequences to take per (spacing, topology) cell")
    ap.add_argument("--analyze", default=None,
                    help="path to the folded .jsonl, to run the analysis pass")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Could not find {args.input}. Run this in the project folder.")
        return
    analyze(args) if args.analyze else build(args)


if __name__ == "__main__":
    main()