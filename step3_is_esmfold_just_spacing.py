#!/usr/bin/env python3
"""
step3_is_esmfold_just_spacing.py
================================
THE DIAGNOSTIC. Uses only data you have already folded — no GPU, no new folding.

The question
------------
Our whole evaluation reads disulfide connectivity out of an ESMFold structure.
But does ESMFold actually infer connectivity from the SEQUENCE, or is it simply
reproducing the canonical topology associated with that CYSTEINE SPACING?

This matters more than anything else in the project:

  * If ESMFold merely reads spacing, then our metric cannot support ANY
    sequence-level claim; every "fidelity" number we have is really a spacing
    lookup, and the null/scramble/swap results are explained without any
    reference to the generator. That is a genuine, publishable limitation of
    the folding models this whole subfield relies on.

  * If ESMFold beats a spacing-only predictor, then it uses sequence context,
    the metric is sound, and steering topology with sequence is worth pursuing.

The test
--------
Build a spacing-only predictor: nearest-neighbour lookup over the curated
dataset, mapping a cysteine spacing pattern to its most common topology, with
NO knowledge of the residues in between. Then, for every peptide already
folded, compare:

      what the spacing-only predictor says   vs   what ESMFold predicted

Very high agreement = ESMFold adds nothing beyond the cysteine framework.

Usage
-----
    python step3_is_esmfold_just_spacing.py
    python step3_is_esmfold_just_spacing.py --folds partb_baseline.jsonl \
        partb_conditioned.jsonl partb_nullcond.jsonl partb_scrambled.jsonl \
        partb_swap.jsonl
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


def spacing_pattern(seq):
    pos = [i for i, a in enumerate(seq) if a == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def as_key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


# --------------------------------------------------------------------------
# spacing-only predictor: knows the framework, knows nothing about residues
# --------------------------------------------------------------------------

class SpacingOnlyPredictor:
    def __init__(self, records):
        self.by_exact = defaultdict(Counter)
        self.by_ncys = defaultdict(list)
        for r in records:
            seq = get_sequence(r)
            sp = spacing_pattern(seq)
            tp = as_key(r["pairs"])
            self.by_exact[sp][tp] += 1
            self.by_ncys[seq.count("C")].append((sp, tp))

    def predict(self, seq):
        sp = spacing_pattern(seq)
        if sp in self.by_exact:                      # exact framework match
            return self.by_exact[sp].most_common(1)[0][0], 0
        best_d, best_tp = None, None                 # else nearest framework
        for sp2, tp2 in self.by_ncys.get(seq.count("C"), []):
            if len(sp2) != len(sp):
                continue
            d = sum(abs(a - b) for a, b in zip(sp, sp2))
            if best_d is None or d < best_d:
                best_d, best_tp = d, tp2
        return best_tp, best_d


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="defensin_connectivity_dataset.json")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="folded .jsonl files; default = every partb_*.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Could not find {args.dataset}. Run in the project folder.")
        return
    records = [r for r in json.load(open(args.dataset))
               if get_sequence(r) and r.get("pairs")]
    predictor = SpacingOnlyPredictor(records)
    print(f"spacing-only predictor built from {len(records)} sequences\n")

    files = args.folds or sorted(glob.glob("partb_*.jsonl"))
    if not files:
        print("No folded .jsonl files found (expected partb_*.jsonl).")
        return

    print("=" * 78)
    print("ESMFold  vs  SPACING-ONLY PREDICTOR")
    print("=" * 78)
    print(f"{'file':<28}{'n':>6}{'agree':>9}{'ESMFold=req':>13}"
          f"{'spacing=req':>13}")
    print("-" * 78)

    grand_agree = grand_n = 0
    disagreements = []

    for f in files:
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        agree = esm_req = sp_req = n = 0
        for r in rows:
            seq = r.get("sequence")
            if not seq:
                continue
            esm_tp = as_key(r["realized"])
            req_tp = as_key(r["intended"])
            sp_tp, dist = predictor.predict(seq)
            if sp_tp is None:
                continue
            n += 1
            same = (esm_tp == sp_tp)
            agree += same
            esm_req += (esm_tp == req_tp)
            sp_req += (sp_tp == req_tp)
            if not same:
                disagreements.append((f, seq, esm_tp, sp_tp, dist))
        if n:
            grand_agree += agree
            grand_n += n
            print(f"{os.path.basename(f):<28}{n:>6}{100*agree/n:>8.1f}%"
                  f"{100*esm_req/n:>12.1f}%{100*sp_req/n:>12.1f}%")

    if not grand_n:
        print("\nNo usable rows. Check that the .jsonl files have "
              "'sequence', 'realized' and 'intended' fields.")
        return

    pct = 100 * grand_agree / grand_n
    print("-" * 78)
    print(f"{'ALL ARMS':<28}{grand_n:>6}{pct:>8.1f}%")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"ESMFold agrees with a spacing-only lookup on {pct:.1f}% "
          f"of {grand_n} folded peptides\n")
    if pct >= 90:
        print("ESMFold IS ESSENTIALLY A SPACING LOOKUP for these peptides.")
        print("  Our connectivity metric cannot distinguish sequence effects")
        print("  from framework effects, so it cannot validate any sequence-level")
        print("  topology claim. This explains the null / scramble / swap results")
        print("  WITHOUT any reference to the generator, and it is a substantive")
        print("  limitation of the single-sequence folding models this subfield")
        print("  depends on. Next: confirm with a second predictor (AlphaFold or")
        print("  Boltz, which use MSAs) and with CRiSP, then write it up as the")
        print("  benchmark + diagnostic contribution.")
    elif pct >= 75:
        print("MOSTLY SPACING, with real exceptions.")
        print("  Examine the disagreements below — those are the cases where")
        print("  ESMFold used something beyond the framework, and they are the")
        print("  only place a sequence-level method could demonstrate control.")
    else:
        print("ESMFold USES SEQUENCE CONTEXT well beyond spacing.")
        print("  The metric is sound for sequence-level claims, and steering")
        print("  topology by sequence is worth pursuing on the ambiguous set.")

    if disagreements:
        print(f"\n--- {len(disagreements)} disagreements (first 12) ---")
        for f, seq, e, s, d in disagreements[:12]:
            print(f"  [{os.path.basename(f)}] framework-dist={d}")
            print(f"     ESMFold : {[list(p) for p in e]}")
            print(f"     spacing : {[list(p) for p in s]}")
        print("\nThese are the informative cases. If there are enough of them,")
        print("they become the evaluation set for any sequence-level claim.")


if __name__ == "__main__":
    main()