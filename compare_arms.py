#!/usr/bin/env python3
"""
compare_arms.py
===============
The paper's results table. Compares all experimental arms on two axes:

  1. CONNECTIVITY FIDELITY -- does the folded peptide realise the REQUESTED
     cysteine pairing? (the headline metric)

  2. SEQUENCE IDENTITY TO TEMPLATE -- the confound check. The adapter was
     trained on masked RECONSTRUCTION, so it could score well simply by
     reproducing the parent sequence. If the conditioned arm sits at much
     higher identity than the baseline, high fidelity may just mean "it copied
     the template", and the result must be reported with that caveat (or
     regenerated at a higher --mask-rate).

Expected pattern if the method is real:
    conditioned fidelity  >  baseline fidelity
    scrambled  fidelity  <<  conditioned fidelity      <- the specificity proof
    identity roughly comparable across arms            <- rules out copying

Usage:
    python compare_arms.py
    python compare_arms.py --baseline partb_baseline.jsonl \\
        --conditioned partb_conditioned.jsonl --scrambled partb_scrambled.jsonl
"""

import argparse
import json
import os


def load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def identity_stats(gen_path):
    """Mean % identity of each generated sequence to the template it came from."""
    if not gen_path or not os.path.exists(gen_path):
        return None
    recs = json.load(open(gen_path))
    vals = []
    for r in recs:
        t = r.get("template_sequence")
        s = r.get("sequence")
        if not t or not s or len(t) != len(s):
            continue
        vals.append(sum(a == b for a, b in zip(s, t)) / len(t))
    return 100 * sum(vals) / len(vals) if vals else None


def fidelity(rows):
    if not rows:
        return None, 0, 0
    m = sum(r["match"] for r in rows)
    return 100 * m / len(rows), m, len(rows)


def per_template(rows):
    d = {}
    for r in rows:
        a, b = d.get(r["template"], (0, 0))
        d[r["template"]] = (a + r["match"], b + 1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="partb_baseline.jsonl")
    ap.add_argument("--conditioned", default="partb_conditioned.jsonl")
    ap.add_argument("--scrambled", default="partb_scrambled.jsonl")
    ap.add_argument("--gen-baseline", default="generated.json")
    ap.add_argument("--gen-conditioned", default="generated_conditioned.json")
    ap.add_argument("--gen-scrambled", default="generated_scrambled.json")
    ap.add_argument("--baseline-controlled", type=float, default=70.9,
                    help="the controlled baseline figure (cysteine-count matched)")
    args = ap.parse_args()

    arms = [
        ("baseline",    load_jsonl(args.baseline),    args.gen_baseline),
        ("conditioned", load_jsonl(args.conditioned), args.gen_conditioned),
        ("scrambled",   load_jsonl(args.scrambled),   args.gen_scrambled),
    ]

    print("=" * 74)
    print("RESULTS BY ARM")
    print("=" * 74)
    print(f"{'arm':<14}{'fidelity':>18}{'identity to template':>24}")
    print("-" * 74)

    fid = {}
    for name, rows, gen in arms:
        pct, m, n = fidelity(rows)
        ident = identity_stats(gen)
        fid[name] = pct
        f_s = f"{m}/{n} = {pct:.1f}%" if pct is not None else "(no data)"
        i_s = f"{ident:.1f}%" if ident is not None else "(no data)"
        print(f"{name:<14}{f_s:>18}{i_s:>24}")
    print("-" * 74)
    print(f"{'baseline (controlled, cysteine-matched)':<44}"
          f"{args.baseline_controlled:.1f}%")

    # ---- per-template breakdown ----
    for name, rows, _ in arms:
        if not rows:
            continue
        print(f"\nper template -- {name}:")
        for t, (h, n) in sorted(per_template(rows).items(),
                                key=lambda kv: -(kv[1][0] / kv[1][1])):
            print(f"   {t:<9} {h:>3}/{n:<3} = {100*h/n:5.1f}%")

    # ---- failure modes: misassignment vs incomplete ----
    for name, rows, _ in arms:
        misses = [r for r in rows if not r["match"]]
        if not misses:
            continue
        wrong = sum(1 for r in misses
                    if len(r["realized"]) == len(r["intended"]))
        incomplete = len(misses) - wrong
        print(f"\n{name} failures: {len(misses)} total -- "
              f"{wrong} MISASSIGNED (wrong pairing formed), "
              f"{incomplete} INCOMPLETE (a bond did not form)")
        print("   (incomplete failures are often a folding-model limit, "
              "not a conditioning failure)")

    # ---- the verdict ----
    print("\n" + "=" * 74)
    b, c, s = fid.get("baseline"), fid.get("conditioned"), fid.get("scrambled")
    ref = args.baseline_controlled
    if c is not None:
        print(f"conditioned vs controlled baseline: {c - ref:+.1f} points")
    if c is not None and s is not None:
        print(f"conditioned vs scrambled           : {c - s:+.1f} points")
        if c - s >= 20:
            print("\n-> STRONG: requesting the wrong topology collapses fidelity,")
            print("   so the model is genuinely using the conditioning signal.")
        elif c - s >= 10:
            print("\n-> MODERATE: scrambling hurts, but less than hoped. Report")
            print("   the gap honestly and discuss partial signal use.")
        else:
            print("\n-> WEAK: scrambling barely changes the outcome. The model is")
            print("   largely IGNORING the conditioning; the conditioned number")
            print("   cannot be claimed as evidence the method works.")
    else:
        print("\nRun the --scramble arm: without it the conditioned number alone")
        print("does not establish that the conditioning is what caused the gain.")
    print("=" * 74)


if __name__ == "__main__":
    main()