#!/usr/bin/env python3
"""
step9_mutagenesis_sensitivity.py
================================
THE CONTROLLED VERSION OF THE WHOLE EXPERIMENT.

The gap this closes
-------------------
Every result so far is "different natural sequences, same prediction". A
reviewer can object that those sequences share something relevant we did not
control for -- a common ancestor, a shared motif, similar composition.

This removes that objection. Start from ONE real peptide whose prediction is
confident and correct. Hold the cysteine positions absolutely fixed. Then
progressively replace the residues BETWEEN them, 10% -> 90%, and fold each
variant. We are not claiming to know the mutants' true topology -- we are
asking a narrower and fully answerable question:

        does the oracle's prediction respond to sequence at all?

Why this is legitimate synthetic data
-------------------------------------
Inventing sequences AND inventing their "true" topology would be circular: you
cannot know a synthetic peptide's real connectivity without folding it, and
folding is the thing under test. Here there is no invented ground truth. The
measurement is the CHANGE in prediction relative to the unmutated seed, which
needs no external label.

What the outcomes mean
----------------------
  prediction never changes, even at 90%
      -> sequence contributes nothing; spacing alone determines the output.
         This is the strongest possible form of the finding, and it is a
         dose-response curve rather than a binary.

  prediction degrades to partial/empty but never switches topology
      -> the model loses confidence without ever changing its mind. Also
         supports spacing-determinism, and pins where the structure collapses.

  prediction switches to a different COMPLETE topology at some threshold
      -> sequence DOES matter above that much divergence. That would qualify
         the claim, and the threshold itself becomes the result.

Usage
-----
  # 1. build the mutant panel from peptides that folded correctly
  python step9_mutagenesis_sensitivity.py --prepare \\
      --from-folds partb_knottin.jsonl partb_discrimination_full.jsonl

  # 2. fold it (same folder as always)
  python step0d_colab_local_esmfold.py --input mutagenesis_panel.json \\
      --force-cpu --arm conditioned --out partb_mutagenesis.jsonl

  # 3. read the dose-response curve
  python step9_mutagenesis_sensitivity.py --analyze partb_mutagenesis.jsonl
"""

import argparse
import json
import os
import random
from collections import defaultdict

# no cysteine -- mutating a residue into C would change the framework itself
SUBS = "ADEFGHIKLMNPQRSTVWY"


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def topo_str(pairs):
    return str(sorted(tuple(sorted(p)) for p in pairs))


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def pick_seeds(fold_files, n_seeds, min_bonds):
    """Seeds must have a COMPLETE and CORRECT prediction -- otherwise a change
    later is uninterpretable (you cannot detect a switch from a broken base)."""
    seeds, seen = [], set()
    for f in fold_files:
        if not os.path.exists(f):
            continue
        for r in load_jsonl(f):
            seq = r.get("sequence")
            if not seq or seq in seen:
                continue
            intended = r.get("intended") or []
            realized = r.get("realized") or []
            if len(intended) < min_bonds:
                continue
            if not r.get("match"):                 # must be correct
                continue
            if len(realized) != len(intended):     # must be complete
                continue
            seen.add(seq)
            seeds.append({
                "accession": r.get("template", "?"),
                "sequence": seq,
                "pairs": [list(p) for p in intended],
                "source": os.path.basename(f),
            })
    return seeds[:n_seeds]


def mutate(seq, frac, rng):
    """Replace `frac` of non-cysteine positions. Cysteines are untouchable and
    no new cysteine is ever introduced, so the framework is exactly preserved."""
    free = [i for i, a in enumerate(seq) if a != "C"]
    k = int(round(len(free) * frac))
    chosen = rng.sample(free, k) if k else []
    out = list(seq)
    for i in chosen:
        out[i] = rng.choice([a for a in SUBS if a != seq[i]])
    return "".join(out)


def prepare(args):
    seeds = pick_seeds(args.from_folds, args.n_seeds, args.min_bonds)
    if not seeds:
        print("No usable seeds found. Seeds need a COMPLETE and CORRECT "
              "prediction in one of the --from-folds files.")
        return
    print(f"selected {len(seeds)} seed peptide(s):")
    for s in seeds:
        print(f"  {s['accession']:<12} len={len(s['sequence']):<3} "
              f"cys={s['sequence'].count('C')}  {topo_str(s['pairs'])}"
              f"   [{s['source']}]")

    rng = random.Random(args.seed)
    fracs = [f / 100 for f in range(0, args.max_pct + 1, args.step_pct)]
    panel, jobmap, n = [], {}, 0

    for s in seeds:
        for frac in fracs:
            reps = 1 if frac == 0 else args.replicates
            for rep in range(reps):
                mut = mutate(s["sequence"], frac, rng)
                # sanity: framework must be identical to the seed
                assert [i for i, a in enumerate(mut) if a == "C"] == \
                       [i for i, a in enumerate(s["sequence"]) if a == "C"]
                n += 1
                job = f"mut{n:04d}"
                ident = sum(a == b for a, b in zip(mut, s["sequence"])) / len(mut)
                rec = {
                    "template": f"{s['accession']}@{int(frac*100)}",
                    "template_sequence": s["sequence"],
                    "sequence": mut,
                    # the seed's topology, used only as a fixed reference point
                    "intended_pairs": s["pairs"],
                    "constraints_respected": True,
                    "n_cys": mut.count("C"),
                    "_seed": s["accession"],
                    "_frac": frac,
                    "_rep": rep,
                    "_identity_to_seed": round(ident, 3),
                }
                panel.append(rec)
                jobmap[job] = {"label": rec["template"], "sequence": mut,
                               "intended_pairs": s["pairs"]}

    json.dump(panel, open(args.out, "w"), indent=2)
    json.dump(jobmap, open("mutagenesis_jobmap.json", "w"), indent=2)
    print(f"\nwrote {len(panel)} variants to {args.out}")
    print(f"  {len(seeds)} seeds x {len(fracs)} mutation levels "
          f"x {args.replicates} replicates (+1 unmutated each)")
    print(f"\nNext: fold {args.out}, then re-run with --analyze")


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def analyze(args):
    panel = {r["sequence"]: r for r in json.load(open(args.out))}
    rows = load_jsonl(args.analyze)
    print(f"analysing {len(rows)} folded variants\n")

    # what did each seed predict when unmutated?
    baseline = {}
    for r in rows:
        meta = panel.get(r["sequence"])
        if meta and meta["_frac"] == 0:
            baseline[meta["_seed"]] = topo_str(r["realized"])

    by = defaultdict(list)
    for r in rows:
        meta = panel.get(r["sequence"])
        if not meta:
            continue
        by[(meta["_seed"], meta["_frac"])].append((r, meta))

    print("=" * 76)
    print("DOSE-RESPONSE: does the prediction move as sequence diverges?")
    print("=" * 76)

    for seed in sorted({s for s, _ in by}):
        base = baseline.get(seed)
        print(f"\n### seed {seed}   unmutated prediction: {base}")
        print(f"  {'mutated':>8}{'n':>4}{'same as base':>14}{'complete':>10}"
              f"{'distinct topologies seen':>28}")
        print("  " + "-" * 72)
        for frac in sorted({f for s, f in by if s == seed}):
            items = by[(seed, frac)]
            preds = [topo_str(r["realized"]) for r, _ in items]
            same = sum(1 for p in preds if p == base)
            comp = sum(1 for r, m in items
                       if len(r["realized"]) == len(m["intended_pairs"]))
            distinct = len(set(preds))
            print(f"  {int(frac*100):>7}%{len(items):>4}"
                  f"{100*same/len(items):>13.0f}%{100*comp/len(items):>9.0f}%"
                  f"{distinct:>28}")

    # ---- overall verdict ----
    allrows = [(r, m) for v in by.values() for r, m in v if m["_frac"] > 0]
    if not allrows:
        print("\nno mutated variants folded.")
        return
    same = sum(1 for r, m in allrows
               if topo_str(r["realized"]) == baseline.get(m["_seed"]))
    comp = [(r, m) for r, m in allrows
            if len(r["realized"]) == len(m["intended_pairs"])]
    switched = [(r, m) for r, m in comp
                if topo_str(r["realized"]) != baseline.get(m["_seed"])]

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print(f"  mutated variants folded          : {len(allrows)}")
    print(f"  prediction unchanged from seed   : {same} "
          f"({100*same/len(allrows):.1f}%)")
    print(f"  complete predictions             : {len(comp)}")
    print(f"  COMPLETE *and* switched topology : {len(switched)}")
    print()
    if not switched:
        print("  The prediction NEVER switched to a different complete topology,")
        print("  at any level of sequence divergence. Sequence identity")
        print("  contributes nothing once the cysteine framework is fixed --")
        print("  the strongest form of the finding, and a dose-response curve")
        print("  rather than a binary comparison.")
    else:
        fr = sorted({m["_frac"] for _, m in switched})
        print(f"  Topology switched in {len(switched)} complete prediction(s),")
        print(f"  first at {int(fr[0]*100)}% mutation. Sequence DOES matter above")
        print("  that divergence -- report the threshold; it qualifies the claim.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--from-folds", nargs="*",
                    default=["partb_knottin.jsonl",
                             "partb_discrimination_full.jsonl"])
    ap.add_argument("--out", default="mutagenesis_panel.json")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--min-bonds", type=int, default=3)
    ap.add_argument("--max-pct", type=int, default=90)
    ap.add_argument("--step-pct", type=int, default=10)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.analyze:
        analyze(args)
    elif args.prepare:
        prepare(args)
    else:
        print("Use --prepare first, then fold, then --analyze. See docstring.")


if __name__ == "__main__":
    main()