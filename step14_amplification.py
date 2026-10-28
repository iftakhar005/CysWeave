#!/usr/bin/env python3
"""
step14_amplification.py  (v2 -- completeness-corrected)
=======================================================
Does the predictor over-produce each spacing group's majority topology?

v1 of this script pooled incomplete predictions with wrong ones and therefore
reported spurious NEGATIVE amplification for knottins. A prediction carrying
three bonds for an eight-cysteine peptide is the model declining to place the
rest, not the model choosing a different topology. Those must be counted
separately -- the same rule the density analysis established.

This version classifies every prediction complete or partial against the
cysteine count of its own sequence, measures amplification on complete
predictions only, and prints the raw figure alongside so the correction is
visible rather than hidden.

Accepts either input format:
  * a folded .jsonl  (one prediction per peptide, e.g. partb_knottin.jsonl)
  * rank_ensemble_results.json  (five ranked predictions per peptide)

Usage
-----
  python step14_amplification.py --dataset knottin_connectivity_dataset.json \\
                                 --folded partb_knottin.jsonl
  python step14_amplification.py --dataset defensin_connectivity_dataset.json \\
                                 --folded rank_ensemble_results.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict


def key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def fmt(k):
    return str([list(p) for p in k]) if k else "[]"


def spacing_of(seq):
    pos = [i for i, c in enumerate(seq.upper()) if c == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def expected_bonds(seq):
    return seq.upper().count("C") // 2


PAIR_KEYS = ("pairs", "disulfides", "connectivity", "topology",
             "intended_pairs", "intended", "true_pairs")
PRED_KEYS = ("realized", "got", "predicted", "realized_pairs",
             "predicted_pairs", "observed")
SEQ_KEYS = ("sequence", "mature_sequence", "mature", "seq")


def first(d, names):
    for n in names:
        v = d.get(n)
        if v:
            return v
    return None


def load_json_records(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("records") or data.get("entries") or list(data.values())
    return [r for r in data if isinstance(r, dict)]


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def load_predictions(path):
    """-> list of (sequence, predicted_pairs). Expands ranked ensembles."""
    recs = (load_jsonl(path) if path.endswith("jsonl")
            else load_json_records(path))
    out, bad = [], None
    for r in recs:
        s = first(r, SEQ_KEYS)
        if not s:
            bad = bad or sorted(r.keys())
            continue
        if isinstance(r.get("preds"), dict):          # ranked ensemble
            for v in r["preds"].values():
                out.append((s, v))
            continue
        g = first(r, PRED_KEYS)
        if g is None:
            bad = bad or sorted(r.keys())
            continue
        out.append((s, g))
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folded", required=True)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--min-pred", type=int, default=2)
    a = ap.parse_args()

    for f in (a.dataset, a.folded):
        if not os.path.exists(f):
            print(f"missing {f}")
            return

    # --- annotations ---
    ann, bad = defaultdict(Counter), None
    for r in load_json_records(a.dataset):
        p, s = first(r, PAIR_KEYS), first(r, SEQ_KEYS)
        if not p or not s:
            bad = bad or sorted(r.keys())
            continue
        ann[spacing_of(s)][key(p)] += 1
    if not ann:
        print(f"Parsed no annotations. Keys on a record: {bad}")
        return
    print(f"annotations: {sum(sum(c.values()) for c in ann.values())} records, "
          f"{len(ann)} distinct spacings")

    # --- predictions, split complete vs partial ---
    preds, bad = load_predictions(a.folded)
    if not preds:
        print(f"Parsed no predictions. Keys on a record: {bad}")
        return
    comp, part = defaultdict(Counter), Counter()
    for s, g in preds:
        sp, need = spacing_of(s), expected_bonds(s)
        if need and len(g) == need:
            comp[sp][key(g)] += 1
        else:
            part[sp] += 1
    print(f"predictions: {len(preds)} folded, "
          f"{sum(sum(c.values()) for c in comp.values())} complete, "
          f"{sum(part.values())} partial\n")

    print("=" * 86)
    print("AMPLIFICATION OF EACH SPACING GROUP'S OWN MAJORITY TOPOLOGY")
    print("complete predictions only; partial counted separately")
    print("=" * 86)

    rows = []
    seen = set(ann) & (set(comp) | set(part))
    for sp in sorted(seen, key=lambda x: -sum(ann[x].values())):
        na = sum(ann[sp].values())
        nc, npt = sum(comp[sp].values()), part[sp]
        ntot = nc + npt
        if na < a.min_n or nc < a.min_pred:
            continue
        maj, cmaj = ann[sp].most_common(1)[0]
        p_data = 100 * cmaj / na
        p_comp = 100 * comp[sp][maj] / nc
        p_raw = 100 * comp[sp][maj] / ntot
        pmaj, pcnt = comp[sp].most_common(1)[0]
        rows.append((sp, p_data, p_comp, p_raw, na, nc, npt, pmaj == maj,
                     len(comp[sp])))

        print(f"\nspacing {list(sp)}   annotated n={na}   folded n={ntot}")
        print(f"  annotated majority : {fmt(maj)}  {cmaj}/{na} = {p_data:.1f}%")
        print(f"  completeness       : {nc}/{ntot} complete, {npt} partial "
              f"({100*nc/ntot:.0f}%)")
        print(f"  among complete, share of that topology : "
              f"{comp[sp][maj]}/{nc} = {p_comp:.1f}%")
        print(f"  amplification      : {p_comp - p_data:+.1f} points")
        print(f"  distinct complete topologies produced  : {len(comp[sp])}")
        print(f"  [uncorrected, partials pooled in : {p_raw:.1f}% "
              f"= {p_raw - p_data:+.1f}; do not report this]")
        if pmaj != maj:
            print(f"  NOTE predicted majority differs : {fmt(pmaj)} "
                  f"({pcnt}/{nc})")

    if not rows:
        print("\nNo spacing met the thresholds. Lower --min-n / --min-pred.")
        return

    print("\n" + "=" * 86)
    print("SUMMARY")
    print("=" * 86)
    print(f"  {'spacing':<24} {'data%':>7} {'compl%':>7} {'ampl':>8} "
          f"{'n_ann':>6} {'cmpl':>5} {'part':>5} {'topos':>6}")
    for sp, pd, pc, pr, na, nc, npt, match, ntop in rows:
        print(f"  {str(list(sp)):<24} {pd:7.1f} {pc:7.1f} {pc-pd:+8.1f} "
              f"{na:6d} {nc:5d} {npt:5d} {ntop:6d}")

    amps = [r[2] - r[1] for r in rows]
    pos = sum(1 for x in amps if x > 0)
    saturated = sum(1 for r in rows if r[8] == 1)
    print(f"\n  groups where complete predictions exceed annotation : "
          f"{pos}/{len(rows)}")
    print(f"  mean amplification : {sum(amps)/len(amps):+.1f} points "
          f"(range {min(amps):+.1f} to {max(amps):+.1f})")
    print(f"  groups where ALL complete predictions are one topology : "
          f"{saturated}/{len(rows)}")

    print("""
HOW TO READ
  GENERAL     amplification positive across groups in both families. The
              predictor sharpens whatever a cysteine framework's majority is,
              and the last column shows how far: one distinct topology across
              every complete prediction means it has gone to absolute.

  BETA-ONLY   positive for defensins, flat or negative for knottins. Then the
              defensin result is a beta-specific prior and the claim must be
              scoped to defensins.

  The uncorrected line is printed for transparency only. Pooling partial
  predictions with wrong ones mixes 'the model was confidently wrong' with
  'the model declined to answer', which are different results; the partial
  count belongs with the density boundary, not with the topology comparison.
""")


if __name__ == "__main__":
    main()