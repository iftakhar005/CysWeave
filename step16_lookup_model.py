#!/usr/bin/env python3
"""
step16_lookup_model.py
======================
How much of the validator's behaviour does a cysteine-spacing lookup explain?

The claim so far is qualitative: structure predictors return one topology per
cysteine framework, so structure-based validation of a topology-conditioned
generator cannot detect the requested topology. This turns that into a
measurement.

Build a lookup from the reference annotations:  spacing -> majority topology.
Then for every peptide we have folded, compare three things:

    ORACLE       what the structure predictor actually returned
    SPACING      what the lookup predicts from cysteine spacing alone
    REQUESTED    the topology that was asked for (or annotated, for naturals)

Two numbers come out.

  1. Accuracy of SPACING at predicting ORACLE. If the predictor is doing
     framework lookup, this is high, and the mechanism stops being a
     description and becomes a model.

  2. On the subset where SPACING and REQUESTED DISAGREE, which one does the
     ORACLE follow? This is the decisive cell. If the oracle tracks SPACING
     there, the requested topology has no influence on the validator's verdict,
     and evaluation-insensitivity is measured rather than inferred.

Leave-one-out
-------------
Natural peptides appear in the reference set, so their own annotation would
otherwise be inside the majority they are scored against. Any peptide whose
sequence is found in the reference has its own record subtracted before the
majority is taken. Without this, naturals are predicted trivially.

Minimum support
---------------
After leave-one-out, some spacings are left with one or two reference records,
and a "majority" over one record is not a prediction. --min-support drops those
peptides into the not-scorable column so the result can be tested for
robustness against thin annotation.

Completeness
------------
Only complete predictions are scored: a prediction with fewer bonds than
cysteines/2 is the model declining to answer, not choosing a topology. Partial
counts are reported separately.

Usage
-----
  python step16_lookup_model.py \\
      --reference defensin_connectivity_dataset.json knottin_connectivity_dataset.json \\
      --folded partb_baseline.jsonl partb_conditioned.jsonl partb_nullcond.jsonl \\
               partb_scrambled.jsonl partb_swap.jsonl \\
               partb_discrimination_full.jsonl partb_knottin.jsonl \\
      --min-support 3 --out lookup_min3.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict


# --------------------------------------------------------------------------

def key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def fmt(k):
    return str([list(p) for p in k]) if k else "[]"


def spacing_of(seq):
    pos = [i for i, c in enumerate(seq.upper()) if c == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


SEQ_KEYS = ("sequence", "mature_sequence", "mature", "seq")
REQ_KEYS = ("intended_pairs", "intended", "requested", "target_pairs",
            "pairs", "disulfides", "connectivity", "topology")
ORA_KEYS = ("realized", "got", "predicted", "realized_pairs",
            "predicted_pairs", "observed")
REF_KEYS = ("pairs", "disulfides", "connectivity", "topology")


def first(d, names):
    for n in names:
        v = d.get(n)
        if v:
            return v
    return None


def load_records(path):
    if path.endswith("jsonl"):
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
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("records") or data.get("entries") or list(data.values())
    return [r for r in data if isinstance(r, dict)]


# --------------------------------------------------------------------------

def build_lookup(paths):
    """spacing -> Counter(topology), plus sequence -> topology for leave-one-out."""
    table, own = defaultdict(Counter), {}
    n = 0
    for p in paths:
        if not os.path.exists(p):
            print(f"  reference missing, skipped: {p}")
            continue
        for r in load_records(p):
            seq, pairs = first(r, SEQ_KEYS), first(r, REF_KEYS)
            if not seq or not pairs:
                continue
            seq = seq.upper()
            k = key(pairs)
            table[spacing_of(seq)][k] += 1
            own[seq] = k
            n += 1
        print(f"  reference {os.path.basename(p):<42} loaded")
    return table, own, n


def predict(table, own, seq, min_support=1):
    """Majority topology for this sequence's spacing, excluding its own record.

    Returns (None, n, False) when the spacing is unseen or the surviving
    support is below min_support -- those peptides are counted as not scorable
    rather than scored against a one-record majority.
    """
    sp = spacing_of(seq)
    c = Counter(table.get(sp, {}))
    mine = own.get(seq.upper())
    if mine and c.get(mine):
        c[mine] -= 1
        if c[mine] == 0:
            del c[mine]
    n = sum(c.values())
    if not c or n < min_support:
        return None, n, False
    top = c.most_common()
    tie = len(top) > 1 and top[0][1] == top[1][1]
    return top[0][0], n, tie


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", nargs="+", required=True)
    ap.add_argument("--folded", nargs="+", required=True)
    ap.add_argument("--min-support", type=int, default=1,
                    help="drop peptides whose spacing lookup rests on fewer "
                         "than this many reference records (after "
                         "leave-one-out)")
    ap.add_argument("--out", default="lookup_model_results.json")
    a = ap.parse_args()

    print("building the spacing lookup")
    table, own, nref = build_lookup(a.reference)
    if not table:
        print("no reference annotations parsed")
        return
    print(f"  {nref} annotated records -> {len(table)} distinct spacings\n")

    rows = []
    for path in a.folded:
        if not os.path.exists(path):
            print(f"folded missing, skipped: {path}")
            continue
        arm = os.path.basename(path).replace("partb_", "").split(".")[0]
        recs = load_records(path)
        for r in recs:
            seq = first(r, SEQ_KEYS)
            if not seq:
                continue
            req = first(r, REQ_KEYS)
            # rank_ensemble format: five predictions, take rank_001
            if isinstance(r.get("preds"), dict):
                ora = r["preds"].get("1")
            else:
                ora = first(r, ORA_KEYS)
            if req is None or ora is None:
                continue
            ncys = seq.upper().count("C")
            need = ncys // 2
            complete = need > 0 and len(ora) == need
            sp_pred, sp_n, tie = predict(table, own, seq, a.min_support)
            rows.append({
                "arm": arm, "sequence": seq, "ncys": ncys,
                "requested": [list(p) for p in key(req)],
                "oracle": [list(p) for p in key(ora)],
                "spacing_pred": [list(p) for p in sp_pred] if sp_pred else None,
                "support": sp_n, "tie": tie, "complete": complete,
            })

    if not rows:
        print("no folded records parsed")
        return

    K = lambda v: key(v) if v else None

    # ---------------- per arm ----------------
    print("=" * 92)
    print("ACCURACY AT PREDICTING WHAT THE ORACLE RETURNED")
    print("complete predictions only; no lookup entry = not scorable")
    print("=" * 92)
    print(f"  {'arm':<20} {'n':>5} {'part':>5} {'nocov':>6} "
          f"{'spacing':>9} {'requested':>10}")

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    tot_sp = tot_rq = tot_n = 0
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        part = sum(1 for r in rs if not r["complete"])
        sc = [r for r in rs if r["complete"] and r["spacing_pred"]]
        nocov = sum(1 for r in rs if r["complete"] and not r["spacing_pred"])
        if not sc:
            print(f"  {arm:<20} {0:>5} {part:>5} {nocov:>6}       n/a        n/a")
            continue
        sp_ok = sum(1 for r in sc if K(r["spacing_pred"]) == K(r["oracle"]))
        rq_ok = sum(1 for r in sc if K(r["requested"]) == K(r["oracle"]))
        tot_sp += sp_ok
        tot_rq += rq_ok
        tot_n += len(sc)
        print(f"  {arm:<20} {len(sc):>5} {part:>5} {nocov:>6} "
              f"{100*sp_ok/len(sc):8.1f}% {100*rq_ok/len(sc):9.1f}%")

    if tot_n:
        print(f"\n  {'ALL':<20} {tot_n:>5} {'':>5} {'':>6} "
              f"{100*tot_sp/tot_n:8.1f}% {100*tot_rq/tot_n:9.1f}%")

    # ---------------- the decisive cell ----------------
    dis = [r for r in rows
           if r["complete"] and r["spacing_pred"]
           and K(r["spacing_pred"]) != K(r["requested"])]

    print("\n" + "=" * 92)
    print("THE DECISIVE CELL — peptides where SPACING and REQUESTED disagree")
    print("=" * 92)
    if not dis:
        print("  none. Every scorable peptide had spacing and request agreeing,")
        print("  so this data cannot separate the two. The swap arm is the one")
        print("  designed to populate this cell -- check it was included, and")
        print("  that --min-support has not removed it.")
    else:
        print(f"  {'arm':<20} {'n':>5} {'-> spacing':>11} {'-> requested':>13} "
              f"{'-> neither':>11}")
        d_by_arm = defaultdict(list)
        for r in dis:
            d_by_arm[r["arm"]].append(r)
        ts = tr = tn = 0
        for arm in sorted(d_by_arm):
            rs = d_by_arm[arm]
            s = sum(1 for r in rs if K(r["spacing_pred"]) == K(r["oracle"]))
            q = sum(1 for r in rs if K(r["requested"]) == K(r["oracle"]))
            nn = len(rs) - s - q
            ts += s; tr += q; tn += nn
            print(f"  {arm:<20} {len(rs):>5} {s:>11} {q:>13} {nn:>11}")
        print(f"\n  {'TOTAL':<20} {len(dis):>5} {ts:>11} {tr:>13} {tn:>11}")
        print(f"\n  oracle followed SPACING   : {100*ts/len(dis):5.1f}%")
        print(f"  oracle followed REQUESTED : {100*tr/len(dis):5.1f}%")

        # same cell with the scrambled arm removed
        ns = [r for r in dis if r["arm"] != "scrambled"]
        if ns and len(ns) != len(dis):
            s = sum(1 for r in ns if K(r["spacing_pred"]) == K(r["oracle"]))
            q = sum(1 for r in ns if K(r["requested"]) == K(r["oracle"]))
            print(f"\n  excluding the scrambled arm, whose request is noise "
                  f"rather than\n  a real alternative topology ({len(ns)} peptides):")
            print(f"    oracle followed SPACING   : {100*s/len(ns):5.1f}%  ({s}/{len(ns)})")
            print(f"    oracle followed REQUESTED : {100*q/len(ns):5.1f}%  ({q}/{len(ns)})")

    # ---------------- coverage notes ----------------
    ties = sum(1 for r in rows if r["complete"] and r["tie"])
    thin = sum(1 for r in rows if r["complete"] and r["spacing_pred"]
               and r["support"] < 3)
    print("\n" + "=" * 92)
    print("COVERAGE")
    print("=" * 92)
    print(f"  minimum support required                    : {a.min_support}")
    print(f"  lookup had a tied majority for              : {ties} peptides")
    print(f"  lookup rested on fewer than 3 records for   : {thin} peptides")
    print(f"  partial predictions excluded from scoring   : "
          f"{sum(1 for r in rows if not r['complete'])}")

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")

    print("""
HOW TO READ
  The first table is the mechanism as a number. If SPACING predicts the oracle
  far better than REQUESTED does, the predictor is keying on the cysteine
  framework and not on the rest of the sequence.

  The second table is the contribution. Both columns predict the oracle well
  when they agree, which says nothing. Only the rows where they disagree carry
  information, and there the oracle can follow only one of them. Read the
  scrambled-excluded figure as the headline: a scrambled request is noise, and
  the oracle failing to return noise is not evidence of anything.

  CAVEAT for the generated arms: cysteine positions were held fixed from the
  template, so a generated peptide inherits its template's spacing. SPACING is
  therefore not independent of the template's native topology, and the right
  reading is 'the validator reports the template's framework answer', not
  'the lookup has predictive power over novel sequences'. Say so in the paper;
  it is the honest version and it is still the claim that matters.
""")


if __name__ == "__main__":
    main()