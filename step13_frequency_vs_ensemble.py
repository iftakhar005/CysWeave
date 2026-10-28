#!/usr/bin/env python3
"""
step13_frequency_vs_ensemble.py
===============================
Does AlphaFold2 reproduce the topology FREQUENCY for a cysteine spacing?

The question left open by step10/step12
---------------------------------------
Spacing determines the prediction; alignment depth does not explain which
prediction. So what does the model appear to be doing? One hypothesis is that
it reproduces the topology distribution its training sources carry for that
spacing pattern -- effectively a lookup keyed on the cysteine framework.

That hypothesis makes a checkable prediction. The composition of the five-model
ensemble for a group should track the topology frequency for that group's
spacing in the sequence databases. If it does, we have a mechanism and not just
an observation. If the ensemble is always beta-dominated regardless of what the
data says for that spacing, the model has a global prior that overrides the
local statistics, which is a different and stronger claim.

SP4 is the interesting case: it is the only collision group whose spacing is
NOT beta-dominated in our dataset, and it is also the only group where the
ensemble discriminated. This script tests whether those two facts line up
across all four groups or whether SP4 is a coincidence.

Caveat built into the output
----------------------------
Frequency estimated from our own ~1100-sequence dataset is only meaningful when
that spacing has enough examples in it. SP3 has two. The script prints n for
every estimate and marks the small ones; it also computes a relaxed estimate
over near-neighbour spacings to give those groups a larger, if looser, sample.

Usage
-----
    python step13_frequency_vs_ensemble.py
    python step13_frequency_vs_ensemble.py --dataset defensin_connectivity_dataset.json
    python step13_frequency_vs_ensemble.py --tol 2
"""

import argparse
import json
import os
from collections import Counter, defaultdict


# --------------------------------------------------------------------------
# topology handling
# --------------------------------------------------------------------------

NAMES = {
    ((1, 5), (2, 4), (3, 6)): "beta",
    ((1, 6), (2, 4), (3, 5)): "alpha",
    ((1, 4), (2, 5), (3, 6)): "knot",
    ((1, 6), (2, 5), (3, 4)): "theta",
}


def key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def label(k):
    if k in NAMES:
        return NAMES[k]
    if not k:
        return "(none)"
    if len(k) < 3:
        return "partial-%d" % len(k)
    return "other"


def show(k):
    return f"{label(k):<9} {[list(p) for p in k]}"


def spacing_of(seq):
    """Residues between consecutive cysteines. Matches the project convention."""
    pos = [i for i, c in enumerate(seq.upper()) if c == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def near(a, b, tol):
    """Same length, every gap within tol."""
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


# --------------------------------------------------------------------------

def load_dataset(path):
    """-> list of (spacing, topology_key). Prefers stored fields."""
    rows = []
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("records") or data.get("entries") or list(data.values())
    for r in data:
        if not isinstance(r, dict):
            continue
        pairs = (r.get("pairs") or r.get("disulfides") or r.get("connectivity")
                 or r.get("topology") or r.get("intended_pairs"))
        if not pairs:
            continue
        seq = (r.get("sequence") or r.get("mature_sequence") or
               r.get("mature") or "")
        sp = r.get("spacing") or r.get("cys_spacing")
        if sp:
            sp = tuple(sp)
        elif seq:
            sp = spacing_of(seq)
        else:
            continue
        try:
            rows.append((sp, key(pairs)))
        except Exception:
            continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="defensin_connectivity_dataset.json")
    ap.add_argument("--ensemble", default="rank_ensemble_results.json")
    ap.add_argument("--pdb", default="pdb_topology_counts.json")
    ap.add_argument("--tol", type=int, default=1,
                    help="gap tolerance for the relaxed neighbour estimate")
    ap.add_argument("--min-n", type=int, default=6,
                    help="below this, an exact-spacing estimate is not trusted")
    a = ap.parse_args()

    for f in (a.dataset, a.ensemble):
        if not os.path.exists(f):
            print(f"missing {f}")
            if f == a.dataset:
                print("  candidates in this folder:")
                for c in sorted(x for x in os.listdir(".") if x.endswith(".json")):
                    print("   ", c)
            return

    rows = load_dataset(a.dataset)
    print(f"dataset: {len(rows)} annotated records from {a.dataset}")
    if not rows:
        d = json.load(open(a.dataset))
        d = d if isinstance(d, list) else list(d.values())
        first = next((x for x in d if isinstance(x, dict)), {})
        print("  Parsed nothing. Keys present on the first record:")
        print("   ", sorted(first.keys()))
        print("  Tell me which key holds the disulfide pairs and I will patch "
              "load_dataset().")
        return

    recs = json.load(open(a.ensemble))
    print(f"ensemble: {len(recs)} peptides from {a.ensemble}\n")

    # global dataset distribution, six-cysteine only
    six = [(s, t) for s, t in rows if len(s) == 5]
    gdist = Counter(t for _, t in six)
    gtot = sum(gdist.values())
    print("=" * 78)
    print(f"GLOBAL topology frequency, six-cysteine records (n={gtot})")
    print("=" * 78)
    for t, c in gdist.most_common(6):
        print(f"  {show(t):<44} {c:5d}  {100*c/gtot:5.1f}%")

    if os.path.exists(a.pdb):
        try:
            pdb = json.load(open(a.pdb))
            print(f"\n  {a.pdb}: {json.dumps(pdb)[:300]}")
        except Exception as e:
            print(f"\n  could not read {a.pdb}: {e}")

    # group the ensemble records and recover each group's spacing
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)

    print("\n" + "=" * 78)
    print("PER GROUP -- dataset frequency for that spacing vs ensemble output")
    print("=" * 78)

    table = []
    for g in sorted(by_g, key=str):
        members = by_g[g]
        sp = None
        for m in members:
            if m.get("sequence"):
                sp = spacing_of(m["sequence"])
                break
        print(f"\n### {g}   spacing = {list(sp) if sp else 'unknown'}")

        # --- what the data says for this spacing ---
        exact = Counter(t for s, t in rows if sp and s == sp)
        n_ex = sum(exact.values())
        print(f"  dataset, exact spacing (n={n_ex}):")
        if n_ex:
            for t, c in exact.most_common(5):
                print(f"     {show(t):<44} {c:4d}  {100*c/n_ex:5.1f}%")
        else:
            print("     none")

        relaxed = Counter(t for s, t in rows if sp and near(s, sp, a.tol))
        n_rx = sum(relaxed.values())
        if n_ex < a.min_n and n_rx > n_ex:
            print(f"  dataset, spacing within +/-{a.tol} (n={n_rx}):")
            for t, c in relaxed.most_common(5):
                print(f"     {show(t):<44} {c:4d}  {100*c/n_rx:5.1f}%")

        # --- what the model produced ---
        ens = Counter()
        for m in members:
            for v in m["preds"].values():
                ens[key(v)] += 1
        n_en = sum(ens.values())
        print(f"  AlphaFold2 ensemble, all models pooled (n={n_en}):")
        for t, c in ens.most_common(5):
            print(f"     {show(t):<44} {c:4d}  {100*c/n_en:5.1f}%")

        # --- the comparison, on beta share ---
        b = ((1, 5), (2, 4), (3, 6))
        if n_ex >= a.min_n:
            src, n_src, tag = exact, n_ex, "exact"
        elif n_rx > n_ex:
            src, n_src, tag = relaxed, n_rx, f"+/-{a.tol}"
        else:
            src, n_src, tag = exact, n_ex, "exact, SMALL"
        d_beta = 100 * src[b] / n_src if n_src else None
        e_beta = 100 * ens[b] / n_en if n_en else None
        if d_beta is not None and e_beta is not None:
            print(f"  -> beta share   data {d_beta:5.1f}% ({tag}, n={n_src})"
                  f"   ensemble {e_beta:5.1f}%   diff {e_beta-d_beta:+6.1f}")
            table.append((g, d_beta, e_beta, n_src, tag))
        else:
            print("  -> beta share   not computable")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'group':<7} {'data beta%':>11} {'ens beta%':>10} {'diff':>8}   basis")
    for g, d, e, n, tag in table:
        print(f"  {g:<7} {d:11.1f} {e:10.1f} {e-d:+8.1f}   {tag}, n={n}")

    if len(table) >= 2:
        ranked_d = [g for g, d, e, n, t in sorted(table, key=lambda x: -x[1])]
        ranked_e = [g for g, d, e, n, t in sorted(table, key=lambda x: -x[2])]
        print(f"\n  groups ordered by data beta share     : {ranked_d}")
        print(f"  groups ordered by ensemble beta share : {ranked_e}")
        print(f"  orders agree: {ranked_d == ranked_e}")

    print("""
HOW TO READ
  The hypothesis is that the model reproduces the topology frequency its
  training sources carry for a given cysteine spacing.

  SUPPORTED  if ensemble beta share tracks data beta share across groups, and
             in particular if the group with the lowest data beta share also
             has the lowest ensemble beta share. That is a mechanism: the
             predictor is doing frequency lookup on the framework.

  REFUTED    if the ensemble is uniformly beta-heavy regardless of what the
             data says for that spacing. That means a global prior overriding
             local statistics -- a stronger claim than frequency matching, and
             it would mean the SP4 result needs a different explanation.

  With four groups this is a direction, not a statistic. Treat any group whose
  estimate rests on a handful of records as indicative only; the model draws on
  databases far larger than ours, so a small local count is a poor proxy for
  what it actually saw.
""")


if __name__ == "__main__":
    main()