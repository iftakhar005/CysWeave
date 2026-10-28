#!/usr/bin/env python3
"""
step15_prepare_knottin_cf.py  (v2)
==================================
Select knottin peptides for a ColabFold run that separates two explanations.

What the defensin data cannot settle
------------------------------------
Every defensin collision group's annotated majority is the beta-defensin
topology. So "the predictor returns the majority for that cysteine spacing" and
"the predictor always returns beta-defensin" make identical predictions on that
data, and it cannot tell them apart. Chapter 4 asserts the first on the strength
of ESMFold behaviour in knottins; AlphaFold2 has never been tested on it.

The test
--------
Knottins contain spacing groups with DIFFERENT unanimous majorities. Fold
several of them:

  SPACING-CONDITIONAL   each group's ensemble returns that group's own
                        majority. The lookup is keyed on the framework, and the
                        mechanism claim is confirmed on AlphaFold2.

  GLOBAL PRIOR          all groups return the same topology regardless of what
                        their annotations say. Then the chapter's central
                        sentence is wrong and needs rewriting.

v1 of this script partitioned groups into "conflicts with the per-cysteine-count
default" and "agrees with it". That does not work here: at six cysteines the
knot pattern is over 99% of records so nothing can conflict, and at eight the
majority is only ~60% so there is no default to conflict with. Majority
diversity is the property worth selecting on instead.

Selection
---------
  DIVERSITY arm   unanimous, well-sampled groups chosen greedily so that every
                  group contributes a majority topology not already covered.
                  Experimental annotation and low cysteine density preferred.
  COLLISION arm   groups carrying more than one annotated topology, with one
                  member of each, so the discrimination test runs on the same
                  structures at no extra cost.

Filters: invalid pairings dropped; cysteine density above --max-density
excluded, since those come back as partial predictions and waste the session.

Output
------
  knottin_cf_input.fasta    feed this to ColabFold
  knottin_cf_jobmap.json    keys match what step10_rank_ensemble.py expects

Usage
-----
    python step15_prepare_knottin_cf.py
    python step15_prepare_knottin_cf.py --max-peptides 30 --per-group 4
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


SEQ_KEYS = ("sequence", "mature_sequence", "mature", "seq")
PAIR_KEYS = ("pairs", "disulfides", "connectivity", "topology")
ACC_KEYS = ("_accession", "accession", "id", "entry")


def first(d, names):
    for n in names:
        v = d.get(n)
        if v:
            return v
    return None


def valid_pairing(pairs, ncys):
    flat = [c for p in pairs for c in p]
    return (len(pairs) * 2 == ncys
            and len(flat) == len(set(flat))
            and sorted(flat) == list(range(1, ncys + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="knottin_connectivity_dataset.json")
    ap.add_argument("--out-fasta", default="knottin_cf_input.fasta")
    ap.add_argument("--out-jobmap", default="knottin_cf_jobmap.json")
    ap.add_argument("--max-peptides", type=int, default=24)
    ap.add_argument("--per-group", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--max-density", type=float, default=0.22)
    ap.add_argument("--max-collision", type=int, default=2)
    ap.add_argument("--existing", default="knottin_discrimination.json")
    a = ap.parse_args()

    if not os.path.exists(a.dataset):
        print(f"missing {a.dataset}")
        return

    data = json.load(open(a.dataset))
    if isinstance(data, dict):
        data = data.get("records") or data.get("entries") or list(data.values())

    recs, drop_bad, drop_dense = [], 0, 0
    for r in data:
        if not isinstance(r, dict):
            continue
        seq, pairs = first(r, SEQ_KEYS), first(r, PAIR_KEYS)
        if not seq or not pairs:
            continue
        seq = seq.upper()
        ncys = seq.count("C")
        if ncys < 4 or not valid_pairing(pairs, ncys):
            drop_bad += 1 if ncys >= 4 else 0
            continue
        dens = ncys / len(seq)
        drop_dense += dens > a.max_density
        recs.append({
            "acc": first(r, ACC_KEYS) or "?", "seq": seq, "topo": key(pairs),
            "ncys": ncys, "density": dens, "spacing": spacing_of(seq),
            "evidence": r.get("evidence_class") or "?",
            "too_dense": dens > a.max_density,
        })

    print(f"parsed {len(recs)} usable records ({drop_bad} invalid pairings "
          f"dropped, {drop_dense} above density {a.max_density})\n")

    groups = defaultdict(list)
    for r in recs:
        groups[r["spacing"]].append(r)

    cand = []
    for sp, members in groups.items():
        foldable = [m for m in members if not m["too_dense"]]
        if len(members) < a.min_n or len(foldable) < 2:
            continue
        tc = Counter(m["topo"] for m in members)
        maj, cmaj = tc.most_common(1)[0]
        cand.append({
            "sp": sp, "members": members, "foldable": foldable, "n": len(members),
            "maj": maj, "maj_share": 100 * cmaj / len(members), "ntopo": len(tc),
            "ncys": members[0]["ncys"],
            "n_exp": sum(1 for m in members if m["evidence"] == "experimental"),
            "mean_dens": sum(m["density"] for m in foldable) / len(foldable),
        })

    # ---------------- selection ----------------
    def pick_members(c, limit):
        pool = sorted(c["foldable"],
                      key=lambda m: (m["evidence"] != "experimental",
                                     m["density"], m["acc"]))
        picked, seen = [], set()
        for m in pool:                      # one of each topology first
            if m["topo"] not in seen:
                picked.append(m)
                seen.add(m["topo"])
        for m in pool:
            if len(picked) >= limit:
                break
            if m not in picked:
                picked.append(m)
        return picked[:limit]

    chosen, total, covered = [], 0, set()

    collisions = sorted([c for c in cand if c["ntopo"] > 1],
                        key=lambda c: (-c["n_exp"], -c["n"]))
    for c in collisions[:a.max_collision]:
        lim = min(a.per_group + 1, a.max_peptides - total)
        p = pick_members(c, lim)
        if len(p) >= 2:
            chosen.append((c, p, "collision"))
            total += len(p)
            covered.add(c["maj"])

    unanimous = sorted([c for c in cand if c["ntopo"] == 1],
                       key=lambda c: (-c["n_exp"], -c["n"], c["mean_dens"]))
    for c in unanimous:
        if total >= a.max_peptides:
            break
        if c["maj"] in covered:             # majority already represented
            continue
        p = pick_members(c, min(a.per_group, a.max_peptides - total))
        if len(p) >= 2:
            chosen.append((c, p, "diversity"))
            total += len(p)
            covered.add(c["maj"])

    if not chosen:
        print("Nothing selected. Loosen --min-n or --max-density.")
        return

    # ---------------- write ----------------
    existing = set()
    if os.path.exists(a.existing):
        try:
            for r in json.load(open(a.existing)):
                s = first(r, SEQ_KEYS)
                if s:
                    existing.add(s.upper())
        except Exception:
            pass

    fasta, jobmap, i = [], {}, 0
    print("=" * 86)
    print("SELECTED FOR FOLDING")
    print("=" * 86)
    for gi, (c, picked, arm) in enumerate(chosen, 1):
        tag = f"K{gi}"
        print(f"\n### {tag}  spacing {list(c['sp'])}  [{arm}]  {c['ncys']} cys, "
              f"group n={c['n']}, {c['ntopo']} topolog"
              f"{'ies' if c['ntopo']>1 else 'y'}")
        print(f"     majority {fmt(c['maj'])}  {c['maj_share']:.1f}%")
        for m in picked:
            i += 1
            job = f"kno{i:04d}"
            fasta.append(f">{job}\n{m['seq']}")
            jobmap[job] = {
                "sequence": m["seq"],
                "intended_pairs": [list(p) for p in m["topo"]],
                "label": tag,
                "accession": m["acc"],
                "evidence": m["evidence"],
                "spacing": list(c["sp"]),
                "group_majority": [list(p) for p in c["maj"]],
                "arm": arm,
                "is_majority": m["topo"] == c["maj"],
                "density": round(m["density"], 3),
            }
            mark = "  (in ESMFold set)" if m["seq"] in existing else ""
            flag = "" if m["topo"] == c["maj"] else "  <-- minority"
            print(f"   {job}  {m['acc']:<12} {m['evidence']:<12} "
                  f"len={len(m['seq']):3d} d={m['density']:.2f}  "
                  f"{fmt(m['topo'])}{flag}{mark}")

    open(a.out_fasta, "w").write("\n".join(fasta) + "\n")
    json.dump(jobmap, open(a.out_jobmap, "w"), indent=2)

    print("\n" + "=" * 86)
    print("THE COMPARISON THIS SETS UP")
    print("=" * 86)
    print(f"  {'group':<6} {'cys':>4} {'n':>5}  annotated majority")
    for gi, (c, p, arm) in enumerate(chosen, 1):
        print(f"  {'K%d' % gi:<6} {c['ncys']:4d} {c['n']:5d}  {fmt(c['maj'])}")
    print(f"\n  distinct majorities across selected groups: {len(covered)}")
    if len(covered) < 3:
        print("  WARNING only a couple of distinct majorities. The test is weak;"
              "\n  raise --max-peptides or lower --min-n to cover more.")

    print(f"\nwrote {a.out_fasta} and {a.out_jobmap}")
    print(f"  {len(jobmap)} peptides across {len(chosen)} groups")
    print(f"""
NEXT
  1. Upload {a.out_fasta} to ColabFold. Match the defensin run:
     msa_mode = mmseqs2_uniref_env, model_type = alphafold2_ptm,
     num_models = 5, default recycles, relaxation OFF. Save results as cf_out.

  2. python step10_rank_ensemble.py --cf-out cf_out \\
            --jobmap {a.out_jobmap} --source-json {a.dataset}

     Ignore step10's closing CAVEAT paragraph; it is written for the defensins.

READING THE RESULT
  Compare each group's rank_001 prediction against the column above.

  SPACING-CONDITIONAL  each group returns its own majority. The lookup is keyed
                       on the cysteine framework, on AlphaFold2 as well as
                       ESMFold, and across two families. Chapter 4's mechanism
                       sentence is then properly supported.

  GLOBAL PRIOR         every group returns the same topology. The mechanism
                       sentence is wrong as written and the finding becomes a
                       fixed preference per predictor, which is a different and
                       narrower claim.

  MIXED                some groups track their majority and others default.
                       Then what separates them is the next question, and the
                       per-group alignment depths in cf_out are the first place
                       to look.
""")


if __name__ == "__main__":
    main()