#!/usr/bin/env python3
"""
step8_family_density_screen.py
==============================
Which peptide family can actually be tested with this method?

Background
----------
Pooling 510 folded peptides showed that CYSTEINE DENSITY (cysteines per
residue), not raw length, predicts whether ESMFold produces a complete
disulfide prediction:

    density   complete
    0.10       100%
    0.15        98%
    0.20        88%     <- usable limit
    0.25        45%
    0.30        31%
    0.40         5%
    0.45         0%

Below ~0.20 the oracle works. Above it, ESMFold declines to place disulfides
rather than placing them wrongly -- so the connectivity metric returns no
signal, not a wrong signal. That is why the conotoxin run (median density
~0.35-0.40) produced 18% complete predictions and could not test anything.

So before building and folding another family, screen it. Density is
computable from sequence alone -- no folding, no GPU, minutes.

What this does
--------------
For each candidate family: fetch from UniProt, compute the density
distribution, estimate what fraction of the family falls in the workable band,
and report whether it is worth pursuing. Also reports how many collision
groups would survive the density filter, since a family is only useful if it
has BOTH workable density AND same-spacing/different-topology groups.

Usage
-----
    python step8_family_density_screen.py
    python step8_family_density_screen.py --families cyclotide knottin "plant defensin"
    python step8_family_density_screen.py --use-existing   # screen files you already built
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

WORKABLE = 0.20        # density at or below which completeness stayed >=88%
MARGINAL = 0.25        # between WORKABLE and this, completeness ~45%


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


def density(seq):
    return seq.count("C") / len(seq) if seq else 0.0


def spacing_pattern(seq):
    pos = [i for i, a in enumerate(seq) if a == "C"]
    return tuple(pos[i + 1] - pos[i] - 1 for i in range(len(pos) - 1))


def topo_key(pairs):
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def band(d):
    if d <= WORKABLE:
        return "workable"
    if d <= MARGINAL:
        return "marginal"
    return "too dense"


# --------------------------------------------------------------------------

def screen(recs, label):
    recs = [r for r in recs if get_sequence(r) and r.get("pairs")]
    if not recs:
        print(f"\n### {label}: no usable records")
        return None

    ds = [density(get_sequence(r)) for r in recs]
    bands = Counter(band(d) for d in ds)
    n = len(recs)
    ds_sorted = sorted(ds)
    med = ds_sorted[n // 2]

    workable = [r for r in recs if density(get_sequence(r)) <= WORKABLE]

    print(f"\n### {label}")
    print(f"  n = {n}   median density = {med:.3f}   "
          f"median length = {sorted(len(get_sequence(r)) for r in recs)[n//2]}")
    print(f"  workable (<={WORKABLE}):   {bands['workable']:>4}  "
          f"({100*bands['workable']/n:>5.1f}%)")
    print(f"  marginal ({WORKABLE}-{MARGINAL}):  {bands['marginal']:>4}  "
          f"({100*bands['marginal']/n:>5.1f}%)")
    print(f"  too dense (>{MARGINAL}):    {bands['too dense']:>4}  "
          f"({100*bands['too dense']/n:>5.1f}%)")

    # collision groups AFTER the density filter -- both conditions must hold
    groups = defaultdict(set)
    for r in workable:
        groups[spacing_pattern(get_sequence(r))].add(topo_key(r["pairs"]))
    collisions = {sp: t for sp, t in groups.items() if len(t) > 1}

    # how many workable sequences actually sit inside a collision group
    in_collision = sum(1 for r in workable
                       if spacing_pattern(get_sequence(r)) in collisions)

    print(f"  collision groups among workable seqs: {len(collisions)}"
          f"   (covering {in_collision} sequences)")

    if bands["workable"] < 30:
        verdict = "NOT VIABLE - too few workable-density sequences"
    elif not collisions:
        verdict = "NOT VIABLE - workable density but no collision groups"
    elif len(collisions) < 3:
        verdict = f"WEAK - only {len(collisions)} collision group(s)"
    else:
        verdict = (f"VIABLE - {bands['workable']} workable seqs, "
                   f"{len(collisions)} collision groups")
    print(f"  -> {verdict}")
    return {"label": label, "n": n, "median_density": med,
            "workable": bands["workable"], "collisions": len(collisions),
            "verdict": verdict}


def fetch_family(term, reviewed=True, min_len=10, max_len=120, limit=1500):
    """Reuses the same UniProt query shape as build_uniprot_dataset.py."""
    import requests
    q = f"({term}) AND (ft_disulfid:*) AND (length:[{min_len} TO {max_len}])"
    if reviewed:
        q += " AND (reviewed:true)"
    params = {
        "query": q, "format": "json", "size": 500,
        "fields": ("accession,sequence,ft_disulfid,ft_signal,ft_propep,"
                   "ft_chain,protein_name"),
    }
    r = requests.get("https://rest.uniprot.org/uniprotkb/stream",
                     params=params, timeout=180)
    r.raise_for_status()
    return r.json().get("results", [])


def parse_entries(entries):
    """Minimal mature-peptide + disulfide extraction (same rules as step 0)."""
    def loc(f, which):
        try:
            v = f["location"][which]["value"]
        except (KeyError, TypeError):
            return None
        return v if isinstance(v, int) else None

    out = []
    for e in entries:
        seq = e.get("sequence", {}).get("value", "")
        if not seq:
            continue
        feats = e.get("features", [])
        chains = [(loc(f, "start"), loc(f, "end")) for f in feats
                  if f.get("type") == "Chain"]
        chains = [c for c in chains if c[0] and c[1]]
        if chains:
            s, t = max(chains, key=lambda c: c[1] - c[0])
        else:
            cut = 0
            for f in feats:
                if f.get("type") in ("Signal", "Propeptide", "Transit peptide"):
                    v = loc(f, "end")
                    if v:
                        cut = max(cut, v)
            s, t = cut + 1, len(seq)
        mature = seq[s - 1:t]
        if not (10 <= len(mature) <= 120):
            continue

        ordinal, k = {}, 0
        for i, a in enumerate(mature):
            if a == "C":
                k += 1
                ordinal[s + i] = k
        pairs = []
        for f in feats:
            if f.get("type") != "Disulfide bond":
                continue
            if "Interchain" in (f.get("description") or ""):
                continue
            a, b = loc(f, "start"), loc(f, "end")
            if a in ordinal and b in ordinal:
                pairs.append(tuple(sorted((ordinal[a], ordinal[b]))))
        pairs = sorted(set(pairs))
        if len(pairs) < 2 or len(pairs) * 2 != mature.count("C"):
            continue
        out.append({"accession": e.get("primaryAccession", ""),
                    "mature_sequence": mature,
                    "pairs": [list(p) for p in pairs]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*",
                    default=["cyclotide", "knottin", "plant defensin",
                             "hepcidin", "toxin-like peptide"])
    ap.add_argument("--use-existing", action="store_true",
                    help="screen *_connectivity_dataset.json files instead of fetching")
    args = ap.parse_args()

    print("=" * 74)
    print("FAMILY DENSITY SCREEN")
    print(f"(workable <= {WORKABLE} cys/residue; above ~{MARGINAL} the oracle "
          "stops predicting)")
    print("=" * 74)

    results = []

    # always screen what you already have, for reference points
    for f in sorted(glob.glob("*_connectivity_dataset.json")):
        try:
            results.append(screen(json.load(open(f)), f"[existing] {f}"))
        except Exception as ex:
            print(f"\n### {f}: could not read ({ex})")

    if not args.use_existing:
        for fam in args.families:
            try:
                entries = fetch_family(fam)
                recs = parse_entries(entries)
                res = screen(recs, f"[uniprot] {fam}")
                if res and res["workable"] >= 30:
                    path = (fam.replace(" ", "_")
                            + "_connectivity_dataset.json")
                    json.dump(recs, open(path, "w"), indent=2)
                    print(f"  saved {len(recs)} records -> {path}")
                results.append(res)
            except Exception as ex:
                print(f"\n### [uniprot] {fam}: fetch failed ({type(ex).__name__}: {ex})")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"{'family':<38}{'n':>6}{'med dens':>10}{'workable':>10}{'collis':>8}")
    print("-" * 74)
    for r in [x for x in results if x]:
        print(f"{r['label'][:37]:<38}{r['n']:>6}{r['median_density']:>10.3f}"
              f"{r['workable']:>10}{r['collisions']:>8}")

    print("""
NEXT
  Pick the family marked VIABLE with the most collision groups, build it with
  build_uniprot_dataset.py (or use the file this script just saved), then run
  step2_metric_discrimination.py on it and fold ONLY the workable-density
  sequences. Filtering by density before folding avoids repeating the conotoxin
  outcome, where 82% of predictions were incomplete and nothing could be
  concluded.""")


if __name__ == "__main__":
    main()