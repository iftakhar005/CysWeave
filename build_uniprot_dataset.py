#!/usr/bin/env python3
"""
build_uniprot_dataset.py
========================
THE DATA-SUFFICIENCY GO/NO-GO for the topology-conditioning method.

Fetches defensin sequences from UniProt that carry disulfide-bond annotations,
extracts the MATURE peptide (so cysteine numbering is correct), converts each
entry's disulfides into a pairing graph over cysteine ordinals (C1..CN), tags
each by EVIDENCE (experimental vs inferred-by-similarity), buckets by topology,
and counts.

The deliverable is one table:  topology -> number of sequences  (per evidence
class, raw and clustered). That table decides the architecture:

    hundreds of distinct sequences per major topology -> LoRA adapter (Option 2)
    only tens                                         -> inference-time guidance
                                                         (Option 3), no training

Outputs:
    defensin_connectivity_dataset.json   (one record per usable entry)
    prints the topology x evidence count table

Usage:
    pip install requests
    python build_uniprot_dataset.py                     # reviewed (Swiss-Prot)
    python build_uniprot_dataset.py --all               # include TrEMBL too
    python build_uniprot_dataset.py --cluster           # add approx non-redundant counts
    python build_uniprot_dataset.py --query "defensin"  # change the search term

NOTE ON CLUSTERING: --cluster uses an approximate, alignment-free identity
(difflib ratio) purely to give a rough non-redundant count. For the REAL
train/test split, run CD-HIT or MMseqs2 on the emitted FASTA (also written).
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict

import requests

STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# ECO evidence codes
EXPERIMENTAL = {"ECO:0000269", "ECO:0000305"}   # experimental / curator-from-exp
INFERRED     = {"ECO:0000250", "ECO:0000255",   # by similarity / by rule
                "ECO:0000256", "ECO:0000312", "ECO:0000313"}


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch(query, reviewed=True, min_len=10, max_len=120):
    q = f"({query}) AND (ft_disulfid:*) AND (length:[{min_len} TO {max_len}])"
    if reviewed:
        q += " AND (reviewed:true)"
    params = {
        "query": q,
        "format": "json",
        "fields": ("accession,id,protein_name,organism_name,sequence,"
                   "ft_disulfid,ft_signal,ft_propep,ft_chain,ft_transit"),
        "size": 500,
    }
    print(f"UniProt query: {q}\n")
    entries, url, first = [], STREAM, True
    # the stream endpoint returns everything in one response; loop is a safety net
    while url:
        r = requests.get(url, params=params if first else None, timeout=180)
        r.raise_for_status()
        data = r.json()
        entries.extend(data.get("results", []))
        # stream has no pagination, but keep a guard in case of the search API
        url = None
        first = False
    print(f"  fetched {len(entries)} raw entries")
    return entries


# --------------------------------------------------------------------------
# Parse (verified offline)
# --------------------------------------------------------------------------

def _feats(entry):
    return entry.get("features", [])


def loc_span(f):
    """(start,end) for a feature, or None if either position is missing/unknown.

    UniProt uses null positions for uncertain/unknown feature boundaries
    (location modifier UNKNOWN), so every position read must tolerate None.
    """
    try:
        a = f["location"]["start"]["value"]
        b = f["location"]["end"]["value"]
    except (KeyError, TypeError):
        return None
    if a is None or b is None:
        return None
    return int(a), int(b)


def mature_region(features, seqlen):
    """(start,end) 1-based inclusive of the mature peptide.
    Prefer the longest explicit Chain; else strip Signal/Propeptide/Transit;
    else use the whole sequence. Features with unknown positions are ignored."""
    chains = []
    for f in features:
        if f.get("type") == "Chain":
            span = loc_span(f)
            if span:
                chains.append(span)
    if chains:
        a, b = max(chains, key=lambda s: s[1] - s[0])
        return a, b

    cut = 0
    for f in features:
        if f.get("type") in ("Signal", "Propeptide", "Transit peptide"):
            span = loc_span(f)
            if span:
                cut = max(cut, span[1])
    return cut + 1, seqlen


def disulfides_in_mature(features, m_start, m_end):
    """-> (list of absolute (a,b) bonds, evidence-code set). Skips interchain
    and any bond with an endpoint outside the mature region."""
    bonds, ev = [], set()
    for f in features:
        if f.get("type") != "Disulfide bond":
            continue
        if "Interchain" in (f.get("description") or ""):
            continue
        span = loc_span(f)
        if span is None:
            continue
        a, b = span
        if not (m_start <= a <= m_end and m_start <= b <= m_end):
            continue
        bonds.append((a, b))
        for e in f.get("evidences", []):
            ev.add(e.get("evidenceCode", ""))
    return bonds, ev


def to_cys_ordinals(mature_seq, bonds_abs, m_start):
    """Absolute residue positions -> 1..N cysteine indices within the mature seq."""
    ordinal, k = {}, 0
    for i, aa in enumerate(mature_seq):
        if aa == "C":
            k += 1
            ordinal[m_start + i] = k
    pairs = []
    for a, b in bonds_abs:
        if a in ordinal and b in ordinal:
            pairs.append(tuple(sorted((ordinal[a], ordinal[b]))))
    return sorted(set(pairs))


def classify_evidence(ev):
    has_exp = bool(ev & EXPERIMENTAL)
    has_inf = bool(ev & INFERRED)
    if has_exp and not has_inf:
        return "experimental"
    if has_exp and has_inf:
        return "mixed"
    if has_inf:
        return "inferred"
    return "unknown"


def parse_entry(entry):
    seq = entry.get("sequence", {}).get("value", "")
    if not seq:
        return None
    feats = _feats(entry)
    try:
        m_start, m_end = mature_region(feats, len(seq))
    except Exception:
        return None
    # clamp to the actual sequence; skip nonsensical spans
    m_start = max(1, m_start)
    m_end = min(len(seq), m_end)
    if m_end < m_start:
        return None
    mature = seq[m_start - 1:m_end]
    if not (10 <= len(mature) <= 120):
        return None

    bonds_abs, ev = disulfides_in_mature(feats, m_start, m_end)
    pairs = to_cys_ordinals(mature, bonds_abs, m_start)
    ncys = mature.count("C")

    # require every cysteine paired and at least 2 bonds (a real disulfide fold)
    if len(pairs) < 2 or len(pairs) * 2 != ncys:
        return None

    name = ""
    try:
        name = (entry["proteinDescription"]["recommendedName"]
                     ["fullName"]["value"])
    except Exception:
        pass

    return {
        "accession": entry.get("primaryAccession", ""),
        "name": name,
        "organism": entry.get("organism", {}).get("scientificName", ""),
        "mature_sequence": mature,
        "n_cys": ncys,
        "pairs": [list(p) for p in pairs],
        "topology": tuple(pairs),                     # canonical signature
        "evidence_codes": sorted(ev),
        "evidence_class": classify_evidence(ev),
    }


# --------------------------------------------------------------------------
# Approximate non-redundant clustering (rough count only)
# --------------------------------------------------------------------------

def approx_cluster(seqs, threshold=0.7):
    """Greedy alignment-free clustering by difflib ratio. Returns #clusters.
    Rough proxy for CD-HIT -- use real clustering for the actual split."""
    from difflib import SequenceMatcher
    reps = []
    for s in sorted(seqs, key=len, reverse=True):
        if not any(SequenceMatcher(None, s, r).ratio() >= threshold
                   for r in reps):
            reps.append(s)
    return len(reps)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="defensin")
    ap.add_argument("--all", action="store_true",
                    help="include unreviewed TrEMBL entries (larger, noisier)")
    ap.add_argument("--cluster", action="store_true",
                    help="also report approximate non-redundant counts")
    ap.add_argument("--out", default="defensin_connectivity_dataset.json")
    args = ap.parse_args()

    try:
        raw = fetch(args.query, reviewed=not args.all)
    except Exception as e:
        print(f"FETCH FAILED: {e}")
        print("Check your network. UniProt REST is at rest.uniprot.org.")
        sys.exit(1)

    records, dropped, errored = [], 0, 0
    for e in raw:
        try:
            rec = parse_entry(e)
        except Exception:
            errored += 1
            continue
        if rec:
            records.append(rec)
        else:
            dropped += 1
    if errored:
        print(f"  note: {errored} entries skipped due to malformed annotations")

    # de-duplicate identical mature sequences (keep the best evidence)
    order = {"experimental": 0, "mixed": 1, "inferred": 2, "unknown": 3}
    best = {}
    for r in records:
        s = r["mature_sequence"]
        if s not in best or order[r["evidence_class"]] < order[best[s]["evidence_class"]]:
            best[s] = r
    unique = list(best.values())

    with open(args.out, "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "topology"}
                   for r in unique], fh, indent=2)
    with open("defensin_mature.fasta", "w") as fh:
        for r in unique:
            fh.write(f">{r['accession']}|{r['evidence_class']}\n"
                     f"{r['mature_sequence']}\n")

    print(f"\n  usable entries: {len(records)}  "
          f"(dropped {dropped}: no/partial disulfides, bad length, unpaired Cys)")
    print(f"  unique mature sequences: {len(unique)}")

    # ---- topology x evidence table ----
    topo_ev = defaultdict(lambda: Counter())
    topo_seqs = defaultdict(list)
    for r in unique:
        topo_ev[r["topology"]][r["evidence_class"]] += 1
        topo_seqs[r["topology"]].append(r["mature_sequence"])

    print("\n" + "=" * 78)
    print("TOPOLOGY  x  EVIDENCE   (unique mature sequences)")
    print("=" * 78)
    hdr = f"{'topology':<34}{'exp':>5}{'mixed':>7}{'infer':>7}{'total':>7}"
    if args.cluster:
        hdr += f"{'~nr':>6}"
    print(hdr)
    print("-" * 78)

    rows = sorted(topo_ev.items(), key=lambda kv: -sum(kv[1].values()))
    for topo, ev in rows:
        total = sum(ev.values())
        line = (f"{str([list(p) for p in topo]):<34}"
                f"{ev['experimental']:>5}{ev['mixed']:>7}"
                f"{ev['inferred'] + ev['unknown']:>7}{total:>7}")
        if args.cluster:
            line += f"{approx_cluster(topo_seqs[topo]):>6}"
        print(line)

    print("-" * 78)
    tot_exp = sum(ev['experimental'] for ev in topo_ev.values())
    tot_mix = sum(ev['mixed'] for ev in topo_ev.values())
    tot_inf = sum(ev['inferred'] + ev['unknown'] for ev in topo_ev.values())
    print(f"{'TOTAL':<34}{tot_exp:>5}{tot_mix:>7}{tot_inf:>7}"
          f"{len(unique):>7}")
    print(f"\ndistinct topologies: {len(topo_ev)}")

    # ---- the decision ----
    print("\n" + "-" * 78)
    big = [t for t, ev in topo_ev.items() if sum(ev.values()) >= 100]
    mid = [t for t, ev in topo_ev.items() if 30 <= sum(ev.values()) < 100]
    print("READING THE RESULT:")
    print(f"  topologies with >=100 sequences: {len(big)}  "
          f"(LoRA adapter viable on these)")
    print(f"  topologies with 30-99 sequences: {len(mid)}  (borderline)")
    if len(big) >= 2:
        print("  -> Enough data on multiple topologies. Option 2 (LoRA) is viable.")
    elif len(big) + len(mid) >= 2:
        print("  -> Marginal. Consider Option 2 on the big topologies only, or")
        print("     augment, or fall back to Option 3 (inference-time guidance).")
    else:
        print("  -> Too thin for training. Favour Option 3 (guidance, no training),")
        print("     or the cysteine-spacing angle. Re-run with --all to see TrEMBL.")
    print("-" * 78)
    print(f"\nWrote {args.out} and defensin_mature.fasta")
    print("For the real non-redundant split, run:  "
          "cd-hit -i defensin_mature.fasta -o nr90 -c 0.9")


if __name__ == "__main__":
    main()