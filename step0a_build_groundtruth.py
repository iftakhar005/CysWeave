#!/usr/bin/env python3
"""
step0a_build_groundtruth.py
===========================
Builds the ground-truth dataset for Step 0: real peptides with EXPERIMENTALLY
DETERMINED disulfide connectivity, taken from RCSB PDB.

It downloads PDB entries, reads their SSBOND records (which state exactly which
cysteine pairs are bonded), extracts the chain sequence, and converts each bond
into a pairing graph over cysteine indices C1..CN in chain order.

Output: ground_truth.json  ->  [{pdb_id, chain, sequence, n_cys, pairs}, ...]
        where pairs looks like [[1,4],[2,5],[3,6]]

Usage:
    pip install requests
    python step0a_build_groundtruth.py                 # search PDB for defensins
    python step0a_build_groundtruth.py 1ZMM 2LR5 ...   # or give explicit PDB IDs

NOTE: verify the search results yourself. A text search for "defensin" will also
return antibody Fabs, complexes, and other things you do not want. Inspect the
output file and delete anything that is not a small disulfide-rich peptide.
"""

import json
import sys
import time

import requests

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILE   = "https://files.rcsb.org/download/{}.pdb"

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


# --------------------------------------------------------------------------
# PDB parsing (column-based; tested)
# --------------------------------------------------------------------------

def parse_chain_residues(pdb_text, chain):
    """Ordered [(resseq, resname)] from CA atoms of one chain."""
    seen, out = set(), []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        line = line.ljust(80)
        if line[12:16].strip() != "CA":
            continue
        if line[21] != chain:
            continue
        if line[16] not in (" ", "A"):       # first altloc only
            continue
        resseq = line[22:27].strip()          # residue number + insertion code
        if resseq in seen:
            continue
        seen.add(resseq)
        out.append((resseq, line[17:20].strip()))
    return out


def parse_ssbond(pdb_text):
    """[(chain1, resseq1, chain2, resseq2)] from SSBOND records."""
    bonds = []
    for line in pdb_text.splitlines():
        if not line.startswith("SSBOND"):
            continue
        line = line.ljust(80)
        bonds.append((line[15], (line[17:21] + line[21]).strip(),
                      line[29], (line[31:35] + line[35]).strip()))
    return bonds


def chains_in(pdb_text):
    return sorted({line.ljust(80)[21] for line in pdb_text.splitlines()
                   if line.startswith("ATOM")})


def ground_truth_for_chain(pdb_text, chain):
    """-> (sequence, pairs, n_cys). Cysteines labelled C1..CN in chain order."""
    residues = parse_chain_residues(pdb_text, chain)
    seq = "".join(AA3TO1.get(rn, "X") for _, rn in residues)

    cys_index, k = {}, 0
    for resseq, rn in residues:
        if rn == "CYS":
            k += 1
            cys_index[resseq] = k

    pairs = []
    for c1, r1, c2, r2 in parse_ssbond(pdb_text):
        if c1 != chain or c2 != chain:
            continue                          # skip inter-chain disulfides
        if r1 in cys_index and r2 in cys_index:
            pairs.append(tuple(sorted((cys_index[r1], cys_index[r2]))))
    return seq, sorted(set(pairs)), len(cys_index)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def search_pdb(term="defensin", max_len=80, limit=100):
    """Full-text search restricted to short polymer entries. Returns PDB IDs."""
    query = {
        "query": {
            "type": "group", "logical_operator": "and", "nodes": [
                {"type": "terminal", "service": "full_text",
                 "parameters": {"value": term}},
                {"type": "terminal", "service": "text",
                 "parameters": {
                     "attribute": "entity_poly.rcsb_sample_sequence_length",
                     "operator": "less_or_equal", "value": max_len}},
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    r = requests.post(RCSB_SEARCH, json=query, timeout=60)
    r.raise_for_status()
    return [hit["identifier"] for hit in r.json().get("result_set", [])]


def fetch_pdb(pdb_id):
    r = requests.get(RCSB_FILE.format(pdb_id), timeout=60)
    r.raise_for_status()
    return r.text


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ids = [a.upper() for a in sys.argv[1:]]
    if not ids:
        print("Searching RCSB PDB for short defensin entries ...")
        ids = search_pdb()
        print(f"  {len(ids)} candidate entries")

    records, skipped = [], []
    for n, pdb_id in enumerate(ids, 1):
        try:
            text = fetch_pdb(pdb_id)
        except Exception as e:
            skipped.append((pdb_id, f"download failed: {e}"))
            continue

        for chain in chains_in(text):
            seq, pairs, ncys = ground_truth_for_chain(text, chain)
            # keep only short peptides with at least 2 disulfides fully resolved
            if not (10 <= len(seq) <= 80):
                continue
            if len(pairs) < 2 or ncys < 4:
                continue
            if len(pairs) * 2 != ncys:        # every cysteine accounted for
                skipped.append((f"{pdb_id}_{chain}", "unpaired cysteines"))
                continue
            records.append({
                "pdb_id": pdb_id, "chain": chain, "sequence": seq,
                "n_cys": ncys, "pairs": [list(p) for p in pairs],
            })
        if n % 10 == 0:
            print(f"  processed {n}/{len(ids)}")
        time.sleep(0.2)                        # be polite to the server

    # drop duplicate sequences
    unique, seen = [], set()
    for rec in records:
        if rec["sequence"] not in seen:
            seen.add(rec["sequence"])
            unique.append(rec)

    with open("ground_truth.json", "w") as fh:
        json.dump(unique, fh, indent=2)

    print(f"\nWrote {len(unique)} unique peptides to ground_truth.json")
    if skipped:
        print(f"Skipped {len(skipped)} (first few): {skipped[:5]}")
    print("\nIMPORTANT: open ground_truth.json and check the entries are really")
    print("small disulfide-rich peptides. Text search returns false positives.")
    for rec in unique[:5]:
        print(f"  {rec['pdb_id']}_{rec['chain']}  len={len(rec['sequence']):<3} "
              f"cys={rec['n_cys']}  pairs={rec['pairs']}")


if __name__ == "__main__":
    main()