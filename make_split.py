#!/usr/bin/env python3
"""
make_split.py
=============
Builds the TRAIN / TEST split for the topology-conditioning adapter.

Two rules, both non-negotiable for a defensible paper:

1. CLUSTER-AWARE. Defensins are highly homologous. If human beta-defensin-1 is
   in train and chimp beta-defensin-1 is in test, the model looks brilliant
   while really just recognising a sequence it memorised. So sequences are
   clustered by identity and WHOLE CLUSTERS go to train or test -- never split.

2. EVIDENCE-AWARE. ~89% of UniProt connectivity here is inferred-by-similarity,
   i.e. assigned by homology to some other peptide. Those must never be the
   thing you evaluate on. Test clusters are drawn preferentially from entries
   with EXPERIMENTAL evidence (ECO:0000269).

Clustering is a pure-Python CD-HIT-style greedy pass (k-mer prefilter + global
alignment identity) so you do not need to install CD-HIT/MMseqs2 on Windows.
~1000 peptides cluster in a couple of seconds.

Usage:
    python make_split.py                       # 90% identity, ~20% test
    python make_split.py --identity 0.7        # stricter non-redundancy
    python make_split.py --test-frac 0.15

Inputs : defensin_connectivity_dataset.json   (from build_uniprot_dataset.py)
Outputs: train.json, test.json, split_report.txt
"""

import argparse
import json
import random
from collections import Counter, defaultdict


# --------------------------------------------------------------------------
# Clustering (benchmarked: ~1080 seqs -> ~1.3 s)
# --------------------------------------------------------------------------

def kmers(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def nw_identity(a, b, match=1, mismatch=-1, gap=-2):
    """Global alignment identity = identical positions / alignment length."""
    n, m = len(a), len(b)
    prev = [j * gap for j in range(m + 1)]
    prev_id = [0] * (m + 1)
    prev_len = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i * gap] + [0] * m
        cur_id = [0] * (m + 1)
        cur_len = [i] + [0] * m
        for j in range(1, m + 1):
            d = prev[j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            u = prev[j] + gap
            l = cur[j - 1] + gap
            best = max(d, u, l)
            cur[j] = best
            if best == d:
                cur_id[j] = prev_id[j - 1] + (1 if a[i - 1] == b[j - 1] else 0)
                cur_len[j] = prev_len[j - 1] + 1
            elif best == u:
                cur_id[j] = prev_id[j]
                cur_len[j] = prev_len[j] + 1
            else:
                cur_id[j] = cur_id[j - 1]
                cur_len[j] = cur_len[j - 1] + 1
        prev, prev_id, prev_len = cur, cur_id, cur_len
    return prev_id[m] / prev_len[m] if prev_len[m] else 0.0


def greedy_cluster(seqs, threshold=0.9, prefilter=0.30, k=3):
    """CD-HIT-style: longest sequences become representatives."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    km = [kmers(s, k) for s in seqs]
    reps, assign = [], [None] * len(seqs)
    for i in order:
        placed = False
        for cid, (ri, rk) in enumerate(reps):
            if jaccard(km[i], rk) < prefilter:
                continue
            if nw_identity(seqs[i], seqs[ri]) >= threshold:
                assign[i] = cid
                placed = True
                break
        if not placed:
            assign[i] = len(reps)
            reps.append((i, km[i]))
    return assign, len(reps)


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="defensin_connectivity_dataset.json")
    ap.add_argument("--identity", type=float, default=0.9,
                    help="cluster identity threshold (0.9 = CD-HIT -c 0.9)")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--min-topology", type=int, default=30,
                    help="only keep topologies with at least this many sequences")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = json.load(open(args.input))
    print(f"loaded {len(data)} sequences")

    # ---- keep only trainable topologies ----
    topo_of = {i: tuple(map(tuple, r["pairs"])) for i, r in enumerate(data)}
    counts = Counter(topo_of.values())
    keep_topos = {t for t, c in counts.items() if c >= args.min_topology}
    kept = [i for i in range(len(data)) if topo_of[i] in keep_topos]
    dropped = len(data) - len(kept)
    print(f"keeping {len(kept)} sequences across {len(keep_topos)} topologies "
          f"(dropped {dropped} in rare topologies with <{args.min_topology})")
    print("  rare topologies are reserved as a GENERALISATION test set\n")

    # ---- cluster ----
    seqs = [data[i]["mature_sequence"] for i in kept]
    print(f"clustering at {args.identity:.0%} identity ...")
    assign, n_clusters = greedy_cluster(seqs, args.identity)
    print(f"  {len(seqs)} sequences -> {n_clusters} independent clusters\n")

    cluster_members = defaultdict(list)
    for pos, i in enumerate(kept):
        cluster_members[assign[pos]].append(i)

    # ---- score clusters: prefer EXPERIMENTAL evidence for the test set ----
    def exp_count(cid):
        return sum(1 for i in cluster_members[cid]
                   if data[i]["evidence_class"] in ("experimental", "mixed"))

    # assign whole clusters, stratified by topology
    by_topo = defaultdict(list)
    for cid, members in cluster_members.items():
        t = topo_of[members[0]]          # clusters are homologous -> one topology
        by_topo[t].append(cid)

    test_clusters = set()
    for t, cids in by_topo.items():
        # experimental-rich clusters first, then random for the remainder
        cids = sorted(cids, key=lambda c: (-exp_count(c), rng.random()))
        n_seq_t = sum(len(cluster_members[c]) for c in cids)
        target = max(1, int(round(n_seq_t * args.test_frac)))
        taken = 0
        for c in cids:
            if taken >= target:
                break
            if len(cids) - len(test_clusters & set(cids)) <= 1:
                break                     # never empty a topology's train side
            test_clusters.add(c)
            taken += len(cluster_members[c])

    train_idx, test_idx = [], []
    for cid, members in cluster_members.items():
        (test_idx if cid in test_clusters else train_idx).extend(members)

    def dump(idxs, path):
        json.dump([data[i] for i in idxs], open(path, "w"), indent=2)

    rare_idx = [i for i in range(len(data)) if topo_of[i] not in keep_topos]
    dump(train_idx, "train.json")
    dump(test_idx, "test.json")
    dump(rare_idx, "rare_topologies.json")

    # ---- report ----
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 74)
    out(f"SPLIT  (identity {args.identity:.0%}, whole clusters only)")
    out("=" * 74)
    out(f"train: {len(train_idx)} sequences")
    out(f"test : {len(test_idx)} sequences")
    out(f"rare-topology holdout: {dropped} sequences\n")

    hdr = f"{'topology':<34}{'train':>7}{'test':>6}{'test_exp':>10}"
    out(hdr); out("-" * 74)
    for t in sorted(by_topo, key=lambda t: -counts[t]):
        tr = sum(1 for i in train_idx if topo_of[i] == t)
        te = sum(1 for i in test_idx if topo_of[i] == t)
        te_exp = sum(1 for i in test_idx if topo_of[i] == t
                     and data[i]["evidence_class"] in ("experimental", "mixed"))
        out(f"{str([list(p) for p in t]):<34}{tr:>7}{te:>6}{te_exp:>10}")
    out("-" * 74)

    ev_test = Counter(data[i]["evidence_class"] for i in test_idx)
    out(f"\ntest evidence: {dict(ev_test)}")
    out("NOTE: report headline accuracy on the EXPERIMENTAL test entries;")
    out("      inferred-by-similarity labels were assigned by homology and")
    out("      are training signal, not ground truth.\n")

    # imbalance warning -- this drives the sampling strategy during training
    tr_counts = Counter(topo_of[i] for i in train_idx)
    if tr_counts:
        mx, mn = max(tr_counts.values()), min(tr_counts.values())
        out(f"train imbalance: largest topology {mx}, smallest {mn} "
            f"({mx/mn:.1f}x)")
        if mx / mn > 3:
            out("  -> OVERSAMPLE the rare topologies during training, or the")
            out("     model will just learn to emit the majority pattern.")

    open("split_report.txt", "w").write("\n".join(lines))
    print("\nwrote train.json, test.json, rare_topologies.json, split_report.txt")


if __name__ == "__main__":
    main()