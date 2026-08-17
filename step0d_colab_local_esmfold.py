#!/usr/bin/env python3
"""
step0d_colab_local_esmfold.py
=============================
Part B analysis WITHOUT the ESMFold API.

The public api.esmatlas.com endpoint is unreliable (long-standing SSL/cert
problems and repeated outages). This script runs ESMFold locally instead --
designed for a free Colab GPU (T4, ~15 GB), which is plenty for 30-50 residue
peptides.

It is RESUMABLE: results are appended to disk after every fold, so a dropped
Colab session costs you minutes, not the whole run.

--------------------------------------------------------------------------
HOW TO RUN IN COLAB
--------------------------------------------------------------------------
1. New notebook -> Runtime -> Change runtime type -> T4 GPU
2. Upload generated.json (or mount Drive and point --input at it)
3. In a cell:
       !pip install -q transformers accelerate
       !python step0d_colab_local_esmfold.py --input generated.json
   (or paste this file's contents into a cell and call main())

Recommended: mount Drive first so results survive a disconnect:
       from google.colab import drive; drive.mount('/content/drive')
   then use --out /content/drive/MyDrive/partb_results.jsonl
--------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
from collections import Counter
from itertools import combinations

import torch


# ==========================================================================
# 1. Connectivity geometry (same tested logic as Part A -- inlined so this
#    file has no local imports and drops straight into Colab)
# ==========================================================================

def parse_cys_sg(pdb_text):
    """[(cys_index, resseq, (x,y,z))] for CYS SG atoms, ordered along chain."""
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        line = line.ljust(80)
        if line[12:16].strip() == "SG" and line[17:20].strip() == "CYS":
            atoms.append((int(line[22:26]),
                          (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))))
    atoms.sort(key=lambda a: a[0])
    return [(i + 1, rs, xyz) for i, (rs, xyz) in enumerate(atoms)]


def _dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def realized_pairing(pdb_text, max_bond_dist=3.0):
    """Greedy nearest-neighbour matching on SG-SG distance."""
    cys = parse_cys_sg(pdb_text)
    cand = sorted((_dist(xi, xj), i, j)
                  for (i, _, xi), (j, _, xj) in combinations(cys, 2))
    used, bonds, dists = set(), set(), {}
    for d, i, j in cand:
        if i in used or j in used or d > max_bond_dist:
            continue
        key = frozenset((i, j))
        bonds.add(key)
        dists[key] = round(d, 2)
        used.update((i, j))
    return bonds, dists, len(cys)


def to_set(pairs):
    return {frozenset(p) for p in pairs}


def fmt(bonds):
    return sorted(tuple(sorted(b)) for b in bonds)


# ==========================================================================
# 2. Local ESMFold
# ==========================================================================

def load_esmfold(device="cuda", chunk_size=64):
    """Load facebook/esmfold_v1. ~5 GB in fp16; fine on a free Colab T4."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    print("Loading facebook/esmfold_v1 (first run downloads ~8.4 GB) ...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    # Load in fp32 (default). Only the language-model trunk is safe in fp16;
    # the folding trunk must stay fp32 or it emits NaNs and produces no output.
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True, use_safetensors=True)
    model = model.to(device).eval()

    if device == "cuda":
        model.esm = model.esm.half()               # LM trunk fp16 (supported)
        torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(chunk_size)     # lower = less memory
    print("ESMFold ready.\n")
    return tokenizer, model


@torch.no_grad()
def fold_local(tokenizer, model, sequence, device="cuda"):
    """Sequence -> PDB text, following the HuggingFace ESMFold usage."""
    input_ids = tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        raise ValueError(f"empty tokenization for: {sequence!r}")

    out = model(input_ids)

    # a NaN in positions means the folding trunk went unstable (usually fp16);
    # output_to_pdb would then return nothing -> report it clearly instead
    if torch.isnan(out["positions"]).any():
        raise ValueError("NaN in predicted coordinates (folding trunk unstable)")

    pdbs = model.output_to_pdb(out)
    if not pdbs:
        raise ValueError("output_to_pdb returned nothing")
    return pdbs[0]


# ==========================================================================
# 3. Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="generated.json")
    ap.add_argument("--out", default="partb_results.jsonl",
                    help="JSONL, appended after every fold (resumable)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cutoff", type=float, default=3.0)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arm", default="baseline",
                    choices=["baseline", "conditioned", "scrambled"],
                    help="which experimental arm this input is, so the label "
                         "and verdict are correct")
    ap.add_argument("--force-cpu", action="store_true",
                    help="run on CPU anyway (slow: hours, not minutes)")
    args = ap.parse_args()

    # ---- fail fast if no GPU: ESMFold on CPU takes hours ----
    if args.device == "cpu" and not args.force_cpu:
        print("=" * 72)
        print("NO GPU DETECTED - refusing to run.")
        print("ESMFold on CPU takes hours for 140 sequences.")
        print()
        print("In Colab: Runtime -> Change runtime type -> T4 GPU -> Save.")
        print("That RESTARTS the session, so re-upload your files afterwards.")
        print("Verify with:  import torch; print(torch.cuda.get_device_name(0))")
        print()
        print("To run on CPU anyway (slow), pass --force-cpu")
        print("=" * 72)
        return

    data = json.load(open(args.input))
    data = [d for d in data if d.get("constraints_respected", True)]
    if args.limit:
        data = data[:args.limit]

    # ---- resume: skip sequences already folded ----
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["sequence"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} already folded, skipping those.\n")

    todo = [d for d in data if d["sequence"] not in done]
    print(f"Folding {len(todo)} sequences on {args.device} "
          f"(cutoff {args.cutoff} A)\n")

    tokenizer, model = load_esmfold(args.device, args.chunk_size)

    with open(args.out, "a") as fh:
        for k, rec in enumerate(todo):
            intended = to_set(tuple(p) for p in rec["intended_pairs"])
            try:
                pdb = fold_local(tokenizer, model, rec["sequence"], args.device)
            except Exception as e:
                print(f"[{k:>3}] FOLD FAILED: {type(e).__name__}: {e}")
                continue

            pred, dists, ncys = realized_pairing(pdb, args.cutoff)
            hit = (pred == intended)
            print(f"[{k:>3}] {rec['template']:<9} cys={ncys} "
                  f"intended={fmt(intended)} got={fmt(pred)} "
                  f"{'MATCH' if hit else 'miss'}")

            fh.write(json.dumps({
                "template": rec["template"], "sequence": rec["sequence"],
                "intended": fmt(intended), "realized": fmt(pred),
                "match": hit, "n_cys": ncys,
                "sg_distances": {str(sorted(kk)): v for kk, v in dists.items()},
            }) + "\n")
            fh.flush()                     # survive a session drop

    # ---- summary over everything on disk ----
    results = [json.loads(l) for l in open(args.out)]
    if not results:
        print("\nNo results.")
        return

    n = len(results)
    m = sum(r["match"] for r in results)
    pct = 100 * m / n

    label = {"baseline": "unconditioned (baseline) generations",
             "conditioned": "CONDITIONED generations",
             "scrambled": "SCRAMBLED-topology generations (specificity control)"}[args.arm]
    print("\n" + "=" * 72)
    print(f"RESULT [{args.arm}]: requested topology reached in "
          f"{m}/{n} = {pct:.1f}% of {label}")
    print("=" * 72)

    per_t = {}
    for r in results:
        a, b = per_t.setdefault(r["template"], [0, 0])
        per_t[r["template"]] = [a + r["match"], b + 1]
    print("\nPer template:")
    for t, (h, tot) in sorted(per_t.items()):
        print(f"   {t:<9} {h:>3}/{tot:<3} = {100*h/tot:5.1f}%")

    print("\nTopologies that actually formed (top 8):")
    for topo, c in Counter(tuple(map(tuple, r["realized"]))
                           for r in results).most_common(8):
        print(f"   {list(topo)}  ->  {c}  ({100*c/n:.0f}%)")

    print("\n" + "-" * 72)
    if args.arm == "baseline":
        if pct < 60:
            print("VERDICT: clear gap -- conditioning has real work to do. PROCEED.")
        elif pct > 85:
            print("VERDICT: little headroom. RECONSIDER before committing.")
        else:
            print("VERDICT: partial gap. Target the weakest topologies.")
    elif args.arm == "conditioned":
        print("Compare this against the BASELINE arm. Higher = conditioning works.")
        print("This number alone proves nothing without the --scramble control.")
    else:  # scrambled
        print("SPECIFICITY CONTROL: this number should be MUCH LOWER than the")
        print("conditioned arm. If it is comparable, the model is IGNORING the")
        print("conditioning and the conditioned result is not meaningful.")
    print("-" * 72)


if __name__ == "__main__":
    main()