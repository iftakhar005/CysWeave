#!/usr/bin/env python3
"""
step4_colabfold_oracle.py
=========================
Bridges ColabFold (AlphaFold2 + real MSAs) into the existing analysis pipeline,
so a second, independent oracle can be tested against ESMFold.

Why
---
Step 3 showed ESMFold behaves ~76% like a cysteine-framework lookup, and that
its deviations are a systematic alpha->beta bias rather than sequence
sensitivity. Every "conditioning failed" conclusion assumed ESMFold was a valid
judge of disulfide connectivity. If it is not, the negative result is a
measurement artefact and has to be re-measured.

ColabFold uses MSAs (ESMFold does not). KnotFold (MLSB 2023) found AlphaFold's
connectivity accuracy tracks MSA depth, so on NATURAL peptides — where MSAs are
deep — it should do better if MSAs are what matters.

Two phases
----------
PHASE 1 (do this first): validate the oracle on the 81 natural PDB defensins,
where the connectivity is experimentally determined. Two questions:
    * does it recover known connectivity better than ESMFold's 91.4%?
    * does it agree with the spacing-only lookup LESS than ESMFold's 76%?
Only an oracle that passes both is worth re-measuring the generations with.

PHASE 2 (only if phase 1 passes): fold the conditioned and null arms with it and
re-run step3_is_esmfold_just_spacing.py against the new file.

Usage
-----
  # 1. build the input
  python step4_colabfold_oracle.py --prepare
  #    -> colabfold_input.fasta  +  colabfold_jobmap.json

  # 2. in Colab (GPU runtime), install and run:
  #      !pip install -q "colabfold[alphafold]"
  #      !colabfold_batch colabfold_input.fasta cf_out --msa-mode mmseqs2_uniref_env
  #    then download the cf_out folder

  # 3. collect the results back into JSONL
  python step4_colabfold_oracle.py --collect cf_out --out partb_colabfold.jsonl

  # 4. compare the two oracles
  python step3_is_esmfold_just_spacing.py --folds partb_colabfold.jsonl

To prepare generated peptides instead of naturals:
  python step4_colabfold_oracle.py --prepare --source generated_conditioned.json
"""

import argparse
import glob
import json
import math
import os
from itertools import combinations


# --------------------------------------------------------------------------
# connectivity from a predicted structure (same tested logic as elsewhere)
# --------------------------------------------------------------------------

def parse_cys_sg(pdb_text):
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


def fmt(bonds):
    return sorted(tuple(sorted(b)) for b in bonds)


def get_sequence(rec):
    return rec.get("mature_sequence") or rec.get("sequence")


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def prepare(args):
    recs = json.load(open(args.source))
    jobmap, lines, n = {}, [], 0
    for r in recs:
        seq = get_sequence(r)
        pairs = r.get("pairs") or r.get("intended_pairs")
        if not seq or not pairs:
            continue
        n += 1
        job = f"pep{n:04d}"                     # safe, ColabFold-friendly name
        label = (r.get("accession")
                 or (f"{r.get('pdb_id')}_{r.get('chain')}"
                     if r.get("pdb_id") else r.get("template", job)))
        jobmap[job] = {"label": label, "sequence": seq,
                       "intended_pairs": [list(p) for p in pairs]}
        lines.append(f">{job}\n{seq}")

    open(args.fasta, "w").write("\n".join(lines) + "\n")
    json.dump(jobmap, open(args.jobmap, "w"), indent=2)
    print(f"wrote {n} sequences to {args.fasta}")
    print(f"wrote job map to {args.jobmap}")
    print("\nIn Colab (GPU runtime):")
    print('  !pip install -q "colabfold[alphafold]"')
    print(f"  !colabfold_batch {args.fasta} cf_out "
          f"--msa-mode mmseqs2_uniref_env")
    print("\nThen download cf_out and run:  --collect cf_out")
    print(f"\nEstimate: ~1-3 min per peptide with the MSA server, so ~{n*2//60}-"
          f"{n*3//60} h for {n}. Split the FASTA if the session times out.")


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def collect(args):
    jobmap = json.load(open(args.jobmap))
    rows, missing = [], []

    for job, meta in jobmap.items():
        # ColabFold names outputs like  <job>_unrelaxed_rank_001_..._seed_000.pdb
        hits = sorted(glob.glob(os.path.join(args.collect, f"{job}*rank_001*.pdb")))
        if not hits:
            hits = sorted(glob.glob(os.path.join(args.collect, f"{job}*.pdb")))
        if not hits:
            missing.append(job)
            continue

        pdb = open(hits[0]).read()
        pred, dists, ncys = realized_pairing(pdb, args.cutoff)
        intended = {frozenset(tuple(p)) for p in meta["intended_pairs"]}
        rows.append({
            "template": meta["label"],
            "sequence": meta["sequence"],
            "intended": [list(p) for p in sorted(
                tuple(sorted(x)) for x in intended)],
            "realized": fmt(pred),
            "match": pred == intended,
            "n_cys": ncys,
            "sg_distances": {str(sorted(k)): v for k, v in dists.items()},
            "oracle": "colabfold",
        })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    if not rows:
        print("No structures found. Check the --collect directory and that "
              "ColabFold finished.")
        return

    m = sum(r["match"] for r in rows)
    print(f"collected {len(rows)} structures"
          + (f" ({len(missing)} missing: {missing[:5]})" if missing else ""))
    print()
    print("=" * 70)
    print(f"ColabFold exact connectivity recovery: {m}/{len(rows)} "
          f"= {100*m/len(rows):.1f}%")
    print("=" * 70)
    print("  ESMFold, same task, 81 natural defensins: 91.4%")
    print()
    if 100 * m / len(rows) > 91.4:
        print("  BETTER than ESMFold on recovery. Now the important question:")
    else:
        print("  NOT better than ESMFold on recovery. Still check the next bit —")
        print("  recovery and framework-independence are different things:")
    print(f"    python step3_is_esmfold_just_spacing.py --folds {args.out}")
    print("  If ColabFold agrees with the spacing lookup MUCH LESS than")
    print("  ESMFold's 76%, it is reading sequence context and is the better")
    print("  oracle — in which case the generation arms must be re-measured.")
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--collect", default=None,
                    help="ColabFold output directory")
    ap.add_argument("--source", default="ground_truth.json",
                    help="naturals (ground_truth.json) or a generated_*.json")
    ap.add_argument("--fasta", default="colabfold_input.fasta")
    ap.add_argument("--jobmap", default="colabfold_jobmap.json")
    ap.add_argument("--out", default="partb_colabfold.jsonl")
    ap.add_argument("--cutoff", type=float, default=3.0)
    args = ap.parse_args()

    if args.prepare:
        prepare(args)
    elif args.collect:
        collect(args)
    else:
        print("Use --prepare first, then --collect <dir>. See the docstring.")


if __name__ == "__main__":
    main()