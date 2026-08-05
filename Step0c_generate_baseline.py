#!/usr/bin/env python3
"""
step0c_generate_baseline.py  (FIXED v2)
=======================================
Step 0 Part B, generation half.

WHAT WAS BROKEN IN v1: it overwrote token IDs by hand and never passed
`partial_masks`, so DPLM had no idea which positions to preserve and generated
completely unconditionally (0/20 constraints respected).

v2 follows DPLM's own generate_dplm.py exactly:
    chars = ["<mask>" if free else residue, ...]   # literal "<mask>" strings
    input_tokens = tokenizer.batch_encode_plus(...)["input_ids"]
    partial_mask = input_tokens.ne(model.mask_id)  # True = keep this position
    model.generate(input_tokens=..., tokenizer=..., partial_masks=partial_mask, ...)

Produces the BASELINE your method must beat: defensin-like sequences with the
gamma-core and cysteines held fixed, but WITHOUT disulfide-topology conditioning.

Usage:
    python step0c_generate_baseline.py --model dplm_150m --n-per-template 5
Confirm it prints "constraints respected in 5/5" before scaling up.
"""

import argparse
import json
import random

import torch


# --------------------------------------------------------------------------
# Choosing which positions to keep
# --------------------------------------------------------------------------

def gamma_core_span(sequence):
    """Approximate the gamma-core: 3rd-from-last cysteine to the last one,
    which brackets the C-terminal beta-hairpin in canonical defensins.
    Heuristic -- for the paper, take motif bounds from an alignment instead."""
    cys = [i for i, a in enumerate(sequence) if a == "C"]
    if len(cys) < 3:
        return None
    return cys[-3], cys[-1]


def build_mask(sequence, mask_rate=0.6, keep_gamma_core=True, rng=None):
    """True = keep fixed, False = let DPLM redesign.
    Cysteines are always kept: their positions define the connectivity graph."""
    rng = rng or random
    keep = [a == "C" for a in sequence]

    if keep_gamma_core:
        span = gamma_core_span(sequence)
        if span:
            for i in range(span[0], span[1] + 1):
                keep[i] = True

    free = [i for i in range(len(sequence)) if not keep[i]]
    n_keep = int(round(len(free) * (1.0 - mask_rate)))
    for i in rng.sample(free, n_keep):
        keep[i] = True
    return keep


# --------------------------------------------------------------------------
# Generation -- mirrors DPLM's generate_dplm.py
# --------------------------------------------------------------------------

def load_model(model_name, device):
    from byprot.models.dplm import DiffusionProteinLanguageModel
    model = DiffusionProteinLanguageModel.from_pretrained(
        f"airkingbd/{model_name}")
    return model.eval().to(device)


@torch.no_grad()
def generate_from_template(model, sequence, keep, n_seqs, max_iter=500,
                           sampling_strategy="gumbel_argmax", device="cuda"):
    """Build a partially-masked string; DPLM fills only the masked positions."""
    tokenizer = model.tokenizer

    # kept positions -> the real residue; free positions -> literal "<mask>"
    chars = [sequence[i] if keep[i] else "<mask>" for i in range(len(sequence))]
    init_seq = ["".join(chars)] * n_seqs

    batch = tokenizer.batch_encode_plus(
        init_seq, add_special_tokens=True, padding="longest",
        return_tensors="pt",
    )
    input_tokens = batch["input_ids"].to(device)

    # True where the token is NOT a mask -> those positions stay frozen
    partial_mask = input_tokens.ne(model.mask_id)

    outputs = model.generate(
        input_tokens=input_tokens,
        tokenizer=tokenizer,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        partial_masks=partial_mask,
    )
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    return ["".join(s.split(" ")) for s in
            tokenizer.batch_decode(outputs, skip_special_tokens=True)]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="ground_truth.json")
    ap.add_argument("--model", default="dplm_150m",
                    choices=["dplm_150m", "dplm_650m", "dplm_3b"])
    ap.add_argument("--n-per-template", type=int, default=20)
    ap.add_argument("--n-templates", type=int, default=8)
    ap.add_argument("--mask-rate", type=float, default=0.6)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--sampling-strategy", default="gumbel_argmax")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="generated.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    gt = json.load(open(args.ground_truth))

    # pick templates spanning DIFFERENT topologies
    by_topo = {}
    for rec in gt:
        by_topo.setdefault(tuple(map(tuple, rec["pairs"])), []).append(rec)
    templates = []
    for _, recs in sorted(by_topo.items(), key=lambda kv: -len(kv[1])):
        templates.append(rng.choice(recs))
        if len(templates) >= args.n_templates:
            break

    print(f"Model: {args.model} on {args.device}")
    print(f"Templates: {len(templates)}   mask rate: {args.mask_rate}")
    print("(cysteines and gamma-core held fixed; no topology conditioning)\n")

    model = load_model(args.model, args.device)

    records = []
    for t in templates:
        seq = t["sequence"]
        keep = build_mask(seq, args.mask_rate, True, rng)
        entry = f"{t['pdb_id']}_{t['chain']}"
        print(f"{entry:<9} len={len(seq):<3} "
              f"redesigning {sum(1 for k in keep if not k)}/{len(seq)}  "
              f"intended={t['pairs']}")

        try:
            seqs = generate_from_template(
                model, seq, keep, args.n_per_template,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                device=args.device)
        except Exception as e:
            print(f"   GENERATION FAILED: {type(e).__name__}: {e}")
            continue

        ok_n = 0
        for s in seqs:
            ok = (len(s) == len(seq)
                  and all(s[i] == seq[i] for i, k in enumerate(keep) if k))
            ok_n += int(ok)
            records.append({
                "template": entry, "template_sequence": seq,
                "intended_pairs": t["pairs"], "sequence": s,
                "mask_rate": args.mask_rate,
                "constraints_respected": ok, "n_cys": s.count("C"),
            })
        print(f"   generated {len(seqs)}, constraints respected in "
              f"{ok_n}/{len(seqs)}")
        if seqs:
            print(f"   template : {seq}")
            print(f"   example  : {seqs[0]}")

    json.dump(records, open(args.out, "w"), indent=2)
    print(f"\nWrote {len(records)} sequences to {args.out}")

    bad = sum(1 for r in records if not r["constraints_respected"])
    if bad:
        print(f"WARNING: {bad}/{len(records)} broke the fixed positions.")
        print("Do NOT run step0d until this is 0 -- results would be meaningless.")
    else:
        print("All constraints respected. Next: python step0d_analyze_partb.py")


if __name__ == "__main__":
    main()