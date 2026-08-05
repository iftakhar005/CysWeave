#!/usr/bin/env python3
"""
generate_conditioned.py
=======================
THE EXPERIMENT. Generates peptides WITH the trained topology conditioning, on
the same templates and the same protocol as the unconditioned baseline, so the
two numbers are directly comparable.

    baseline (no conditioning)  -> 70.9% reach the intended topology
    this script (conditioned)   -> ?

Also runs the SPECIFICITY CONTROL, which matters as much as the headline:

    --scramble   request a deliberately WRONG topology for each template.

If conditioning is real, fidelity must DROP when you ask for the wrong wiring.
If asking for pattern A versus pattern B produces the same output, the model is
ignoring the signal and a high headline number would mean nothing.

Output is written in exactly the format step0d_colab_local_esmfold.py expects,
so you fold it with the same command you already used.

Usage:
    python generate_conditioned.py --adapter topology_adapter.pt
    python generate_conditioned.py --adapter topology_adapter.pt --scramble \\
           --out generated_scrambled.json

    # then fold, exactly as before:
    python step0d_colab_local_esmfold.py --input generated_conditioned.json \\
           --out partb_conditioned.jsonl
"""

import argparse
import json
import random

import torch
import torch.nn as nn


# ==========================================================================
# Conditioning (identical to training -- keep these in sync)
# ==========================================================================

def build_condition_vector(sequence, pairs):
    """0 at non-cysteines; at the k-th cysteine, the ordinal of its partner."""
    partner = {}
    for a, b in pairs:
        partner[a] = b
        partner[b] = a
    vec, k = [], 0
    for aa in sequence:
        if aa == "C":
            k += 1
            vec.append(partner.get(k, 0))
        else:
            vec.append(0)
    return vec


class TopologyConditioner(nn.Module):
    def __init__(self, hidden_dim, max_cys=16):
        super().__init__()
        self.emb = nn.Embedding(max_cys + 1, hidden_dim, padding_idx=0)
        self.current = None
        self.offset = 1

    def hook(self, module, inputs, output):
        if self.current is None:
            return output
        cond = self.current
        B_out = output.shape[0]
        # DPLM's generate() changes batch size during iterative decoding (it
        # drops finished rows and makes some batch-1 calls), so the stored
        # conditioning will not always match. In generation every row shares the
        # SAME conditioning, so expanding row 0 is correct.
        if cond.shape[0] != B_out:
            cond = cond[:1].expand(B_out, -1)
        L = cond.shape[1]
        add = self.emb(cond)
        out = output.clone()
        s = self.offset
        n = min(L, out.shape[1] - s)
        if n <= 0:
            return output
        out[:, s:s + n, :] = out[:, s:s + n, :] + add[:, :n, :]
        return out


# ==========================================================================
# Masking (identical to the baseline run -- keep the comparison fair)
# ==========================================================================

def gamma_core_span(sequence):
    cys = [i for i, a in enumerate(sequence) if a == "C"]
    return (cys[-3], cys[-1]) if len(cys) >= 3 else None


def build_mask(sequence, mask_rate=0.6, keep_gamma_core=True, rng=None):
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


def scramble_topology(pairs, rng):
    """A DIFFERENT valid perfect matching over the same cysteines."""
    n = 2 * len(pairs)
    original = {tuple(sorted(p)) for p in pairs}
    for _ in range(200):
        ords = list(range(1, n + 1))
        rng.shuffle(ords)
        cand = {tuple(sorted((ords[i], ords[i + 1]))) for i in range(0, n, 2)}
        if cand != original:
            return sorted(cand)
    return sorted(original)          # give up (only possible for n == 2)


# ==========================================================================
# Loading
# ==========================================================================

def load_model_and_adapter(model_name, adapter_path, device):
    from byprot.models.dplm import DiffusionProteinLanguageModel
    base = DiffusionProteinLanguageModel.from_pretrained(f"airkingbd/{model_name}")
    base = base.to(device).eval()

    ck = torch.load(adapter_path, map_location=device)
    print(f"adapter: hidden={ck['hidden']} max_cys={ck['max_cys']} "
          f"offset={ck['offset']} lora_used={ck.get('lora_used')}")

    # re-create the LoRA structure, then load the trained LoRA tensors
    if ck.get("lora_used") and ck.get("lora"):
        try:
            from peft import LoraConfig, get_peft_model
            targets = ck["lora_targets"].split(",")
            wrapped = get_peft_model(base, LoraConfig(
                r=ck["lora_r"], lora_alpha=2 * ck["lora_r"],
                target_modules=targets, lora_dropout=0.0, bias="none"))
            missing, unexpected = wrapped.load_state_dict(ck["lora"], strict=False)
            loaded = len(ck["lora"]) - len(unexpected)
            print(f"  loaded {loaded}/{len(ck['lora'])} LoRA tensors "
                  f"({len(unexpected)} unexpected)")
            if loaded == 0:
                print("  WARNING: no LoRA tensors matched -- the adapter is NOT "
                      "being applied. Check that --model matches training.")
        except ImportError:
            print("  WARNING: peft not installed; LoRA weights NOT applied.")
    else:
        print("  note: checkpoint has no LoRA weights (conditioner only)")

    # find the module to hook (same rule as training)
    emb_mod = None
    for name, mod in base.named_modules():
        if name.endswith("esm.embeddings") and not isinstance(mod, nn.Embedding):
            emb_mod = mod
            break
    if emb_mod is None:
        raise RuntimeError("could not locate net.esm.embeddings to hook")

    cond = TopologyConditioner(ck["hidden"], ck["max_cys"]).to(device)
    cond.load_state_dict(ck["conditioner"])
    cond.offset = ck["offset"]
    emb_mod.register_forward_hook(cond.hook)
    print(f"  conditioner loaded; |emb| mean abs = "
          f"{cond.emb.weight.abs().mean().item():.5f}")
    if cond.emb.weight.abs().mean().item() < 1e-8:
        print("  WARNING: conditioner weights are all zero -- it never trained.")
    return base, cond


# ==========================================================================
# Generation
# ==========================================================================

@torch.no_grad()
def generate(base, cond, sequence, keep, pairs, n_seqs, max_iter,
             sampling_strategy, device, forbid_new_cys=True, oversample=4,
             max_rounds=6):
    tokenizer = base.tokenizer
    free = [i for i, k in enumerate(keep) if not k]

    def valid(s):
        if len(s) != len(sequence):
            return False
        if any(s[i] != sequence[i] for i, k in enumerate(keep) if k):
            return False
        if forbid_new_cys and any(s[i] == "C" for i in free):
            return False
        return True

    cv = build_condition_vector(sequence, pairs)
    kept, rounds = [], 0
    while len(kept) < n_seqs and rounds < max_rounds:
        want = n_seqs - len(kept)
        bn = want * (oversample if forbid_new_cys else 1)

        chars = [sequence[i] if keep[i] else "<mask>" for i in range(len(sequence))]
        init = ["".join(chars)] * bn
        enc = tokenizer.batch_encode_plus(init, add_special_tokens=True,
                                          padding="longest", return_tensors="pt")
        inp = enc["input_ids"].to(device)
        partial = inp.ne(base.mask_id)

        # one row: the hook expands it to whatever batch size DPLM uses
        cond.current = torch.tensor([cv], dtype=torch.long, device=device)
        out = base.generate(input_tokens=inp, tokenizer=tokenizer,
                            max_iter=max_iter,
                            sampling_strategy=sampling_strategy,
                            partial_masks=partial)
        cond.current = None
        if isinstance(out, (list, tuple)):
            out = out[0]

        for s in ["".join(x.split(" ")) for x in
                  tokenizer.batch_decode(out, skip_special_tokens=True)]:
            if valid(s):
                kept.append(s)
                if len(kept) >= n_seqs:
                    break
        rounds += 1
    return kept[:n_seqs], rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="topology_adapter.pt")
    ap.add_argument("--ground-truth", default="ground_truth.json")
    ap.add_argument("--model", default="dplm_150m")
    ap.add_argument("--n-per-template", type=int, default=20)
    ap.add_argument("--n-templates", type=int, default=8)
    ap.add_argument("--mask-rate", type=float, default=0.6)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--sampling-strategy", default="gumbel_argmax")
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--scramble", action="store_true",
                    help="SPECIFICITY CONTROL: request a wrong topology")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or ("generated_scrambled.json" if args.scramble
                            else "generated_conditioned.json")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    gt = json.load(open(args.ground_truth))
    by_topo = {}
    for r in gt:
        by_topo.setdefault(tuple(map(tuple, r["pairs"])), []).append(r)
    templates = []
    for _, recs in sorted(by_topo.items(), key=lambda kv: -len(kv[1])):
        templates.append(rng.choice(recs))
        if len(templates) >= args.n_templates:
            break

    print(f"model {args.model} on {args.device}")
    print(f"mode: {'SCRAMBLED (specificity control)' if args.scramble else 'CONDITIONED'}")
    print(f"templates: {len(templates)}\n")

    base, cond = load_model_and_adapter(args.model, args.adapter, args.device)
    print()

    records = []
    for t in templates:
        seq = t["sequence"]
        entry = f"{t['pdb_id']}_{t['chain']}"
        true_pairs = [list(p) for p in t["pairs"]]
        req_pairs = ([list(p) for p in scramble_topology(t["pairs"], rng)]
                     if args.scramble else true_pairs)
        keep = build_mask(seq, args.mask_rate, True, rng)

        print(f"{entry:<9} len={len(seq):<3} redesigning "
              f"{sum(1 for k in keep if not k)}/{len(seq)}")
        print(f"   requesting {req_pairs}"
              + (f"   (true is {true_pairs})" if args.scramble else ""))

        try:
            seqs, rounds = generate(base, cond, seq, keep, req_pairs,
                                    args.n_per_template, args.max_iter,
                                    args.sampling_strategy, args.device,
                                    oversample=args.oversample)
        except Exception as e:
            print(f"   GENERATION FAILED: {type(e).__name__}: {e}")
            continue
        if len(seqs) < args.n_per_template:
            print(f"   NOTE: only {len(seqs)}/{args.n_per_template} valid after "
                  f"{rounds} rounds")

        for s in seqs:
            records.append({
                "template": entry, "template_sequence": seq,
                # what we ASKED for -- this is what fidelity is scored against
                "intended_pairs": req_pairs,
                "true_pairs": true_pairs,
                "scrambled": args.scramble,
                "sequence": s, "mask_rate": args.mask_rate,
                "constraints_respected": True, "n_cys": s.count("C"),
            })
        if seqs:
            print(f"   template: {seq}")
            print(f"   example : {seqs[0]}")

    json.dump(records, open(out_path, "w"), indent=2)
    print(f"\nwrote {len(records)} sequences to {out_path}")
    print(f"\nnext:  python step0d_colab_local_esmfold.py --input {out_path} "
          f"--out {out_path.replace('.json', '.jsonl')}")
    if not args.scramble:
        print("then rerun with --scramble for the specificity control.")


if __name__ == "__main__":
    main()