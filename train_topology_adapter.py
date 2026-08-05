#!/usr/bin/env python3
"""
train_topology_adapter.py
=========================
THE METHOD. Teaches DPLM to generate a peptide that folds with a SPECIFIED
disulfide connectivity.

Conditioning design (per-cysteine partner embedding):
    Every position gets an integer. 0 everywhere except at cysteines, where the
    k-th cysteine carries THE ORDINAL OF ITS PARTNER. For alpha-defensin
    [[1,6],[2,4],[3,5]] over 6 cysteines that is [6,4,5,2,3,1].
    A learned embedding of that integer is ADDED to the token embedding at each
    position, so the model sees "this cysteine must bond to that one".

    Why not just 4 class tokens? Because this encoding can express ANY pairing,
    including ones never seen in training -- which is what makes the
    rare_topologies.json holdout a real generalisation test, and what backs the
    "full multi-bond topology conditioning" claim rather than 4-way classification.
    (Run --mode class-token to train that ablation for comparison.)

Trainable parameters: the conditioning embedding table (tiny) + LoRA on the
attention projections. The base model stays frozen. Fits a 4 GB GPU at 150M.

--------------------------------------------------------------------------
RUN THIS FIRST -- it does not train, it just prints your model's structure so
the LoRA target modules can be verified instead of guessed:

    python train_topology_adapter.py --inspect

Send that output before training. Then:

    python train_topology_adapter.py --train --epochs 10
--------------------------------------------------------------------------
"""

import argparse
import json
import random
from collections import Counter

import torch
import torch.nn as nn


# ==========================================================================
# 1. Conditioning encoding (framework-free; unit-tested)
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


def condition_vector_to_pairs(vec, sequence):
    """Inverse -- recover the pairing graph (used to sanity-check batches)."""
    ords = [v for aa, v in zip(sequence, vec) if aa == "C"]
    pairs = set()
    for k, p in enumerate(ords, start=1):
        if p:
            pairs.add(tuple(sorted((k, p))))
    return sorted(pairs)


def balanced_indices(records, seed=0, target=None):
    """Oversample rare topologies so every topology is seen equally per epoch.
    Without this the model just learns to emit the majority (CSab) pattern --
    which is the very bias this project exists to fix."""
    rng = random.Random(seed)
    by_topo = {}
    for i, r in enumerate(records):
        by_topo.setdefault(tuple(map(tuple, r["pairs"])), []).append(i)
    if target is None:
        target = max(len(v) for v in by_topo.values())
    out = []
    for _, idxs in by_topo.items():
        reps = idxs * (target // len(idxs))
        reps += rng.sample(idxs, target - len(reps))
        out.extend(reps)
    rng.shuffle(out)
    return out


# ==========================================================================
# 2. The conditioning module
# ==========================================================================

class TopologyConditioner(nn.Module):
    """Learned embedding of the partner-ordinal, added to token embeddings.

    Installed as a forward hook on the model's embedding layer so DPLM itself
    needs no code changes. `self.current` is set just before each forward pass.
    """

    def __init__(self, hidden_dim, max_cys=16):
        super().__init__()
        # padding_idx=0 pins the "no constraint" row at zero PERMANENTLY, so
        # non-cysteine positions receive exactly nothing. Without it, index 0
        # becomes a learned constant added to every residue, which wastes
        # capacity and blurs the signal.
        self.emb = nn.Embedding(max_cys + 1, hidden_dim, padding_idx=0)
        nn.init.zeros_(self.emb.weight)      # start as a no-op: identical to base
        self.current = None                  # (B, L) LongTensor, set per batch
        self.offset = 0                      # BOS offset, set after inspection

    def hook(self, module, inputs, output):
        if self.current is None:
            return output
        cond = self.current
        B_out = output.shape[0]
        # DPLM's generate() changes the batch size during iterative decoding
        # (it drops finished rows, and makes some batch-1 calls), so the stored
        # conditioning will not always match. During generation every row shares
        # the SAME conditioning, so expanding row 0 is correct; during training
        # the sizes already agree and this is a no-op.
        if cond.shape[0] != B_out:
            cond = cond[:1].expand(B_out, -1)
        L = cond.shape[1]
        add = self.emb(cond)                             # (B, L, H)
        out = output.clone()
        s = self.offset
        n = min(L, out.shape[1] - s)
        if n <= 0:
            return output
        out[:, s:s + n, :] = out[:, s:s + n, :] + add[:, :n, :]
        return out


# ==========================================================================
# 3. Model loading / inspection
# ==========================================================================

def load_dplm(model_name, device):
    from byprot.models.dplm import DiffusionProteinLanguageModel
    model = DiffusionProteinLanguageModel.from_pretrained(f"airkingbd/{model_name}")
    return model.to(device).eval()


def inspect(model):
    """Print what we need to verify before training: embedding layer, hidden
    size, tokenizer offset, and candidate LoRA target module names."""
    print("=" * 74)
    print("MODEL INSPECTION -- send this output before training")
    print("=" * 74)

    print("\n--- top-level attributes ---")
    for a in ("net", "tokenizer", "mask_id", "pad_id", "bos_id", "eos_id"):
        if hasattr(model, a):
            v = getattr(model, a)
            print(f"  model.{a}: {type(v).__name__ if not isinstance(v,int) else v}")

    print("\n--- candidate EMBEDDING layers ---")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            print(f"  {name:<60} {tuple(mod.weight.shape)}")

    print("\n--- candidate LoRA targets (Linear layers, name counts) ---")
    leaf = Counter()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            leaf[name.split(".")[-1]] += 1
    for n, c in leaf.most_common(15):
        print(f"  {n:<24} x{c}")

    print("\n--- tokenizer offset check ---")
    try:
        tok = model.tokenizer
        ids = tok.batch_encode_plus(["ACDEF"], add_special_tokens=True,
                                    return_tensors="pt")["input_ids"]
        print(f"  'ACDEF' (len 5) -> {ids.shape[1]} tokens: {ids.tolist()[0]}")
        print(f"  => offset (BOS tokens before residue 1) = "
              f"{(ids.shape[1] - 5) // 2 if ids.shape[1] > 5 else 0}")
    except Exception as e:
        print(f"  tokenizer probe failed: {e}")

    print("\n" + "=" * 74)
    print("Send the above. Then training targets can be set, not guessed.")
    print("=" * 74)


# ==========================================================================
# 4. Training
# ==========================================================================

def make_batch(records, idxs, tokenizer, device, mask_rate, rng, offset):
    """Masked-LM style batch: mask a fraction of NON-cysteine positions and ask
    the model to recover them, with the topology signal supplied."""
    seqs = [records[i]["mature_sequence"] for i in idxs]
    L = max(len(s) for s in seqs)

    conds, masked, targets = [], [], []
    for s, i in zip(seqs, idxs):
        cv = build_condition_vector(s, records[i]["pairs"])
        keep = [a == "C" or rng.random() > mask_rate for a in s]
        m = "".join(s[j] if keep[j] else "<mask>" for j in range(len(s)))
        masked.append(m)
        targets.append(s)
        conds.append(cv + [0] * (L - len(cv)))

    enc = tokenizer.batch_encode_plus(masked, add_special_tokens=True,
                                      padding="longest", return_tensors="pt")
    tgt = tokenizer.batch_encode_plus(targets, add_special_tokens=True,
                                      padding="longest", return_tensors="pt")
    return (enc["input_ids"].to(device), tgt["input_ids"].to(device),
            torch.tensor(conds, dtype=torch.long, device=device))


def train(args):
    device = args.device
    print(f"loading {args.model} on {device} ...")
    model = load_dplm(args.model, device)

    if args.inspect:
        inspect(model)
        return

    train_recs = json.load(open(args.train_file))
    print(f"train records: {len(train_recs)}")
    print("topology counts:",
          dict(Counter(tuple(map(tuple, r['pairs'])) for r in train_recs)))

    # --- find the module to hook ---
    # IMPORTANT: hook the whole embeddings MODULE (net.esm.embeddings), not
    # word_embeddings. In ESM the word-embedding output passes through LayerNorm
    # before the encoder, which would normalise our conditioning signal away.
    # Hooking the parent module means the addition happens AFTER LayerNorm and
    # after position embeddings, so the signal survives.
    emb_mod, emb_name, hidden = None, None, None
    for name, mod in model.named_modules():
        if name.endswith("esm.embeddings") and not isinstance(mod, nn.Embedding):
            emb_mod, emb_name = mod, name
            break
    if emb_mod is not None:
        for n2, m2 in emb_mod.named_modules():
            if isinstance(m2, nn.Embedding) and m2.weight.shape[0] < 100:
                hidden = m2.weight.shape[1]
    if emb_mod is None:                      # fallback: word embedding itself
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding) and mod.weight.shape[0] < 100:
                emb_mod, emb_name, hidden = mod, name, mod.weight.shape[1]
        print("WARNING: falling back to hooking word_embeddings; the signal may "
              "be attenuated by LayerNorm.")
    if emb_mod is None or hidden is None:
        raise RuntimeError("Could not locate the embedding module. "
                           "Run --inspect and report the output.")
    print(f"hooking '{emb_name}' (hidden={hidden})")

    cond = TopologyConditioner(hidden, args.max_cys).to(device)
    cond.offset = args.offset
    emb_mod.register_forward_hook(cond.hook)

    # --- freeze base; train conditioner (+ optional LoRA) ---
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(cond.parameters())

    base = model            # keep a handle: PEFT wrapping hides .net/.tokenizer
    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model
            targets = args.lora_targets.split(",")
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=2 * args.lora_r,
                target_modules=targets, lora_dropout=0.05, bias="none"))
            params += [p for p in model.parameters() if p.requires_grad]
            print(f"LoRA on {targets}, r={args.lora_r}")
        except Exception as e:
            print(f"LoRA setup failed ({e}); training the conditioner only.")

    n_train = sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable parameters: {n_train:,}")

    opt = torch.optim.AdamW(params, lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(args.seed)
    tokenizer = base.tokenizer

    val_recs = []
    if args.val_file:
        try:
            val_recs = json.load(open(args.val_file))
            print(f"validation records: {len(val_recs)}")
        except Exception as e:
            print(f"(no validation: {e})")

    def val_loss():
        if not val_recs:
            return None
        model.eval()
        tot, nb = 0.0, 0
        vr = random.Random(1234)
        with torch.no_grad():
            for b in range(0, min(len(val_recs), 200), args.batch_size):
                idxs = list(range(b, min(b + args.batch_size, len(val_recs))))
                if len(idxs) < 2:
                    continue
                inp, tgt, cv = make_batch(val_recs, idxs, tokenizer, device,
                                          args.mask_rate, vr, args.offset)
                cond.current = cv
                res = base.net(inp)
                out = res["logits"] if isinstance(res, dict) else getattr(res, "logits", res)
                lab = tgt.clone(); lab[inp != base.mask_id] = -100
                tot += lossf(out.reshape(-1, out.shape[-1]), lab.reshape(-1)).item()
                nb += 1
                cond.current = None
        return tot / max(nb, 1)

    best_val, best_state = [float("inf"), -1], {}
    for ep in range(args.epochs):
        order = balanced_indices(train_recs, seed=args.seed + ep)
        model.train()
        tot, nb = 0.0, 0
        for b in range(0, len(order), args.batch_size):
            idxs = order[b:b + args.batch_size]
            if len(idxs) < 2:
                continue
            inp, tgt, cv = make_batch(train_recs, idxs, tokenizer, device,
                                      args.mask_rate, rng, args.offset)
            if ep == 0 and nb == 0:      # one-time conditioning sanity check
                s0 = train_recs[idxs[0]]["mature_sequence"]
                rt = condition_vector_to_pairs(cv[0].tolist()[:len(s0)], s0)
                want = sorted(tuple(sorted(p)) for p in train_recs[idxs[0]]["pairs"])
                print(f"  [sanity] conditioning round-trip {rt} vs target {want}"
                      f" -> {'OK' if rt == want else 'MISMATCH!'}")
            cond.current = cv
            res = base.net(inp)
            out = res["logits"] if isinstance(res, dict) else getattr(res, "logits", res)
            labels = tgt.clone()
            labels[inp != base.mask_id] = -100       # score only masked spots
            loss = lossf(out.reshape(-1, out.shape[-1]), labels.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            cond.current = None
            tot += loss.item(); nb += 1
        vl = val_loss()
        msg = f"epoch {ep+1}/{args.epochs}  train {tot/max(nb,1):.4f}"
        if vl is not None:
            msg += f"  val {vl:.4f}"
            if vl < best_val[0]:
                best_val[0], best_val[1] = vl, ep + 1
                best_state["conditioner"] = {k: v.detach().cpu().clone()
                                             for k, v in cond.state_dict().items()}
                best_state["lora"] = {n: q.detach().cpu().clone()
                                      for n, q in model.named_parameters()
                                      if q.requires_grad}
                msg += "  <- best"
        print(msg)

    # save EVERY trainable tensor: the conditioner AND the LoRA weights.
    # (Saving only the conditioner loses ~99% of what was learned.)
    # prefer the BEST-validation epoch, not the last: once val starts rising the
    # final epoch is a worse model than the best one.
    if best_state:
        print(f"using best-validation checkpoint from epoch {best_val[1]} "
              f"(val {best_val[0]:.4f})")
        cond.load_state_dict(best_state["conditioner"])
        lora_state = best_state["lora"]
    else:
        lora_state = {n: q.detach().cpu()
                      for n, q in model.named_parameters() if q.requires_grad}
    torch.save({"conditioner": cond.state_dict(),
                "lora": lora_state,
                "lora_used": bool(args.lora and lora_state),
                "lora_r": args.lora_r,
                "lora_targets": args.lora_targets,
                "max_cys": args.max_cys, "offset": args.offset,
                "hidden": hidden}, args.out)
    print(f"\nsaved adapter to {args.out} "
          f"(conditioner + {len(lora_state)} LoRA tensors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="print model structure and exit (RUN THIS FIRST)")
    ap.add_argument("--train", action="store_true", dest="do_train")
    ap.add_argument("--model", default="dplm_150m")
    ap.add_argument("--train-file", dest="train_file", default="train.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-rate", type=float, default=0.5)
    ap.add_argument("--max-cys", type=int, default=16)
    ap.add_argument("--offset", type=int, default=1,
                    help="tokens before residue 1 (verify with --inspect)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-targets", default="query,key,value",
                    help="ESM naming: query,key,value(,dense). "
                         "Adding dense also adapts the FFN.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-file", dest="val_file", default="test.json",
                    help="held-out set for validation loss "
                         "(watch for train falling while val rises)")
    ap.add_argument("--out", default="topology_adapter.pt")
    args = ap.parse_args()

    if not (args.inspect or args.do_train):
        print("Run --inspect first, then --train. See the docstring.")
        return
    train(args)


if __name__ == "__main__":
    main()