# Copyright 2026. Apache License 2.0.
#
# Does the forgetting-attention win survive the lenses the IBNN paper actually cared about -
# ROBUSTNESS to input perturbation and ANTI-MEMORIZATION - rather than just clean accuracy?
# (Neither the IBNN paper nor the Forgetting Transformer checked this for forgetting attention.)
#
# Two probes, softmax vs forgetting attention, matched everything (standard FFN):
#
#  (1) Input-noise robustness. Take the trained model and evaluate next-token BPC when a
#      fraction eps of the *context* tokens are randomly corrupted (targets stay clean). A more
#      robust model degrades less. We report BPC(eps) and the degradation BPC(0.2)-BPC(0).
#
#  (2) Memorization / generalization gap. gap = val_BPC - train_BPC after training. A model that
#      memorizes the training data more has a larger gap. The IBNN paper claims its neuron
#      memorizes *less*; here we ask whether the forget gate does.
#
# Hypothesis (open): the forget gate's recency bias makes the model lean on recent context and
# discount the distant past - which could cut either way for robustness, and could *reduce*
# memorization of long-range arbitrary patterns. We measure rather than guess.
#
#   python -m ibnn_lm.robustness --dataset tinyshakespeare --seeds 0 1 2 --steps 2000

import argparse
import json
import math
import os
import statistics as stats
import time

import torch
import torch.nn.functional as F

from . import data as data_mod
from .train import build_arg_parser, train_run, load_gpt_from_checkpoint
from .evaluate import full_loss
from .utils import get_device

LN2 = math.log(2.0)


@torch.no_grad()
def noisy_bpc(model, data, block_size, batch_size, device, eps, vocab_size, seed=0):
    """Exact next-token BPC when an eps fraction of CONTEXT tokens are randomly corrupted
    (targets stay clean). eps=0 reproduces the clean held-out BPC."""
    g = torch.Generator().manual_seed(seed)
    n = (len(data) - 1) // block_size
    tot_l, tot_t = 0.0, 0
    was_training = model.training
    model.eval()
    for b in range(0, n, batch_size):
        idxs = range(b, min(b + batch_size, n))
        x = torch.stack([data[i * block_size:i * block_size + block_size] for i in idxs]).clone()
        y = torch.stack([data[i * block_size + 1:i * block_size + 1 + block_size] for i in idxs])
        if eps > 0:
            mask = torch.rand(x.shape, generator=g) < eps
            n_corrupt = int(mask.sum())
            if n_corrupt:
                x[mask] = torch.randint(0, vocab_size, (n_corrupt,), generator=g, dtype=x.dtype)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        tot_l += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                                 reduction="sum").item()
        tot_t += y.numel()
    if was_training:
        model.train()
    return tot_l / tot_t / LN2


def main():
    ap = argparse.ArgumentParser(description="Robustness + memorization: softmax vs forgetting attn.")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    device = get_device(args.device)
    ds = data_mod.load(args.dataset, train_frac=1.0)
    vocab = ds.tokenizer.vocab_size
    train_slice = ds.train[: len(ds.val)]   # equal-size slice for a fair train-vs-val gap
    shared = dict(d_model=args.d_model, d_ff=args.d_ff, n_layer=args.n_layer, n_head=args.n_head,
                  block_size=args.block_size, batch_size=args.batch_size, dropout=0.1,
                  steps=args.steps, lr=args.lr, min_lr=args.lr / 10, warmup=100,
                  eval_interval=200, eval_iters=40, ffn="sm")

    print(f"robustness + memorization on {args.dataset}: softmax vs forget  seeds={args.seeds}")
    print(f"eps sweep = {args.eps}\n")
    t0 = time.time()
    res = {"softmax": {"curve": {e: [] for e in args.eps}, "gap": []},
           "forget":  {"curve": {e: [] for e in args.eps}, "gap": []}}

    for attn in ("softmax", "forget"):
        for seed in args.seeds:
            out = f"checkpoints/rob_{attn}_s{seed}.pt"
            a = build_arg_parser().parse_args([])
            for k, v in {**shared, "attn": attn, "seed": seed, "dataset": args.dataset,
                         "device": device, "out": out, "sample_interval": 0}.items():
                setattr(a, k, v)
            train_run(a, quiet=True)
            model, cfg, tok, _ = load_gpt_from_checkpoint(out, device)
            bs = args.block_size
            for e in args.eps:
                res[attn]["curve"][e].append(
                    noisy_bpc(model, ds.val, bs, 128, device, e, vocab, seed=1234))
            val_bpc = full_loss(model, ds.val, bs, 128, device)["bpc"]
            tr_bpc = full_loss(model, train_slice, bs, 128, device)["bpc"]
            res[attn]["gap"].append(val_bpc - tr_bpc)
            clean = res[attn]["curve"][args.eps[0]][-1]
            noisy = res[attn]["curve"][args.eps[-1]][-1]
            print(f"  {attn} seed {seed}: clean {clean:.4f}  noisy@{args.eps[-1]} {noisy:.4f}  "
                  f"gap {val_bpc - tr_bpc:+.4f}", flush=True)
            for p in (out, out.replace(".pt", "_last.pt")):
                if os.path.exists(p):
                    os.remove(p)

    def ms(xs):
        return (stats.mean(xs), stats.stdev(xs) if len(xs) > 1 else 0.0)

    print("\n" + "=" * 74)
    print("ROBUSTNESS TO CONTEXT NOISE  -  held-out BPC at corruption rate eps (lower better)")
    print("=" * 74)
    header = f"{'eps':>6} | " + " | ".join(f"{a:>16}" for a in ("softmax", "forget")) + " |  gap(forget-soft)"
    print(header); print("-" * len(header))
    for e in args.eps:
        sm_m, sm_s = ms(res["softmax"]["curve"][e])
        fo_m, fo_s = ms(res["forget"]["curve"][e])
        print(f"{e:>6.2f} | {sm_m:>7.4f} +/-{sm_s:5.3f} | {fo_m:>7.4f} +/-{fo_s:5.3f} | {fo_m-sm_m:>+8.4f}")
    print("-" * len(header))
    # degradation from clean to eps=0.2 (or the largest eps)
    big = 0.2 if 0.2 in res["softmax"]["curve"] else args.eps[-1]
    sm_deg = ms(res["softmax"]["curve"][big])[0] - ms(res["softmax"]["curve"][0.0])[0]
    fo_deg = ms(res["forget"]["curve"][big])[0] - ms(res["forget"]["curve"][0.0])[0]
    print(f"degradation (BPC@{big} - BPC@0):  softmax {sm_deg:+.4f}   forget {fo_deg:+.4f}   "
          f"-> {'forget more robust' if fo_deg < sm_deg else 'softmax more robust' if sm_deg < fo_deg else 'equal'}")

    print("\n" + "=" * 74)
    print("MEMORIZATION  -  generalization gap (val BPC - train BPC; smaller = less memorization)")
    print("=" * 74)
    sm_g = ms(res["softmax"]["gap"]); fo_g = ms(res["forget"]["gap"])
    print(f"  softmax gap: {sm_g[0]:+.4f} +/- {sm_g[1]:.4f}")
    print(f"  forget  gap: {fo_g[0]:+.4f} +/- {fo_g[1]:.4f}")
    print(f"  -> {'forget memorizes LESS' if fo_g[0] < sm_g[0] - (sm_g[1]+fo_g[1]) else 'softmax memorizes less' if sm_g[0] < fo_g[0]-(sm_g[1]+fo_g[1]) else 'similar (within noise)'}")
    print(f"\ndone in {time.time()-t0:.0f}s")

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"robustness_{args.dataset}.json"), "w") as f:
        json.dump({"shared": shared, "eps": args.eps, "results": res}, f, indent=2)
    print(f"saved runs/robustness_{args.dataset}.json")


if __name__ == "__main__":
    main()
