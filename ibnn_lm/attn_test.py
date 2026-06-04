# Copyright 2026. Apache License 2.0.
#
# Does a content-gated forgetting mechanism on the token axis beat standard softmax attention?
#
# Motivation from this repo's own results: a char-level LSTM (which has a forget gate) clearly
# beat the transformer here, and the post-mortem on IBNN concluded a learnable "decay" coupling
# needs a STRUCTURED axis - which in a transformer is the token/time axis, i.e. attention's job.
# So we add a per-head, content-dependent forget gate to attention (forgetting attention), built
# as a strict SUPERSET of softmax attention (gates open => identical), and test it head to head.
#
# Everything else is held fixed (same standard FFN, depth, width, data, optimizer, seeds); the
# only change is attn='softmax' vs attn='forget' (+~0.4% params for the gate). Exact held-out
# bits-per-char, 3 seeds.
#
#   python -m ibnn_lm.attn_test --dataset tinyshakespeare --seeds 0 1 2 --steps 1500

import argparse
import json
import os
import statistics as stats
import time

from .train import build_arg_parser, train_run
from .evaluate import evaluate_checkpoint

MODELS = [
    ("softmax_attn", dict(attn="softmax", ffn="sm")),
    ("forget_attn",  dict(attn="forget",  ffn="sm")),
]


def _args(over):
    a = build_arg_parser().parse_args([])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def train_eval(over, dataset, device):
    out = over["out"]
    r = train_run(_args({**over, "dataset": dataset, "device": device,
                         "sample_interval": 0}), quiet=True)
    ev = evaluate_checkpoint(out, dataset=dataset, device=device)
    r["bpc"] = ev["bpc"]
    for p in (out, out.replace(".pt", "_last.pt")):
        if os.path.exists(p):
            os.remove(p)
    return r


def main():
    ap = argparse.ArgumentParser(description="Softmax vs forgetting attention, matched A/B.")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--byte_level", action="store_true")
    ap.add_argument("--max_mb", type=float, default=0.0)
    ap.add_argument("--lstm_ref", type=float, default=None,
                    help="optional char-LSTM BPC to print as a reference line")
    args = ap.parse_args()

    shared = dict(d_model=args.d_model, d_ff=args.d_ff, n_layer=args.n_layer,
                  n_head=args.n_head, block_size=args.block_size, batch_size=args.batch_size,
                  dropout=0.1, steps=args.steps, lr=args.lr, min_lr=args.lr / 10, warmup=100,
                  patience=6, eval_interval=150, eval_iters=40,
                  byte_level=args.byte_level, max_mb=args.max_mb)

    print(f"attention A/B on {args.dataset}: {[m[0] for m in MODELS]} seeds={args.seeds}\n")
    t0 = time.time()
    results = {}
    for name, mspec in MODELS:
        bpcs, params = [], None
        print(f"== {name} ==", flush=True)
        for seed in args.seeds:
            r = train_eval({**shared, **mspec, "seed": seed,
                            "out": f"checkpoints/attn_{name}_s{seed}.pt"},
                           args.dataset, args.device)
            bpcs.append(r["bpc"])
            params = r["params"]
            print(f"   seed {seed}: bpc={r['bpc']:.4f}  ({r['elapsed_s']:.0f}s)", flush=True)
        results[name] = dict(bpcs=bpcs, mean=stats.mean(bpcs),
                             std=stats.stdev(bpcs) if len(bpcs) > 1 else 0.0, params=params)
        print(f"   -> {name}: {results[name]['mean']:.4f} +/- {results[name]['std']:.4f} bpc "
              f"({params:,} params)\n", flush=True)

    base = results["softmax_attn"]
    new = results["forget_attn"]
    delta = new["mean"] - base["mean"]
    noise = new["std"] + base["std"]
    print("=" * 70)
    print("FORGETTING ATTENTION vs SOFTMAX ATTENTION  -  held-out bits-per-char")
    print("=" * 70)
    for name, _ in MODELS:
        r = results[name]
        print(f"   {name:>14}: {r['mean']:.4f} +/- {r['std']:.4f}   ({r['params']:,} params)")
    if args.lstm_ref:
        print(f"   {'char-LSTM (ref)':>14}: {args.lstm_ref:.4f}")
    if abs(delta) <= noise:
        verdict = "TIE (within seed noise)"
    elif delta < 0:
        verdict = f"FORGETTING ATTENTION WINS by {-delta:.4f} bpc (beyond noise)"
    else:
        verdict = f"softmax better by {delta:.4f}"
    print(f"\n   delta(forget - softmax) = {delta:+.4f}   ->   {verdict}")
    print(f"\ndone in {time.time() - t0:.0f}s")

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"attn_{args.dataset}.json"), "w") as f:
        json.dump({"shared": shared, "results": results}, f, indent=2)
    print(f"saved runs/attn_{args.dataset}.json")


if __name__ == "__main__":
    main()
