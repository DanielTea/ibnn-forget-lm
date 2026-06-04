# Copyright 2026. Apache License 2.0.
#
# Bake-off of NEW IBNN-FFN variants against the standard FFN and the plain (mean-field) IBNN.
# These are ideas aimed at the diagnosis from the parent study (a uniform mean-field coupling
# over UNORDERED FFN channels is structureless and only smooths), none of which are - as far as
# we know - in the literature in this form:
#
#   sm              standard Transformer FFN (baseline)
#   ibnn_meanfield  paper IBNN: additive, uniform 1/D coupling, lambda<0      (the known tie)
#   ibnn_gate       #1 competition-as-GATE: v = phi(y) * 2*sigmoid(lambda*L)  (GLU/divisive-norm flavour)
#   ibnn_topo       #2 learned channel TOPOLOGY: w_ik = softmax kernel over learned coordinates
#   ibnn_sharpen    #3 SHARPENING instead of smoothing: lambda>0, num_iters=1 (soft winner-take-all)
#
# All share depth/width/data/optimizer/seeds; each variant adds <1% params. Exact held-out BPC,
# mean +/- std over seeds. Honest prior: FFN-channel tricks have tied/lost in 8+ prior runs.
#
#   python -m ibnn_lm.ideas_test --dataset tinyshakespeare --seeds 0 1 2 --steps 1200

import argparse
import json
import os
import statistics as stats
import time

from .train import build_arg_parser, train_run
from .evaluate import evaluate_checkpoint

# (label, overrides applied on top of the shared config)
VARIANTS = [
    ("sm",             dict(ffn="sm")),
    ("ibnn_meanfield", dict(ffn="ibnn", coupling="meanfield", interaction="additive", lam=-0.05)),
    ("ibnn_gate",      dict(ffn="ibnn", coupling="meanfield", interaction="gate",     lam=0.0)),
    ("ibnn_topo",      dict(ffn="ibnn", coupling="topo",      interaction="additive", lam=-0.05)),
    ("ibnn_sharpen",   dict(ffn="ibnn", coupling="meanfield", interaction="additive", lam=0.05,
                           num_iters=1)),
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
    ap = argparse.ArgumentParser(description="Bake-off of new IBNN-FFN variants vs SM and plain IBNN.")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=192)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    shared = dict(d_model=args.d_model, d_ff=args.d_ff, n_layer=args.n_layer,
                  n_head=args.n_head, block_size=args.block_size, batch_size=args.batch_size,
                  dropout=0.1, steps=args.steps, lr=args.lr, min_lr=args.lr / 10, warmup=100,
                  patience=6, eval_interval=150, eval_iters=40)

    print(f"ideas bake-off on {args.dataset}: {[v[0] for v in VARIANTS]}  seeds={args.seeds}\n")
    t0 = time.time()
    results = {}
    for label, over in VARIANTS:
        bpcs, params = [], None
        print(f"== {label} ==", flush=True)
        for seed in args.seeds:
            r = train_eval({**shared, **over, "seed": seed,
                            "out": f"checkpoints/idea_{label}_s{seed}.pt"},
                           args.dataset, args.device)
            bpcs.append(r["bpc"]); params = r["params"]
            print(f"   seed {seed}: bpc={r['bpc']:.4f}  ({r['elapsed_s']:.0f}s)", flush=True)
        results[label] = dict(bpcs=bpcs, mean=stats.mean(bpcs),
                             std=stats.stdev(bpcs) if len(bpcs) > 1 else 0.0, params=params)
        print(f"   -> {label}: {results[label]['mean']:.4f} +/- {results[label]['std']:.4f} bpc "
              f"({params:,} params)\n", flush=True)

    sm = results["sm"]
    print("=" * 76)
    print("NEW IBNN-FFN IDEAS  -  exact held-out bits-per-char (lower is better)")
    print("=" * 76)
    print(f"{'variant':>16} | {'params':>9} | {'BPC (mean+/-std)':>18} | {'Δ vs sm':>9} | verdict")
    print("-" * 76)
    winner = None
    for label, _ in VARIANTS:
        r = results[label]
        if label == "sm":
            print(f"{label:>16} | {r['params']:>9,} | {r['mean']:>8.4f} +/- {r['std']:6.4f} | "
                  f"{'—':>9} | baseline")
            continue
        d = r["mean"] - sm["mean"]
        noise = r["std"] + sm["std"]
        verdict = ("BEATS sm" if d < -noise else "loses to sm" if d > noise else "~tie")
        if d < -noise and (winner is None or r["mean"] < results[winner]["mean"]):
            winner = label
        print(f"{label:>16} | {r['params']:>9,} | {r['mean']:>8.4f} +/- {r['std']:6.4f} | "
              f"{d:>+9.4f} | {verdict}")
    print("-" * 76)
    print(f"result: {'a variant beats SM: ' + winner if winner else 'no variant beats SM beyond noise (null holds)'}")
    print(f"\ndone in {time.time()-t0:.0f}s")

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"ideas_{args.dataset}.json"), "w") as f:
        json.dump({"shared": shared, "results": results}, f, indent=2)
    print(f"saved runs/ideas_{args.dataset}.json")


if __name__ == "__main__":
    main()
