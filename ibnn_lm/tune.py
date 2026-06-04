# Copyright 2026. Apache License 2.0.
#
# Per-model hyperparameter tuning + final comparison. A "shared neutral HP" A/B can quietly
# disadvantage one model; this gives each variant its own best learning rate before the
# head-to-head, which is the fair way to ask "is the IBNN neuron better?".
#
# Protocol, for each model in {SM, IBNN-lite (num_iters=1), IBNN-implicit (num_iters=3)}:
#   1) LEARNING-RATE SEARCH: train 1 seed at each candidate LR for a short budget, score the
#      best checkpoint with the exact deterministic evaluator, keep the LR with the lowest BPC.
#   2) FINAL: retrain at that LR for the full budget across several seeds; report mean +/- std.
# Then print the tuned head-to-head (delta vs the tuned SM, with a noise-aware verdict).
#
# Only LR is swept (the dominant knob, and a shared axis all models have). num_iters is the
# IBNN-specific axis, handled by treating n=1 and n=3 as separate models. lambda is trainable
# so it self-tunes; p is left at the paper's 10. Sweeping those too is a longer follow-up.
#
#   python -m ibnn_lm.tune --lrs 1e-3 3e-3 1e-2 --seeds 0 1 2 --search_steps 500 --final_steps 1500

import argparse
import json
import math
import os
import statistics as stats
import time

from .train import build_arg_parser, train_run
from .evaluate import evaluate_checkpoint

MODELS = [
    ("sm",       dict(ffn="sm",   num_iters=1)),
    ("ibnn_n1",  dict(ffn="ibnn", num_iters=1)),
    ("ibnn_n3",  dict(ffn="ibnn", num_iters=3)),
]


def _args(overrides):
    a = build_arg_parser().parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def train_eval(overrides, dataset, device):
    """Train one config and return its exact held-out BPC (deletes the checkpoint after)."""
    out = overrides["out"]
    overrides = {**overrides, "dataset": dataset, "device": device, "sample_interval": 0,
                 "warmup": min(100, max(10, overrides["steps"] // 10))}
    r = train_run(_args(overrides), quiet=True)
    ev = evaluate_checkpoint(out, dataset=dataset, device=device)
    r["bpc_exact"], r["ppl_exact"] = ev["bpc"], ev["ppl"]
    for p in (out, out.replace(".pt", "_last.pt")):
        if os.path.exists(p):
            os.remove(p)
    return r


def main():
    ap = argparse.ArgumentParser(description="Per-model tuned IBNN(n=1,3)-vs-SM comparison.")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=[m[0] for m in MODELS],
                    help="subset of model names to run (e.g. just ibnn_n3)")
    ap.add_argument("--fixed_lr", type=float, default=None,
                    help="skip the LR search and use this LR for all models")
    ap.add_argument("--search_steps", type=int, default=500)
    ap.add_argument("--final_steps", type=int, default=1500)
    ap.add_argument("--device", type=str, default="auto")
    # smaller model so num_iters=3 stays tractable on a laptop GPU
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=192)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=50)
    args = ap.parse_args()

    shared = dict(d_model=args.d_model, d_ff=args.d_ff, n_layer=args.n_layer,
                  n_head=args.n_head, block_size=args.block_size, batch_size=args.batch_size,
                  dropout=args.dropout, patience=args.patience,
                  eval_interval=args.eval_interval, eval_iters=args.eval_iters)

    print(f"tuned comparison on {args.dataset}: models={[m[0] for m in MODELS]} "
          f"lrs={args.lrs} seeds={args.seeds}")
    print(f"model: d_model={args.d_model} d_ff={args.d_ff} n_layer={args.n_layer} "
          f"block={args.block_size}; search {args.search_steps} steps, final {args.final_steps}\n")

    t0 = time.time()
    results = {}
    run_models = [(n, s) for n, s in MODELS if n in args.models]
    for name, mspec in run_models:
        # ---- LR search (1 seed, short) ----
        if args.fixed_lr is not None:
            best_lr, search = args.fixed_lr, [(args.fixed_lr, None)]
            print(f"== {name}: using fixed lr {best_lr:.0e} (search skipped) ==", flush=True)
        else:
            print(f"== {name}: learning-rate search ==", flush=True)
            search = []
            for lr in args.lrs:
                r = train_eval({**shared, **mspec, "lr": lr, "min_lr": lr / 10,
                                "steps": args.search_steps, "seed": 0,
                                "out": f"checkpoints/tune_{name}_lr{lr:.0e}.pt"},
                               args.dataset, args.device)
                search.append((lr, r["bpc_exact"]))
                print(f"   lr={lr:.0e}  bpc={r['bpc_exact']:.4f}  ({r['elapsed_s']:.0f}s)",
                      flush=True)
            best_lr = min(search, key=lambda t: t[1])[0]
            print(f"   -> best lr for {name}: {best_lr:.0e}\n", flush=True)

        # ---- final (multiple seeds, full budget) ----
        print(f"== {name}: final @ lr={best_lr:.0e}, {len(args.seeds)} seeds ==", flush=True)
        bpcs, params = [], None
        for seed in args.seeds:
            r = train_eval({**shared, **mspec, "lr": best_lr, "min_lr": best_lr / 10,
                            "steps": args.final_steps, "seed": seed,
                            "out": f"checkpoints/tune_{name}_s{seed}.pt"},
                           args.dataset, args.device)
            bpcs.append(r["bpc_exact"])
            params = r["params"]
            print(f"   seed {seed}: bpc={r['bpc_exact']:.4f}  ({r['elapsed_s']:.0f}s)", flush=True)
        mean = stats.mean(bpcs)
        sd = stats.stdev(bpcs) if len(bpcs) > 1 else 0.0
        results[name] = dict(best_lr=best_lr, bpcs=bpcs, mean=mean, std=sd,
                             params=params, search=search)
        print(f"   -> {name}: {mean:.4f} +/- {sd:.4f} bpc\n", flush=True)

    # ---- report ----
    print("=" * 70)
    print("TUNED COMPARISON  -  exact held-out bits-per-char (LOWER is better)")
    print("=" * 70)
    print(f"dataset: {args.dataset}   model: d_model={args.d_model} d_ff={args.d_ff} "
          f"n_layer={args.n_layer} block={args.block_size}   final_steps={args.final_steps}\n")
    base = results.get("sm", {}).get("mean")
    print(f"{'model':>10} | {'tuned lr':>9} | {'params':>9} | {'BPC (mean+/-std)':>18} | {'vs SM':>16}")
    print("-" * 76)
    for name, _ in MODELS:
        if name not in results:
            continue
        r = results[name]
        cell = f"{r['mean']:.4f} +/- {r['std']:.4f}"
        vs = ""
        if base is not None and name != "sm":
            delta = r["mean"] - base
            noise = r["std"] + results["sm"]["std"]
            tag = "~tie" if abs(delta) <= noise else ("IBNN better" if delta < 0 else "SM better")
            vs = f"{delta:+.4f} {tag}"
        print(f"{name:>10} | {r['best_lr']:>9.0e} | {r['params']:>9,} | {cell:>18} | {vs:>16}")
    print("-" * 76)
    print("(delta = model minus tuned-SM; |delta| <= sum of stds => tie. lambda self-tunes; "
          "p=10 fixed.)")

    os.makedirs("runs", exist_ok=True)
    out_json = os.path.join("runs", f"tune_{args.dataset}.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\ndone in {time.time() - t0:.0f}s. full results -> {out_json}")


if __name__ == "__main__":
    main()
