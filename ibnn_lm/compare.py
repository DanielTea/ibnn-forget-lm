# Copyright 2026. Apache License 2.0.
#
# Controlled head-to-head: is the IBNN FFN actually better than a standard transformer FFN?
#
# This trains matched models that differ ONLY in the FFN (ffn="sm" vs ffn="ibnn") - identical
# depth, width, attention, data, optimizer, LR schedule, step budget, and seed - then scores
# each on the exact held-out validation set (bits-per-char). Because IBNN adds only +n_layer
# scalar parameters, this isolates the neuron model as the single independent variable, which
# is the whole point of the fork.
#
# It sweeps seeds (for error bars) and optionally training-set fractions (the paper's headline
# data-efficiency claim: the updated neuron should help MORE when data is scarce). Results are
# printed as a table with mean +/- std and saved to runs/compare_*.json.
#
#   python -m ibnn_lm.compare --seeds 0 1 2 --steps 1500
#   python -m ibnn_lm.compare --seeds 0 1 2 --train_fracs 1.0 0.1 --steps 1500
#
# Runtime is N_ffn * N_seeds * N_fracs training runs; keep the model small and use early
# stopping (on by default here) to keep it to tens of minutes on a laptop GPU.

import argparse
import json
import math
import os
import statistics as stats
import time
from copy import deepcopy

from .train import build_arg_parser, train_run
from .evaluate import evaluate_checkpoint


def _base_args(overrides):
    """A full train.py args namespace at its defaults, with `overrides` applied."""
    args = build_arg_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def run_grid(dataset, ffns, seeds, train_fracs, shared, device, keep_ckpts):
    os.makedirs("runs", exist_ok=True)
    rows = []
    total = len(ffns) * len(seeds) * len(train_fracs)
    done = 0
    t0 = time.time()
    for frac in train_fracs:
        for ffn in ffns:
            for seed in seeds:
                done += 1
                tag = f"{ffn}_f{frac}_s{seed}"
                out = f"checkpoints/cmp_{tag}.pt"
                args = _base_args({
                    **shared,
                    "dataset": dataset, "ffn": ffn, "seed": seed,
                    "train_frac": frac, "device": device, "out": out,
                    "sample_interval": 0,
                })
                print(f"[{done}/{total}] training {tag} "
                      f"(steps<= {args.steps}, patience {args.patience}) ...", flush=True)
                r = train_run(args, quiet=True)
                # Exact, deterministic held-out score on the best checkpoint (apples-to-apples).
                ev = evaluate_checkpoint(out, dataset=dataset, device=device)
                r["bpc_exact"], r["ppl_exact"], r["val_exact"] = ev["bpc"], ev["ppl"], ev["loss"]
                rows.append(r)
                print(f"        -> best_val {r['best_val']:.4f} @ {r['best_step']}  "
                      f"exact bpc {r['bpc_exact']:.4f}  ppl {r['ppl_exact']:.3f}  "
                      f"({r['elapsed_s']:.0f}s, params {r['params']:,})", flush=True)
                if not keep_ckpts:
                    for p in (out, out.replace(".pt", "_last.pt")):
                        if os.path.exists(p):
                            os.remove(p)
    print(f"\nall {total} runs done in {time.time() - t0:.0f}s")
    return rows


def aggregate(rows, ffns, train_fracs):
    """Group exact-BPC results by (train_frac, ffn) -> (mean, std, n)."""
    agg = {}
    for frac in train_fracs:
        for ffn in ffns:
            vals = [r["bpc_exact"] for r in rows
                    if r["train_frac"] == frac and r["ffn"] == ffn]
            if vals:
                mean = stats.mean(vals)
                sd = stats.stdev(vals) if len(vals) > 1 else 0.0
                agg[(frac, ffn)] = (mean, sd, len(vals))
    return agg


def report(rows, ffns, train_fracs, shared, dataset):
    agg = aggregate(rows, ffns, train_fracs)
    params = {r["ffn"]: r["params"] for r in rows}
    print("\n" + "=" * 74)
    print("CONTROLLED COMPARISON  -  IBNN FFN vs standard transformer FFN")
    print("=" * 74)
    print(f"dataset   : {dataset}")
    print(f"model     : d_model={shared['d_model']} d_ff={shared['d_ff']} "
          f"n_layer={shared['n_layer']} n_head={shared['n_head']} block={shared['block_size']}")
    print(f"budget    : <= {shared['steps']} steps, early stop patience {shared['patience']}, "
          f"lr {shared['lr']}->{shared['min_lr']}")
    if "sm" in params and "ibnn" in params:
        print(f"params    : sm={params['sm']:,}  ibnn={params['ibnn']:,} "
              f"(+{params['ibnn'] - params['sm']} = one lambda per layer)")
    print("\nExact held-out bits-per-character (LOWER is better), mean +/- std:\n")
    header = f"{'train_frac':>10} | " + " | ".join(f"{f'{x} FFN':>20}" for x in ffns)
    if {"sm", "ibnn"} <= set(ffns):
        header += " | " + f"{'delta(ibnn-sm)':>16} | winner"
    print(header)
    print("-" * len(header))
    for frac in train_fracs:
        cells = []
        for ffn in ffns:
            if (frac, ffn) in agg:
                m, sd, n = agg[(frac, ffn)]
                cells.append(f"{m:>8.4f} +/- {sd:6.4f}")
            else:
                cells.append(f"{'-':>20}")
        line = f"{frac:>10.2f} | " + " | ".join(f"{c:>20}" for c in cells)
        if {"sm", "ibnn"} <= set(ffns) and (frac, "sm") in agg and (frac, "ibnn") in agg:
            ms, ss, _ = agg[(frac, "sm")]
            mi, si, _ = agg[(frac, "ibnn")]
            delta = mi - ms
            noise = ss + si
            if abs(delta) <= noise:
                verdict = "~tie (within noise)"
            elif delta < 0:
                verdict = "IBNN better"
            else:
                verdict = "SM better"
            line += f" | {delta:>+16.4f} | {verdict}"
        print(line)
    print("-" * len(header))
    print("(delta is IBNN minus SM; |delta| <= sum of stds is reported as a tie. "
          "Few seeds => treat small gaps as noise.)")
    return agg


def main():
    ap = argparse.ArgumentParser(description="Controlled IBNN-vs-SM comparison with error bars.")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--ffns", nargs="+", default=["sm", "ibnn"], choices=["sm", "ibnn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train_fracs", nargs="+", type=float, default=[1.0])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--keep_ckpts", action="store_true", help="don't delete per-run checkpoints")
    # shared model/optim budget (small + early stop = laptop-friendly)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--num_iters", type=int, default=1, help="IBNN fixed-point iters")
    args = ap.parse_args()

    shared = dict(
        d_model=args.d_model, d_ff=args.d_ff, n_layer=args.n_layer, n_head=args.n_head,
        block_size=args.block_size, dropout=args.dropout, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
        patience=args.patience, eval_interval=args.eval_interval, eval_iters=args.eval_iters,
        num_iters=args.num_iters,
    )
    print(f"comparison grid: ffns={args.ffns} seeds={args.seeds} "
          f"train_fracs={args.train_fracs}  ({len(args.ffns) * len(args.seeds) * len(args.train_fracs)} runs)")
    rows = run_grid(args.dataset, args.ffns, args.seeds, args.train_fracs,
                    shared, args.device, args.keep_ckpts)
    report(rows, args.ffns, args.train_fracs, shared, args.dataset)

    stamp = f"{args.dataset}_{'_'.join(args.ffns)}_{len(args.seeds)}seeds"
    out_json = os.path.join("runs", f"compare_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({"shared": shared, "dataset": args.dataset, "rows": rows}, f, indent=2)
    print(f"\nfull per-run results saved to {out_json}")


if __name__ == "__main__":
    main()
