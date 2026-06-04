# Copyright 2026. Apache License 2.0.
#
# Does breaking the IBNN's mean-field symmetry help? The paper's neuron uses a uniform 1/D
# coupling (parameter-free), which over an FFN's unordered hidden channels is a structureless
# all-to-all (mean-field) interaction. This experiment swaps in a *learned* (D, D) coupling
# matrix w_ik so the interaction can become structured, and asks whether that closes the gap
# to a standard FFN.
#
# The learned coupling adds d_ff^2 weights/layer, so it is NOT parameter-matched to SM. To tell
# "structure helped" apart from "more parameters helped", we include sm_wide: a standard FFN
# widened to the SAME parameter count as ibnn_learned. The decisive comparison is therefore
# ibnn_learned vs sm_wide, not ibnn_learned vs sm.
#
#   python -m ibnn_lm.coupling_test --seeds 0 1 2 --steps 1500
#
# Exact held-out bits-per-char, mean +/- std, identical recipe for every model.

import argparse
import json
import os
import statistics as stats
import time

from .train import build_arg_parser, train_run
from .evaluate import evaluate_checkpoint

# (label, overrides). d_ff=512 makes sm_wide match ibnn_learned's parameter count.
MODELS = [
    ("sm",             dict(ffn="sm",   d_ff=256)),
    ("sm_wide",        dict(ffn="sm",   d_ff=512)),
    ("ibnn_meanfield", dict(ffn="ibnn", d_ff=256, coupling="meanfield", num_iters=1)),
    ("ibnn_learned",   dict(ffn="ibnn", d_ff=256, coupling="learned",   num_iters=1)),
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
    ap = argparse.ArgumentParser(description="Mean-field vs learned IBNN coupling (+ param control).")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    shared = dict(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                  block_size=args.block_size, batch_size=args.batch_size, dropout=0.1,
                  steps=args.steps, lr=args.lr, min_lr=args.lr / 10, warmup=100,
                  patience=args.patience, eval_interval=150, eval_iters=40)

    print(f"coupling test on {args.dataset}: models={[m[0] for m in MODELS]} seeds={args.seeds}\n")
    t0 = time.time()
    results = {}
    for name, mspec in MODELS:
        bpcs, params = [], None
        print(f"== {name} ==", flush=True)
        for seed in args.seeds:
            r = train_eval({**shared, **mspec, "seed": seed,
                            "out": f"checkpoints/coup_{name}_s{seed}.pt"},
                           args.dataset, args.device)
            bpcs.append(r["bpc"])
            params = r["params"]
            print(f"   seed {seed}: bpc={r['bpc']:.4f}  ({r['elapsed_s']:.0f}s)", flush=True)
        mean = stats.mean(bpcs)
        sd = stats.stdev(bpcs) if len(bpcs) > 1 else 0.0
        results[name] = dict(bpcs=bpcs, mean=mean, std=sd, params=params)
        print(f"   -> {name}: {mean:.4f} +/- {sd:.4f} bpc  ({params:,} params)\n", flush=True)

    # ---- report ----
    sm = results["sm"]["mean"]
    smw = results.get("sm_wide", {}).get("mean")
    print("=" * 78)
    print("MEAN-FIELD vs LEARNED IBNN COUPLING  -  exact held-out bits-per-char (lower better)")
    print("=" * 78)
    print(f"{'model':>16} | {'params':>9} | {'BPC (mean+/-std)':>18} | {'vs sm':>9} | {'vs sm_wide':>10}")
    print("-" * 78)
    for name, _ in MODELS:
        r = results[name]
        d_sm = f"{r['mean'] - sm:+.4f}" if name != "sm" else "-"
        d_smw = f"{r['mean'] - smw:+.4f}" if (smw is not None and name != "sm_wide") else "-"
        print(f"{name:>16} | {r['params']:>9,} | {r['mean']:>8.4f} +/- {r['std']:6.4f} | "
              f"{d_sm:>9} | {d_smw:>10}")
    print("-" * 78)
    # verdict on the decisive comparison
    if smw is not None:
        d = results["ibnn_learned"]["mean"] - smw
        noise = results["ibnn_learned"]["std"] + results["sm_wide"]["std"]
        if abs(d) <= noise:
            verdict = "TIE: structured coupling does NOT beat a param-matched plain FFN"
        elif d < 0:
            verdict = "IBNN_LEARNED WINS: the mean-field structurelessness WAS the limiter"
        else:
            verdict = "WORSE: the learned coupling is a poor use of the extra parameters"
        print(f"decisive (ibnn_learned vs sm_wide, param-matched): delta={d:+.4f}  ->  {verdict}")
    print(f"\ndone in {time.time() - t0:.0f}s")

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"coupling_{args.dataset}.json"), "w") as f:
        json.dump({"shared": shared, "results": results}, f, indent=2)
    print(f"saved runs/coupling_{args.dataset}.json")


if __name__ == "__main__":
    main()
