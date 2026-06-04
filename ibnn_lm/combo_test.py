# Copyright 2026. Apache License 2.0.
#
# The headline experiment of this repo: a 2x2 factorial over {FFN neuron} x {attention type}.
#
#   FFN:        sm   (standard Linear->GELU->Linear)   vs  ibnn (implicit-bias lateral neuron)
#   attention:  softmax (standard causal attention)    vs  forget (content-gated forgetting attn)
#
# Background (from the parent ibnn-lm study): swapping the FFN neuron to IBNN did NOT help at
# char-LM scale, but adding a forget gate to attention (an independent re-derivation of the
# Forgetting Transformer / FoX, Lin et al. ICLR 2025) DID, by a wide margin. The open question
# this repo asks: once the forget gate is doing the heavy lifting, does the IBNN neuron add
# anything on top? The factorial isolates each main effect and their interaction.
#
# Everything else is held identical (depth, width, data, optimizer, LR, steps, seeds). Exact
# held-out bits-per-char, mean +/- std over seeds.
#
#   python -m ibnn_lm.combo_test --dataset tinyshakespeare --seeds 0 1 2 --steps 1500

import argparse
import json
import os
import statistics as stats
import time

from .train import build_arg_parser, train_run
from .evaluate import evaluate_checkpoint

# (label, ffn, attn). Order: the four cells of the 2x2.
CELLS = [
    ("sm+softmax",   "sm",   "softmax"),
    ("sm+forget",    "sm",   "forget"),
    ("ibnn+softmax", "ibnn", "softmax"),
    ("ibnn+forget",  "ibnn", "forget"),
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
    ap = argparse.ArgumentParser(description="2x2 factorial: FFN neuron x attention type.")
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
    args = ap.parse_args()

    shared = dict(d_model=args.d_model, d_ff=args.d_ff, n_layer=args.n_layer,
                  n_head=args.n_head, block_size=args.block_size, batch_size=args.batch_size,
                  dropout=0.1, steps=args.steps, lr=args.lr, min_lr=args.lr / 10, warmup=100,
                  patience=6, eval_interval=150, eval_iters=40,
                  byte_level=args.byte_level, max_mb=args.max_mb)

    print(f"2x2 factorial on {args.dataset}: {[c[0] for c in CELLS]}  seeds={args.seeds}\n")
    t0 = time.time()
    results = {}
    for label, ffn, attn in CELLS:
        bpcs, params = [], None
        print(f"== {label} ==", flush=True)
        for seed in args.seeds:
            r = train_eval({**shared, "ffn": ffn, "attn": attn, "seed": seed,
                            "out": f"checkpoints/combo_{label.replace('+', '_')}_s{seed}.pt"},
                           args.dataset, args.device)
            bpcs.append(r["bpc"])
            params = r["params"]
            print(f"   seed {seed}: bpc={r['bpc']:.4f}  ({r['elapsed_s']:.0f}s)", flush=True)
        results[label] = dict(bpcs=bpcs, mean=stats.mean(bpcs),
                             std=stats.stdev(bpcs) if len(bpcs) > 1 else 0.0, params=params)
        print(f"   -> {label}: {results[label]['mean']:.4f} +/- {results[label]['std']:.4f} bpc\n",
              flush=True)

    m = {k: results[k]["mean"] for k in results}
    s = {k: results[k]["std"] for k in results}

    print("=" * 72)
    print("2x2 FACTORIAL  -  exact held-out bits-per-char (lower is better)")
    print("=" * 72)
    print(f"{'':>12} | {'softmax attn':>16} | {'forget attn':>16} | {'forget effect':>14}")
    print("-" * 72)
    for ffn in ("sm", "ibnn"):
        sa, fa = f"{ffn}+softmax", f"{ffn}+forget"
        row = (f"{ffn+' FFN':>12} | {m[sa]:>7.4f} +/-{s[sa]:5.3f} | "
               f"{m[fa]:>7.4f} +/-{s[fa]:5.3f} | {m[fa]-m[sa]:>+14.4f}")
        print(row)
    print("-" * 72)
    ibnn_eff_soft = m["ibnn+softmax"] - m["sm+softmax"]
    ibnn_eff_forg = m["ibnn+forget"] - m["sm+forget"]
    print(f"{'IBNN effect':>12} | {ibnn_eff_soft:>+15.4f} | {ibnn_eff_forg:>+15.4f} |")
    print("=" * 72)
    # interpretation
    best = min(m, key=m.get)
    forget_helps = (m["sm+forget"] - m["sm+softmax"]) < -(s["sm+forget"] + s["sm+softmax"])
    ibnn_adds = ibnn_eff_forg < -(s["ibnn+forget"] + s["sm+forget"])
    print(f"best cell: {best} ({m[best]:.4f} bpc)")
    print(f"forget gate helps (vs softmax): {forget_helps}")
    print(f"IBNN adds value ON TOP of forget: {ibnn_adds}  "
          f"(ibnn+forget vs sm+forget = {ibnn_eff_forg:+.4f})")
    print(f"\ndone in {time.time()-t0:.0f}s")

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"combo_{args.dataset}.json"), "w") as f:
        json.dump({"shared": shared, "results": results}, f, indent=2)
    print(f"saved runs/combo_{args.dataset}.json")


if __name__ == "__main__":
    main()
