# Copyright 2026. Apache License 2.0.
#
# Full local training harness for the IBNN-LM.
#
# Beyond the minimal train_demo.py, this driver:
#   - downloads/caches a real corpus (default: tinyshakespeare) via ibnn_lm.data
#   - trains on the best available device (Apple-Silicon MPS GPU, else CUDA, else CPU)
#   - uses AdamW + linear-warmup/cosine-decay LR, gradient clipping, grad accumulation
#   - periodically reports train/val loss + perplexity and samples text so you can eyeball it
#   - checkpoints the best-val and last models, with tokenizer + config embedded so that
#     `ibnn_lm.generate` can run with nothing but the .pt file
#   - can warm-start an IBNN model from a trained SM checkpoint (the paper's surrogate trick)
#
# Examples:
#   python -m ibnn_lm.train --ffn ibnn --dataset tinyshakespeare --steps 2000
#   python -m ibnn_lm.train --ffn sm   --dataset tinyshakespeare --steps 2000 --out checkpoints/sm.pt
#   python -m ibnn_lm.train --ffn ibnn --init_from checkpoints/sm.pt --steps 1000   # warm start

import argparse
import math
import os
import sys
import time
from dataclasses import asdict, fields

import torch

from . import data as data_mod
from .data import CharTokenizer, get_batch
from .model import GPT, GPTConfig, copy_sm_weights_into_ibnn
from .utils import count_params, get_device, set_seed


def build_config(args, vocab_size: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        ffn=args.ffn,
        attn=args.attn,
        ibnn_lambda=args.lam,
        ibnn_lambda_trainable=not args.lam_frozen,
        ibnn_p=args.p,
        ibnn_num_iters=args.num_iters,
        ibnn_chunk_size=args.chunk_size,
        ibnn_coupling=args.coupling,
        ibnn_interaction=args.interaction,
        ibnn_topo_dim=args.topo_dim,
    )


def lr_at(step, base_lr, warmup, total, min_lr):
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters):
    """Average loss over a few random batches from each split."""
    model.eval()
    out = {}
    for name, d in splits.items():
        if len(d) <= block_size + 1:
            out[name] = float("nan")
            continue
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(d, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample(model, tokenizer, device, prompt="\n", max_new_tokens=240, temperature=0.8, top_k=40):
    model.eval()
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tokenizer.decode(out[0].tolist())


def save_checkpoint(path, model, cfg, tokenizer, step, val_loss, args, optimizer=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "val_loss": val_loss,
        "args": vars(args),
    }
    # Optimizer moments roughly double the file size, so only the resumable "_last" checkpoint
    # carries them; the lean best-val checkpoint is meant for inference (generate).
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    torch.save(blob, path)


def load_gpt_from_checkpoint(path, device, override_iters=None):
    """Rebuild a GPT (and its tokenizer) from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    known = {f.name for f in fields(GPTConfig)}
    cfg = GPTConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
    if override_iters is not None:
        cfg.ibnn_num_iters = override_iters
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer.from_dict(ckpt["tokenizer"])
    return model, cfg, tok, ckpt


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Train the IBNN-LM locally.")
    # data
    ap.add_argument("--dataset", type=str, default="tinyshakespeare",
                    help="tinyshakespeare | tinystories | synth | path/to.txt")
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="shrink the TRAIN split (data-efficiency probe)")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_mb", type=float, default=0.0,
                    help="truncate the corpus to the first N MB (0=use all; e.g. enwik8)")
    ap.add_argument("--byte_level", action="store_true",
                    help="byte-level tokenization (vocab<=256); standard for enwik8")
    # model
    ap.add_argument("--ffn", choices=["ibnn", "sm"], default="ibnn")
    ap.add_argument("--attn", choices=["softmax", "forget"], default="softmax",
                    help="attention type: standard softmax or content-gated forgetting")
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    # ibnn knobs
    ap.add_argument("--num_iters", type=int, default=1, help="fixed-point iters (1=lite)")
    ap.add_argument("--lam", type=float, default=-0.05)
    ap.add_argument("--lam_frozen", action="store_true", help="do not train lambda")
    ap.add_argument("--p", type=float, default=10.0)
    ap.add_argument("--chunk_size", type=int, default=0,
                    help="chunk the O(D^2) lateral term over this many hidden units (0=off)")
    ap.add_argument("--coupling", choices=["meanfield", "learned", "topo"], default="meanfield",
                    help="IBNN lateral weights: meanfield (1/D) | learned w_ik | topo (learned topology)")
    ap.add_argument("--interaction", choices=["additive", "gate"], default="additive",
                    help="how the lateral term acts: additive (z=y-λL) or multiplicative gate")
    ap.add_argument("--topo_dim", type=int, default=4, help="channel-coordinate dim for coupling=topo")
    # optim
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # early stopping
    ap.add_argument("--patience", type=int, default=0,
                    help="stop if val hasn't improved for this many evals (0=off)")
    ap.add_argument("--min_delta", type=float, default=1e-3,
                    help="minimum val-loss improvement that counts as progress")
    # bookkeeping
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--sample_interval", type=int, default=0,
                    help="generate a text sample every N steps (0=only at the end)")
    ap.add_argument("--out", type=str, default=None, help="checkpoint path (best val)")
    ap.add_argument("--init_from", type=str, default=None,
                    help="warm-start from a checkpoint (e.g. an SM model -> IBNN)")
    ap.add_argument("--resume", type=str, default=None, help="resume optimizer+model from ckpt")
    ap.add_argument("--device", type=str, default="auto", help="auto|mps|cuda|cpu")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main():
    args = build_arg_parser().parse_args()

    # Line-buffer stdout so `... > train.log` shows progress live instead of in one late dump.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    train_run(args)


def train_run(args, quiet=False):
    """Train one model from a fully-populated args namespace and return a results dict.

    Importable so the comparison harness (ibnn_lm.compare) can run many configs in-process.
    The returned dict carries the headline metrics (best/final val loss, bits-per-char,
    perplexity) plus where the checkpoint landed.
    """
    def say(*a):
        if not quiet:
            print(*a)

    set_seed(args.seed)
    device = get_device(args.device)
    out_path = args.out or f"checkpoints/{args.ffn}_{args.dataset.replace('/', '_')}.pt"

    # ---- data ----
    ds = data_mod.load(args.dataset, train_frac=args.train_frac, val_split=args.val_split,
                       max_mb=getattr(args, "max_mb", 0.0),
                       byte_level=getattr(args, "byte_level", False))
    tok = ds.tokenizer
    splits = {"train": ds.train, "val": ds.val}
    cfg = build_config(args, tok.vocab_size)

    # ---- model ----
    model = GPT(cfg).to(device)
    start_step = 0
    best_val = float("inf")

    if args.init_from:
        src, _, src_tok, _ = load_gpt_from_checkpoint(args.init_from, device)
        if src_tok.chars != tok.chars:
            say("WARNING: init_from tokenizer differs from current dataset's; "
                "embeddings/head may transfer poorly.")
        copied, skipped = copy_sm_weights_into_ibnn(src, model)
        say(f"warm-started from {args.init_from}: copied {copied} tensors, "
            f"{len(skipped)} left at init (e.g. lambda).")

    resume_ckpt = None
    if args.resume:
        model, cfg, tok, resume_ckpt = load_gpt_from_checkpoint(
            args.resume, device, args.num_iters)
        start_step = resume_ckpt.get("step", 0)
        best_val = resume_ckpt.get("val_loss", float("inf"))
        splits = {"train": ds.train, "val": ds.val}
        say(f"resumed from {args.resume} at step {start_step} (val {best_val:.4f})")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        opt.load_state_dict(resume_ckpt["optimizer"])
        say("restored optimizer state (AdamW moments) from checkpoint")

    run_model = model
    if args.compile:
        try:
            run_model = torch.compile(model)
            say("torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            say(f"torch.compile unavailable ({e}); continuing eagerly.")

    n_params = count_params(model)
    say(f"\n== IBNN-LM training ==")
    say(f"ffn={cfg.ffn}  params={n_params:,}  device={device}")
    say(f"dataset={args.dataset}  vocab={tok.vocab_size}  "
        f"train_tokens={len(ds.train):,}  val_tokens={len(ds.val):,}")
    say(f"d_model={cfg.d_model} d_ff={cfg.d_ff or 4 * cfg.d_model} "
        f"n_layer={cfg.n_layer} n_head={cfg.n_head} block={cfg.block_size} "
        f"num_iters={cfg.ibnn_num_iters}")
    say(f"steps={args.steps} batch={args.batch_size}x{args.grad_accum} "
        f"lr={args.lr}->{args.min_lr}  out={out_path}\n")

    model.train()
    t0 = time.time()
    running = None
    best_step = start_step
    evals_without_improve = 0
    stopped_early = False
    last_step = start_step
    last_val = float("nan")
    for step in range(start_step + 1, args.steps + 1):
        last_step = step
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(ds.train, cfg.block_size, args.batch_size, device)
            _, loss = run_model(x, y)
            (loss / args.grad_accum).backward()
            loss_accum += loss.item() / args.grad_accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running = loss_accum if running is None else 0.9 * running + 0.1 * loss_accum

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            stats = estimate_loss(model, splits, cfg.block_size, args.batch_size,
                                  device, args.eval_iters)
            last_val = stats["val"]
            dt = time.time() - t0
            lam = _lambda_summary(model)
            say(f"step {step:5d}/{args.steps}  lr {lr:.2e}  "
                f"train {stats['train']:.4f}  val {stats['val']:.4f}  "
                f"ppl {math.exp(min(20, stats['val'])):.2f}  "
                f"{lam}  {dt:.1f}s")
            if stats["val"] < best_val - args.min_delta:
                best_val, best_step = stats["val"], step
                evals_without_improve = 0
                save_checkpoint(out_path, model, cfg, tok, step, best_val, args)
            else:
                evals_without_improve += 1
                if args.patience and evals_without_improve >= args.patience:
                    say(f"early stop at step {step}: no val improvement for "
                        f"{evals_without_improve} evals (best {best_val:.4f} @ {best_step})")
                    stopped_early = True
                    break

        if args.sample_interval and step % args.sample_interval == 0:
            say("  --- sample " + "-" * 50)
            say("  " + sample(model, tok, device).replace("\n", "\n  "))
            say("  " + "-" * 60)

    # final save (last) carries optimizer state so `--resume <..._last.pt>` continues exactly
    last_path = out_path.replace(".pt", "_last.pt")
    save_checkpoint(last_path, model, cfg, tok, last_step, best_val, args, optimizer=opt)
    elapsed = time.time() - t0
    say(f"\ndone in {elapsed:.1f}s. best val {best_val:.4f} @ step {best_step} -> {out_path}")
    say(f"last model -> {last_path}")
    if not quiet:
        say("\n=== final sample ===")
        say(sample(model, tok, device, max_new_tokens=400))

    ln2 = math.log(2.0)
    results = {
        "ffn": cfg.ffn, "seed": args.seed, "train_frac": args.train_frac,
        "params": n_params, "best_val": best_val, "best_step": best_step,
        "final_val": last_val, "best_bpc": best_val / ln2, "best_ppl": math.exp(best_val),
        "stopped_early": stopped_early, "last_step": last_step,
        "elapsed_s": elapsed, "out": out_path, "device": device,
        "lambdas": _lambda_values(model),
    }
    # free GPU memory between in-process runs (compare.py calls this in a loop)
    del model, run_model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return results


def _lambda_values(model):
    """Per-layer lambda values (empty list for an SM model)."""
    return [round(m.lam.item(), 4) for m in model.modules()
            if hasattr(m, "lam") and hasattr(m, "p")]


def _lambda_summary(model):
    """Compact readout of the (trainable) lambdas, to watch the IBNN coupling evolve."""
    lams = _lambda_values(model)
    if not lams:
        return "lam=n/a"
    return f"lam[{min(lams):+.3f},{max(lams):+.3f}]"


if __name__ == "__main__":
    main()
