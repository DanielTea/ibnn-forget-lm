# Copyright 2026. Apache License 2.0.
#
# Deterministic held-out evaluation for a trained IBNN-LM. Unlike the random-batch estimate
# used for progress logging during training, this sweeps the ENTIRE validation split once in
# contiguous, non-overlapping windows and reports an exact, reproducible number. That makes it
# a fair benchmark to compare two models (same protocol applied to both).
#
# Reported metrics for a character-level LM:
#   loss  - mean next-token negative log-likelihood (nats/char)
#   bpc   - bits per character = loss / ln(2)   (the standard char-LM benchmark metric)
#   ppl   - perplexity = exp(loss)
#
#   python -m ibnn_lm.evaluate --ckpt checkpoints/ibnn_tinyshakespeare.pt
#   python -m ibnn_lm.evaluate --ckpt checkpoints/sm_tinyshakespeare.pt --dataset tinyshakespeare

import argparse
import math

import torch
import torch.nn.functional as F

from . import data as data_mod
from .train import load_gpt_from_checkpoint
from .utils import get_device


@torch.no_grad()
def full_loss(model, data, block_size, batch_size=64, device="cpu"):
    """Exact mean next-token NLL over `data`, in contiguous non-overlapping blocks.

    Every model sees the same windows, so the number is directly comparable across models.
    Returns a dict with loss (nats/char), bits-per-char, perplexity, and the token count.
    """
    was_training = model.training
    model.eval()
    n_windows = (len(data) - 1) // block_size
    if n_windows == 0:
        return {"loss": float("nan"), "bpc": float("nan"), "ppl": float("nan"), "tokens": 0}

    total_loss, total_tok = 0.0, 0
    for b in range(0, n_windows, batch_size):
        idxs = range(b, min(b + batch_size, n_windows))
        x = torch.stack([data[i * block_size:i * block_size + block_size] for i in idxs])
        y = torch.stack([data[i * block_size + 1:i * block_size + 1 + block_size] for i in idxs])
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tok += y.numel()

    if was_training:
        model.train()
    mean = total_loss / total_tok
    return {"loss": mean, "bpc": mean / math.log(2), "ppl": math.exp(mean), "tokens": total_tok}


def evaluate_checkpoint(ckpt_path, dataset=None, val_split=0.1, batch_size=64, device="auto"):
    device = get_device(device)
    model, cfg, tok, ckpt = load_gpt_from_checkpoint(ckpt_path, device)
    # Match the corpus/tokenization the checkpoint was trained with (dataset, subset, bytes).
    cargs = ckpt.get("args", {})
    dataset = dataset or cargs.get("dataset", "tinyshakespeare")
    ds = data_mod.load(dataset, train_frac=1.0, val_split=val_split,
                       max_mb=cargs.get("max_mb", 0.0),
                       byte_level=cargs.get("byte_level", False))
    if ds.tokenizer.chars != tok.chars:
        print("WARNING: dataset vocabulary differs from the checkpoint's tokenizer; "
              "re-encoding val text with the checkpoint's tokenizer for a valid comparison.")
        with open(data_mod.prepare(dataset), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        val = ids[int(len(ids) * (1 - val_split)):]
    else:
        val = ds.val
    metrics = full_loss(model, val, cfg.block_size, batch_size, device)
    metrics.update(ffn=cfg.ffn, params=sum(p.numel() for p in model.parameters()),
                   dataset=dataset, ckpt=ckpt_path)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Exact held-out BPC/perplexity for a checkpoint.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset", type=str, default=None,
                    help="defaults to the dataset stored in the checkpoint")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    m = evaluate_checkpoint(args.ckpt, args.dataset, args.val_split, args.batch_size, args.device)
    print(f"\ncheckpoint : {m['ckpt']}")
    print(f"ffn        : {m['ffn']}   params: {m['params']:,}")
    print(f"dataset    : {m['dataset']}   val tokens: {m['tokens']:,}")
    print(f"val loss   : {m['loss']:.4f} nats/char")
    print(f"bits/char  : {m['bpc']:.4f}")
    print(f"perplexity : {m['ppl']:.3f}")


if __name__ == "__main__":
    main()
