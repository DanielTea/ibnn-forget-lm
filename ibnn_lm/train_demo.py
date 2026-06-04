# Copyright 2026. Apache License 2.0.
#
# Tiny char-level training demo: the LM equivalent of the paper's 2D toy example. Trains a
# small decoder on a text file and reports validation loss. Use it to compare the IBNN FFN
# against the SM baseline at equal parameter count, or to A/B the data-efficiency claim by
# shrinking --train_frac.
#
#   python -m ibnn_lm.train_demo --ffn ibnn --steps 500
#   python -m ibnn_lm.train_demo --ffn sm   --steps 500
#
# With no --data given, it falls back to a small built-in synthetic corpus so the script runs
# anywhere with zero setup.

import argparse
import math
import torch

from .model import GPT, GPTConfig

SYNTH = (
    "the quick brown fox jumps over the lazy dog. " * 200
    + "pack my box with five dozen liquor jugs. " * 200
    + "how vexingly quick daft zebras jump! " * 200
)


def get_data(path, train_frac):
    text = open(path).read() if path else SYNTH
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(len(data) * 0.9)
    train, val = data[:n_train], data[n_train:]
    train = train[: max(1, int(len(train) * train_frac))]
    return train, val, len(chars)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="path to a .txt file")
    ap.add_argument("--ffn", choices=["ibnn", "sm"], default="ibnn")
    ap.add_argument("--num_iters", type=int, default=1, help="IBNN fixed-point iters (1=lite)")
    ap.add_argument("--lam", type=float, default=-0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val, vocab = get_data(args.data, args.train_frac)
    cfg = GPTConfig(
        vocab_size=vocab, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model,
        ffn=args.ffn, ibnn_lambda=args.lam, ibnn_num_iters=args.num_iters,
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"ffn={args.ffn} params={model.num_params():,} device={device} "
          f"train_tokens={len(train):,} vocab={vocab}")
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            vl = estimate_loss(model, val, args.block_size, args.batch_size, device)
            print(f"step {step:4d}  train {loss.item():.4f}  val {vl:.4f}  ppl {math.exp(vl):.2f}")

    print("done.")


if __name__ == "__main__":
    main()
