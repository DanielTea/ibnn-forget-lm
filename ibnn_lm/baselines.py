# Copyright 2026. Apache License 2.0.
#
# Non-transformer reference baselines, so an IBNN/SM transformer's bits-per-char can be read
# against "other models of the same size." The classic char-level baseline is an LSTM; we size
# it to match the transformer's parameter count and score it with the SAME deterministic
# evaluator (ibnn_lm.evaluate.full_loss), so all three numbers are directly comparable.
#
#   python -m ibnn_lm.baselines --dataset tinyshakespeare --seeds 0 1 2 --steps 1500
#   python -m ibnn_lm.baselines --hidden 256 --emb 128 --layers 1   # ~420k params (match the GPT)

import argparse
import math
import statistics as stats
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as data_mod
from .data import get_batch
from .evaluate import full_loss
from .utils import count_params, get_device, set_seed


class CharLSTM(nn.Module):
    """A minimal char-level LSTM LM with the same call signature as the GPT (so it shares the
    evaluator and sampler): forward(idx, targets=None) -> (logits, loss)."""

    def __init__(self, vocab_size, emb=128, hidden=256, layers=1, dropout=0.1,
                 block_size=128):
        super().__init__()
        self.block_size = block_size            # only used to bound the generation context
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    class _Cfg:  # tiny shim so generate()-style code can read .block_size
        pass

    @property
    def cfg(self):
        c = CharLSTM._Cfg()
        c.block_size = self.block_size
        return c

    def forward(self, idx, targets=None, hidden=None):
        x = self.drop(self.emb(idx))
        out, hidden = self.lstm(x, hidden)
        logits = self.head(self.drop(out))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def train_lstm(ds, seed, steps, emb, hidden, layers, block_size, batch_size,
               lr, device, eval_interval=100, eval_iters=50, patience=8, min_delta=1e-3,
               quiet=True):
    set_seed(seed)
    tok = ds.tokenizer
    model = CharLSTM(tok.vocab_size, emb=emb, hidden=hidden, layers=layers,
                     block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    best_val, best_state, no_improve = float("inf"), None, 0
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(ds.train, block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_interval == 0 or step == steps:
            v = full_loss(model, ds.val, block_size, batch_size, device)["loss"]
            if not quiet:
                print(f"  step {step}/{steps} val {v:.4f} bpc {v/math.log(2):.4f} "
                      f"{time.time()-t0:.0f}s")
            if v < best_val - min_delta:
                best_val, no_improve = v, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                no_improve += 1
                if patience and no_improve >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = full_loss(model, ds.val, block_size, batch_size, device)
    metrics.update(params=count_params(model), best_val=best_val, seed=seed,
                   elapsed_s=time.time() - t0)
    del model, opt
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Char-LSTM baseline at matched parameter count.")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ds = data_mod.load(args.dataset, train_frac=1.0)
    bpcs = []
    print(f"char-LSTM baseline: emb={args.emb} hidden={args.hidden} layers={args.layers} "
          f"on {args.dataset} ({device})")
    for seed in args.seeds:
        m = train_lstm(ds, seed, args.steps, args.emb, args.hidden, args.layers,
                       args.block_size, args.batch_size, args.lr, device,
                       patience=args.patience, quiet=False)
        bpcs.append(m["bpc"])
        print(f"  seed {seed}: bpc {m['bpc']:.4f}  ppl {m['ppl']:.3f}  "
              f"params {m['params']:,}  ({m['elapsed_s']:.0f}s)")
    mean = stats.mean(bpcs)
    sd = stats.stdev(bpcs) if len(bpcs) > 1 else 0.0
    print(f"\nchar-LSTM  bits/char: {mean:.4f} +/- {sd:.4f}  (n={len(bpcs)}, "
          f"params {m['params']:,})")


if __name__ == "__main__":
    main()
