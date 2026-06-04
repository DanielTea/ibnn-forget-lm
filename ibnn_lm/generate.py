# Copyright 2026. Apache License 2.0.
#
# Local inference for a trained IBNN-LM. Loads a checkpoint (model weights + config +
# tokenizer all embedded by ibnn_lm.train) and generates text. No dataset needed.
#
# Examples:
#   python -m ibnn_lm.generate --ckpt checkpoints/ibnn_tinyshakespeare.pt --prompt "ROMEO:"
#   python -m ibnn_lm.generate --ckpt checkpoints/ibnn_tinyshakespeare.pt --interactive
#   # stream tokens as they're produced:
#   python -m ibnn_lm.generate --ckpt checkpoints/ibnn_tinyshakespeare.pt --prompt "To be" --stream

import argparse
import sys

import torch
import torch.nn.functional as F

from .train import load_gpt_from_checkpoint
from .utils import get_device, set_seed


@torch.no_grad()
def stream_generate(model, tokenizer, device, prompt, max_new_tokens,
                    temperature=0.8, top_k=40):
    """Yield decoded text one token at a time (so the terminal can print as it goes)."""
    model.eval()
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    block = model.cfg.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        yield tokenizer.decode([nxt.item()])


def generate_text(model, tokenizer, device, prompt, max_new_tokens,
                  temperature=0.8, top_k=40):
    pieces = list(stream_generate(model, tokenizer, device, prompt,
                                  max_new_tokens, temperature, top_k))
    return prompt + "".join(pieces)


def main():
    ap = argparse.ArgumentParser(description="Generate text from a trained IBNN-LM.")
    ap.add_argument("--ckpt", type=str, required=True, help="path to a training checkpoint")
    ap.add_argument("--prompt", type=str, default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--stream", action="store_true", help="print tokens as they are produced")
    ap.add_argument("--interactive", action="store_true", help="REPL: type prompts, get text")
    ap.add_argument("--num_iters", type=int, default=None,
                    help="override IBNN fixed-point iters at inference (e.g. solve harder)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
    device = get_device(args.device)
    model, cfg, tok, ckpt = load_gpt_from_checkpoint(args.ckpt, device, args.num_iters)
    model.eval()

    trained_step = ckpt.get("step", "?")
    val = ckpt.get("val_loss", float("nan"))
    print(f"loaded {args.ckpt}: ffn={cfg.ffn} params={sum(p.numel() for p in model.parameters()):,} "
          f"vocab={tok.vocab_size} step={trained_step} val={val:.4f} device={device}",
          file=sys.stderr)

    if args.interactive:
        print("Interactive mode. Type a prompt and press enter (Ctrl-D / 'quit' to exit).",
              file=sys.stderr)
        while True:
            try:
                prompt = input("\nprompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip() in {"quit", "exit"}:
                break
            sys.stdout.write(prompt)
            for piece in stream_generate(model, tok, device, prompt or "\n",
                                         args.max_new_tokens, args.temperature, args.top_k):
                sys.stdout.write(piece)
                sys.stdout.flush()
            print()
        return

    for s in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n===== sample {s + 1}/{args.num_samples} =====")
        if args.stream:
            sys.stdout.write(args.prompt)
            for piece in stream_generate(model, tok, device, args.prompt,
                                         args.max_new_tokens, args.temperature, args.top_k):
                sys.stdout.write(piece)
                sys.stdout.flush()
            print()
        else:
            print(generate_text(model, tok, device, args.prompt,
                                args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
