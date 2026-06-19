# Copyright 2026. Apache License 2.0.
#
# A from-scratch toy Vision-Language Model on a REAL dataset (CIFAR-10), reusing this repo's
# Transformer decoder (so it can use the IBNN neuron and/or forgetting attention) and a small
# ViT vision encoder (which can ALSO use the IBNN neuron in its FFN).
#
#   image --[ViT encoder]--> visual tokens --[project]--> prefix
#   prefix + "a photo of a " --[GPT decoder]--> generates "...a photo of a {class}."
#
# Why this is interesting beyond "can we build a VLM": IBNN tied/lost on TEXT (the FFN's channels
# are unordered). Here the model is processing IMAGES - the domain IBNN was actually designed for
# (the paper validated on CIFAR-10) - so we can ask whether ffn="ibnn" finally helps when the
# input has spatial structure. (Caveat: the IBNN coupling still runs over FFN *channels*, not the
# spatial patch axis, so the unordered-channel critique still applies; this measures it.)
#
#   python -m ibnn_lm.vlm --ffn ibnn --steps 3000      # IBNN neuron in encoder + decoder FFNs
#   python -m ibnn_lm.vlm --ffn sm   --steps 3000      # standard FFN baseline

import argparse
import gzip
import os
import struct
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DATA_DIR
from .layers import IBNNMLP
from .ibnn_cnn import IBNNConv2d
from .model import GPT, GPTConfig
from .utils import get_device, set_seed, count_params

# Fashion-MNIST (Xiao et al. 2017) - real 28x28 grayscale clothing images, 10 classes. It is one
# of the datasets the IBNN paper itself validated on, and is reliably hosted on GitHub.
FMNIST_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FMNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz",
}
CLASSES = ["tshirt", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "boot"]
IMG_SIZE, IN_CH, PATCH = 28, 1, 7   # 28/7 = 4 -> 16 patches


# --------------------------------------------------------------------------- data
def _read_idx(path, images):
    with gzip.open(path, "rb") as f:
        if images:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            buf = bytearray(f.read(n * r * c))
            return torch.frombuffer(buf, dtype=torch.uint8).view(n, 1, r, c).float() / 255.0
        _, n = struct.unpack(">II", f.read(8))
        return torch.frombuffer(bytearray(f.read(n)), dtype=torch.uint8).long()


def load_fashion_mnist():
    """Download/cache Fashion-MNIST -> (train_x, train_y, test_x, test_y).
    Images float [0,1], shape (N, 1, 28, 28); labels long (N,)."""
    root = os.path.join(DATA_DIR, "fashion_mnist")
    os.makedirs(root, exist_ok=True)
    paths = {}
    for key, fn in FMNIST_FILES.items():
        dest = os.path.join(root, fn)
        if not os.path.isfile(dest):
            print(f"downloading {fn}")
            urllib.request.urlretrieve(FMNIST_BASE + fn, dest)
        paths[key] = dest
    return (_read_idx(paths["train_x"], True), _read_idx(paths["train_y"], False),
            _read_idx(paths["test_x"], True), _read_idx(paths["test_y"], False))


class CharVocab:
    """Tiny char vocab for the captions, with BOS/EOS/PAD specials."""
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.itos = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad, self.bos, self.eos = 0, 1, 2

    @property
    def size(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i >= 3:
                out.append(self.itos[i])
        return "".join(out)


def caption_for(label):
    return f"a photo of a {CLASSES[label]}."


# --------------------------------------------------------------------------- vision encoder
def make_ffn(ffn, d_model, d_ff):
    if ffn == "ibnn":
        return IBNNMLP(d_model, d_ff=d_ff, activation="gelu")
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))


class BiAttention(nn.Module):
    """Plain (bidirectional) multi-head self-attention for the vision encoder."""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head, self.d_model = n_head, d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / (hs ** 0.5), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class ViTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = BiAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model); self.mlp = make_ffn(ffn, d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, d_model, n_layer=3, n_head=4, patch=PATCH, img=IMG_SIZE, ffn="sm",
                 d_ff=None):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.patch_embed = nn.Conv2d(IN_CH, d_model, kernel_size=patch, stride=patch)
        self.n_patches = (img // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(d_model, n_head, d_ff, ffn)
                                     for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img):
        x = self.patch_embed(img).flatten(2).transpose(1, 2)   # (B, P, d_model)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x)


class ConvVisionEncoder(nn.Module):
    """A convolutional vision encoder whose convs use the paper's SPATIAL IBNN coupling
    (coupling='ibnn') or are plain convs (coupling='standard'). This is the principled home for
    IBNN: the conv couples over a pixel's spatial neighbours (a structured axis), unlike the ViT
    encoder whose FFN couples over unordered channels. Output: out_grid^2 visual tokens."""
    def __init__(self, d_model, coupling="standard", ch=32, num_iters=1):
        super().__init__()
        self.c1 = IBNNConv2d(IN_CH, ch, 3, coupling=coupling, num_iters=num_iters)
        self.bn1 = nn.BatchNorm2d(ch)
        self.c2 = IBNNConv2d(ch, 2 * ch, 3, coupling=coupling, num_iters=num_iters)
        self.bn2 = nn.BatchNorm2d(2 * ch)
        self.proj = nn.Conv2d(2 * ch, d_model, 1)
        self.n_patches = 7 * 7   # 28 -> 14 -> 7 (two 2x pools)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, img):
        x = F.max_pool2d(F.gelu(self.bn1(self.c1(img))), 2)    # 28 -> 14
        x = F.max_pool2d(F.gelu(self.bn2(self.c2(x))), 2)      # 14 -> 7
        x = self.proj(x)                                       # (B, d_model, 7, 7)
        return x.flatten(2).transpose(1, 2) + self.pos         # (B, 49, d_model)


# --------------------------------------------------------------------------- VLM
class VLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, enc_layers=3, dec_layers=3, n_head=6,
                 block_size=64, ffn="sm", attn="softmax", dropout=0.1, d_ff=None,
                 encoder="vit", enc_coupling="standard"):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        if encoder == "conv":
            # convolutional encoder with the paper's SPATIAL IBNN coupling (the right home for it)
            self.encoder = ConvVisionEncoder(d_model, coupling=enc_coupling)
        else:
            self.encoder = VisionEncoder(d_model, n_layer=enc_layers, n_head=n_head,
                                         ffn=ffn, d_ff=d_ff)
        self.proj = nn.Linear(d_model, d_model)
        self.n_patches = self.encoder.n_patches
        block_size = max(block_size, self.n_patches + 40)   # cover visual prefix + caption
        cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=dec_layers,
                        n_head=n_head, d_model=d_model, d_ff=d_ff, dropout=dropout,
                        ffn=ffn, attn=attn)
        self.decoder = GPT(cfg)
        self.d_model = d_model

    def _visual_prefix(self, img):
        return self.proj(self.encoder(img))                    # (B, P, d_model)

    def forward(self, img, cap_in, cap_tgt, pad_id):
        B = img.shape[0]
        vis = self._visual_prefix(img)
        txt = self.decoder.tok_emb(cap_in)                     # (B, L, d_model)
        emb = torch.cat([vis, txt], dim=1)                     # (B, P+L, d_model)
        # targets: dummy over the P visual positions, real over the L text positions
        P = vis.shape[1]
        dummy = torch.full((B, P), pad_id, dtype=torch.long, device=img.device)
        targets = torch.cat([dummy, cap_tgt], dim=1)
        mask = torch.cat([torch.zeros(B, P, device=img.device),
                          (cap_tgt != pad_id).float()], dim=1)
        _, loss = self.decoder.forward_embeds(emb, targets=targets, loss_mask=mask)
        return loss

    @torch.no_grad()
    def caption(self, img, vocab, max_len=32):
        self.eval()
        vis = self._visual_prefix(img)                         # (B, P, d_model)
        B = img.shape[0]
        toks = torch.full((B, 1), vocab.bos, dtype=torch.long, device=img.device)
        for _ in range(max_len):
            txt = self.decoder.tok_emb(toks)
            emb = torch.cat([vis, txt], dim=1)
            logits, _ = self.decoder.forward_embeds(emb)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            toks = torch.cat([toks, nxt], dim=1)
            if (nxt == vocab.eos).all():
                break
        return [vocab.decode(toks[i, 1:].tolist()) for i in range(B)]


# --------------------------------------------------------------------------- train / eval
def make_batch(x, y, idx, vocab, device):
    imgs = x[idx].to(device)
    caps = [caption_for(int(y[i])) for i in idx]
    enc = [[vocab.bos] + vocab.encode(c) + [vocab.eos] for c in caps]
    L = max(len(e) for e in enc)
    cap_in = torch.full((len(idx), L - 1), vocab.pad, dtype=torch.long)
    cap_tgt = torch.full((len(idx), L - 1), vocab.pad, dtype=torch.long)
    for r, e in enumerate(enc):
        cap_in[r, :len(e) - 1] = torch.tensor(e[:-1])
        cap_tgt[r, :len(e) - 1] = torch.tensor(e[1:])
    return imgs, cap_in.to(device), cap_tgt.to(device)


@torch.no_grad()
def accuracy(model, x, y, vocab, device, n=1000, batch=250):
    model.eval()
    correct = 0
    for b in range(0, n, batch):
        idx = list(range(b, min(b + batch, n)))
        caps = model.caption(x[idx].to(device), vocab)
        for j, i in enumerate(idx):
            # exact caption match (avoids "shirt" matching inside "tshirt")
            if caps[j].strip() == caption_for(int(y[i])).strip():
                correct += 1
    return correct / n


def main():
    ap = argparse.ArgumentParser(description="Toy VLM on CIFAR-10 (IBNN-capable).")
    ap.add_argument("--ffn", choices=["ibnn", "sm"], default="ibnn")
    ap.add_argument("--attn", choices=["softmax", "forget"], default="softmax")
    ap.add_argument("--encoder", choices=["vit", "conv"], default="vit",
                    help="vision encoder: ViT (transformer) or conv (spatial-IBNN-capable)")
    ap.add_argument("--enc_coupling", choices=["standard", "ibnn"], default="standard",
                    help="conv encoder coupling: plain conv or the paper's spatial IBNN coupling")
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--d_ff", type=int, default=256, help="FFN hidden width (keep modest for IBNN)")
    ap.add_argument("--enc_layers", type=int, default=3)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/vlm.pt")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"loading Fashion-MNIST ...")
    tr_x, tr_y, te_x, te_y = load_fashion_mnist()
    vocab = CharVocab([caption_for(c) for c in range(10)])
    model = VLM(vocab.size, d_model=args.d_model, d_ff=args.d_ff, enc_layers=args.enc_layers,
                dec_layers=args.dec_layers, n_head=args.n_head, ffn=args.ffn,
                attn=args.attn, encoder=args.encoder, enc_coupling=args.enc_coupling).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    print(f"VLM ffn={args.ffn} attn={args.attn} params={count_params(model):,} device={device}")
    print(f"train images={len(tr_x):,}  vocab={vocab.size}  patches={model.n_patches}\n")

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(tr_x), (args.batch_size,))
        imgs, cap_in, cap_tgt = make_batch(tr_x, tr_y, idx, vocab, device)
        loss = model(imgs, cap_in, cap_tgt, vocab.pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_interval == 0 or step == 1:
            acc = accuracy(model, te_x, te_y, vocab, device, n=1000)
            model.train()
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"test_acc {acc*100:.1f}%  {time.time()-t0:.0f}s", flush=True)
        elif step % 100 == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "vocab": vocab.itos}, args.out)
    acc = accuracy(model, te_x, te_y, vocab, device, n=5000)
    print(f"\nfinal test accuracy (5000 imgs): {acc*100:.2f}%   -> {args.out}")
    # show a few example captions
    print("\nsample captions (held-out test images):")
    idx = list(range(8))
    caps = model.caption(te_x[idx].to(device), vocab)
    for j, i in enumerate(idx):
        print(f"  true={CLASSES[int(te_y[i])]:<11} model=\"{caps[j]}\"")


if __name__ == "__main__":
    main()
