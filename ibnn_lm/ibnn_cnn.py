# Copyright 2026. Apache License 2.0.
#
# THE faithful test: replicate the IBNN paper's *spatial* coupling (a cross-difference
# convolution) in a small CNN image classifier, and compare to a standard CNN. Everything in
# this repo so far ported the IBNN neuron into an FFN, where the coupling runs over UNORDERED
# channels and always tied/lost. The paper's actual win comes from coupling over SPATIAL
# neighbours - which are ordered and local, i.e. they have the structure the FFN port lacked.
#
# Spatial IBNN neuron (this file), within-channel over a 3x3 neighbourhood N(i):
#     z_i = (conv x)_i - b_i  -  lambda * (1/|N|) * sum_{k in N(i)} tanh( p * (z_k - z_i) )
# This is O(|N|) per pixel (8 shifted differences), far cheaper than the FFN port's O(D^2).
#
# If IBNN finally helps HERE - on images, in a CNN, with spatial coupling - it confirms the whole
# diagnosis: the neuron needs a structured axis. If it still ties, that's a negative result on
# the paper's own claim, reproduced locally.
#
#   python -m ibnn_lm.ibnn_cnn --seeds 0 1 2 --steps 2000
#   python -m ibnn_lm.ibnn_cnn --seeds 0 1 2 --train_fracs 1.0 0.05   # data-efficiency (paper's headline)

import argparse
import statistics as stats
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vlm import load_fashion_mnist, CLASSES
from .utils import get_device, set_seed, count_params

# 3x3 neighbourhood minus the centre (the centre's own difference is tanh(0)=0 anyway)
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def spatial_lateral(z, p):
    """(1/8) * sum over 3x3 spatial neighbours of tanh(p*(z_k - z_i)), within-channel.
    z: (B, C, H, W) -> same shape. Replicate-padded so edges have no wraparound."""
    H, W = z.shape[-2:]
    zp = F.pad(z, (1, 1, 1, 1), mode="replicate")
    L = torch.zeros_like(z)
    for dy, dx in NEIGHBORS:
        z_k = zp[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        L = L + torch.tanh(p * (z_k - z))
    return L / len(NEIGHBORS)


class IBNNConv2d(nn.Module):
    """Conv -> spatial IBNN lateral coupling. coupling='standard' is a plain conv (no coupling);
    coupling='ibnn' adds the spatial cross-difference term (+1 scalar lambda)."""
    def __init__(self, in_ch, out_ch, k=3, lam=-0.05, p=10.0, num_iters=1, coupling="standard"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
        self.p, self.num_iters, self.coupling = float(p), int(num_iters), coupling
        if coupling == "ibnn":
            self.lam = nn.Parameter(torch.tensor(float(lam)))
        elif coupling != "standard":
            raise ValueError(coupling)

    def forward(self, x):
        y = self.conv(x)
        if self.coupling == "standard":
            return y
        z = y
        for _ in range(self.num_iters):
            z = y - self.lam * spatial_lateral(z, self.p)
        return z


class SmallCNN(nn.Module):
    def __init__(self, coupling="standard", lam=-0.05, num_iters=1, ch=32, n_classes=10):
        super().__init__()
        self.c1 = IBNNConv2d(1, ch, 3, lam, 10.0, num_iters, coupling)
        self.bn1 = nn.BatchNorm2d(ch)
        self.c2 = IBNNConv2d(ch, 2 * ch, 3, lam, 10.0, num_iters, coupling)
        self.bn2 = nn.BatchNorm2d(2 * ch)
        self.c3 = IBNNConv2d(2 * ch, 2 * ch, 3, lam, 10.0, num_iters, coupling)
        self.bn3 = nn.BatchNorm2d(2 * ch)
        self.head = nn.Linear(2 * ch, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.gelu(self.bn1(self.c1(x))), 2)   # 28 -> 14
        x = F.max_pool2d(F.gelu(self.bn2(self.c2(x))), 2)   # 14 -> 7
        x = F.gelu(self.bn3(self.c3(x)))
        x = x.mean(dim=(-2, -1))                            # global average pool
        return self.head(x)


@torch.no_grad()
def evaluate(model, x, y, device, n=10000, batch=1000):
    model.eval()
    correct = 0
    n = min(n, len(x))
    for b in range(0, n, batch):
        xb = x[b:b + batch].to(device)
        pred = model(xb).argmax(dim=-1).cpu()
        correct += (pred == y[b:b + batch]).sum().item()
    return correct / n


def train_one(coupling, seed, tr_x, tr_y, te_x, te_y, device, steps, batch, lr,
              num_iters, train_frac):
    set_seed(seed)
    n_train = int(len(tr_x) * train_frac)
    idx_all = torch.randperm(len(tr_x))[:n_train]
    sx, sy = tr_x[idx_all], tr_y[idx_all]
    model = SmallCNN(coupling=coupling, num_iters=num_iters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for step in range(steps):
        bi = torch.randint(0, len(sx), (batch,))
        xb, yb = sx[bi].to(device), sy[bi].to(device)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    acc = evaluate(model, te_x, te_y, device)
    params = count_params(model)
    del model, opt
    return acc, params


def main():
    ap = argparse.ArgumentParser(description="Spatial IBNN conv vs standard CNN (Fashion-MNIST).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--num_iters", type=int, default=1)
    ap.add_argument("--train_fracs", nargs="+", type=float, default=[1.0])
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    tr_x, tr_y, te_x, te_y = load_fashion_mnist()
    print(f"spatial IBNN conv vs standard CNN on Fashion-MNIST  device={device}")
    print(f"seeds={args.seeds}  steps={args.steps}  train_fracs={args.train_fracs}\n")

    t0 = time.time()
    results = {}
    for frac in args.train_fracs:
        for coupling in ("standard", "ibnn"):
            accs, params = [], None
            for seed in args.seeds:
                acc, params = train_one(coupling, seed, tr_x, tr_y, te_x, te_y, device,
                                        args.steps, args.batch, args.lr, args.num_iters, frac)
                accs.append(acc)
                print(f"  frac={frac:<4} {coupling:>8} seed{seed}: {acc*100:.2f}%  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            results[(frac, coupling)] = (stats.mean(accs),
                                         stats.stdev(accs) if len(accs) > 1 else 0.0, params)

    print("\n" + "=" * 64)
    print("SPATIAL IBNN CONV vs STANDARD CNN  -  test accuracy (higher better)")
    print("=" * 64)
    print(f"{'train data':>11} | {'standard CNN':>16} | {'IBNN-conv CNN':>16} | {'Δ (ibnn-std)':>12}")
    print("-" * 64)
    for frac in args.train_fracs:
        sm_m, sm_s, sm_p = results[(frac, "standard")]
        ib_m, ib_s, ib_p = results[(frac, "ibnn")]
        d = ib_m - sm_m
        noise = sm_s + ib_s
        tag = "" if abs(d) <= noise else ("  IBNN wins" if d > 0 else "  std wins")
        print(f"{frac*100:>9.0f}% | {sm_m*100:>7.2f} +/-{sm_s*100:5.2f} | "
              f"{ib_m*100:>7.2f} +/-{ib_s*100:5.2f} | {d*100:>+9.2f}%{tag}")
    print("-" * 64)
    print(f"params: standard={results[(args.train_fracs[0],'standard')][2]:,}  "
          f"ibnn={results[(args.train_fracs[0],'ibnn')][2]:,} "
          f"(+{results[(args.train_fracs[0],'ibnn')][2]-results[(args.train_fracs[0],'standard')][2]} "
          f"= one lambda per conv)")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
