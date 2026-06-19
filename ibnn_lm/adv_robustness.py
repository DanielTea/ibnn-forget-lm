# Copyright 2026. Apache License 2.0.
#
# The IBNN paper's HEADLINE claim was adversarial robustness (image CNNs). This tests it
# faithfully: a DEEPER CNN (5 conv layers) using the paper's spatial cross-difference coupling
# (IBNNConv2d, coupling='ibnn') vs the identical CNN with plain convs, under FGSM and PGD
# attacks, over many seeds. Same architecture for both; IBNN adds only +1 scalar per conv.
#
# Clean accuracy alone has been a tie/slight-edge throughout this repo. The paper's gains were on
# *robustness*, so this measures accuracy under adversarial perturbation - where, if the spatial
# coupling's stability signal we keep seeing is real, it should finally show up beyond noise.
#
#   python -m ibnn_lm.adv_robustness --seeds 0 1 2 3 4 5 6 7 --steps 1500

import argparse
import statistics as stats
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ibnn_cnn import IBNNConv2d
from .utils import get_device, set_seed, count_params


class DeeperCNN(nn.Module):
    """5-conv-layer CNN; every conv is an IBNNConv2d (coupling='standard' -> plain conv,
    coupling='ibnn' -> spatial cross-difference coupling). Identical graph for both."""
    def __init__(self, coupling="standard", lam=-0.05, num_iters=1, ch=32, n_classes=10):
        super().__init__()
        def cv(i, o):
            return IBNNConv2d(i, o, 3, lam, 10.0, num_iters, coupling)
        self.c1, self.b1 = cv(1, ch), nn.BatchNorm2d(ch)
        self.c2, self.b2 = cv(ch, ch), nn.BatchNorm2d(ch)
        self.c3, self.b3 = cv(ch, 2 * ch), nn.BatchNorm2d(2 * ch)
        self.c4, self.b4 = cv(2 * ch, 2 * ch), nn.BatchNorm2d(2 * ch)
        self.c5, self.b5 = cv(2 * ch, 4 * ch), nn.BatchNorm2d(4 * ch)
        self.head = nn.Linear(4 * ch, n_classes)

    def forward(self, x):
        x = F.gelu(self.b1(self.c1(x)))
        x = F.max_pool2d(F.gelu(self.b2(self.c2(x))), 2)    # 28 -> 14
        x = F.gelu(self.b3(self.c3(x)))
        x = F.max_pool2d(F.gelu(self.b4(self.c4(x))), 2)    # 14 -> 7
        x = F.max_pool2d(F.gelu(self.b5(self.c5(x))), 2)    # 7 -> 3
        return self.head(x.mean(dim=(-2, -1)))              # global avg pool


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, steps):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + alpha * grad.sign()).detach()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv


def adv_accuracy(model, x, y, device, attack, n=2000, batch=500, **kw):
    """attack in {'clean','fgsm','pgd'}. Returns accuracy over the first n test images."""
    model.eval()
    n = min(n, len(x))
    correct = 0
    for b in range(0, n, batch):
        xb, yb = x[b:b + batch].to(device), y[b:b + batch].to(device)
        if attack == "fgsm":
            xb = fgsm(model, xb, yb, kw["eps"])
        elif attack == "pgd":
            xb = pgd(model, xb, yb, kw["eps"], kw["alpha"], kw["steps"])
        with torch.no_grad():
            pred = model(xb).argmax(dim=-1)
        correct += (pred == yb).sum().item()
    return correct / n


def train_cnn(coupling, seed, tr_x, tr_y, device, steps, batch, lr, num_iters):
    set_seed(seed)
    model = DeeperCNN(coupling=coupling, num_iters=num_iters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for step in range(steps):
        bi = torch.randint(0, len(tr_x), (batch,))
        loss = F.cross_entropy(model(tr_x[bi].to(device)), tr_y[bi].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(description="Deeper spatial-IBNN CNN vs standard CNN: adversarial robustness.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--num_iters", type=int, default=1)
    ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    from .vlm import load_fashion_mnist
    device = get_device(args.device)
    tr_x, tr_y, te_x, te_y = load_fashion_mnist()
    # the perturbation sets we report
    PROBES = [("clean", {}), ("fgsm", {"eps": 0.1}), ("fgsm", {"eps": 0.2}),
              ("pgd", {"eps": 0.1, "alpha": 0.02, "steps": 7})]
    labels = ["clean", "FGSM .1", "FGSM .2", "PGD .1"]

    print(f"DEEPER CNN ({count_params(DeeperCNN('standard')):,} params) adversarial robustness on "
          f"Fashion-MNIST  device={device}")
    print(f"seeds={args.seeds}  steps={args.steps}  eval_n={args.eval_n}\n")

    t0 = time.time()
    res = {c: {lab: [] for lab in labels} for c in ("standard", "ibnn")}
    for coupling in ("standard", "ibnn"):
        for seed in args.seeds:
            model = train_cnn(coupling, seed, tr_x, tr_y, device, args.steps, args.batch,
                              args.lr, args.num_iters)
            row = []
            for (atk, kw), lab in zip(PROBES, labels):
                a = adv_accuracy(model, te_x, te_y, device, atk, n=args.eval_n, **kw)
                res[coupling][lab].append(a)
                row.append(f"{lab} {a*100:.1f}")
            print(f"  {coupling:>8} seed{seed}: " + "  ".join(row) + f"   ({time.time()-t0:.0f}s)",
                  flush=True)
            del model

    def ms(xs):
        return stats.mean(xs) * 100, (stats.stdev(xs) * 100 if len(xs) > 1 else 0.0)

    print("\n" + "=" * 76)
    print("ADVERSARIAL ROBUSTNESS  -  test accuracy under attack (higher = more robust)")
    print("=" * 76)
    print(f"{'attack':>10} | {'standard CNN':>16} | {'spatial-IBNN CNN':>18} | {'Δ (ibnn-std)':>12}")
    print("-" * 76)
    for lab in labels:
        sm_m, sm_s = ms(res["standard"][lab])
        ib_m, ib_s = ms(res["ibnn"][lab])
        d = ib_m - sm_m
        tag = "" if abs(d) <= sm_s + ib_s else ("  IBNN more robust" if d > 0 else "  std more robust")
        print(f"{lab:>10} | {sm_m:>7.2f} +/-{sm_s:5.2f} | {ib_m:>8.2f} +/-{ib_s:6.2f} | {d:>+9.2f}%{tag}")
    print("-" * 76)
    print(f"params: standard={count_params(DeeperCNN('standard')):,}  "
          f"ibnn={count_params(DeeperCNN('ibnn')):,}  (+5 = one lambda per conv)")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
