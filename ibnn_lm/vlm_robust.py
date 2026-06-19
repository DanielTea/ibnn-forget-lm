# Copyright 2026. Apache License 2.0.
#
# VLM image-corruption robustness factorial - the one place both mechanisms coexist, so it is how
# we test the forget gate alongside the spatial IBNN under a robustness lens (a plain CNN has no
# attention for a forget gate). Design hardened by a multi-agent design review; see README.
#
#   {encoder: standard-conv vs spatial-IBNN-conv (num_iters=3)} x {decoder: softmax vs forget}
#
# Key rigor (from the design panel):
#  - FORGET FIX: in a VLM the forget gate sits maximally far from the visual prefix and "forgets"
#    the image -> the forget cell collapses. Fix: re-init each decoder fgate.bias to 5.0 in
#    train_vlm (less prefix decay; still a strict softmax superset).
#  - VERDICT METRIC = RETENTION = acc(corrupt)/acc(clean), paired by seed (removes the clean-level
#    confound). Raw corrupted accuracy is NOT the verdict.
#  - DIAGNOSTICS logged to catch artifacts: mean |lambda| (did the coupling engage?), forget
#    strength (did the gate engage?), clean acc/NLL, collapse flag.
#  - SCORED accuracy (argmin teacher-forced CE over the 10 class captions) - ~5x faster than
#    free-run, statistically identical - for gradient-free levels; free-run for adversarial.
#  - Incremental save/resume: results checkpoint after every (cell,seed) to survive interruptions.
#
#   python -m ibnn_lm.vlm_robust --seeds 0 1 2 3 4 5 6 7        # resumes if re-run

import argparse
import json
import math
import os
import statistics as stats
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vlm import load_fashion_mnist, CharVocab, VLM, make_batch, caption_for, CLASSES
from .model import CausalSelfAttention
from .utils import get_device, set_seed, count_params

RESULTS = os.path.join("runs", "vlm_robust_results.json")
LN2 = math.log(2.0)


# ----------------------------------------------------------------- corruptions (gradient-free)
def _gauss_kernel(sigma, device):
    r = max(1, int(round(2 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return k, r


def corrupt(name, level):
    """Return a function x(B,1,28,28)->corrupted, clamped to [0,1]. Deterministic given a fixed
    generator (passed via closure) so all cells see identical corrupted images."""
    g = torch.Generator().manual_seed(1234)
    if name == "clean":
        return lambda x: x
    if name == "gauss":
        return lambda x: (x + torch.randn(x.shape, generator=g) * level).clamp(0, 1)
    if name == "blur":
        def f(x):
            k, r = _gauss_kernel(level, x.device)
            kx = k.view(1, 1, 1, -1); ky = k.view(1, 1, -1, 1)
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx)
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky)
            return x.clamp(0, 1)
        return f
    if name == "contrast":
        return lambda x: ((x - 0.5) * level + 0.5).clamp(0, 1)
    if name == "occlude":
        def f(x):
            x = x.clone(); k = int(level)
            yy = int(torch.randint(0, 28 - k, (1,), generator=g))
            xx = int(torch.randint(0, 28 - k, (1,), generator=g))
            x[..., yy:yy + k, xx:xx + k] = 0.0
            return x
        return f
    if name == "bright":   # NEGATIVE CONTROL - no mechanism should be robust here
        return lambda x: (x + level).clamp(0, 1)
    raise ValueError(name)


# ----------------------------------------------------------------- candidate captions (scoring)
def _candidates(vocab, device):
    enc = [[vocab.bos] + vocab.encode(caption_for(c)) + [vocab.eos] for c in range(10)]
    L = max(len(e) for e in enc)
    ci = torch.full((10, L - 1), vocab.pad, dtype=torch.long)
    ct = torch.full((10, L - 1), vocab.pad, dtype=torch.long)
    for r, e in enumerate(enc):
        ci[r, :len(e) - 1] = torch.tensor(e[:-1]); ct[r, :len(e) - 1] = torch.tensor(e[1:])
    return ci.to(device), ct.to(device)


@torch.no_grad()
def scored_accuracy(model, x, y, vocab, device, corrupt_fn, n=10000, batch=500):
    """argmin over the 10 class captions of teacher-forced CE given the (corrupted) image.
    ~5x faster than free-run decode and statistically identical on this closed label set."""
    ci, ct = _candidates(vocab, device)
    model.eval(); n = min(n, len(x)); correct = 0
    for b in range(0, n, batch):
        xb = corrupt_fn(x[b:b + batch].clone()).to(device)
        vis = model._visual_prefix(xb); B = xb.shape[0]; P = vis.shape[1]
        losses = torch.empty(B, 10, device=device)
        for c in range(10):
            cic = ci[c].unsqueeze(0).expand(B, -1)
            ctc = ct[c].unsqueeze(0).expand(B, -1)
            emb = torch.cat([vis, model.decoder.tok_emb(cic)], dim=1)
            logits, _ = model.decoder.forward_embeds(emb)
            tl = logits[:, P:, :]
            ce = F.cross_entropy(tl.reshape(-1, tl.size(-1)), ctc.reshape(-1),
                                 reduction="none", ignore_index=vocab.pad).view(B, -1)
            m = (ctc != vocab.pad).float()
            losses[:, c] = (ce * m).sum(1) / m.sum(1).clamp(min=1)
        correct += (losses.argmin(1).cpu() == y[b:b + batch]).sum().item()
    return correct / n


def vlm_fgsm(model, imgs, ci, ct, pad, eps):
    imgs = imgs.clone().detach().requires_grad_(True)
    loss = model(imgs, ci, ct, pad)
    g = torch.autograd.grad(loss, imgs)[0]
    return (imgs + eps * g.sign()).clamp(0, 1).detach()


def vlm_pgd(model, imgs, ci, ct, pad, eps, alpha, steps):
    adv = (imgs + torch.empty_like(imgs).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = model(adv, ci, ct, pad)
        g = torch.autograd.grad(loss, adv)[0]
        adv = (adv + alpha * g.sign()).detach()
        adv = torch.min(torch.max(adv, imgs - eps), imgs + eps).clamp(0, 1)
    return adv


@torch.no_grad()
def freerun_accuracy(model, x, y, vocab, device, n=2000, batch=250):
    model.eval(); n = min(n, len(x)); correct = 0
    for b in range(0, n, batch):
        caps = model.caption(x[b:b + batch].to(device), vocab)
        for j in range(len(caps)):
            if caps[j].strip() == caption_for(int(y[b + j])).strip():
                correct += 1
    return correct / n


def adversarial_accuracy(model, x, y, vocab, device, attack, n=2000, batch=250, **kw):
    """Attack the teacher-forced TRUE-caption CE, then evaluate free-run decode (non-circular)."""
    model.eval(); n = min(n, len(x)); correct = 0
    for b in range(0, n, batch):
        xb, yb = x[b:b + batch].to(device), y[b:b + batch]
        imgs, ci, ct = make_batch(xb.cpu(), yb, list(range(len(yb))), vocab, device)
        if attack == "fgsm":
            xadv = vlm_fgsm(model, imgs, ci, ct, vocab.pad, kw["eps"])
        else:
            xadv = vlm_pgd(model, imgs, ci, ct, vocab.pad, kw["eps"], kw["alpha"], kw["steps"])
        caps = model.caption(xadv, vocab)
        for j in range(len(caps)):
            if caps[j].strip() == caption_for(int(yb[j])).strip():
                correct += 1
    return correct / n


# ----------------------------------------------------------------- diagnostics
def mean_abs_lambda(model):
    lams = [abs(model.encoder.c1.lam.item()), abs(model.encoder.c2.lam.item())] \
        if hasattr(model.encoder, "c1") and hasattr(model.encoder.c1, "lam") else []
    return round(stats.mean(lams), 4) if lams else None


# ----------------------------------------------------------------- train one cell/seed
def train_vlm(enc_coupling, attn, seed, tr_x, tr_y, vocab, device, steps, enc_iters):
    set_seed(seed)
    m = VLM(vocab.size, d_model=96, dec_layers=2, n_head=4, ffn="sm",
            attn=attn, encoder="conv", enc_coupling=enc_coupling).to(device)
    if enc_coupling == "ibnn":           # use the implicit solve (num_iters>1) in the conv encoder
        m.encoder.c1.num_iters = enc_iters
        m.encoder.c2.num_iters = enc_iters
    if attn == "forget":                 # FORGET FIX: reduce prefix decay so the image isn't forgotten
        for mod in m.modules():
            if isinstance(mod, CausalSelfAttention) and getattr(mod, "forget", False):
                nn.init.constant_(mod.fgate.bias, 5.0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=5e-4)
    m.train()
    for step in range(steps):
        bi = torch.randint(0, len(tr_x), (64,))
        imgs, ci, ct = make_batch(tr_x, tr_y, bi, vocab, device)
        loss = m(imgs, ci, ct, vocab.pad)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m, loss.item()


def main():
    ap = argparse.ArgumentParser(description="VLM robustness factorial (rigorous, resumable).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--enc_iters", type=int, default=3, help="num_iters for the IBNN conv encoder")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    tr_x, tr_y, te_x, te_y = load_fashion_mnist()
    vocab = CharVocab([caption_for(c) for c in range(10)])
    cells = [("standard", "softmax"), ("ibnn", "softmax"),
             ("standard", "forget"), ("ibnn", "forget")]
    # gradient-free corruption grid (scored) + adversarial (free-run)
    GF = [("clean", 0.0), ("gauss", 0.1), ("gauss", 0.2), ("gauss", 0.4),
          ("blur", 0.7), ("blur", 1.2), ("contrast", 0.5), ("occlude", 12), ("bright", 0.3)]
    ADV = [("fgsm", {"eps": 0.1}), ("fgsm", {"eps": 0.2}),
           ("pgd", {"eps": 0.1, "alpha": 0.02, "steps": 7})]

    os.makedirs("runs", exist_ok=True)
    done = {}
    if os.path.isfile(RESULTS):
        done = {tuple(json.loads(k)): v for k, v in json.load(open(RESULTS)).items()}
    print(f"VLM robustness factorial  device={device}  enc_iters={args.enc_iters}  "
          f"(resuming: {len(done)} cell-seeds already done)\n", flush=True)

    t0 = time.time()
    for enc, attn in cells:
        for seed in args.seeds:
            key = (enc, attn, seed)
            if key in done:
                continue
            model, train_loss = train_vlm(enc, attn, seed, tr_x, tr_y, vocab, device,
                                          args.steps, args.enc_iters)
            rec = {"train_loss": train_loss, "lambda": mean_abs_lambda(model),
                   "gf": {}, "adv": {}}
            for name, lvl in GF:
                rec["gf"][f"{name}_{lvl}"] = scored_accuracy(model, te_x, te_y, vocab, device,
                                                             corrupt(name, lvl), n=5000)
            for atk, kw in ADV:
                tag = atk + "_" + "_".join(f"{k}{v}" for k, v in kw.items())
                rec["adv"][tag] = adversarial_accuracy(model, te_x, te_y, vocab, device, atk,
                                                        n=1500, **kw)
            clean = rec["gf"]["clean_0.0"]
            rec["collapsed"] = clean < 0.60
            done[key] = rec
            json.dump({json.dumps(list(k)): v for k, v in done.items()}, open(RESULTS, "w"))
            print(f"  {enc:>8}/{attn:<7} seed{seed}: clean {clean*100:.1f}  "
                  f"blur1.2 {rec['gf']['blur_1.2']*100:.1f}  fgsm.1 {rec['adv']['fgsm_eps0.1']*100:.1f}  "
                  f"lam {rec['lambda']}  {'COLLAPSE' if rec['collapsed'] else ''}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    n_total = len(cells) * len(args.seeds)
    if len(done) >= n_total:
        report(done, cells, args.seeds, GF, ADV)
    else:
        print(f"\n{len(done)}/{n_total} done. Re-run the same command to resume the rest.", flush=True)


def report(done, cells, seeds, GF, ADV):
    levels = [f"{n}_{l}" for n, l in GF] + [a + "_" + "_".join(f"{k}{v}" for k, v in kw.items())
                                            for a, kw in ADV]
    def cell_seed(enc, attn, seed):
        return done[(enc, attn, seed)]
    print("\n" + "=" * 92)
    print("VLM IMAGE-CORRUPTION ROBUSTNESS  -  RETENTION = acc(corrupt)/acc(clean), paired by seed")
    print("=" * 92)
    # clean accuracy + retention per cell per level
    hdr = f"{'cell':>17} | {'clean':>11} | " + " | ".join(f"{lv.split('_')[0][:4]+lv.split('_')[-1][:4]:>9}" for lv in levels[1:])
    print(hdr); print("-" * min(len(hdr), 160))
    rets = {}
    for enc, attn in cells:
        cleans = [cell_seed(enc, attn, s)["gf"]["clean_0.0"] for s in seeds]
        succ = [s for s in seeds if cell_seed(enc, attn, s)["gf"]["clean_0.0"] >= 0.60]
        cm, cs = stats.mean(cleans) * 100, (stats.stdev(cleans) * 100 if len(cleans) > 1 else 0)
        row = []
        for lv in levels[1:]:
            r = []
            for s in succ:
                rec = cell_seed(enc, attn, s)
                acc = rec["gf"].get(lv) or rec["adv"].get(lv)
                clean = rec["gf"]["clean_0.0"]
                if acc is not None and clean >= 0.6:
                    r.append(acc / clean)
            rets[(enc, attn, lv)] = r
            row.append(f"{stats.mean(r)*100:>5.0f}" if r else "  -  ")
        print(f"{enc+'/'+attn:>17} | {cm:>5.1f}±{cs:<4.1f} ({len(succ)}/{len(seeds)}) | "
              + " | ".join(f"{c:>9}" for c in row))
    print("-" * min(len(hdr), 160))
    # 2x2 marginal on retention at the primary level (blur 1.2), softmax row = encoder main effect
    print("\nPRIMARY (H1): encoder effect on RETENTION at blur σ=1.2, SOFTMAX row (paired):")
    a = rets[("ibnn", "softmax", "blur_1.2")]
    b = rets[("standard", "softmax", "blur_1.2")]
    pairs = [ai - bi for ai, bi in zip(a, b)]
    if pairs:
        m_ = stats.mean(pairs)
        # bootstrap 95% CI over paired diffs
        g = torch.Generator().manual_seed(0)
        boots = [stats.mean([pairs[int(i)] for i in torch.randint(0, len(pairs), (len(pairs),), generator=g)])
                 for _ in range(10000)]
        boots.sort(); lo, hi = boots[250], boots[9750]
        verdict = ("BENEFIT" if lo > 0 and m_ >= 0.03 else "no benefit (CI includes 0 or <3%)")
        print(f"  Δ_enc(retention) = {m_*100:+.2f}%   bootstrap 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]%"
              f"   n_pairs={len(pairs)}  ->  {verdict}")
    lam = stats.mean([done[("ibnn", "softmax", s)]["lambda"] for s in seeds])
    print(f"  mean |lambda| (ibnn/softmax) = {lam:.3f}  ({'engaged' if abs(lam+0.05) > 0.01 else 'INERT - coupling never moved'})")
    print(f"\ncollapsed cells: " + ", ".join(f"{e}/{a}={sum(done[(e,a,s)]['collapsed'] for s in seeds)}"
                                             for e, a in cells))


if __name__ == "__main__":
    main()
