# Copyright 2026. Apache License 2.0.
#
# Minimal nanoGPT-style decoder LM. The only non-standard piece is the FFN: set
# ffn="ibnn" to use the IBNN neuron on the MLP hidden layer, or ffn="sm" for an
# ordinary transformer MLP (the Standard Model baseline). Everything else - token and
# positional embeddings, causal self-attention, residuals, LayerNorm, the LM head and the
# next-token cross-entropy objective - is untouched, which is the whole point: it isolates
# the effect of swapping the neuron model while holding parameter count essentially equal
# (IBNN adds exactly one scalar, lambda, per layer).

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import IBNNMLP


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = None            # defaults to 4 * d_model
    dropout: float = 0.0
    ffn: str = "ibnn"           # "ibnn" or "sm"
    attn: str = "softmax"       # "softmax" (standard) or "forget" (content-gated decay)
    # IBNN hyperparameters (ignored when ffn == "sm")
    ibnn_lambda: float = -0.05
    ibnn_lambda_trainable: bool = True
    ibnn_p: float = 10.0
    ibnn_num_iters: int = 1     # 1 == lite (forward only); >1 unrolls the implicit solve
    ibnn_chunk_size: int = 0    # >0 computes the O(D^2) lateral term in chunks to save memory
    ibnn_coupling: str = "meanfield"  # "meanfield" (paper) | "learned" (w_ik) | "topo" (learned topology)
    ibnn_interaction: str = "additive"  # "additive" (z=y-λL) | "gate" (v=φ(y)·2σ(λL))
    ibnn_topo_dim: int = 4            # channel-coordinate dim for coupling="topo"


class CausalSelfAttention(nn.Module):
    """Causal self-attention, optionally with a content-gated forget mechanism.

    attn="softmax": standard scaled dot-product attention.
    attn="forget":  each (head, position) emits a forget gate f_t = sigmoid(W_f x_t) in (0,1);
                    the attention from query t to key s<=t is multiplicatively decayed by the
                    product of forgets in between, prod_{r=s+1..t} f_r. Implemented additively in
                    log space via a cumulative sum, so it costs one extra small linear + a cumsum
                    and stays fully parallel. With f_r -> 1 the decay vanishes and this is EXACTLY
                    softmax attention, so it is a strict superset (it can always fall back to it).

    Motivation: an LSTM (which has a forget gate) beat plain attention on this char-LM, and the
    token axis is the one axis with sequential structure - so a learnable decay belongs here, on
    the keys, rather than inside the position-less FFN.

    Prior art: this is essentially the Forgetting Transformer / Forgetting Attention (FoX),
    Lin et al., ICLR 2025 (arXiv:2503.02130) - a data-dependent forget gate that decays the
    unnormalized attention scores via a cumulative-log-forget bias. The implementation here is an
    independent re-derivation from this repo's measurements, not a new design.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.forget = (cfg.attn == "forget")
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        if self.forget:
            # one forget logit per head per position; bias init positive so f~1 at start
            # (=> training begins as ordinary attention and only adds decay if it helps).
            self.fgate = nn.Linear(cfg.d_model, cfg.n_head)
            nn.init.zeros_(self.fgate.weight)
            nn.init.constant_(self.fgate.bias, 3.0)   # sigmoid(3)~0.95
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)   # (B, nh, T, T)
        if self.forget:
            # log f_r = log sigmoid(logit_r); cumulative C_t = sum_{r<=t} log f_r (non-increasing)
            logf = F.logsigmoid(self.fgate(x)).transpose(1, 2)   # (B, nh, T)
            cum = logf.cumsum(dim=-1)                             # (B, nh, T)
            decay = cum.unsqueeze(-1) - cum.unsqueeze(-2)         # (B,nh,T,T): [t,s]=C_t - C_s <=0
            att = att + decay                                    # additive log-space decay bias
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class StandardMLP(nn.Module):
    """The SM baseline FFN: Linear -> GELU -> Linear."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d_ff = cfg.d_ff or 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, d_ff)
        self.down = nn.Linear(d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.ffn == "ibnn":
            self.mlp = IBNNMLP(
                cfg.d_model, d_ff=cfg.d_ff, dropout=cfg.dropout,
                lam=cfg.ibnn_lambda, lam_trainable=cfg.ibnn_lambda_trainable,
                p=cfg.ibnn_p, num_iters=cfg.ibnn_num_iters, activation="gelu",
                chunk_size=cfg.ibnn_chunk_size, coupling=cfg.ibnn_coupling,
                interaction=cfg.ibnn_interaction, topo_dim=cfg.ibnn_topo_dim,
            )
        elif cfg.ffn == "sm":
            self.mlp = StandardMLP(cfg)
        else:
            raise ValueError(f"unknown ffn {cfg.ffn}")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        # GPT-2 style init: small normal everywhere, zeros on biases, and an extra 1/sqrt(2L)
        # shrink on the two residual-path output projections so the residual stream does not
        # blow up with depth. Without this the (weight-tied) head produces huge initial logits
        # and the first-step loss is enormous.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def copy_sm_weights_into_ibnn(sm_model: GPT, ibnn_model: GPT):
    """Warm-start an IBNN model from a trained SM model of identical shape.

    This is the LM analog of the paper's surrogate warmup: train a normal transformer, then
    transfer every matching parameter (embeddings, attention, layernorms, head, and the FFN
    up/down projections) into the IBNN model and switch lambda on. Only lambda has no source
    and keeps its init.
    """
    sm_sd = sm_model.state_dict()
    tgt_sd = ibnn_model.state_dict()
    copied, skipped = 0, []
    for k, v in tgt_sd.items():
        # FFN keys differ: IBNN up-projection lives at ...mlp.up.weight/.bias (IBNNLinear),
        # SM up lives at ...mlp.up.weight/.bias (nn.Linear) - same names, same shapes.
        if k in sm_sd and sm_sd[k].shape == v.shape:
            tgt_sd[k] = sm_sd[k].clone()
            copied += 1
        else:
            skipped.append(k)
    ibnn_model.load_state_dict(tgt_sd, strict=False)
    return copied, skipped
