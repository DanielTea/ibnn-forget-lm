# Copyright 2026. Apache License 2.0.
#
# Fast correctness checks for the IBNN neuron and its wiring into the GPT. These are not a
# claim about model quality; they verify the *math* matches the paper's neuron equation and
# that the implicit solve, gradients, and parameter accounting behave as documented.
#
#   python -m ibnn_lm.sanity
#
# Each check prints PASS/FAIL and the script exits non-zero if any fail.

import torch

from .layers import IBNNLinear, _lateral_term
from .model import GPT, GPTConfig
from .utils import count_params

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def lam_zero_reduces_to_standard():
    """With lambda=0 the lateral term vanishes => IBNN-lite == Linear + activation."""
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    lin = IBNNLinear(16, 32, lam=0.0, lam_trainable=False, num_iters=1, activation="gelu")
    y_ibnn = lin(x)
    y_std = torch.nn.functional.gelu(torch.nn.functional.linear(x, lin.weight, lin.bias))
    err = (y_ibnn - y_std).abs().max().item()
    return err < 1e-6, f"max|IBNN(lam=0) - (Linear+GELU)| = {err:.2e}"


@check
def lateral_term_matches_definition():
    """_lateral_term(z) must equal (1/D) * sum_k tanh(p*(z_k - z_i)), computed naively."""
    torch.manual_seed(1)
    z = torch.randn(3, 7)
    p = 10.0
    fast = _lateral_term(z, p)
    D = z.shape[-1]
    slow = torch.zeros_like(z)
    for b in range(z.shape[0]):
        for i in range(D):
            # definition: L_i = (1/D) * sum_k tanh(p * (z_k - z_i))
            slow[b, i] = torch.tanh(p * (z[b] - z[b, i])).mean()
    err = (fast - slow).abs().max().item()
    return err < 1e-5, f"max|fast - naive lateral| = {err:.2e}"


@check
def chunked_lateral_matches_full():
    """The memory-saving chunked path must give the same result as the single-shot path."""
    torch.manual_seed(7)
    z = torch.randn(2, 5, 64)
    full = _lateral_term(z, 10.0, chunk_size=0)
    for c in (1, 8, 16, 63, 64, 100):
        chunked = _lateral_term(z, 10.0, chunk_size=c)
        err = (full - chunked).abs().max().item()
        if err > 1e-6:
            return False, f"chunk_size={c}: max abs diff {err:.2e}"
    return True, "identical for chunk_size in {1,8,16,63,64,100}"


@check
def initial_loss_is_near_uniform():
    """Good init => first-step loss ~ ln(vocab), not the huge value default init produces."""
    import math as _m
    vocab = 65
    cfg = GPTConfig(vocab_size=vocab, block_size=32, n_layer=4, n_head=4, d_model=128,
                    d_ff=256, ffn="ibnn")
    model = GPT(cfg)
    x = torch.randint(0, vocab, (8, 32))
    _, loss = model(x, x)
    expected = _m.log(vocab)
    ok = loss.item() < expected + 1.0  # within ~1 nat of uniform
    return ok, f"loss={loss.item():.3f}  ln(vocab)={expected:.3f}"


@check
def fixed_point_residual_shrinks():
    """Unrolling more iterations must reduce the residual of z = y - lam*L(z)."""
    torch.manual_seed(2)
    x = torch.randn(8, 24)
    lin1 = IBNNLinear(24, 48, lam=-0.05, num_iters=1, tau=1.0, activation="identity")
    with torch.no_grad():
        y = torch.nn.functional.linear(x, lin1.weight, lin1.bias)

        def residual(num_iters):
            z = y.clone()
            for _ in range(num_iters):
                z = y - lin1.lam * _lateral_term(z, lin1.p)
            # residual of the *converged* equation at this z
            return (z - (y - lin1.lam * _lateral_term(z, lin1.p))).abs().max().item()

        r1, r5, r20 = residual(1), residual(5), residual(20)
    ok = r20 < r5 < r1 and r20 < 1e-4
    return ok, f"residual: iters1={r1:.2e} > iters5={r5:.2e} > iters20={r20:.2e}"


@check
def lambda_receives_gradient():
    """lambda must be a leaf parameter that gets a non-zero gradient from a real loss."""
    torch.manual_seed(3)
    x = torch.randn(4, 12)
    lin = IBNNLinear(12, 20, lam=-0.05, lam_trainable=True, num_iters=2, activation="gelu")
    out = lin(x).pow(2).mean()
    out.backward()
    g = lin.lam.grad
    ok = g is not None and torch.isfinite(g).all() and g.abs().item() > 0
    return ok, f"d(loss)/d(lambda) = {None if g is None else g.item():.3e}"


@check
def ibnn_adds_exactly_one_param_per_layer():
    """An IBNN GPT must have exactly +n_layer params vs its SM twin (one lambda per block)."""
    base = dict(vocab_size=50, block_size=32, n_layer=3, n_head=4, d_model=64, d_ff=128)
    sm = GPT(GPTConfig(ffn="sm", **base))
    ibnn = GPT(GPTConfig(ffn="ibnn", ibnn_lambda_trainable=True, **base))
    diff = count_params(ibnn) - count_params(sm)
    return diff == base["n_layer"], f"param diff = {diff} (expected {base['n_layer']})"


@check
def full_gpt_ibnn_lambda0_equals_sm():
    """Integration check: an IBNN GPT with lambda=0 must be bit-identical to its SM twin.
    This proves the IBNN wiring adds nothing except the lateral coupling (no spurious change),
    so any train/eval gap between ffn='sm' and ffn='ibnn' is attributable to the coupling alone.
    """
    from .model import copy_sm_weights_into_ibnn
    torch.manual_seed(0)
    base = dict(vocab_size=80, block_size=32, n_layer=3, n_head=4, d_model=128, d_ff=256)
    sm = GPT(GPTConfig(ffn="sm", **base)).eval()
    ibnn = GPT(GPTConfig(ffn="ibnn", ibnn_num_iters=1, **base)).eval()
    copy_sm_weights_into_ibnn(sm, ibnn)
    for m in ibnn.modules():
        if hasattr(m, "lam") and hasattr(m, "p"):
            m.lam.data.zero_()
    x = torch.randint(0, 80, (4, 32))
    with torch.no_grad():
        err = (sm(x)[0] - ibnn(x)[0]).abs().max().item()
    return err < 1e-6, f"max|logits_sm - logits_ibnn(lam=0)| = {err:.2e}"


@check
def forgetting_attention_superset_of_softmax():
    """Forgetting attention with gates forced open (f->1) must equal softmax attention exactly,
    so the forget mechanism is a strict, fair superset (it can always fall back to attention)."""
    torch.manual_seed(0)
    base = dict(vocab_size=80, block_size=48, n_layer=3, n_head=4, d_model=128, d_ff=256,
                ffn="sm")
    soft = GPT(GPTConfig(attn="softmax", **base)).eval()
    forg = GPT(GPTConfig(attn="forget", **base)).eval()
    forg.load_state_dict(soft.state_dict(), strict=False)   # copy shared weights
    for m in forg.modules():
        if hasattr(m, "fgate"):
            m.fgate.weight.data.zero_()
            m.fgate.bias.data.fill_(30.0)                   # sigmoid(30) ~ 1 -> no decay
    x = torch.randint(0, 80, (4, 48))
    with torch.no_grad():
        err = (soft(x)[0] - forg(x)[0]).abs().max().item()
    return err < 1e-6, f"max|softmax - forget(gates open)| = {err:.2e}"


@check
def forward_backward_runs_on_device():
    """A full GPT forward+backward must run and produce finite grads on the default device."""
    from .utils import get_device
    device = get_device("auto")
    cfg = GPTConfig(vocab_size=40, block_size=16, n_layer=2, n_head=4, d_model=64,
                    d_ff=128, ffn="ibnn", ibnn_num_iters=2)
    model = GPT(cfg).to(device)
    x = torch.randint(0, 40, (2, 16), device=device)
    y = torch.randint(0, 40, (2, 16), device=device)
    _, loss = model(x, y)
    loss.backward()
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return bool(torch.isfinite(loss)) and finite, f"loss={loss.item():.4f} device={device}"


def main():
    print("IBNN-LM sanity checks\n" + "=" * 40)
    failures = 0
    for fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"raised {type(e).__name__}: {e}"
        flag = "PASS" if ok else "FAIL"
        failures += not ok
        print(f"[{flag}] {fn.__name__:<38} {detail}")
    print("=" * 40)
    if failures:
        print(f"{failures} check(s) FAILED")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
