# ibnn-forget-lm

Does the **IBNN neuron** (an implicit-bias lateral-coupling FFN) add anything on top of a
**forget gate in attention** — and what happens when you combine them? A small, fully-local,
controlled char-LM harness to find out.

This continues the investigation in [ibnn-lm](https://github.com/DanielTea/ibnn-lm), which
established two things at character-LM scale (with verified-correct implementations):

1. **Swapping the FFN neuron to IBNN does not help.** Across data-efficiency, tuning, the full
   implicit solve, a 13× scale-up, and a learned (non–mean-field) coupling, IBNN ties — or with
   extra parameters, loses to — a standard Transformer FFN.
2. **A forget gate in attention helps a lot.** Adding a content-gated multiplicative decay to
   softmax attention beat plain attention by **0.19–0.25 bits/char** at matched parameters. That
   mechanism is **not novel** — it is an independent re-derivation of the **Forgetting
   Transformer (FoX)**, Lin, He, Nikishin & Courville, ICLR 2025
   ([arXiv:2503.02130](https://arxiv.org/abs/2503.02130)) — a "learnable, data-dependent ALiBi."

The diagnosis was *structural*: a learnable-decay coupling is wasted on the FFN's **unordered
channels** but pays off on the **structured token axis** (attention's domain). This repo puts
both knobs in one model and runs the clean factorial.

## The experiment

A 2×2, identical everything except the two layers under test:

|            | softmax attention | forgetting attention |
|------------|-------------------|----------------------|
| **SM FFN**   | the plain Transformer | the FoX-style win |
| **IBNN FFN** | the IBNN study's tie  | **the open question** |

The question of interest is the bottom-right cell vs `sm+forget`: **once the forget gate is
doing the heavy lifting, does the IBNN neuron contribute anything?** Prior is "no" (IBNN tied
everywhere), but the forget gate changes the attention dynamics, so it's worth a direct test.

## Quick start (fully local, MPS/CUDA/CPU)

```bash
./setup.sh            # venv + torch (uses uv if available)
make sanity           # correctness checks (incl. λ=0 ≡ SM, forget ⊇ softmax)
make combo            # the 2×2 factorial, 3 seeds  (tinyshakespeare)
make combo-enwik8     # the same at larger scale on enwik8 byte-level
```

Train / generate a single combined model directly:

```bash
python -m ibnn_lm.train --dataset tinyshakespeare --ffn ibnn --attn forget --steps 2500 \
  --out checkpoints/ibnn_forget.pt
python -m ibnn_lm.generate --ckpt checkpoints/ibnn_forget.pt --prompt "ROMEO:" --stream
```

## Results

### The 2×2 factorial (`make combo`, tinyshakespeare, 3 seeds)

Exact held-out **bits-per-char** (lower is better):

|              | softmax attention | forgetting attention | forget effect |
|--------------|-------------------|----------------------|---------------|
| **SM FFN**   | 2.5245 ± 0.018    | **2.3357 ± 0.007**   | **−0.189**    |
| **IBNN FFN** | 2.5432 ± 0.013    | 2.3567 ± 0.017       | −0.187        |
| _IBNN effect_| _+0.019_          | _+0.021_             |               |

**The verdict is clean and a little brutal for IBNN:**
- **Forget gate = the entire win** (−0.19 bpc, essentially identical on both FFN rows).
- **IBNN ≈ 0** — in fact +0.02 bpc (slightly *worse*), within noise, under *both* attention types.
- **No interaction.** The effects are independent; IBNN adds nothing on top of the forget gate.

So combining them doesn't rescue IBNN: the best model is plain **`sm + forget`**. The lateral
neuron is null regardless of the attention it's paired with — consistent with the parent study's
finding that competition over the FFN's *unordered* channels has nothing to exploit.

### New IBNN-FFN ideas (`make ideas`)

_Bake-off of three new variants (below) running — table here once complete; raw numbers in
`runs/ideas_*.json`._

## Can IBNN be fixed? (new ideas, not in the literature)

The diagnosis says a fix must either give the channels *structure*, change competition from
*smoothing* to something useful, or move it to a meaningful axis. Three variants implemented here
(`ibnn_lm/ideas_test.py`, `make ideas`), each adding <1% params:

- **#1 `ibnn_gate` — competition-as-gate.** Instead of an additive nudge `z = y − λL`, use the
  lateral signal as a multiplicative gate: `v = φ(y) · 2σ(λL)`. Rides the one FFN trick that
  *does* help Transformers (GLU/SwiGLU); `λ=0` is bit-identical to a standard FFN.
- **#2 `ibnn_topo` — learned channel topology.** Give each hidden channel a learned coordinate
  `eᵢ`; set `w_ik = softmax(−‖eᵢ−e_k‖²/τ)`. The unordered channels self-organize into a learned
  geometry and the coupling becomes *local in that space* — the structured locality the CNN
  version exploits, but learned. A constrained middle-ground between mean-field and the (failed)
  full `D×D` learned coupling.
- **#3 `ibnn_sharpen` — sharpen, don't smooth.** The paper only uses `λ≤0` (homogenizing). With
  the lite layer there's no fixed point to destabilize, so flip to `λ>0`: soft winner-take-all
  that *sparsifies* activations instead of averaging them.

Honest prior: FFN-channel tricks have tied or lost in 8+ prior runs, so these are quick
falsification probes, not confident bets.

## Files

```
ibnn_lm/model.py         GPT; ffn="ibnn"|"sm", attn="softmax"|"forget"; GPT-2 init
ibnn_lm/layers.py        IBNNLinear (mean-field or learned coupling) + IBNNMLP
ibnn_lm/combo_test.py    the 2×2 factorial driver (FFN neuron × attention type)
ibnn_lm/attn_test.py     softmax vs forgetting attention (standard FFN)
ibnn_lm/train.py         training harness (cosine LR, early stop, checkpoints)
ibnn_lm/evaluate.py      deterministic held-out BPC / perplexity
ibnn_lm/generate.py      inference: prompt / stream / interactive REPL
ibnn_lm/data.py          corpus download/cache (tinyshakespeare, enwik8, …) + tokenizer
ibnn_lm/baselines.py     char-LSTM baseline at matched parameters
ibnn_lm/sanity.py        correctness checks (neuron math + both superset properties)
ibnn_lm/{compare,tune,coupling_test}.py   the parent study's IBNN experiments (kept)
Makefile, setup.sh       one-command setup / experiment workflow
```

## Credits & license

- **Forgetting attention** = the Forgetting Transformer (FoX), Lin et al., ICLR 2025,
  [arXiv:2503.02130](https://arxiv.org/abs/2503.02130). The implementation here is an
  independent re-derivation, not claimed as novel.
- **IBNN neuron**: Mohedano et al., *Updating the standard neuron model in artificial neural
  networks*, [arXiv:2605.30370](https://arxiv.org/abs/2605.30370) (2026); upstream CNN code
  [github.com/vmg-io-csic/ibnn](https://github.com/vmg-io-csic/ibnn).
- Related decay/forget lineage: ALiBi (Press et al. 2021), Mamba, GLA, HGRN, RetNet.

Apache 2.0. A research probe, not a validated recipe.
