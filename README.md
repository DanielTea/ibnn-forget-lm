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

_Populated by `make combo` — see `runs/combo_*.json`. (Filled in after the first run.)_

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
