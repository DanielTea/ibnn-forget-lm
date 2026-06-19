# ibnn-forget-lm — local harness for the FFN-neuron × attention-type factorial.
# `make help` lists targets. All use the project-local ./.venv (created by `make setup`).
#   override knobs on the CLI, e.g.  make combo STEPS=2500 DATASET=tinyshakespeare

PY := .venv/bin/python
DATASET ?= tinyshakespeare
STEPS ?= 1500
CKPT ?= checkpoints/best.pt
PROMPT ?= "\n"

.DEFAULT_GOAL := help

.PHONY: help setup sanity combo combo-enwik8 attn-test train sample chat clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install torch + deps (uv if present)
	@bash setup.sh

sanity: ## Run the correctness checks (neuron math, λ=0≡SM, forget⊇softmax)
	$(PY) -m ibnn_lm.sanity

combo: ## THE experiment: 2×2 factorial {sm,ibnn} FFN × {softmax,forget} attn, 3 seeds
	$(PY) -m ibnn_lm.combo_test --dataset $(DATASET) --seeds 0 1 2 --steps $(STEPS)

combo-enwik8: ## The 2×2 factorial on enwik8 byte-level at larger scale (single/few seed)
	$(PY) -m ibnn_lm.combo_test --dataset enwik8 --byte_level --max_mb 25 --seeds 0 1 \
		--d_model 256 --d_ff 256 --n_layer 6 --n_head 8 --block_size 256 --batch_size 16 \
		--steps 2500

attn-test: ## Just softmax vs forgetting attention (standard FFN), 3 seeds
	$(PY) -m ibnn_lm.attn_test --dataset $(DATASET) --seeds 0 1 2 --steps $(STEPS)

ideas: ## Bake-off of NEW IBNN-FFN variants (gate / topology / sharpen) vs sm + plain IBNN
	$(PY) -m ibnn_lm.ideas_test --dataset $(DATASET) --seeds 0 1 2 --steps $(STEPS)

robustness: ## Does forgetting attention survive input-noise robustness + the memorization gap?
	$(PY) -m ibnn_lm.robustness --dataset $(DATASET) --seeds 0 1 2 --steps 2000

vlm: ## Train a toy Vision-Language Model on Fashion-MNIST with the IBNN neuron
	$(PY) -m ibnn_lm.vlm --ffn ibnn --steps 800

cnn: ## The real test: the paper's SPATIAL IBNN conv vs a standard CNN (+ data-efficiency)
	$(PY) -m ibnn_lm.ibnn_cnn --seeds 0 1 2 3 4 5 --train_fracs 1.0 0.05 0.02

adv: ## The paper's HEADLINE claim: deeper spatial-IBNN CNN vs standard CNN under FGSM/PGD
	$(PY) -m ibnn_lm.adv_robustness --seeds 0 1 2 3 4 5 6 7 --steps 1500

train: ## Train one model (override FFN=ibnn ATTN=forget etc. via env if extended)
	$(PY) -m ibnn_lm.train --dataset $(DATASET) --ffn ibnn --attn forget \
		--d_model 128 --d_ff 256 --n_layer 3 --n_head 4 --block_size 128 \
		--steps $(STEPS) --eval_interval 250 --sample_interval 500 --out $(CKPT)

sample: ## Generate from a checkpoint (override CKPT=, PROMPT=)
	$(PY) -m ibnn_lm.generate --ckpt $(CKPT) --prompt $(PROMPT) --stream --max_new_tokens 500

chat: ## Interactive generation REPL
	$(PY) -m ibnn_lm.generate --ckpt $(CKPT) --interactive

clean: ## Remove checkpoints and run logs (keeps data and venv)
	rm -rf checkpoints runs runs_*.log

vlm-robust: ## VLM robustness factorial: encoder coupling x decoder forget gate (8 seeds, resumable)
	$(PY) -m ibnn_lm.vlm_robust --seeds 0 1 2 3 4 5 6 7
