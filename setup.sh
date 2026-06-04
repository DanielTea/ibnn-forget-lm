#!/usr/bin/env bash
# Create the local virtualenv for the IBNN-LM harness and install dependencies.
# Prefers `uv` (fast); falls back to the stdlib `venv` + `pip`. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [ ! -d .venv ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "creating venv with uv (python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" .venv
  else
    echo "uv not found; creating venv with python3 -m venv"
    python3 -m venv .venv
  fi
fi

echo "installing dependencies into .venv"
if command -v uv >/dev/null 2>&1; then
  VIRTUAL_ENV=.venv uv pip install -r requirements.txt
else
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

echo
echo "verifying torch + device:"
./.venv/bin/python - <<'PY'
import torch
dev = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")
print(f"  torch {torch.__version__}  ->  device: {dev}")
PY

echo
echo "setup complete. Next:"
echo "  make sanity        # verify the IBNN neuron math"
echo "  make train-ibnn    # train on tinyshakespeare"
echo "  make sample        # generate from the trained model"
