# Copyright 2026. Apache License 2.0.
#
# Small shared helpers for the harness: device selection and reproducibility.

import os

import torch


def get_device(prefer: str = "auto") -> str:
    """Pick the best available torch device.

    prefer: "auto" (mps > cuda > cpu), or force one of "mps" | "cuda" | "cpu".
    On Apple Silicon this selects the MPS (Metal) GPU. We also enable the CPU fallback for
    the handful of ops MPS does not yet implement, so a run never hard-crashes on a missing
    kernel - it silently runs that op on the CPU instead.
    """
    if prefer != "cpu":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
