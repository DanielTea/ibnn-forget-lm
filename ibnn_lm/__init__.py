# Copyright 2026. Apache License 2.0.
#
# A transformer/language-model fork of the IBNN neuron model from Mohedano et al.,
# "Updating the standard neuron model in artificial neural networks" (2026).

from .layers import IBNNLinear, IBNNMLP
from .model import GPT, GPTConfig, StandardMLP, copy_sm_weights_into_ibnn
from . import data
from .data import CharTokenizer
from .utils import get_device, set_seed, count_params

__all__ = [
    "IBNNLinear",
    "IBNNMLP",
    "GPT",
    "GPTConfig",
    "StandardMLP",
    "copy_sm_weights_into_ibnn",
    "data",
    "CharTokenizer",
    "get_device",
    "set_seed",
    "count_params",
]
