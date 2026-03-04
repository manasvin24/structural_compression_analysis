"""Programmatic allowlist / denylist builder for selective quantization.

Enumerates all ``nn.Linear`` layers via ``model.named_modules()`` and
classifies them by regex into *attn*, *mlp*, or *other*.  Then builds
the correct allowlist / denylist for each ablation variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import torch.nn as nn

ATTN_PATTERN = re.compile(r"self_attn\.(q|k|v|o)_proj")
MLP_PATTERN = re.compile(r"mlp\.(gate|up|down)_proj")


@dataclass
class ModuleClassification:
    attn: list[str] = field(default_factory=list)
    mlp: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)

    @property
    def all_linear(self) -> list[str]:
        return self.attn + self.mlp + self.other


def classify_linear_modules(model: nn.Module) -> ModuleClassification:
    """Walk the model and bucket every ``nn.Linear`` by family."""
    cls = ModuleClassification()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if ATTN_PATTERN.search(name):
            cls.attn.append(name)
        elif MLP_PATTERN.search(name):
            cls.mlp.append(name)
        else:
            cls.other.append(name)
    return cls


def build_denylist(
    classification: ModuleClassification,
    variant: str,
) -> list[str]:
    """Return module names that should **not** be quantized for *variant*.

    Parameters
    ----------
    variant : str
        One of ``"full_quant"``, ``"attn_only_quant"``, ``"mlp_only_quant"``.
    """
    if variant == "full_quant":
        return []
    if variant == "attn_only_quant":
        return classification.mlp + classification.other
    if variant == "mlp_only_quant":
        return classification.attn + classification.other
    raise ValueError(f"Unknown ablation variant: {variant}")


def build_allowlist(
    classification: ModuleClassification,
    variant: str,
) -> list[str]:
    """Return module names that **should** be quantized for *variant*."""
    if variant == "full_quant":
        return classification.all_linear
    if variant == "attn_only_quant":
        return classification.attn
    if variant == "mlp_only_quant":
        return classification.mlp
    raise ValueError(f"Unknown ablation variant: {variant}")


def filter_modules_for_variant(
    all_modules: Sequence[str],
    variant: str,
) -> tuple[list[str], list[str]]:
    """Convenience wrapper: returns ``(to_quantize, to_skip)``."""
    cls = ModuleClassification()
    for name in all_modules:
        if ATTN_PATTERN.search(name):
            cls.attn.append(name)
        elif MLP_PATTERN.search(name):
            cls.mlp.append(name)
        else:
            cls.other.append(name)
    return build_allowlist(cls, variant), build_denylist(cls, variant)
