"""RTN 8-bit per-channel symmetric quantization with bitsandbytes fallback.

Path A (preferred): Use bitsandbytes ``load_in_8bit`` for a practical INT8
baseline with real INT8 GEMM kernels.

Path B (analytical): Custom symmetric per-channel RTN that stores INT8
weights + FP16 scales, but relies on dequantization at inference time.
Results from Path B should be labelled *analytical* and are not directly
comparable for latency / bandwidth numbers.

Exports checkpoints to ``artifacts/rtn_w8_perchannel/<variant>/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.quantize.module_utils import classify_linear_modules, build_denylist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path B helpers: analytical per-channel symmetric RTN
# ---------------------------------------------------------------------------

class RTNQuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` storing INT8 weights + FP16 scales.

    Forward pass dequantizes on the fly: ``y = x @ (W_int8 * scale).T + bias``.
    This is *analytical*; no optimized INT8 GEMM kernel is used.
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self.weight_int8.float() * self.scale.float().unsqueeze(1)
        out = torch.nn.functional.linear(x, w_deq.to(x.dtype), self.bias)
        return out


def _quantize_linear_rtn(linear: nn.Linear) -> RTNQuantizedLinear:
    """Per-output-channel symmetric INT8 quantization of a single Linear."""
    w = linear.weight.data.float()
    scale = w.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-8)
    w_int8 = (w / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return RTNQuantizedLinear(
        w_int8, scale.half(), linear.bias.data.clone() if linear.bias is not None else None
    )


def _apply_rtn_to_model(
    model: nn.Module,
    skip_names: list[str],
) -> nn.Module:
    """Replace selected ``nn.Linear`` layers with ``RTNQuantizedLinear``."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name in skip_names:
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], _quantize_linear_rtn(module))
    return model


# ---------------------------------------------------------------------------
# Path A: bitsandbytes INT8 loading
# ---------------------------------------------------------------------------

def _bnb_load(
    cfg: dict[str, Any],
    variant: str,
) -> tuple[nn.Module, AutoTokenizer]:
    """Load model with bitsandbytes 8-bit quantization."""
    from transformers import BitsAndBytesConfig

    classification_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",
    )
    classification = classify_linear_modules(classification_model)
    denylist = build_denylist(classification, variant)
    del classification_model
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    modules_to_skip = denylist if denylist else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        llm_int8_skip_modules=modules_to_skip,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_rtn(
    cfg: dict[str, Any],
    variant: str = "full_quant",
    use_bitsandbytes: bool = True,
) -> Path:
    """Quantize the base model to INT8 and save the artifact.

    Parameters
    ----------
    cfg : dict
        Project configuration.
    variant : str
        Ablation variant.
    use_bitsandbytes : bool
        If ``True`` (default / Path A), use bitsandbytes for real INT8 GEMM.
        If ``False`` (Path B), use analytical per-channel RTN.

    Returns
    -------
    Path
        Directory containing the saved checkpoint.
    """
    output_dir = Path(cfg["paths"]["artifacts_dir"]) / "rtn_w8_perchannel" / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )

    if use_bitsandbytes:
        logger.info(
            "RTN Path A (bitsandbytes): loading model in 8-bit (%s)…", variant
        )
        model, tokenizer = _bnb_load(cfg, variant)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    else:
        logger.info(
            "RTN Path B (analytical): per-channel symmetric INT8 (%s)…", variant
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
        )
        classification = classify_linear_modules(model)
        skip = build_denylist(classification, variant)
        model = _apply_rtn_to_model(model, skip)

        torch.save(model.state_dict(), str(output_dir / "model_rtn_int8.pt"))
        tokenizer.save_pretrained(str(output_dir))

        with open(output_dir / "rtn_config.txt", "w") as f:
            f.write(
                "ANALYTICAL RTN: per-channel symmetric INT8, clamp [-127,127].\n"
                "No optimized INT8 GEMM kernel. Latency/bandwidth numbers are "
                "NOT directly comparable to GPTQ/AWQ with packed kernels.\n"
            )

    logger.info("Saved RTN checkpoint to %s", output_dir)
    return output_dir
