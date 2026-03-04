"""GPTQ 4-bit quantization via AutoGPTQ with ablation support.

Exports quantized checkpoints to ``artifacts/gptq_w4_g128/<variant>/``.
Uses programmatic denylists from ``module_utils`` for selective quantization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

from src.quantize.module_utils import classify_linear_modules, build_denylist
from src.utils.calibration import get_calibration_data

logger = logging.getLogger(__name__)


def _prepare_calibration_examples(
    cfg: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> list[dict[str, torch.Tensor]]:
    """Return calibration data formatted for AutoGPTQ."""
    samples = get_calibration_data(cfg, tokenizer)
    examples = []
    for s in samples:
        examples.append({
            "input_ids": s["input_ids"].squeeze(0),
            "attention_mask": s["attention_mask"].squeeze(0),
        })
    return examples


def quantize_gptq(
    cfg: dict[str, Any],
    variant: str = "full_quant",
) -> Path:
    """Quantize the base model with GPTQ and save the artifact.

    Parameters
    ----------
    cfg : dict
        Project configuration (from ``config/default.yaml``).
    variant : str
        One of ``"full_quant"``, ``"attn_only_quant"``, ``"mlp_only_quant"``.

    Returns
    -------
    Path
        Directory containing the quantized checkpoint.
    """
    gptq_cfg = next(c for c in cfg["quant_configs"] if c["method"] == "gptq")
    output_dir = Path(cfg["paths"]["artifacts_dir"]) / "gptq_w4_g128" / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )

    quantize_config = BaseQuantizeConfig(
        bits=gptq_cfg["bits"],
        group_size=gptq_cfg["group_size"],
        desc_act=gptq_cfg.get("desc_act", False),
    )

    logger.info("Loading base model for GPTQ quantization (%s)…", variant)
    model = AutoGPTQForCausalLM.from_pretrained(
        cfg["base_model"],
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    classification = classify_linear_modules(model.model)
    denylist = build_denylist(classification, variant)

    if denylist:
        logger.info(
            "Variant %s: skipping %d modules from quantization",
            variant,
            len(denylist),
        )
        quantize_config.modules_to_not_convert = denylist

    examples = _prepare_calibration_examples(cfg, tokenizer)
    logger.info(
        "Running GPTQ with %d calibration samples (bits=%d, g=%d)…",
        len(examples),
        gptq_cfg["bits"],
        gptq_cfg["group_size"],
    )
    model.quantize(examples)

    logger.info("Saving GPTQ checkpoint to %s", output_dir)
    model.save_quantized(str(output_dir), use_safetensors=True)
    tokenizer.save_pretrained(str(output_dir))

    return output_dir
