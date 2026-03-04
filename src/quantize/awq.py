"""AWQ 4-bit quantization via AutoAWQ.

Exports quantized checkpoints to ``artifacts/awq_w4_g128/<variant>/``.

AutoAWQ does *not* cleanly support fine-grained selective quantization
(attn_only / mlp_only).  This module attempts it but falls back to
``full_quant`` only, logging a warning, if the tooling blocks it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from src.utils.calibration import get_calibration_data

logger = logging.getLogger(__name__)

_SUPPORTED_VARIANTS = {"full_quant", "attn_only_quant", "mlp_only_quant"}


def _build_calibration_texts(
    cfg: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> list[str]:
    """AWQ expects raw text strings for calibration, not tensors."""
    samples = get_calibration_data(cfg, tokenizer)
    texts: list[str] = []
    for s in samples:
        ids = s["input_ids"].squeeze(0)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def quantize_awq(
    cfg: dict[str, Any],
    variant: str = "full_quant",
) -> Path | None:
    """Quantize the base model with AWQ and save the artifact.

    Parameters
    ----------
    cfg : dict
        Project configuration.
    variant : str
        One of ``"full_quant"``, ``"attn_only_quant"``, ``"mlp_only_quant"``.
        If selective quantization is not supported by AutoAWQ, only
        ``full_quant`` is executed and ``None`` is returned for the others.

    Returns
    -------
    Path or None
        Path to the saved checkpoint, or ``None`` if the variant was skipped.
    """
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    awq_cfg = next(c for c in cfg["quant_configs"] if c["method"] == "awq")
    output_dir = Path(cfg["paths"]["artifacts_dir"]) / "awq_w4_g128" / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    if variant != "full_quant":
        logger.warning(
            "AutoAWQ does not cleanly support selective quantization. "
            "Skipping variant '%s'. Only full_quant is produced for AWQ.",
            variant,
        )
        return None

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )

    quant_config = {
        "zero_point": awq_cfg.get("zero_point", True),
        "q_group_size": awq_cfg["group_size"],
        "w_bit": awq_cfg["bits"],
        "version": awq_cfg.get("version", "GEMM"),
    }

    logger.info("Loading base model for AWQ quantization (%s)…", variant)
    model = AutoAWQForCausalLM.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
        safetensors=True,
    )

    calib_texts = _build_calibration_texts(cfg, tokenizer)
    logger.info(
        "Running AWQ with %d calibration samples (bits=%d, g=%d)…",
        len(calib_texts),
        awq_cfg["bits"],
        awq_cfg["group_size"],
    )
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    logger.info("Saving AWQ checkpoint to %s", output_dir)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return output_dir
