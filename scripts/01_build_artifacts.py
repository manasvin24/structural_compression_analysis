#!/usr/bin/env python3
"""Step 1: Build all quantized artifacts.

For each (method x variant), quantize the base model and export to
``artifacts/<method>/<variant>/``.

Usage::

    python scripts/01_build_artifacts.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.quantize.gptq import quantize_gptq
from src.quantize.awq import quantize_awq
from src.quantize.rtn import quantize_rtn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _save_fp16_reference(cfg: dict) -> None:
    """Symlink or copy base model into artifacts/fp16/ for clarity."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    fp16_dir = Path(cfg["paths"]["artifacts_dir"]) / "fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)

    if (fp16_dir / "config.json").exists():
        logger.info("FP16 reference already exists at %s", fp16_dir)
        return

    logger.info("Saving FP16 reference to %s", fp16_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float16, trust_remote_code=True,
    )
    model.save_pretrained(str(fp16_dir))
    tokenizer.save_pretrained(str(fp16_dir))
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quantized artifacts")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    variants = cfg["ablation_variants"]

    _save_fp16_reference(cfg)

    for qcfg in cfg["quant_configs"]:
        method = qcfg["method"]
        for variant in variants:
            logger.info("=== %s / %s ===", method, variant)
            try:
                if method == "gptq":
                    quantize_gptq(cfg, variant=variant)
                elif method == "awq":
                    result = quantize_awq(cfg, variant=variant)
                    if result is None:
                        logger.warning(
                            "AWQ skipped variant %s (tooling limitation)", variant
                        )
                elif method == "rtn":
                    quantize_rtn(cfg, variant=variant, use_bitsandbytes=True)
                else:
                    logger.error("Unknown method: %s", method)
            except Exception:
                logger.exception("FAILED: %s / %s", method, variant)

    logger.info("Artifact build complete.")


if __name__ == "__main__":
    main()
