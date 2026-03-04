#!/usr/bin/env python3
"""Step 3: Run layer diagnostics (FP16 vs quantized).

For each quantized artifact, collects per-layer activation percentiles,
output MSE, and logit-level metrics (KL divergence, cosine similarity,
top-k agreement).

Outputs:
  results/layer_stats.parquet
  results/logit_diagnostics.json

Usage::

    python scripts/03_layer_diagnostics.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.diagnostics.layer_probe import (
    run_layer_diagnostics,
    save_layer_stats,
    save_logit_diagnostics,
    LayerStats,
    LogitDiagnostics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_QUANT_FLAG = {
    "gptq": "gptq",
    "awq": "awq",
    "rtn": None,
}


def _discover_artifacts(cfg: dict) -> list[dict]:
    base = Path(cfg["paths"]["artifacts_dir"])
    found: list[dict] = []
    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "fp16":
            continue
        method_key = method_dir.name.split("_")[0]
        for variant_dir in sorted(method_dir.iterdir()):
            if not variant_dir.is_dir() or not any(variant_dir.iterdir()):
                continue
            found.append({
                "method": method_key,
                "variant": variant_dir.name,
                "path": str(variant_dir),
                "quant_flag": _QUANT_FLAG.get(method_key),
            })
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer diagnostics")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--variant-filter", type=str, default=None,
        help="Only run diagnostics for this variant (e.g. full_quant).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fp16_path = str(Path(cfg["paths"]["artifacts_dir"]) / "fp16")
    results_dir = Path(cfg["paths"]["results_dir"])

    artifacts = _discover_artifacts(cfg)
    if not artifacts:
        logger.error("No artifacts found. Run 01_build_artifacts.py first.")
        return

    all_layer_stats: list[LayerStats] = []
    all_logit_diags: list[LogitDiagnostics] = []

    for art in artifacts:
        if args.variant_filter and art["variant"] != args.variant_filter:
            continue
        logger.info(
            "=== Diagnostics: %s / %s ===", art["method"], art["variant"]
        )
        try:
            layer_stats, logit_diag = run_layer_diagnostics(
                fp16_path=fp16_path,
                quant_path=art["path"],
                cfg=cfg,
                method=art["method"],
                variant=art["variant"],
                quantization=art["quant_flag"],
            )
            all_layer_stats.extend(layer_stats)
            all_logit_diags.append(logit_diag)
        except Exception:
            logger.exception(
                "Diagnostics failed for %s / %s", art["method"], art["variant"]
            )

    if all_layer_stats:
        save_layer_stats(all_layer_stats, results_dir / "layer_stats.parquet")
    if all_logit_diags:
        save_logit_diagnostics(
            all_logit_diags, results_dir / "logit_diagnostics.json"
        )

    logger.info("Layer diagnostics complete.")


if __name__ == "__main__":
    main()
