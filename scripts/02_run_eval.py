#!/usr/bin/env python3
"""Step 2: Evaluate all artifacts.

For each artifact directory, run:
  - Perplexity (HuggingFace teacher-forcing)
  - HumanEval, GSM8K, MMLU (vLLM generation)
  - Throughput benchmark (tok/s, ms/tok, peak VRAM)
  - Actual checkpoint size (bytes/param)

Writes one JSONL row per artifact to ``results/eval.jsonl``.

Usage::

    python scripts/02_run_eval.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.checkpoint_size import bytes_per_param_from_safetensors
from src.eval.perplexity import evaluate_perplexity
from src.eval.humaneval import evaluate_humaneval
from src.eval.gsm8k import evaluate_gsm8k
from src.eval.mmlu import evaluate_mmlu
from src.eval.benchmark import benchmark_throughput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Maps artifact directory prefixes to vLLM quantization flags
_QUANT_FLAG = {
    "gptq": "gptq",
    "awq": "awq",
    "rtn": None,    # bitsandbytes-loaded or analytical
    "fp16": None,
}


def _discover_artifacts(cfg: dict) -> list[dict]:
    """Walk artifacts/ and yield (method, variant, path, quant_flag)."""
    base = Path(cfg["paths"]["artifacts_dir"])
    found: list[dict] = []

    fp16_dir = base / "fp16"
    if fp16_dir.exists() and any(fp16_dir.iterdir()):
        found.append({
            "method": "fp16",
            "variant": "baseline",
            "path": str(fp16_dir),
            "quant_flag": None,
        })

    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "fp16":
            continue
        method_key = method_dir.name.split("_")[0]
        quant_flag = _QUANT_FLAG.get(method_key)
        for variant_dir in sorted(method_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            if not any(variant_dir.iterdir()):
                continue
            found.append({
                "method": method_key,
                "variant": variant_dir.name,
                "path": str(variant_dir),
                "quant_flag": quant_flag,
            })
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all artifacts")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--skip-tasks", nargs="*", default=[],
        help="Tasks to skip (perplexity, humaneval, gsm8k, mmlu, benchmark)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = cfg["accuracy_weights"]
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval.jsonl"

    artifacts = _discover_artifacts(cfg)
    if not artifacts:
        logger.error("No artifacts found. Run 01_build_artifacts.py first.")
        return

    rows: list[dict] = []
    for art in artifacts:
        logger.info("=== Evaluating %s / %s ===", art["method"], art["variant"])
        row: dict = {
            "method": art["method"],
            "variant": art["variant"],
        }

        # Checkpoint size
        row["bytes_per_param_actual"] = bytes_per_param_from_safetensors(
            art["path"]
        )

        skip = set(args.skip_tasks)

        if "perplexity" not in skip:
            try:
                row["ppl"] = evaluate_perplexity(
                    art["path"], cfg, quantization=art["quant_flag"]
                )
            except Exception:
                logger.exception("Perplexity failed for %s", art["path"])
                row["ppl"] = None

        if "humaneval" not in skip:
            try:
                row["humaneval"] = evaluate_humaneval(
                    art["path"], cfg, quantization=art["quant_flag"]
                )
            except Exception:
                logger.exception("HumanEval failed for %s", art["path"])
                row["humaneval"] = None

        if "gsm8k" not in skip:
            try:
                row["gsm8k"] = evaluate_gsm8k(
                    art["path"], cfg, quantization=art["quant_flag"]
                )
            except Exception:
                logger.exception("GSM8K failed for %s", art["path"])
                row["gsm8k"] = None

        if "mmlu" not in skip:
            try:
                row["mmlu"] = evaluate_mmlu(
                    art["path"], cfg, quantization=art["quant_flag"]
                )
            except Exception:
                logger.exception("MMLU failed for %s", art["path"])
                row["mmlu"] = None

        # Composite score
        he = row.get("humaneval")
        gs = row.get("gsm8k")
        mm = row.get("mmlu")
        if he is not None and gs is not None and mm is not None:
            row["score"] = (
                weights["humaneval"] * he
                + weights["gsm8k"] * gs
                + weights["mmlu"] * mm
            )
        else:
            row["score"] = None

        # Deployment benchmark
        if "benchmark" not in skip:
            try:
                bench = benchmark_throughput(
                    art["path"], cfg, quantization=art["quant_flag"]
                )
                row["tok_s"] = bench.tokens_per_sec
                row["ms_per_token"] = bench.ms_per_token
                row["peak_vram_gb"] = bench.peak_vram_gb
            except Exception:
                logger.exception("Benchmark failed for %s", art["path"])

        rows.append(row)

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    logger.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
