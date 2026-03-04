"""Deployment-relevant benchmarks: tok/s, ms/tok, peak VRAM.

Uses vLLM for generation-based throughput measurement, ensuring kernel
realism for GPTQ / AWQ packed formats.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    tokens_per_sec: float
    ms_per_token: float
    peak_vram_gb: float


def benchmark_throughput(
    model_path: str,
    cfg: dict[str, Any],
    *,
    quantization: str | None = None,
) -> BenchmarkResult:
    """Measure generation throughput and peak VRAM.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint.
    cfg : dict
        Project configuration.
    quantization : str or None
        vLLM quantization flag.

    Returns
    -------
    BenchmarkResult
    """
    from vllm import LLM, SamplingParams

    bench_cfg = cfg["deployment_benchmark"]
    prompt_len = bench_cfg["prompt_length"]
    gen_len = bench_cfg["generation_length"]
    warmup = bench_cfg["warmup_iters"]
    iters = bench_cfg["bench_iters"]

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        quantization=quantization,
        dtype="auto",
        enforce_eager=True,
    )

    dummy_prompt = "Hello " * (prompt_len // 2)
    sampling = SamplingParams(temperature=0.0, max_tokens=gen_len)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        llm.generate([dummy_prompt], sampling)

    start = time.perf_counter()
    for _ in range(iters):
        outputs = llm.generate([dummy_prompt], sampling)
    elapsed = time.perf_counter() - start

    total_tokens = sum(
        len(o.outputs[0].token_ids)
        for o in outputs
    ) * iters
    tok_s = total_tokens / elapsed
    ms_tok = (elapsed / total_tokens) * 1000

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    result = BenchmarkResult(
        tokens_per_sec=tok_s,
        ms_per_token=ms_tok,
        peak_vram_gb=peak_vram,
    )
    logger.info(
        "Benchmark: %.1f tok/s, %.2f ms/tok, %.2f GB peak VRAM",
        tok_s,
        ms_tok,
        peak_vram,
    )
    return result
