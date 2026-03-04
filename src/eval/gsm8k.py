"""GSM8K evaluation via vLLM generation.

Generates chain-of-thought answers and extracts the final numeric answer,
comparing against the ground truth.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FEWSHOT_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: Let's think step by step.\n"
)

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def evaluate_gsm8k(
    model_path: str,
    cfg: dict[str, Any],
    *,
    quantization: str | None = None,
) -> float:
    """Return GSM8K accuracy.

    Parameters
    ----------
    model_path : str
        Path to model checkpoint.
    cfg : dict
        Project configuration.
    quantization : str or None
        vLLM quantization flag.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    gsm_cfg = cfg["eval"]["gsm8k"]

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        quantization=quantization,
        dtype="auto",
    )

    ds = load_dataset("openai/gsm8k", "main", split="test")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    prompts: list[str] = []
    for row in ds:
        prompts.append(
            _FEWSHOT_PROMPT_TEMPLATE.format(question=row["question"])
        )

    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0
    for row, output in zip(ds, outputs):
        pred_text = output.outputs[0].text
        pred_answer = _extract_answer(pred_text)
        gold_answer = _extract_answer(row["answer"])
        if pred_answer is not None and gold_answer is not None:
            if abs(pred_answer - gold_answer) < 1e-3:
                correct += 1
        total += 1

    acc = correct / max(total, 1)
    logger.info("GSM8K accuracy: %.3f (%d/%d)", acc, correct, total)
    return acc


def _extract_answer(text: str) -> float | None:
    """Pull the numeric answer from ``#### <number>`` or the last number."""
    m = _ANSWER_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None
