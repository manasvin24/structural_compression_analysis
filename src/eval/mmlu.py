"""MMLU (5-shot) evaluation via vLLM generation.

Formats each MMLU question as a multiple-choice prompt with few-shot
exemplars and checks whether the model predicts the correct letter.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_CHOICES = ["A", "B", "C", "D"]


def _format_example(row: dict, include_answer: bool = True) -> str:
    prompt = row["question"] + "\n"
    for i, choice in enumerate(_CHOICES):
        prompt += f"{choice}. {row['choices'][i]}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += f" {_CHOICES[row['answer']]}\n\n"
    return prompt


def _build_fewshot_prompt(
    subject_examples: list[dict],
    test_row: dict,
    n_shot: int,
) -> str:
    header = f"The following are multiple choice questions about {test_row.get('subject', 'general knowledge')}.\n\n"
    prompt = header
    for ex in subject_examples[:n_shot]:
        prompt += _format_example(ex, include_answer=True)
    prompt += _format_example(test_row, include_answer=False)
    return prompt


def evaluate_mmlu(
    model_path: str,
    cfg: dict[str, Any],
    *,
    quantization: str | None = None,
) -> float:
    """Return MMLU 5-shot accuracy.

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

    mmlu_cfg = cfg["eval"]["mmlu"]
    n_shot = mmlu_cfg.get("num_fewshot", 5)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        quantization=quantization,
        dtype="auto",
    )

    test_ds = load_dataset("cais/mmlu", "all", split="test")
    val_ds = load_dataset("cais/mmlu", "all", split="validation")

    val_by_subject: dict[str, list[dict]] = {}
    for row in val_ds:
        subj = row.get("subject", "other")
        val_by_subject.setdefault(subj, []).append(row)

    sampling = SamplingParams(temperature=0.0, max_tokens=1)

    prompts: list[str] = []
    gold_labels: list[int] = []
    for row in test_ds:
        subj = row.get("subject", "other")
        exemplars = val_by_subject.get(subj, list(val_ds)[:n_shot])
        prompts.append(_build_fewshot_prompt(exemplars, row, n_shot))
        gold_labels.append(row["answer"])

    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0
    for gold, output in zip(gold_labels, outputs):
        pred_text = output.outputs[0].text.strip().upper()
        pred_idx = _CHOICES.index(pred_text) if pred_text in _CHOICES else -1
        correct += int(pred_idx == gold)
        total += 1

    acc = correct / max(total, 1)
    logger.info("MMLU 5-shot accuracy: %.3f (%d/%d)", acc, correct, total)
    return acc
