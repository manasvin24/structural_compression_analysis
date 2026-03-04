"""HumanEval (pass@1) evaluation via vLLM generation.

Uses the ``openai/humaneval`` dataset (or the ``human-eval`` package).
Generates code completions with vLLM and evaluates functional correctness
by executing the generated code against unit tests.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_humaneval(
    model_path: str,
    cfg: dict[str, Any],
    *,
    quantization: str | None = None,
) -> float:
    """Return HumanEval pass@1 score.

    Parameters
    ----------
    model_path : str
        Path to quantized (or FP16) checkpoint.
    cfg : dict
        Project configuration.
    quantization : str or None
        vLLM ``quantization`` argument (``"gptq"``, ``"awq"``, etc.).

    Returns
    -------
    float
        pass@1 in [0, 1].
    """
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    he_cfg = cfg["eval"]["humaneval"]

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        quantization=quantization,
        dtype="auto",
    )

    ds = load_dataset("openai/openai_humaneval", split="test")

    sampling = SamplingParams(
        temperature=he_cfg.get("temperature", 0.0),
        max_tokens=512,
        n=he_cfg.get("num_samples_per_task", 1),
    )

    prompts = [row["prompt"] for row in ds]
    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0
    for row, output in zip(ds, outputs):
        generated = output.outputs[0].text
        # truncate after the first standalone return / end of function
        generated = _truncate_completion(generated)
        full_code = row["prompt"] + generated + "\n" + row["test"] + "\n"
        full_code += f"\ncheck({row['entry_point']})\n"
        passed = _execute_code(full_code)
        correct += int(passed)
        total += 1

    score = correct / max(total, 1)
    logger.info("HumanEval pass@1: %.3f (%d/%d)", score, correct, total)
    return score


def _truncate_completion(code: str) -> str:
    """Stop generation at the likely end of the function body."""
    lines = code.split("\n")
    result: list[str] = []
    for line in lines:
        if result and line.strip() != "" and not line.startswith(" ") and not line.startswith("\t"):
            break
        result.append(line)
    return "\n".join(result)


def _execute_code(code: str, timeout: int = 10) -> bool:
    """Run *code* in a subprocess and return True if exit code is 0."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        f.write(code)
        f.flush()
        try:
            subprocess.run(
                ["python", f.name],
                timeout=timeout,
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
