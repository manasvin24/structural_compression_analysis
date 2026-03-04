"""Perplexity evaluation via HuggingFace teacher-forcing.

vLLM is not suitable for perplexity because it is optimised for generation,
not for computing per-token cross-entropy loss.  We therefore use the
``transformers`` forward pass with ``labels=input_ids``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model_path: str,
    cfg: dict[str, Any],
    *,
    device: str = "cuda",
    quantization: str | None = None,
) -> float:
    """Compute perplexity on the WikiText-2 test split.

    Parameters
    ----------
    model_path : str
        Path (or HF hub id) of the model / quantized checkpoint.
    cfg : dict
        Project configuration.
    device : str
        Target device for inference.
    quantization : str or None
        If ``"gptq"`` or ``"awq"``, loads with the appropriate library.

    Returns
    -------
    float
        Perplexity value.
    """
    ppl_cfg = cfg["eval"]["perplexity"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs: dict[str, Any] = dict(trust_remote_code=True)
    if quantization == "gptq":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            model_path, device_map="auto", trust_remote_code=True
        )
    elif quantization == "awq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_path, fuse_layers=False, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto",
            **load_kwargs,
        )

    model.eval()

    ds = load_dataset(
        ppl_cfg["dataset"],
        ppl_cfg["dataset_name"],
        split=ppl_cfg["split"],
    )
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    encodings = tokenizer(text, return_tensors="pt")

    max_len = ppl_cfg["max_length"]
    stride = ppl_cfg["stride"]
    seq_len = encodings.input_ids.size(1)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        # mask tokens already evaluated so we don't double-count
        target_ids[:, : max(0, prev_end - begin)] = -100

        with torch.no_grad():
            out = model(input_ids, labels=target_ids)
            neg_log_likelihood = out.loss

        nlls.append(neg_log_likelihood)
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(torch.stack(nlls).mean().item())
    logger.info("Perplexity: %.2f", ppl)
    return ppl
