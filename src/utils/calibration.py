"""Prepare and cache a calibration dataset for quantization."""

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_calibration_data(
    cfg: dict[str, Any],
    tokenizer: AutoTokenizer | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Return a list of tokenised calibration samples.

    Loads from cache when available; otherwise builds from HuggingFace
    ``datasets`` and persists to ``calibration_dir``.
    """
    cal_cfg = cfg["calibration"]
    cache_dir = Path(cfg["paths"]["calibration_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"calibration_{cal_cfg['num_samples']}.pt"

    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_model"], trust_remote_code=True
        )

    ds = load_dataset(
        cal_cfg["dataset"],
        cal_cfg["dataset_name"],
        split=cal_cfg["split"],
    )

    all_text = "\n\n".join(t for t in ds["text"] if t.strip())

    rng = torch.Generator().manual_seed(cal_cfg["seed"])
    max_len = cal_cfg["max_length"]
    num_samples = cal_cfg["num_samples"]

    encoded = tokenizer(all_text, return_tensors="pt")
    total_tokens = encoded.input_ids.shape[1]

    samples: list[dict[str, torch.Tensor]] = []
    starts = torch.randint(
        0, total_tokens - max_len, (num_samples,), generator=rng
    )
    for start in starts:
        end = start + max_len
        input_ids = encoded.input_ids[:, start:end]
        attention_mask = torch.ones_like(input_ids)
        samples.append(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

    torch.save(samples, cache_path)
    return samples
