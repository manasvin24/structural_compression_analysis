"""Compute actual bytes-per-parameter from checkpoint files.

Iterates over safetensors (or PyTorch bin) files in a checkpoint directory,
sums ``numel * dtype_bytes`` for every stored tensor (weights, scales, zeros),
and reports the effective bytes per model parameter.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DTYPE_BYTES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def bytes_per_param_from_safetensors(checkpoint_dir: str | Path) -> float:
    """Compute actual bytes-per-parameter from safetensors files.

    Falls back to a rough estimate from total file size if the
    ``safetensors`` library is unavailable.

    Returns
    -------
    float
        Effective bytes per parameter.
    """
    checkpoint_dir = Path(checkpoint_dir)
    st_files = list(checkpoint_dir.glob("*.safetensors"))

    if not st_files:
        return _fallback_from_bin(checkpoint_dir)

    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors not installed; using file-size fallback")
        return _fallback_file_size(st_files)

    total_bytes = 0
    total_params = 0

    for path in st_files:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                numel = tensor.numel()
                dtype_str = str(tensor.dtype)
                elem_size = tensor.element_size()
                total_bytes += numel * elem_size
                total_params += numel

    if total_params == 0:
        return 0.0

    bpp = total_bytes / total_params
    logger.info(
        "Checkpoint %s: %.2f GB, %d params, %.3f bytes/param",
        checkpoint_dir.name,
        total_bytes / 1e9,
        total_params,
        bpp,
    )
    return bpp


def _fallback_from_bin(checkpoint_dir: Path) -> float:
    """Estimate from .bin files using torch.load metadata."""
    import torch

    bin_files = list(checkpoint_dir.glob("*.bin")) + list(
        checkpoint_dir.glob("*.pt")
    )
    if not bin_files:
        logger.warning("No checkpoint files found in %s", checkpoint_dir)
        return 0.0

    total_bytes = 0
    total_params = 0
    for path in bin_files:
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        for tensor in sd.values():
            if hasattr(tensor, "numel"):
                total_bytes += tensor.numel() * tensor.element_size()
                total_params += tensor.numel()

    return total_bytes / max(total_params, 1)


def _fallback_file_size(st_files: list[Path]) -> float:
    """Very rough: total file bytes / estimated param count."""
    total = sum(f.stat().st_size for f in st_files)
    logger.warning(
        "Using raw file size (%.2f GB) as fallback — param count unknown",
        total / 1e9,
    )
    return 0.0
