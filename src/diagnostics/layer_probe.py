"""Layer-level diagnostics: activation stats, output MSE, and logit metrics.

Runs a fixed probe batch through both the FP16 reference and a quantized
model, collecting per-layer statistics via forward hooks.

Metrics
-------
* **Per-layer activation percentiles** (p50, p90, p99, p99.9, max)
* **Per-layer output MSE** between FP16 and quantized outputs
* **Logit-level metrics** (KL divergence, cosine similarity, top-k agreement)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.quantize.module_utils import ATTN_PATTERN, MLP_PATTERN
from src.utils.calibration import get_calibration_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    method: str
    variant: str
    layer_idx: int
    module_family: str
    module_name: str
    mse: float
    p50: float
    p90: float
    p99: float
    p99_9: float
    act_max: float


@dataclass
class LogitDiagnostics:
    method: str
    variant: str
    kl_div: float
    cosine_sim: float
    top1_agreement: float
    top5_agreement: float


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _module_family(name: str) -> str:
    if ATTN_PATTERN.search(name):
        return "attn"
    if MLP_PATTERN.search(name):
        return "mlp"
    return "other"


def _layer_index(name: str) -> int:
    """Try to extract ``.layers[i].`` index from the module name."""
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def _activation_percentiles(
    tensor: torch.Tensor,
) -> dict[str, float]:
    flat = tensor.float().abs().flatten()
    if flat.numel() == 0:
        return {"p50": 0, "p90": 0, "p99": 0, "p99_9": 0, "act_max": 0}
    quantiles = torch.quantile(
        flat, torch.tensor([0.50, 0.90, 0.99, 0.999], device=flat.device)
    )
    return {
        "p50": quantiles[0].item(),
        "p90": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "p99_9": quantiles[3].item(),
        "act_max": flat.max().item(),
    }


class _HookCollector:
    """Collects inputs and outputs of targeted ``nn.Linear`` modules."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, torch.Tensor]] = {}
        self._handles: list[Any] = []

    def register(self, model: nn.Module) -> None:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                fam = _module_family(name)
                if fam == "other":
                    continue
                handle = mod.register_forward_hook(
                    self._make_hook(name)
                )
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self.data[name] = {
                "input": x.detach().cpu(),
                "output": out.detach().cpu(),
            }
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.data.clear()


# ---------------------------------------------------------------------------
# Main probe routine
# ---------------------------------------------------------------------------

def _load_model(path: str, quantization: str | None = None):
    """Load a model for probing (HF path or checkpoint dir)."""
    load_kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if quantization == "gptq":
        from auto_gptq import AutoGPTQForCausalLM
        return AutoGPTQForCausalLM.from_quantized(path, **load_kwargs)
    if quantization == "awq":
        from awq import AutoAWQForCausalLM
        return AutoAWQForCausalLM.from_quantized(
            path, fuse_layers=False, trust_remote_code=True,
        )
    load_kwargs["torch_dtype"] = torch.float16
    return AutoModelForCausalLM.from_pretrained(path, **load_kwargs)


def run_layer_diagnostics(
    fp16_path: str,
    quant_path: str,
    cfg: dict[str, Any],
    method: str,
    variant: str,
    *,
    quantization: str | None = None,
    probe_seqs: int = 32,
    probe_len: int = 256,
    device: str = "cuda",
) -> tuple[list[LayerStats], LogitDiagnostics]:
    """Run diagnostics comparing FP16 vs quantized model.

    Returns
    -------
    tuple[list[LayerStats], LogitDiagnostics]
        Per-layer stats and aggregate logit-level metrics.
    """
    tokenizer = AutoTokenizer.from_pretrained(fp16_path, trust_remote_code=True)
    cal_samples = get_calibration_data(cfg, tokenizer)
    probe_batch = cal_samples[:probe_seqs]

    input_ids = torch.cat(
        [s["input_ids"][:, :probe_len] for s in probe_batch], dim=0
    ).to(device)

    # --- FP16 reference ---
    logger.info("Loading FP16 reference model…")
    fp16_model = _load_model(fp16_path)
    fp16_model.eval()

    fp16_hooks = _HookCollector()
    fp16_hooks.register(fp16_model)

    with torch.no_grad():
        fp16_out = fp16_model(input_ids)
    fp16_logits = fp16_out.logits.detach().cpu()
    fp16_layer_data = dict(fp16_hooks.data)
    fp16_hooks.remove()
    del fp16_model
    torch.cuda.empty_cache()

    # --- Quantized model ---
    logger.info("Loading quantized model (%s / %s)…", method, variant)
    q_model = _load_model(quant_path, quantization=quantization)
    q_model.eval()

    q_hooks = _HookCollector()
    q_hooks.register(q_model)

    with torch.no_grad():
        q_out = q_model(input_ids)
    q_logits = q_out.logits.detach().cpu()
    q_layer_data = dict(q_hooks.data)
    q_hooks.remove()
    del q_model
    torch.cuda.empty_cache()

    # --- Per-layer stats ---
    layer_stats: list[LayerStats] = []
    common_names = set(fp16_layer_data) & set(q_layer_data)
    for name in sorted(common_names):
        fp16_inp = fp16_layer_data[name]["input"]
        fp16_outl = fp16_layer_data[name]["output"]
        q_outl = q_layer_data[name]["output"]

        mse = F.mse_loss(q_outl.float(), fp16_outl.float()).item()
        act_stats = _activation_percentiles(fp16_inp)

        layer_stats.append(LayerStats(
            method=method,
            variant=variant,
            layer_idx=_layer_index(name),
            module_family=_module_family(name),
            module_name=name,
            mse=mse,
            **act_stats,
        ))

    # --- Logit-level metrics ---
    logit_diag = _compute_logit_diagnostics(
        fp16_logits, q_logits, method, variant
    )

    return layer_stats, logit_diag


def _compute_logit_diagnostics(
    fp16_logits: torch.Tensor,
    q_logits: torch.Tensor,
    method: str,
    variant: str,
) -> LogitDiagnostics:
    """KL divergence, cosine similarity, top-k agreement."""
    fp = fp16_logits.float().flatten(0, 1)
    qp = q_logits.float().flatten(0, 1)

    log_p = F.log_softmax(fp, dim=-1)
    log_q = F.log_softmax(qp, dim=-1)
    p = log_p.exp()
    kl = F.kl_div(log_q, p, reduction="batchmean", log_target=False).item()

    cos = F.cosine_similarity(fp, qp, dim=-1).mean().item()

    top1_fp = fp.argmax(dim=-1)
    top1_q = qp.argmax(dim=-1)
    top1_agree = (top1_fp == top1_q).float().mean().item()

    top5_fp = fp.topk(5, dim=-1).indices
    top5_q = qp.topk(5, dim=-1).indices
    top5_overlap = sum(
        len(set(a.tolist()) & set(b.tolist()))
        for a, b in zip(top5_fp, top5_q)
    ) / (top5_fp.size(0) * 5)

    return LogitDiagnostics(
        method=method,
        variant=variant,
        kl_div=kl,
        cosine_sim=cos,
        top1_agreement=top1_agree,
        top5_agreement=top5_overlap,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_layer_stats(
    stats: list[LayerStats],
    output_path: str | Path,
) -> None:
    df = pd.DataFrame([asdict(s) for s in stats])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    logger.info("Saved %d layer stats to %s", len(stats), output_path)


def save_logit_diagnostics(
    diags: list[LogitDiagnostics],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(d) for d in diags], f, indent=2)
    logger.info("Saved logit diagnostics to %s", output_path)
