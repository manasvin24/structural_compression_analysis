# Quantization Comparative Study

Comparative study of **GPTQ (4-bit W4A16)**, **AWQ (4-bit W4A16)**, and **RTN (8-bit W8A16)** weight-only quantization on **Qwen2-7B**, with:

- Module-level ablations: MLP projections vs attention projections
- Layer diagnostics: activation outlier stats, output MSE, logit KL / cosine / top-k agreement
- Accuracy-vs-bandwidth trade-off visualization (actual checkpoint sizes, not estimates)
- Deployment metrics: tokens/sec, ms/token, peak VRAM

---

## Table of Contents

1. [Where to Run This](#1-where-to-run-this)
2. [Environment Setup](#2-environment-setup)
3. [Configuration](#3-configuration)
4. [Step-by-Step Execution](#4-step-by-step-execution)
   - [Step 0 — Verify Setup](#step-0--verify-setup)
   - [Step 1 — Build Artifacts](#step-1--build-artifacts)
   - [Step 2 — Evaluate](#step-2--evaluate)
   - [Step 3 — Layer Diagnostics](#step-3--layer-diagnostics)
   - [Step 4 — Generate Plots](#step-4--generate-plots)
5. [Output Files](#5-output-files)
6. [Project Structure](#6-project-structure)
7. [Quantization Methods](#7-quantization-methods)
8. [Evaluation Details](#8-evaluation-details)
9. [Diagnostics Details](#9-diagnostics-details)
10. [Plots Reference](#10-plots-reference)
11. [Common Issues](#11-common-issues)

---

## 1. Where to Run This

**This project requires a cloud GPU instance.** All four scripts are designed to run on a single server, not on a laptop or CPU-only machine.

### Recommended hardware

| Configuration | Minimum | Recommended |
|--------------|---------|-------------|
| GPU | NVIDIA A100 40GB | NVIDIA A100 80GB |
| GPU count | 1 | 1 |
| CUDA | 11.8+ | 12.1+ |
| RAM | 64 GB | 128 GB |
| Disk | 200 GB free | 400 GB free |

> **Why A100?** GPTQ and AWQ both require Turing+ (SM 7.5+). The FP16 base model alone is ~14 GB. With all artifacts (7 variants × ~4–14 GB each) and calibration data, you need ~200–400 GB of disk.

### Suitable cloud platforms

- **Google Cloud**: `a2-highgpu-1g` (A100 40GB) or `a2-ultragpu-1g` (A100 80GB)
- **AWS**: `p4d.24xlarge` (8× A100) or `p3.2xlarge` (V100, GPTQ only — AWQ requires Turing+)
- **Lambda Labs**: A100 instance
- **RunPod / CoreWeave**: Single A100 pod

### HuggingFace token requirement

`Qwen/Qwen2-7B` is a gated model. You must accept the license on HuggingFace and authenticate:

```bash
huggingface-cli login
# paste your HF token when prompted
```

Or set the environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

---

## 2. Environment Setup

All commands below should be run from the **project root directory** (`227_Project/`), unless stated otherwise.

### 2.1 Clone / copy the project

```bash
cd ~
git clone <your-repo-url> 227_Project   # or copy the directory
cd 227_Project
```

### 2.2 Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `vllm` is the heaviest dependency and installs its own CUDA kernels. This step can take 10–20 minutes. If you are on CUDA 12.x:

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2.4 Verify GPU access

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory // 1e9, 'GB')"
```

Expected output (A100):

```
NVIDIA A100-SXM4-80GB
80.0 GB
```

---

## 3. Configuration

All parameters are controlled by a single YAML file: `config/default.yaml`.

```yaml
base_model: "Qwen/Qwen2-7B"      # HF model ID or local path

calibration:
  dataset: "wikitext"
  dataset_name: "wikitext-2-raw-v1"
  num_samples: 512                 # increase to 2048 for higher quality GPTQ
  max_length: 2048
  seed: 42                         # fixed seed — keep consistent across runs

quant_configs:
  - method: gptq
    bits: 4
    group_size: 128
    desc_act: false
  - method: awq
    bits: 4
    group_size: 128
  - method: rtn
    bits: 8
    per_channel: true

ablation_variants:
  - full_quant
  - attn_only_quant
  - mlp_only_quant

accuracy_weights:
  humaneval: 0.4
  gsm8k: 0.3
  mmlu: 0.3

deployment_benchmark:
  prompt_length: 128
  generation_length: 128
  batch_size: 1
  warmup_iters: 3
  bench_iters: 10

paths:
  artifacts_dir: "artifacts"       # where quantized checkpoints are saved
  results_dir: "results"           # where eval.jsonl and parquet are saved
  calibration_dir: "calibration_data"
  plots_dir: "results/plots"
```

### Using a local model checkpoint

If you have already downloaded Qwen2-7B locally (to avoid re-downloading):

```bash
# Download once
huggingface-cli download Qwen/Qwen2-7B --local-dir ./models/Qwen2-7B
```

Then update `config/default.yaml`:

```yaml
base_model: "./models/Qwen2-7B"
```

### Using a custom config file

All scripts accept `--config`:

```bash
python scripts/01_build_artifacts.py --config config/my_custom.yaml
```

---

## 4. Step-by-Step Execution

Run **from the project root** (`227_Project/`). Steps must be run in order.

---

### Step 0 — Verify Setup

Check that Python can find all modules and the GPU is visible:

```bash
python -c "
from src.utils.config import load_config
cfg = load_config()
print('Base model:', cfg['base_model'])
print('Variants:', cfg['ablation_variants'])
"
```

Expected output:

```
Base model: Qwen/Qwen2-7B
Variants: ['full_quant', 'attn_only_quant', 'mlp_only_quant']
```

---

### Step 1 — Build Artifacts

**Estimated time: 3–6 hours** (7 checkpoints × 20–50 min each)

```bash
python scripts/01_build_artifacts.py
```

What this does:

1. Downloads Qwen2-7B base weights (FP16) and saves to `artifacts/fp16/`
2. For each method (`gptq`, `awq`, `rtn`) × each variant (`full_quant`, `attn_only_quant`, `mlp_only_quant`):
   - Loads the base model
   - Builds a calibration corpus (512 samples from WikiText-2, cached to `calibration_data/`)
   - Quantizes with the appropriate library
   - Exports the checkpoint to `artifacts/<method>_<config>/<variant>/`

> **AWQ note:** AutoAWQ does not support selective quantization. `attn_only_quant` and `mlp_only_quant` variants will be **skipped** for AWQ with a warning logged. Only `awq_w4_g128/full_quant/` is produced.

> **RTN note:** By default, `use_bitsandbytes=True` (Path A). This uses bitsandbytes `load_in_8bit` for real INT8 GEMM kernels. If bitsandbytes is unavailable, edit `scripts/01_build_artifacts.py` and set `use_bitsandbytes=False` for Path B (analytical, no optimized kernels — latency numbers will not be kernel-realistic).

**Artifacts directory after Step 1:**

```
artifacts/
├── fp16/
├── gptq_w4_g128/
│   ├── full_quant/
│   ├── attn_only_quant/
│   └── mlp_only_quant/
├── awq_w4_g128/
│   └── full_quant/             # only this variant for AWQ
└── rtn_w8_perchannel/
    ├── full_quant/
    ├── attn_only_quant/
    └── mlp_only_quant/
```

**To run only one method or variant** (useful for debugging or partial runs):

```bash
# Edit scripts/01_build_artifacts.py temporarily, or call the library directly:
python -c "
from src.utils.config import load_config
from src.quantize.gptq import quantize_gptq
cfg = load_config()
quantize_gptq(cfg, variant='full_quant')
"
```

---

### Step 2 — Evaluate

**Estimated time: 4–10 hours** (depends on number of artifacts and GPU speed)

```bash
python scripts/02_run_eval.py
```

What this does for **each artifact**:

1. **Perplexity** — HuggingFace forward pass on WikiText-2 test split (teacher-forcing, not vLLM)
2. **HumanEval** — vLLM generation, pass@1 evaluated by executing the code
3. **GSM8K** — vLLM generation, accuracy on test set
4. **MMLU** — vLLM 5-shot, single-token prediction
5. **Deployment benchmark** — tok/s, ms/token, peak VRAM via vLLM
6. **Checkpoint size** — actual bytes/param from safetensors metadata

Results are appended to `results/eval.jsonl` (one JSON line per artifact).

#### Skip individual tasks

Use `--skip-tasks` to skip costly tasks during debugging:

```bash
# Skip HumanEval and the benchmark (fastest run for checking correctness)
python scripts/02_run_eval.py --skip-tasks humaneval benchmark

# Run only perplexity and GSM8K
python scripts/02_run_eval.py --skip-tasks humaneval mmlu benchmark
```

Available task names: `perplexity`, `humaneval`, `gsm8k`, `mmlu`, `benchmark`

#### Example output row in `results/eval.jsonl`

```json
{
  "method": "gptq",
  "variant": "full_quant",
  "bytes_per_param_actual": 0.52,
  "ppl": 8.74,
  "humaneval": 0.41,
  "gsm8k": 0.49,
  "mmlu": 0.62,
  "score": 0.511,
  "tok_s": 82.3,
  "ms_per_token": 12.1,
  "peak_vram_gb": 5.8
}
```

---

### Step 3 — Layer Diagnostics

**Estimated time: 1–3 hours** (one forward pass per artifact on 32 probe sequences)

```bash
python scripts/03_layer_diagnostics.py
```

What this does:

1. Loads the FP16 reference from `artifacts/fp16/`
2. For each quantized artifact, runs a fixed probe batch (32 sequences × 256 tokens) through both models simultaneously using forward hooks
3. Computes per-layer:
   - Activation percentiles on layer inputs (p50, p90, p99, p99.9, max)
   - Output MSE between FP16 and quantized layer outputs
4. Computes logit-level metrics (over the full output sequence):
   - KL divergence between FP16 and quantized next-token distributions
   - Cosine similarity between logit vectors
   - Top-1 and top-5 prediction agreement

Outputs:
- `results/layer_stats.parquet` — per-layer stats, one row per (method, variant, layer, module)
- `results/logit_diagnostics.json` — logit-level aggregate per (method, variant)

#### Run diagnostics for full_quant only (faster)

```bash
python scripts/03_layer_diagnostics.py --variant-filter full_quant
```

---

### Step 4 — Generate Plots

**Estimated time: < 1 minute**

```bash
python scripts/04_generate_plots.py
```

Reads `results/eval.jsonl` and `results/layer_stats.parquet`, writes all figures to `results/plots/`:

| File | Description |
|------|-------------|
| `accuracy_vs_bandwidth.png` | Scatter: composite score vs bytes/param — sweet-spot plot |
| `accuracy_vs_deployment.png` | Scatter panels: score vs tok/s, ms/tok, peak VRAM |
| `ablation_sensitivity.png` | Grouped bar: score drop from attn-only and mlp-only quantization |
| `layer_mse_<method>.png` | Heatmap: per-layer MSE by transformer layer index and module family |

---

## 5. Output Files

```
calibration_data/
└── calibration_512.pt           # Cached calibration tensors (auto-generated)

artifacts/
├── fp16/                        # FP16 baseline (safetensors)
├── gptq_w4_g128/full_quant/     # GPTQ W4 g128 full
├── gptq_w4_g128/attn_only_quant/
├── gptq_w4_g128/mlp_only_quant/
├── awq_w4_g128/full_quant/      # AWQ W4 g128 (full only)
├── rtn_w8_perchannel/full_quant/
├── rtn_w8_perchannel/attn_only_quant/
└── rtn_w8_perchannel/mlp_only_quant/

results/
├── eval.jsonl                   # One row per artifact: all metrics
├── layer_stats.parquet          # Per-layer activation stats + MSE
├── logit_diagnostics.json       # Logit-level KL / cosine / top-k
└── plots/
    ├── accuracy_vs_bandwidth.png
    ├── accuracy_vs_deployment.png
    ├── ablation_sensitivity.png
    ├── layer_mse_gptq.png
    ├── layer_mse_awq.png
    └── layer_mse_rtn.png
```

---

## 6. Project Structure

```
227_Project/
├── config/
│   └── default.yaml              Single config driving everything
├── src/
│   ├── quantize/
│   │   ├── module_utils.py       Enumerate nn.Linear layers; build allowlist/denylist
│   │   ├── gptq.py               AutoGPTQ W4 g128, supports full ablations
│   │   ├── awq.py                AutoAWQ W4 g128, full_quant only
│   │   └── rtn.py                bitsandbytes INT8 (Path A) or analytical RTN (Path B)
│   ├── eval/
│   │   ├── perplexity.py         HuggingFace teacher-forcing on WikiText-2
│   │   ├── humaneval.py          HumanEval pass@1 via vLLM + subprocess execution
│   │   ├── gsm8k.py              GSM8K accuracy via vLLM
│   │   ├── mmlu.py               MMLU 5-shot via vLLM
│   │   └── benchmark.py          Throughput and VRAM measurement via vLLM
│   ├── utils/
│   │   ├── config.py             YAML loader with path resolution
│   │   ├── calibration.py        Tokenize and cache calibration corpus
│   │   └── checkpoint_size.py    Actual bytes/param from safetensors metadata
│   ├── diagnostics/
│   │   └── layer_probe.py        Forward hooks: MSE, activation stats, logit metrics
│   └── viz/
│       └── plots.py              Scatter, bar, heatmap figure generation
├── scripts/
│   ├── 01_build_artifacts.py     Orchestrate all quantization exports
│   ├── 02_run_eval.py            Run full eval suite, write eval.jsonl
│   ├── 03_layer_diagnostics.py   FP16 vs quant probing, write parquet + json
│   └── 04_generate_plots.py      Read results, write all figures
├── artifacts/                    (generated) Quantized checkpoints
├── calibration_data/             (generated) Cached calibration tensors
├── results/                      (generated) Metrics and plots
├── requirements.txt
└── README.md
```

---

## 7. Quantization Methods

| Method | Precision | Group Size | Library | Ablation variants | vLLM flag |
|--------|-----------|------------|---------|-------------------|-----------|
| GPTQ | W4A16 | 128 | `auto-gptq` | full, attn_only, mlp_only | `"gptq"` |
| AWQ | W4A16 | 128 | `autoawq` | full_quant only | `"awq"` |
| RTN | W8A16 | per-channel | `bitsandbytes` | full, attn_only, mlp_only | `None` |
| FP16 | FP16 | — | `transformers` | baseline | `None` |

### Selective quantization (ablations)

Ablations enumerate all `nn.Linear` layers at runtime via `model.named_modules()`, classify them as `attn` (q/k/v/o_proj) or `mlp` (gate/up/down_proj), and build a denylist that is passed to the quantizer. This means **no module names are hardcoded**.

- **GPTQ**: uses `modules_to_not_convert` in `BaseQuantizeConfig`
- **AWQ**: selective quantization is not cleanly supported — only `full_quant` is run; this limitation is logged and documented
- **RTN (bitsandbytes)**: uses `llm_int8_skip_modules` in `BitsAndBytesConfig`

### RTN implementation note

By default the project uses **bitsandbytes Path A** (real INT8 GEMM kernels). A custom analytical Path B (symmetric per-channel RTN, clamp to `[-127, 127]`) is also available but its latency and bandwidth numbers are **not kernel-realistic** and must be labelled as such. To switch:

```python
# in scripts/01_build_artifacts.py
quantize_rtn(cfg, variant=variant, use_bitsandbytes=False)
```

---

## 8. Evaluation Details

### Perplexity

Uses HuggingFace `transformers` forward pass with `labels=input_ids` on the WikiText-2 test set. Computed with a sliding window (`stride=512`, `max_length=2048`) to avoid padding artifacts.

**vLLM is not used for perplexity** — it is optimized for generation, not teacher-forcing cross-entropy loss.

### HumanEval

164 Python programming problems. For each problem:
1. vLLM generates a completion (greedy, `temperature=0`)
2. The completion is concatenated with the problem's unit tests
3. Executed in a subprocess with a 10-second timeout
4. Pass@1 = fraction of problems where the code passes all tests

### GSM8K

Grade-school math word problems (~1,319 test examples). 8-shot prompting. Final numeric answer extracted from `#### <number>` pattern in model output.

### MMLU

~14,000 multiple-choice questions across 57 subjects. 5-shot prompting. Correct answer if model's first generated token matches the gold letter (A/B/C/D).

### Composite Score

```
score = 0.4 × HumanEval + 0.3 × GSM8K + 0.3 × MMLU
```

Weights are configurable in `config/default.yaml` under `accuracy_weights`.

### Bandwidth (bytes/param)

Computed from actual safetensors file metadata — not fixed estimates:

```python
bytes_per_param_actual = sum(numel × dtype_bytes for all tensors) / total_params
```

This accounts for quantized weights (INT4/INT8), scales, zero-points, and any FP16 layers left unquantized in ablations.

---

## 9. Diagnostics Details

Layer diagnostics run a fixed probe batch (32 sequences × 256 tokens, drawn from the calibration corpus) through the FP16 model and each quantized model, with forward hooks on every `nn.Linear` in the attn and MLP families.

### Per-layer metrics (saved to `layer_stats.parquet`)

| Column | Description |
|--------|-------------|
| `method` | Quantization method |
| `variant` | Ablation variant |
| `layer_idx` | Transformer layer index |
| `module_family` | `attn` or `mlp` |
| `module_name` | Full dotted module name (e.g. `model.layers.3.self_attn.q_proj`) |
| `mse` | Mean squared error between FP16 and quantized layer outputs |
| `p50/p90/p99/p99_9` | Activation magnitude percentiles on layer inputs |
| `act_max` | Maximum absolute activation value |

### Logit-level metrics (saved to `logit_diagnostics.json`)

| Field | Description |
|-------|-------------|
| `kl_div` | KL(P_fp16 ‖ P_quant) averaged over probe tokens |
| `cosine_sim` | Cosine similarity between FP16 and quantized logit vectors |
| `top1_agreement` | Fraction of tokens where argmax prediction matches FP16 |
| `top5_agreement` | Fraction of top-5 predicted tokens that overlap with FP16 |

---

## 10. Plots Reference

### `accuracy_vs_bandwidth.png`

Scatter plot with `bytes_per_param_actual` on the x-axis and composite `score` on the y-axis. Each point is one (method, variant). Identifies the efficiency "sweet spot" — the knee where accuracy degrades steeply relative to the storage savings.

### `accuracy_vs_deployment.png`

Three-panel scatter: accuracy vs tokens/sec, ms/token, and peak VRAM. Shows whether bandwidth savings translate to real deployment gains.

### `ablation_sensitivity.png`

Grouped bar chart per method. For each method:
- `delta_attn = score(full_quant) − score(attn_only_quant)`
- `delta_mlp = score(full_quant) − score(mlp_only_quant)`

A **larger delta** means that module family causes more degradation when quantized — i.e. it is more sensitive. This directly answers the "MLP vs attention" research question.

### `layer_mse_<method>.png`

Heatmap with transformer layer index on the y-axis and module family (attn/mlp) on the x-axis, coloured by mean MSE. Identifies which layers accumulate quantization error — typically later layers or layers that receive large activation outliers.

---

## 11. Common Issues

### Out of memory during quantization

GPTQ and AWQ load the model in FP16 before quantizing, requiring ~14 GB. If you OOM:

```yaml
# config/default.yaml
calibration:
  num_samples: 128      # reduce from 512
  max_length: 512       # reduce from 2048
```

Or run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/01_build_artifacts.py
```

### HuggingFace authentication error

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

### `auto-gptq` kernel not found

AutoGPTQ requires a compatible CUDA version. Install with CUDA 12.1:

```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

### `vllm` import error on Volta (V100)

AWQ is not supported on Volta (SM 7.0). Set `quantization=None` or skip AWQ. GPTQ and RTN work on Volta.

### Partial run / resume

Each script discovers artifacts by scanning the `artifacts/` directory. If a checkpoint already exists, the quantization step will be skipped automatically. Similarly, `02_run_eval.py` writes rows progressively — re-running will re-evaluate all found artifacts (deduplication is not automatic; move or delete `results/eval.jsonl` to start fresh).

### Slow HumanEval execution

Code execution happens in a subprocess per problem. If the environment has restricted subprocess access, HumanEval will fail silently. Skip it:

```bash
python scripts/02_run_eval.py --skip-tasks humaneval
```
