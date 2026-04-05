# Uni-Layer

**Understand your layers before you optimize them.**

[![PyPI](https://img.shields.io/pypi/v/uni-layer.svg)](https://pypi.org/project/uni-layer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-402%20passed-brightgreen.svg)]()

Uni-Layer is a PyTorch toolkit that scores every layer in your neural network across **26 metrics in 9 theoretical categories**. It tells you which layers matter most — so you can prune smarter, fine-tune better, distill more effectively, and audit model security.

**[English](#quick-start)** | **[中文](#中文说明)**

---

## Why Uni-Layer?

Most compression and fine-tuning tools treat all layers equally or rely on simple magnitude heuristics. Uni-Layer replaces guesswork with principled, multi-metric layer analysis.

There is no other library that does this. Captum does input attribution. Torch-Pruning does structural pruning. TransformerLens does mechanistic interpretability. **Uni-Layer is the only tool that unifies 26 layer importance metrics under one API and bridges them to 7 downstream tools.**

| You want to... | Uni-Layer provides | Works with |
|---|---|---|
| **Prune** a model | Per-layer importance scores & pruning ratios | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| **LoRA fine-tune** | Which layers to target, adaptive rank allocation | [HuggingFace PEFT](https://github.com/huggingface/peft) |
| **Distill** knowledge | Layer pairing & per-layer distillation weights | Any distillation framework |
| **Quantize** selectively | Per-layer INT8/FP16 sensitivity + mixed-precision plan | ONNX / TensorRT |
| **Audit** security | Adversarial sensitivity, backdoor detection, privacy risk | Red-team workflows |
| **Understand** a model | Auto-generated layer profile with actionable insights | Standalone |

---

## Quick Start

```bash
pip install uni-layer
```

```python
from uni_layer import LayerAnalyzer, LayerProfile

analyzer = LayerAnalyzer(model, task_type='classification')

# One-line analysis with preset (recommended for LLMs)
contributions = analyzer.compute_metrics(
    data_loader=train_loader, preset="llm_fast",
)

# Or pick metrics manually
from uni_layer.metrics import GradientNorm, CKA, BlockInfluence
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA(), BlockInfluence()],
    data_loader=train_loader,
)

# Auto-generate insights (pure CPU, milliseconds)
profile = LayerProfile(contributions, model_name="my-model")
print(profile.summary())
print(profile.pruning_suggestion(target_ratio=0.3))
print(profile.lora_suggestion(base_rank=8))
```

### Presets

| Preset | Metrics | Use case |
|---|---|---|
| `"llm_fast"` | BlockInfluence, EffectiveRank, CKA, ActivationEntropy, AttentionFlow | LLM quick scan (seconds) |
| `"llm_full"` | + GradientNorm, FisherInformation | LLM full analysis (minutes) |
| `"quick"` | GradientNorm, BlockInfluence, EffectiveRank | Fastest overview |
| `"full"` | All 26 metrics | Complete deep analysis |

---

## LayerProfile: Auto Insights

`LayerProfile` takes the raw numbers from `compute_metrics()` and turns them into actionable insights. No extra GPU cost — all analysis runs on CPU in milliseconds.

```python
from uni_layer import LayerProfile

profile = LayerProfile(contributions, model_name="Llama-3.2-3B")

profile.redundant_layers        # ["layers.14", "layers.15"] — safe to prune
profile.bottleneck_layers       # ["layers.30"] — representation bottleneck
profile.consensus_ranking       # multi-metric Borda count ranking
profile.depth_trends            # {"gradient_norm": {"trend": "U-shaped", ...}}
profile.anomalies               # layers with z-score > 2 on any metric
profile.layer_clusters          # {"high_contribution": [...], "low_contribution": [...]}

profile.pruning_suggestion(0.3) # {"safe_to_remove": [...], "estimated_speedup": "25%"}
profile.lora_suggestion(8)      # {"target_layers": [...], "adaptive_ranks": {...}}
profile.summary()               # one-paragraph natural language summary
profile.to_dict()               # full JSON export
```

Example summary output:
> "bert-base-uncased (12-layer model). analyzed with 10 metrics. U-shaped gradient norm distribution. 2 redundant layers (encoder.layer.5, encoder.layer.6) safe to prune. most important: encoder.layer.9, least important: encoder.layer.3."

---

## Output Format

Every call to `compute_metrics()` returns a structured dict:

```json
{
  "encoder.layer.0": {
    "layer_idx": 0,
    "layer_type": "transformer_block",
    "gradient_norm": 0.0193,
    "cka_score": 0.4161,
    "block_influence": 0.1465,
    "effective_rank": 10.54
  },
  ...
}
```

`rank_layers()` returns sorted `(name, score)` tuples:

```python
[("encoder.layer.9", 0.1094), ("encoder.layer.0", 0.0624), ...]
```

### Supported Architectures (7 families)

| Architecture | Models | Layer Extraction |
|---|---|---|
| **Transformer** | BERT, GPT-2, LLaMA, Qwen, T5, ViT, DINOv2, Wav2Vec2 (20+) | Block-level (auto) |
| **Mamba/SSM** | Mamba, S4, S6 | Block-level (auto) |
| **GNN** | GCNConv, GATConv, SAGEConv (PyG) | Conv-level |
| **Diffusion** | UNet, DDPM, DiT | down/mid/up blocks |
| **MoE** | Mixtral, Switch Transformer, DeepSeek-V3 | Router + expert analysis |
| **Multi-Modal** | CLIP, LLaVA, Mistral-Small-3.1 | Per-branch analysis |
| **CNN** | ResNet, ConvNeXt, EfficientNet | Block/layer level |

### Model Compatibility

Uni-Layer uses **pattern-based BFS** to detect transformer blocks — no model-specific code. Any model using standard `nn.ModuleList` block containers works automatically. Tested on **33 model families** with real HuggingFace checkpoints.

<details>
<summary><b>Tested Model Families — 38 PASS, 5 PARTIAL (click to expand)</b></summary>

#### Full Pass (layer detection + metrics + profile)

| Model Family | Tested Model | Block Path | Status |
|---|---|---|---|
| **Gemma 4 E2B** ⭐ (Apr 2026, multimodal) | `google/gemma-4-E2B` | `model.language_model.layers` | PASS |
| **Gemma 4 E4B** ⭐ (Apr 2026, multimodal) | `google/gemma-4-E4B` | `model.language_model.layers` | PASS |
| **Gemma 4 31B** ⭐ (Apr 2026, multimodal) | `google/gemma-4-31B` | `model.language_model.layers` | PASS |
| **Gemma 4 26B-A4B** ⭐ (Apr 2026, multimodal MoE) | `google/gemma-4-26B-A4B` | `model.language_model.layers` | PASS |
| **Llama 4 Scout** ⭐ (Apr 2026, multimodal MoE) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `language_model.model.layers` | PASS |
| **Qwen3** | `Qwen/Qwen3-0.6B` | `layers` | PASS |
| **Qwen2.5** | `Qwen/Qwen2.5-0.5B` | `layers` | PASS |
| **Qwen2** | `Qwen/Qwen2-0.5B` | `layers` | PASS |
| **Gemma 3** | `google/gemma-3-1b-pt` | `layers` | PASS |
| **Gemma 2** | `google/gemma-2-2b` | `layers` | PASS |
| **Gemma** | `google/gemma-7b` | `layers` | PASS |
| **LLaMA** | `hf-internal-testing/tiny-random-LlamaForCausalLM` | `layers` | PASS |
| **TinyLlama** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `layers` | PASS |
| **Mistral** | `hf-internal-testing/tiny-random-MistralForCausalLM` | `layers` | PASS |
| **Mistral-3.1** (multimodal) | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `language_model.layers` | PASS |
| **Mixtral** (MoE) | `hf-internal-testing/tiny-random-MixtralForCausalLM` | `layers` | PASS |
| **Yi-1.5** | `01-ai/Yi-1.5-6B` | `layers` | PASS |
| **InternLM-2.5** | `internlm/internlm2_5-1_8b` | `model.layers` | PASS |
| **GLM-4** | `THUDM/glm-4-9b` | `transformer.encoder.layers` | PASS |
| **StarCoder2** | `bigcode/starcoder2-3b` | `layers` | PASS |
| **Falcon-3** | `tiiuae/Falcon3-1B-Base` | `layers` | PASS |
| **Falcon** (classic) | `hf-internal-testing/tiny-random-FalconForCausalLM` | `h` | PASS |
| **Nemotron** | `nvidia/Nemotron-Mini-4B-Instruct` | `layers` | PASS |
| **OLMo-2** | `allenai/OLMo-2-0425-1B` | `layers` | PASS |
| **StableLM** | `stabilityai/stablelm-3b-4e1t` | `layers` | PASS |
| **Pythia** | `EleutherAI/pythia-70m` | `layers` | PASS |
| **GPT-2** | `hf-internal-testing/tiny-random-gpt2` | `h` | PASS |
| **GPT-Neo** | `hf-internal-testing/tiny-random-GPTNeoForCausalLM` | `h` | PASS |
| **GPT-J** | `hf-internal-testing/tiny-random-GPTJForCausalLM` | `h` | PASS |
| **GPT-NeoX** | `hf-internal-testing/tiny-random-GPTNeoXForCausalLM` | `layers` | PASS |
| **BLOOM** | `hf-internal-testing/tiny-random-BloomForCausalLM` | `h` | PASS |
| **BERT** | `hf-internal-testing/tiny-random-BertModel` | `encoder.layer` | PASS |
| **RoBERTa** | `hf-internal-testing/tiny-random-RobertaModel` | `encoder.layer` | PASS |
| **ELECTRA** | `hf-internal-testing/tiny-random-ElectraModel` | `encoder.layer` | PASS |
| **DeBERTa-v2** | `hf-internal-testing/tiny-random-DebertaV2Model` | `encoder.layer` | PASS |
| **T5** | `hf-internal-testing/tiny-random-t5` | `encoder.block` + `decoder.block` | PASS |
| **BART** | `hf-internal-testing/tiny-random-BartModel` | `encoder.layers` + `decoder.layers` | PASS |
| **ALBERT** | `hf-internal-testing/tiny-random-AlbertModel` | type-based (weight sharing) | PASS |

> ⭐ = verified April 2026 releases. Gemma 4 requires `transformers>=5.0.0`. Llama 4 requires `transformers>=4.51.0`.

#### Partial Pass (layer detection OK, metrics need specific deps)

| Model Family | Tested Model | Block Path | Issue |
|---|---|---|---|
| **DeepSeek-V3** (MoE) | `deepseek-ai/DeepSeek-V3-0324` | `layers` | Needs transformers>=4.48 (DynamicCache API) |
| **DeepSeek-V2** (MoE) | `deepseek-ai/DeepSeek-V2-Lite` | `layers` | Needs transformers>=4.48 (DynamicCache API) |
| **Jamba** (Mamba+Attn hybrid) | `ai21labs/Jamba-v0.1` | `layers` | Needs [mamba-ssm](https://github.com/state-spaces/mamba) CUDA kernels |
| **Phi-2** | `microsoft/phi-2` | `layers` | Layer detection works; metric computation needs matching config |
| **ViT** | `hf-internal-testing/tiny-random-ViTModel` | `encoder.layer` | Needs image tensor input (not token IDs) |

> **Note**: "Partial" means layer detection and profiling work correctly. The metric computation issues are from external dependency versions, not Uni-Layer code.

#### Not Tested (access restrictions)

| Model Family | Reason |
|---|---|
| **Command-R/R+** | Gated repo (requires HuggingFace access token) |
| **Phi-4** | Requires `transformers>=4.49` (custom code import) |
| **DBRX** | Model removed from HuggingFace Hub |

</details>

**Custom models**: Any `nn.Module` with blocks under attributes named `layer`, `layers`, `h`, `blocks`, or `block` will be auto-detected. For non-standard structures, blocks fall back to type-based extraction (individual Linear/Conv layers).

Run the compatibility test yourself:
```bash
python tests/test_model_compatibility.py
```

---

## 26 Metrics in 9 Categories

| Category | Metrics | What it measures |
|---|---|---|
| **Optimization** (5) | `GradientNorm`, `HessianTrace`, `FisherInformation`, `WandaImportance`, `IGSensitivity` | Loss landscape, weight x activation importance, IG attribution |
| **Spectral** (3) | `CKA`, `EffectiveRank`, `NTKTrace` | Representation similarity, diversity, kernel influence |
| **Information Theory** (2) | `ActivationEntropy`, `MutualInformation` | Information content and task relevance |
| **Representation** (2) | `JacobianRank`, `BlockInfluence` | Expressiveness and layer redundancy |
| **Robustness** (2) | `DropLayerRobustness`, `ResidualDropLayer` | Ablation with/without residual preservation |
| **Bayesian** (1) | `LaplacePosterior` | Parameter uncertainty |
| **Efficiency** (4) | `EfficiencyProfiler`, `WeightDistribution`, `IntrinsicDimensionality`, `QuantizationSensitivity` | FLOPs, sparsity, manifold dim, quantization noise |
| **Security** (4) | `AdversarialSensitivity`, `ActivationAnomalyScore`, `MembershipInferenceRisk`, `AttentionPathTrace` | Adversarial robustness, backdoor, privacy, injection |
| **Arch-Specific** (3) | `AttentionFlow`, `MoERouterAnalysis`, `DiffusionTimestepAnalysis` | Attention heads, MoE routing, diffusion timesteps |

See [docs/METRICS.md](docs/METRICS.md) for full output keys and usage examples.

---

## Integration Bridges

### Torch-Pruning

```python
from uni_layer.integrations import TorchPruningBridge

bridge = TorchPruningBridge(model, contributions)
pruning_ratios = bridge.as_layer_pruning_ratios(
    metric_name='gradient_norm', target_sparsity=0.5
)
protected = bridge.get_protected_layers(top_k=3)
```

### HuggingFace PEFT

```python
from uni_layer.integrations import HuggingFacePEFTBridge
from peft import LoraConfig, get_peft_model

bridge = HuggingFacePEFTBridge(model, contributions)
config_params = bridge.recommend_lora_config_params(metric_name='gradient_norm')
peft_model = get_peft_model(model, LoraConfig(**config_params))
```

### Knowledge Distillation

```python
from uni_layer.integrations import DistillationBridge

bridge = DistillationBridge(teacher, student, contributions)
pairs = bridge.recommend_layer_pairs(top_k=4)
weights = bridge.recommend_layer_weights()
```

### ONNX / TensorRT

```python
from uni_layer.integrations import ExportHintsBridge

bridge = ExportHintsBridge(model, contributions)
plan = bridge.quantization_plan(target="int8", protect_ratio=0.2)
config = bridge.tensorrt_config()
```

### LLM Frameworks (Axolotl / LLaMA-Factory)

```python
from uni_layer.integrations import AxolotlConfigBridge, LLaMAFactoryConfigBridge

AxolotlConfigBridge(model, contributions).save_yaml("config.yml", base_model="meta-llama/Llama-2-7b")
LLaMAFactoryConfigBridge(model, contributions).save_json("config.json", model_name="my-model")
```

### Security Audit

```python
from uni_layer.integrations import CompressionSafetyAudit

audit = CompressionSafetyAudit(pre_contributions, post_contributions)
report = audit.audit()
print(report["overall_degradation"], report["recommendations"])
```

---

## CLI

```bash
uni-layer info                                    # version, PyTorch, CUDA, metrics
uni-layer list-metrics                            # all 26 metrics with keys
uni-layer list-metrics --format json              # machine-readable
uni-layer analyze bert-base-uncased               # analyze a HuggingFace model
uni-layer analyze bert-base-uncased -p llm_fast --profile   # preset + insights
uni-layer analyze bert-base-uncased -m GradientNorm,BlockInfluence -o results.json
```

### CLI Options

| Flag | Description |
|------|-------------|
| `-p`, `--preset` | Use a preset: `quick`, `llm_fast`, `llm_full`, `full` |
| `--profile` | Generate LayerProfile summary with pruning/LoRA suggestions |
| `-m`, `--metrics` | Comma-separated metric names (overridden by `--preset`) |
| `-o`, `--output` | Save results to JSON file |
| `-d`, `--device` | `cpu` or `cuda` |
| `-n`, `--num-batches` | Number of data batches (default: 3) |

---

## MCP Server (Claude Code / AI Assistants)

Uni-Layer includes an [MCP](https://modelcontextprotocol.io/) server for integration with Claude Code, Codex, and other MCP-compatible assistants.

```bash
pip install uni-layer[mcp]
```

### Configure in Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "uni-layer": {
      "command": "python",
      "args": ["-m", "uni_layer.mcp_server"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `list_metrics` | List all 26 metrics (optionally filtered by category) |
| `analyze_model` | Analyze a HuggingFace model with a preset or specific metrics |
| `layer_profile` | Generate actionable insights (pruning, LoRA, anomalies) |
| `suggest_pruning` | Torch-Pruning compatible per-layer pruning ratios |
| `suggest_lora` | PEFT/LoRA config with adaptive rank allocation |
| `suggest_quantization` | Mixed-precision quantization plan |
| `security_audit` | Per-layer security vulnerability report |

### Example conversation with Claude Code

```
You: Analyze bert-base-uncased and tell me which layers to prune
Claude: [calls analyze_model → layer_profile → suggest_pruning]
        Layers 5 and 6 are redundant (low block_influence, high CKA similarity).
        Safe to prune with ~25% speedup. Here's the Torch-Pruning code...
```

---

## HuggingFace Model Support

Uni-Layer natively handles HuggingFace models — dict/dataclass outputs, `attention_mask`, `decoder_input_ids` (Seq2Seq) are all handled automatically:

```python
from transformers import AutoModel
from uni_layer import LayerAnalyzer, LayerProfile

model = AutoModel.from_pretrained("bert-base-uncased")
analyzer = LayerAnalyzer(model, task_type='classification')
contributions = analyzer.compute_metrics(
    data_loader=tokenized_loader, preset="llm_fast",
)
profile = LayerProfile(contributions)
print(profile.summary())
```

---

## Examples

| Example | Model | File |
|---|---|---|
| ResNet layer analysis | ResNet-18 (CNN) | [`examples/resnet_layer_analysis.py`](examples/resnet_layer_analysis.py) |
| ViT attention analysis | Vision Transformer | [`examples/vit_layer_analysis.py`](examples/vit_layer_analysis.py) |
| BERT layer analysis + LoRA | BERT-style Transformer | [`examples/bert_layer_analysis.py`](examples/bert_layer_analysis.py) |
| Torch-Pruning integration | Any model | [`examples/integrate_torch_pruning.py`](examples/integrate_torch_pruning.py) |
| HuggingFace PEFT integration | Any model | [`examples/integrate_huggingface_peft.py`](examples/integrate_huggingface_peft.py) |
| Knowledge distillation | Teacher-Student | [`examples/integrate_distillation.py`](examples/integrate_distillation.py) |

---

## Installation

```bash
pip install uni-layer                    # core
pip install uni-layer[integrations]      # + torch-pruning, peft, transformers
pip install uni-layer[dev]               # + pytest, black, flake8, mypy
pip install uni-layer[all]               # everything
```

From source:

```bash
git clone https://github.com/GeoffreyWang1117/Uni-Layer.git
cd Uni-Layer && pip install -e ".[dev]"
```

---

## Roadmap

### v0.4.0
- [x] Diffusion model support (UNet timestep-aware analysis)
- [x] Mamba / SSM architecture support
- [x] MoE router layer analysis
- [x] Residual-aware DropLayer metric
- [x] Layer-to-layer CKA similarity matrix
- [x] GNN support (PyG MessagePassing layers)

### v0.5.0
- [x] Multi-modal model branch analysis (vision encoder + language decoder)
- [x] Wanda-style importance (weight x activation norm)
- [x] IG-based sensitivity scoring (IGU-LoRA style)
- [x] Integration with LLM training frameworks (Axolotl / LLaMA-Factory)
- [x] Export to ONNX / TensorRT optimization hints

### v0.6.0
- [x] Security & red-team analysis metrics (`security/` category)
  - [x] AdversarialSensitivity, ActivationAnomalyScore, MembershipInferenceRisk, AttentionPathTrace
- [x] Compression safety audit + `LayerProfile.security_report()`

### v0.6.1
- [x] Efficiency metrics (`efficiency/` category)
  - [x] EfficiencyProfiler: per-layer FLOPs, param count, memory, compute ratio
  - [x] WeightDistribution: sparsity, norms, rank deficiency, kurtosis
  - [x] IntrinsicDimensionality: MLE manifold dimension for LoRA rank selection
  - [x] QuantizationSensitivity: INT8/FP16 noise tolerance per layer

### v0.7.0 (Current)
- [x] **38-family model compatibility** — Gemma 4 (E2B/E4B/31B/26B-A4B), Llama 4 Scout, DeepSeek-R1-Distill, Qwen3.5
- [x] **MCP Server** — 7-tool FastMCP server for Claude Code / AI-assistant integration
- [x] **CLI** — all 26 metrics exposed, `--preset`, `--profile` flags
- [x] **Claude Code skill** — `/layer-analyze` in-editor analysis
- [x] Bug fixes: CausalLM label injection, Gemma 4 `mm_token_type_ids`, audio tower deprioritisation

### v0.8.0 (Planned)
- [ ] KV Cache analysis for LLM inference
  - [ ] KV cache redundancy detection (cross-layer / cross-head similarity)
  - [ ] KV cache compression recommendations (per-layer budget allocation)
  - [ ] Cache information leakage scoring
- [ ] Inference framework integration (SGLang / vLLM)
  - [ ] Runtime layer profiling via inference engine hooks
  - [ ] Serving-aware importance scoring (latency vs. quality trade-off)
- [ ] Adversarial prompt attack path analysis
  - [ ] Layer-level vulnerability to jailbreak / injection prompts
  - [ ] Attention hijacking detection across decoding steps
- [ ] Inference attack surface analysis
  - [ ] Model inversion risk per layer
  - [ ] Embedding extraction vulnerability scoring

### v1.0.0
- [ ] Stable API with full backward compatibility
- [ ] Interactive web dashboard for layer analysis
- [ ] Distributed analysis for large models (FSDP/DeepSpeed)
- [ ] Pre-computed analysis for popular models (BERT, LLaMA, ViT, etc.)
- [ ] Academic paper and comprehensive benchmark suite

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICKSTART.md) | Get started in 5 minutes |
| [Metric Reference](docs/METRICS.md) | All 26 metrics with output keys |
| [API Reference](docs/API.md) | Complete class and method reference |
| [Architecture Guide](docs/ARCHITECTURE.md) | Per-architecture examples |
| [Security Guide](docs/SECURITY.md) | Red-team analysis workflows |
| [Compression Guide](docs/COMPRESSION.md) | Pruning, distillation, LoRA |

---

## Citation

```bibtex
@software{unilayer2025,
  title={Uni-Layer: A Universal Framework for Layer Contribution Analysis},
  author={Geoffrey Wang},
  year={2025},
  url={https://github.com/GeoffreyWang1117/Uni-Layer}
}
```

## License

MIT License. See [LICENSE](LICENSE).

---

<a id="中文说明"></a>

# 中文说明

## Uni-Layer：神经网络层贡献度分析框架

**先理解你的层，再优化它们。**

Uni-Layer 是一个 PyTorch 工具库，通过 **9 大理论类别的 26 种指标** 为神经网络的每一层打分。支持 Transformer、Mamba/SSM、GNN、Diffusion、MoE、多模态等 7 种架构。提供安全审计、效率分析、量化敏感度等全面的模型分析能力。

### 快速开始

```python
from uni_layer import LayerAnalyzer, LayerProfile

analyzer = LayerAnalyzer(model, task_type='classification')

# 一行分析（推荐大模型使用 preset）
contributions = analyzer.compute_metrics(data_loader=loader, preset="llm_fast")

# 自动生成洞察（纯 CPU，毫秒级）
profile = LayerProfile(contributions, model_name="my-model")
print(profile.summary())          # 一段话总结
print(profile.redundant_layers)   # 可安全剪除的层
print(profile.pruning_suggestion(0.3))  # 剪枝建议
print(profile.lora_suggestion(8))       # LoRA 建议
```

### Presets（预设）

| 预设 | 包含指标 | 用途 |
|---|---|---|
| `"llm_fast"` | BlockInfluence, EffectiveRank, CKA, Entropy, AttentionFlow | 大模型快速扫描（秒级） |
| `"llm_full"` | + GradientNorm, FisherInformation | 大模型完整分析（分钟级） |
| `"quick"` | GradientNorm, BlockInfluence, EffectiveRank | 最快概览 |
| `"full"` | 全部 26 指标 | 完整深度分析 |

### LayerProfile 自动分析

```python
profile = LayerProfile(contributions)

profile.redundant_layers        # 冗余层（可剪枝）
profile.bottleneck_layers       # 瓶颈层（信息流受限）
profile.consensus_ranking       # 多指标共识排名
profile.depth_trends            # 深度趋势（U 形/递增/递减/平坦）
profile.anomalies               # 统计异常层
profile.pruning_suggestion(0.3) # 剪枝方案 + 预估加速
profile.lora_suggestion(8)      # LoRA 目标层 + 自适应秩
profile.summary()               # 自然语言摘要
profile.to_dict()               # 完整 JSON 导出
```

### 支持的架构（7 大类）

Transformer (20+ HF 模型) | Mamba/SSM | GNN (PyG) | Diffusion (UNet/DiT) | MoE | 多模态 (CLIP/LLaVA) | CNN

**已测试 33 个模型家族**：Qwen3, Gemma 3, LLaMA, Mistral-3.1, DeepSeek-V3, GLM-4, InternLM-2.5, Yi-1.5, Falcon-3, Nemotron, StarCoder2, OLMo-2 等（详见[兼容性列表](README_CN.md#模型兼容性)）

### 26 种指标（9 大类别）

优化 (5) | 谱方法 (3) | 信息论 (2) | 表征 (2) | 鲁棒性 (2) | 贝叶斯 (1) | 效率 (4) | 安全 (4) | 架构特定 (3)

### 7 个集成桥

Torch-Pruning | HuggingFace PEFT | 知识蒸馏 | ONNX/TensorRT | Axolotl | LLaMA-Factory | 压缩安全审计

### CLI

```bash
uni-layer info              # 版本、PyTorch、CUDA
uni-layer list-metrics      # 列出全部 26 种指标
uni-layer analyze bert-base-uncased   # 分析 HuggingFace 模型
```

### 路线图

- **v0.4.0**：Diffusion/Mamba/MoE/GNN 支持 + CKA 矩阵 + 残差感知 DropLayer
- **v0.5.0**：Wanda/IG 指标 + 多模态 + ONNX/TensorRT + Axolotl/LLaMA-Factory
- **v0.6.0**：安全/红队指标 + 压缩安全审计
- **v0.6.1**（当前）：效率指标（FLOPs、权重分布、内在维度、量化敏感度）
- **v0.7.0**：KV Cache 分析 + SGLang/vLLM 集成 + 推理攻击面分析
- **v1.0.0**：稳定 API + Web 可视化 + 分布式分析 + 学术论文

> 完整中文文档：[README_CN.md](README_CN.md) | [docs/QUICKSTART_CN.md](docs/QUICKSTART_CN.md) | [docs/METRICS_CN.md](docs/METRICS_CN.md)

### 许可证

MIT License。详见 [LICENSE](LICENSE)。
