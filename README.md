# Uni-Layer

**Understand your layers before you optimize them.**

[![PyPI](https://img.shields.io/pypi/v/uni-layer.svg)](https://pypi.org/project/uni-layer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-205%20passed-brightgreen.svg)]()

Uni-Layer is a PyTorch toolkit that scores every layer in your neural network across **13 metrics in 7 theoretical categories**. It tells you which layers matter most — so you can prune smarter, fine-tune better, and distill more effectively.

**[English](#quick-start)** | **[中文](#中文说明)**

---

## Why Uni-Layer?

Most compression and fine-tuning tools treat all layers equally or rely on simple magnitude heuristics. Uni-Layer replaces guesswork with principled, multi-metric layer analysis.

There is no other library that does this. Captum does input attribution. Torch-Pruning does structural pruning. TransformerLens does mechanistic interpretability. **Uni-Layer is the only tool that unifies 13 layer importance metrics under one API and bridges them to downstream tools.**

| You want to... | Uni-Layer provides | Works with |
|---|---|---|
| **Prune** a model | Per-layer importance scores & pruning ratios | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| **LoRA fine-tune** | Which layers to target, adaptive rank allocation | [HuggingFace PEFT](https://github.com/huggingface/peft) |
| **Distill** knowledge | Layer pairing & per-layer distillation weights | Any distillation framework |
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
| `"full"` | All 13 metrics | Small model deep analysis |

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

### Verified on 20+ HuggingFace Models

| Architecture | Models tested | Block-level extraction |
|---|---|---|
| NLP Encoder | BERT, RoBERTa, DeBERTa-v3, DistilBERT, SciBERT, MiniLM | `encoder.layer.N` |
| NLP Decoder | GPT-2, Pythia, BLOOM, Falcon, TinyLlama, Llama-3.2-3B, Qwen2.5-3B | `model.layers.N` |
| Seq2Seq | ByT5 | `encoder.block.N` + `decoder.block.N` |
| Vision | DINOv2 | `encoder.layer.N` |
| Speech | Wav2Vec2, HuBERT | `encoder.layers.N` |

---

## 13 Metrics in 7 Categories

| Category | Metrics | What it measures |
|---|---|---|
| **Optimization** | `GradientNorm`, `HessianTrace`, `FisherInformation` | How much the layer affects the loss landscape |
| **Spectral** | `CKA`, `EffectiveRank`, `NTKTrace` | Representation similarity, diversity, kernel influence |
| **Information Theory** | `ActivationEntropy`, `MutualInformation` | Information content and task relevance |
| **Representation** | `JacobianRank`, `BlockInfluence` | Expressiveness and layer redundancy |
| **Robustness** | `DropLayerRobustness` | Performance impact of removing the layer |
| **Bayesian** | `LaplacePosterior` | Parameter uncertainty (Laplace approximation) |
| **Architecture** | `AttentionFlow` | Attention entropy, head diversity (Transformers) |

<details>
<summary>Full output keys per metric</summary>

| Metric | Primary Key | Additional Keys |
|---|---|---|
| GradientNorm | `gradient_norm` | `gradient_norm_std`, `_max`, `_min` |
| HessianTrace | `hessian_trace` | `hessian_trace_std` |
| FisherInformation | `fisher_information` | `fisher_mean` |
| CKA | `cka_score` | |
| EffectiveRank | `effective_rank` | `stable_rank`, `rank_ratio` |
| NTKTrace | `ntk_trace` | `ntk_trace_per_param` |
| ActivationEntropy | `activation_entropy` | `activation_mean`, `_std`, `_sparsity` |
| MutualInformation | `mutual_information` | `mi_max`, `mi_std` |
| JacobianRank | `jacobian_rank` | `jacobian_rank_ratio`, `_condition`, `_max_sv` |
| BlockInfluence | `block_influence` | `block_similarity` |
| DropLayerRobustness | `droplayer_loss_increase` | `droplayer_loss_ratio` |
| LaplacePosterior | `laplace_posterior` | `laplace_posterior_std` |
| AttentionFlow | `attention_entropy` | `attention_max_weight`, `head_diversity`, `attention_distance` |

</details>

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

---

## CLI

```bash
uni-layer info                                    # version, PyTorch, CUDA, metrics
uni-layer list-metrics                            # all 13 metrics with keys
uni-layer list-metrics --format json              # machine-readable
uni-layer analyze bert-base-uncased               # analyze a HuggingFace model
uni-layer analyze bert-base-uncased -m GradientNorm,BlockInfluence -o results.json
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

### v0.6.0 (Current)
- [x] Security & red-team analysis metrics (`security/` category)
  - [x] AdversarialSensitivity: per-layer FGSM/PGD perturbation sensitivity
  - [x] ActivationAnomalyScore: backdoor detection via activation pattern analysis
  - [x] MembershipInferenceRisk: per-layer gradient leakage scoring
  - [x] AttentionPathTrace: adversarial attention flow / prompt injection path analysis
- [x] Compression safety audit (safety degradation pre/post pruning/quantization)
- [x] `LayerProfile.security_report()` for automated vulnerability summary

### v0.7.0
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

Uni-Layer 是一个 PyTorch 工具库，通过 **7 大理论类别的 13 种指标** 为神经网络的每一层打分。它自动生成可操作的洞察——告诉你哪些层冗余可以剪枝、哪些层重要应该用 LoRA 微调、哪些层是表征瓶颈。

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
| `"full"` | 全部 13 指标 | 小模型深度分析 |

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

### 已验证 20+ 个 HuggingFace 模型

BERT / RoBERTa / DeBERTa / DistilBERT / SciBERT / MiniLM / GPT-2 / Pythia / BLOOM / Falcon / TinyLlama / Llama-3.2-3B / Qwen2.5-3B / ByT5 / DINOv2 / Wav2Vec2 / HuBERT — 全部通过。

### CLI

```bash
uni-layer info              # 版本、PyTorch、CUDA、已安装指标
uni-layer list-metrics      # 列出全部 13 种指标
uni-layer analyze bert-base-uncased   # 分析 HuggingFace 模型
```

### 路线图

**v0.4.0**：Diffusion/Mamba/MoE 支持 / 残差感知 DropLayer / 层间 CKA 矩阵 / GNN 支持

**v0.5.0**：多模态分支分析 / Wanda 重要性 / IG 灵敏度 / LLM 训练框架集成

**v1.0.0**：稳定 API / Web 可视化 / 分布式分析 / 预计算模型库 / 学术论文

### 许可证

MIT License。详见 [LICENSE](LICENSE)。
