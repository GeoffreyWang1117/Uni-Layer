# Uni-Layer

**Understand your layers before you optimize them.**

[![PyPI](https://img.shields.io/pypi/v/uni-layer.svg)](https://pypi.org/project/uni-layer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-168%20passed-brightgreen.svg)]()

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
| **Understand** a model | Multi-metric layer contribution profile | Standalone |

---

## Quick Start

```bash
pip install uni-layer
```

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, BlockInfluence

analyzer = LayerAnalyzer(model, task_type='classification')
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA(), BlockInfluence()],
    data_loader=train_loader,
)

# Rank layers by importance
for name, score in analyzer.rank_layers(contributions, 'gradient_norm'):
    print(f"  {name}: {score:.4f}")
```

---

## Output Format

Every call to `compute_metrics()` returns a structured dict. Here is a real example from a 4-layer MLP:

```json
{
  "0": {
    "layer_idx": 0,
    "layer_type": "linear",
    "gradient_norm": 0.0193,
    "gradient_norm_std": 0.0016,
    "cka_score": 0.4161,
    "effective_rank": 10.54,
    "block_influence": 1.0,
    "fisher_information": 0.0001
  },
  "2": {
    "layer_idx": 1,
    "layer_type": "linear",
    "gradient_norm": 0.0494,
    "cka_score": 0.5449,
    "effective_rank": 20.18,
    "block_influence": 1.0,
    "fisher_information": 0.0002
  },
  "4": {
    "layer_idx": 2,
    "layer_type": "linear",
    "gradient_norm": 0.0624,
    "cka_score": 0.6233,
    "effective_rank": 9.58,
    "block_influence": 1.0,
    "fisher_information": 0.0003
  },
  "6": {
    "layer_idx": 3,
    "layer_type": "linear",
    "gradient_norm": 0.1094,
    "cka_score": 1.0,
    "effective_rank": 2.36,
    "block_influence": 1.0,
    "fisher_information": 0.0009
  }
}
```

`rank_layers()` returns sorted `(name, score)` tuples:

```python
[("6", 0.1094), ("4", 0.0624), ("2", 0.0494), ("0", 0.0193)]
# Layer 6 (output head) contributes most; Layer 0 (input) contributes least.
```

And here is a 4-block Transformer analyzed with `GradientNorm`, `BlockInfluence`, and `EffectiveRank`:

```
Layer              Type                  GradNorm  BlockInfluence  EffectiveRank
--------------------------------------------------------------------------------
blocks.0           transformer_block       0.1425          0.0278          94.47
blocks.1           transformer_block       0.1404          0.0275          94.21
blocks.2           transformer_block       0.1319          0.0265          93.92
blocks.3           transformer_block       0.1276          0.0269          93.66
```

> Early blocks have slightly higher gradient norms — they are adapting more. BlockInfluence is low everywhere (all ~0.027) because residual connections dominate, meaning each block's transformation is small relative to the skip path. EffectiveRank is uniformly high (~94), indicating rich, non-degenerate representations.

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

Each metric returns a dict with a **primary key** (used for ranking) and optional secondary keys:

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

---

## Integration Bridges

### Torch-Pruning

```python
from uni_layer.integrations import TorchPruningBridge

bridge = TorchPruningBridge(model, contributions)

# Important layers get low pruning ratios, unimportant layers get high ratios
pruning_ratios = bridge.as_layer_pruning_ratios(
    metric_name='gradient_norm', target_sparsity=0.5
)
protected = bridge.get_protected_layers(top_k=3)

# Use with torch-pruning
import torch_pruning as tp
pruner = tp.pruner.MetaPruner(
    model, example_inputs,
    importance=tp.importance.MagnitudeImportance(),
    pruning_ratio_dict=pruning_ratios,
)
```

### HuggingFace PEFT

```python
from uni_layer.integrations import HuggingFacePEFTBridge
from peft import LoraConfig, get_peft_model

bridge = HuggingFacePEFTBridge(model, contributions)

# Auto-select LoRA targets and adaptive rank
config_params = bridge.recommend_lora_config_params(metric_name='gradient_norm')
peft_model = get_peft_model(model, LoraConfig(**config_params))

# Or fine-grained control: different rank per layer
ranks = bridge.recommend_adaptive_ranks(base_rank=8, max_rank=64)
```

### Knowledge Distillation

```python
from uni_layer.integrations import DistillationBridge

bridge = DistillationBridge(teacher, student, contributions)

pairs = bridge.recommend_layer_pairs(top_k=4)    # teacher-student layer mapping
weights = bridge.recommend_layer_weights()         # per-layer distillation weights
```

---

## HuggingFace Model Support

Uni-Layer natively handles HuggingFace models that return dataclass/dict outputs, with automatic `attention_mask` injection:

```python
from transformers import AutoModel
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, BlockInfluence

model = AutoModel.from_pretrained("bert-base-uncased")
analyzer = LayerAnalyzer(model, task_type='classification')

# Just works -- dict outputs, attention_mask, labels all handled automatically
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), BlockInfluence()],
    data_loader=tokenized_loader,
)
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

### v0.3.0 (Next)
- [ ] Diffusion model support (UNet timestep-aware analysis)
- [ ] Mamba / SSM architecture support
- [ ] MoE router layer analysis
- [ ] Residual-aware DropLayer metric (understand skip connections)
- [ ] Layer-to-layer CKA similarity matrix

### v0.4.0
- [ ] GNN support (PyG MessagePassing layers)
- [ ] Multi-modal model branch analysis (vision encoder + language decoder)
- [ ] Wanda-style importance (weight x activation norm)
- [ ] IG-based sensitivity scoring (IGU-LoRA style)
- [ ] Export to ONNX / TensorRT optimization hints

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

Uni-Layer 是一个 PyTorch 工具库，通过 **7 大理论类别的 13 种指标** 为神经网络的每一层打分，告诉你哪些层最重要——从而实现更精准的剪枝、更高效的微调和更有效的蒸馏。

### 核心优势

- **唯一的层重要性通用评分库**：Captum 做输入归因，Torch-Pruning 做剪枝，TransformerLens 做机制解释——只有 Uni-Layer 把 13 种层重要性指标统一到一个 API 中
- **与下游工具解耦**：通过 Bridge 模式无缝连接 Torch-Pruning / PEFT / 蒸馏框架
- **兼容 HuggingFace**：自动处理 dict/dataclass 输出、attention_mask、labels 透传

### 快速开始

```bash
pip install uni-layer
```

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, BlockInfluence

analyzer = LayerAnalyzer(model, task_type='classification')
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA(), BlockInfluence()],
    data_loader=train_loader,
)

# 按重要性排序
for name, score in analyzer.rank_layers(contributions, 'gradient_norm'):
    print(f"  {name}: {score:.4f}")
```

### 输出格式

`compute_metrics()` 返回结构化字典：

```python
{
  "layer_name": {
    "layer_idx": 0,                  # 层索引
    "layer_type": "linear",          # 层类型
    "gradient_norm": 0.0193,         # 各指标值
    "cka_score": 0.4161,
    "block_influence": 1.0,
    ...
  },
  ...
}
```

`rank_layers()` 返回排序后的元组列表：

```python
[("layer_6", 0.1094), ("layer_4", 0.0624), ...]  # 降序
```

### 13 种指标

| 类别 | 指标 | 衡量内容 |
|---|---|---|
| 优化几何 | GradientNorm, HessianTrace, FisherInformation | 层对损失曲面的影响 |
| 谱方法 | CKA, EffectiveRank, NTKTrace | 表征相似性、多样性、核影响力 |
| 信息论 | ActivationEntropy, MutualInformation | 信息含量与任务相关性 |
| 表征结构 | JacobianRank, BlockInfluence | 表达能力与层冗余度 |
| 鲁棒性 | DropLayerRobustness | 移除该层后的性能损失 |
| 贝叶斯 | LaplacePosterior | 参数不确定性 |
| 架构特定 | AttentionFlow | 注意力熵、头多样性 (Transformer) |

### 集成桥

```python
# Torch-Pruning：重要层少剪，不重要层多剪
from uni_layer.integrations import TorchPruningBridge
bridge = TorchPruningBridge(model, contributions)
ratios = bridge.as_layer_pruning_ratios(target_sparsity=0.5)

# PEFT：自动选择 LoRA 目标层和自适应秩
from uni_layer.integrations import HuggingFacePEFTBridge
bridge = HuggingFacePEFTBridge(model, contributions)
config = bridge.recommend_lora_config_params()

# 蒸馏：教师-学生层配对和权重分配
from uni_layer.integrations import DistillationBridge
bridge = DistillationBridge(teacher, student, contributions)
pairs = bridge.recommend_layer_pairs(top_k=4)
```

### 路线图

**v0.3.0**：扩散模型支持 / Mamba-SSM / MoE 路由层分析 / 残差感知 DropLayer / 层间 CKA 矩阵

**v0.4.0**：GNN 支持 / 多模态分支分析 / Wanda 重要性 / IG 灵敏度 / ONNX 导出

**v1.0.0**：稳定 API / Web 可视化面板 / 分布式分析 / 预计算热门模型 / 学术论文

### 许可证

MIT License。详见 [LICENSE](LICENSE)。
