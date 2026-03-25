# Uni-Layer：神经网络层贡献度分析框架

**先理解你的层，再优化它们。**

[![PyPI](https://img.shields.io/pypi/v/uni-layer.svg)](https://pypi.org/project/uni-layer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-168%20passed-brightgreen.svg)]()

Uni-Layer 是一个 PyTorch 工具库，通过 **7 大理论类别的 13 种指标** 为神经网络的每一层打分。它告诉你哪些层最重要，从而实现更精准的剪枝、更高效的微调和更有效的蒸馏。

> 完整英文文档见 [README.md](README.md)

---

## 为什么需要 Uni-Layer？

大多数压缩和微调工具对所有层一视同仁，或依赖简单的权重大小启发式。Uni-Layer 用原则性的多指标分析替代了猜测。

**在"层重要性通用评分"这个赛道上没有直接竞品：**
- Captum 做输入归因，不做层排序
- Torch-Pruning 的重要性指标只有 3-4 种，且与剪枝操作耦合
- TransformerLens 只做 LLM 机制解释
- **Uni-Layer 是唯一将 13 种层重要性指标统一到一个 API 并桥接到下游工具的库**

| 你想做... | Uni-Layer 提供 | 配合使用 |
|---|---|---|
| **剪枝** | 每层重要性分数和剪枝比例 | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| **LoRA 微调** | 目标层选择、自适应秩分配 | [HuggingFace PEFT](https://github.com/huggingface/peft) |
| **知识蒸馏** | 层配对、每层蒸馏权重 | 任意蒸馏框架 |
| **理解模型** | 多指标层贡献度画像 | 独立使用 |

---

## 快速开始

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

---

## 输出格式

### compute_metrics() 输出

返回结构化字典，每一层包含所有请求的指标值：

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

### rank_layers() 输出

返回按分数降序排列的 `(层名, 分数)` 元组列表：

```python
[("6", 0.1094), ("4", 0.0624), ("2", 0.0494), ("0", 0.0193)]
# 第 6 层（输出头）贡献最大；第 0 层（输入层）贡献最小
```

### Transformer 模型输出示例

4 层 Transformer 的分析结果：

```
Layer              Type                  GradNorm  BlockInfluence  EffectiveRank
--------------------------------------------------------------------------------
blocks.0           transformer_block       0.1425          0.0278          94.47
blocks.1           transformer_block       0.1404          0.0275          94.21
blocks.2           transformer_block       0.1319          0.0265          93.92
blocks.3           transformer_block       0.1276          0.0269          93.66
```

> 浅层 block 梯度范数稍高——它们在更积极地适应。BlockInfluence 全局较低（~0.027），因为残差连接占主导地位，每个 block 相对于跳跃路径的变换很小。EffectiveRank 均匀很高（~94），表明表征丰富、未退化。

---

## 13 种指标（7 大类别）

| 类别 | 指标 | 衡量内容 |
|---|---|---|
| **优化几何** | `GradientNorm`, `HessianTrace`, `FisherInformation` | 层对损失曲面的影响 |
| **谱与核方法** | `CKA`, `EffectiveRank`, `NTKTrace` | 表征相似性、多样性、核影响力 |
| **信息论** | `ActivationEntropy`, `MutualInformation` | 信息含量与任务相关性 |
| **表征结构** | `JacobianRank`, `BlockInfluence` | 表达能力与层冗余度 |
| **鲁棒性** | `DropLayerRobustness` | 移除该层后的性能损失 |
| **贝叶斯** | `LaplacePosterior` | 参数不确定性（Laplace 近似） |
| **架构特定** | `AttentionFlow` | 注意力熵、头多样性（Transformer 专属） |

### 每个指标的输出键

| 指标 | 主键（用于排序） | 附属键 |
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

## 集成桥

### Torch-Pruning（剪枝）

```python
from uni_layer.integrations import TorchPruningBridge

bridge = TorchPruningBridge(model, contributions)

# 重要层少剪，不重要层多剪
pruning_ratios = bridge.as_layer_pruning_ratios(
    metric_name='gradient_norm', target_sparsity=0.5
)
protected = bridge.get_protected_layers(top_k=3)  # 保护最重要的 3 层
```

### HuggingFace PEFT（参数高效微调）

```python
from uni_layer.integrations import HuggingFacePEFTBridge
from peft import LoraConfig, get_peft_model

bridge = HuggingFacePEFTBridge(model, contributions)

# 一键生成 LoRA 配置
config_params = bridge.recommend_lora_config_params(metric_name='gradient_norm')
peft_model = get_peft_model(model, LoraConfig(**config_params))

# 或精细控制：每层不同的 LoRA 秩
ranks = bridge.recommend_adaptive_ranks(base_rank=8, max_rank=64)
```

### 知识蒸馏

```python
from uni_layer.integrations import DistillationBridge

bridge = DistillationBridge(teacher, student, contributions)

pairs = bridge.recommend_layer_pairs(top_k=4)    # 教师-学生层映射
weights = bridge.recommend_layer_weights()         # 每层蒸馏权重
```

---

## HuggingFace 模型支持

Uni-Layer 原生兼容 HuggingFace 模型（dict/dataclass 输出），自动注入 `attention_mask`：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
analyzer = LayerAnalyzer(model, task_type='classification')

# 直接使用——dict 输出、attention_mask、labels 全自动处理
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), BlockInfluence()],
    data_loader=tokenized_loader,
)
```

---

## 示例

| 示例 | 模型 | 文件 |
|---|---|---|
| ResNet 层分析 | ResNet-18 (CNN) | [`examples/resnet_layer_analysis.py`](examples/resnet_layer_analysis.py) |
| ViT 注意力分析 | Vision Transformer | [`examples/vit_layer_analysis.py`](examples/vit_layer_analysis.py) |
| BERT 层分析 + LoRA | BERT 风格 Transformer | [`examples/bert_layer_analysis.py`](examples/bert_layer_analysis.py) |
| Torch-Pruning 集成 | 通用模型 | [`examples/integrate_torch_pruning.py`](examples/integrate_torch_pruning.py) |
| HuggingFace PEFT 集成 | 通用模型 | [`examples/integrate_huggingface_peft.py`](examples/integrate_huggingface_peft.py) |
| 知识蒸馏 | 教师-学生 | [`examples/integrate_distillation.py`](examples/integrate_distillation.py) |

---

## 安装

```bash
pip install uni-layer                    # 核心
pip install uni-layer[integrations]      # + torch-pruning, peft, transformers
pip install uni-layer[dev]               # + pytest, black, flake8, mypy
pip install uni-layer[all]               # 全部
```

从源码安装：

```bash
git clone https://github.com/GeoffreyWang1117/Uni-Layer.git
cd Uni-Layer && pip install -e ".[dev]"
```

---

## 路线图

### v0.3.0（下一版本）
- [ ] 扩散模型支持（UNet 时间步感知分析）
- [ ] Mamba / SSM 架构支持
- [ ] MoE 路由层分析
- [ ] 残差感知 DropLayer 指标
- [ ] 层间 CKA 相似度矩阵

### v0.4.0
- [ ] GNN 支持（PyG MessagePassing 层）
- [ ] 多模态模型分支分析（视觉编码器 + 语言解码器）
- [ ] Wanda 风格重要性（权重 x 激活范数）
- [ ] IG 灵敏度评分（IGU-LoRA 风格）
- [ ] ONNX / TensorRT 优化提示导出

### v1.0.0
- [ ] 稳定 API，完全向后兼容
- [ ] Web 交互式可视化面板
- [ ] 大模型分布式分析（FSDP / DeepSpeed）
- [ ] 热门模型预计算分析（BERT、LLaMA、ViT 等）
- [ ] 学术论文与完整基准评测套件

---

## 添加自定义指标

```python
from uni_layer.core.base_metric import LayerMetric

class MyMetric(LayerMetric):
    def __init__(self, **kwargs):
        super().__init__(name="my_metric", category="custom",
                         requires_gradient=True, requires_data=True, **kwargs)

    def compute(self, model, layer, layer_name, layer_idx,
                data_loader, device, **kwargs):
        # 你的指标计算逻辑
        return {"my_metric": value}
```

---

## 引用

```bibtex
@software{unilayer2025,
  title={Uni-Layer: A Universal Framework for Layer Contribution Analysis},
  author={Geoffrey Wang},
  year={2025},
  url={https://github.com/GeoffreyWang1117/Uni-Layer}
}
```

## 许可证

MIT License。详见 [LICENSE](LICENSE)。

---

**GitHub**: [GeoffreyWang1117/Uni-Layer](https://github.com/GeoffreyWang1117/Uni-Layer) | **Issues**: [问题反馈](https://github.com/GeoffreyWang1117/Uni-Layer/issues)
