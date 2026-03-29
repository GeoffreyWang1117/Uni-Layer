# Uni-Layer：神经网络层贡献度分析框架

**先理解你的层，再优化它们。**

[![PyPI](https://img.shields.io/pypi/v/uni-layer.svg)](https://pypi.org/project/uni-layer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-387%20passed-brightgreen.svg)]()

Uni-Layer 是一个 PyTorch 工具库，通过 **9 大理论类别的 26 种指标** 为神经网络的每一层打分。支持 Transformer、Mamba/SSM、GNN、Diffusion、MoE、多模态等 7 种架构。提供安全审计、效率分析、量化敏感度等全面分析能力。

> 完整英文文档见 [README.md](README.md)

---

## 快速开始

```bash
pip install uni-layer
```

```python
from uni_layer import LayerAnalyzer, LayerProfile

analyzer = LayerAnalyzer(model, task_type='classification')

# 一行分析（推荐大模型使用 preset）
contributions = analyzer.compute_metrics(data_loader=loader, preset="llm_fast")

# 自动生成洞察（纯 CPU，毫秒级）
profile = LayerProfile(contributions, model_name="my-model")
print(profile.summary())
print(profile.pruning_suggestion(0.3))
print(profile.lora_suggestion(8))
```

---

## Presets（预设）

| 预设 | 包含指标 | 用途 |
|---|---|---|
| `"llm_fast"` | BlockInfluence, EffectiveRank, CKA, Entropy, AttentionFlow | 大模型快速扫描（秒级） |
| `"llm_full"` | + GradientNorm, FisherInformation | 大模型完整分析（分钟级） |
| `"quick"` | GradientNorm, BlockInfluence, EffectiveRank | 最快概览 |
| `"full"` | 全部 13 指标 | 小模型深度分析 |

---

## LayerProfile 自动分析

`LayerProfile` 从 `compute_metrics()` 的原始数字中自动提取洞察，不需要额外 GPU 计算：

```python
profile = LayerProfile(contributions, model_name="Llama-3.2-3B")

profile.redundant_layers        # ["layers.14", "layers.15"] — 可安全剪除
profile.bottleneck_layers       # ["layers.30"] — 表征瓶颈
profile.consensus_ranking       # 多指标共识排名（Borda count）
profile.depth_trends            # {"gradient_norm": {"trend": "U-shaped", ...}}
profile.anomalies               # z-score > 2 的异常层
profile.layer_clusters          # {"high_contribution": [...], "low_contribution": [...]}

profile.pruning_suggestion(0.3) # {"safe_to_remove": [...], "estimated_speedup": "25%"}
profile.lora_suggestion(8)      # {"target_layers": [...], "adaptive_ranks": {...}}
profile.summary()               # 一段话自然语言总结
profile.to_dict()               # 完整 JSON 导出
```

示例摘要输出：
> "bert-base-uncased (12-layer model). analyzed with 10 metrics. U-shaped gradient norm distribution. 2 redundant layers (encoder.layer.5, encoder.layer.6) safe to prune. most important: encoder.layer.9."

---

## 输出格式

`compute_metrics()` 返回结构化字典：

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

`rank_layers()` 返回排序后的 `(层名, 分数)` 元组列表：

```python
[("encoder.layer.9", 0.1094), ("encoder.layer.0", 0.0624), ...]
```

---

## 已验证 20+ 个 HuggingFace 模型

| 架构 | 模型 | 层提取 |
|---|---|---|
| NLP Encoder | BERT, RoBERTa, DeBERTa-v3, DistilBERT, SciBERT, MiniLM | `encoder.layer.N` |
| NLP Decoder | GPT-2, Pythia, BLOOM, Falcon, TinyLlama, Llama-3.2-3B, Qwen2.5-3B | `model.layers.N` |
| Seq2Seq | ByT5 | `encoder.block.N` + `decoder.block.N` |
| Vision | DINOv2 | `encoder.layer.N` |
| Speech | Wav2Vec2, HuBERT | `encoder.layers.N` |

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

---

## 集成桥

### Torch-Pruning（剪枝）

```python
from uni_layer.integrations import TorchPruningBridge

bridge = TorchPruningBridge(model, contributions)
pruning_ratios = bridge.as_layer_pruning_ratios(target_sparsity=0.5)
protected = bridge.get_protected_layers(top_k=3)
```

### HuggingFace PEFT（参数高效微调）

```python
from uni_layer.integrations import HuggingFacePEFTBridge
from peft import LoraConfig, get_peft_model

bridge = HuggingFacePEFTBridge(model, contributions)
config_params = bridge.recommend_lora_config_params(metric_name='gradient_norm')
peft_model = get_peft_model(model, LoraConfig(**config_params))
```

### 知识蒸馏

```python
from uni_layer.integrations import DistillationBridge

bridge = DistillationBridge(teacher, student, contributions)
pairs = bridge.recommend_layer_pairs(top_k=4)
weights = bridge.recommend_layer_weights()
```

---

## CLI 命令行

```bash
uni-layer info                                    # 版本、PyTorch、CUDA、指标数
uni-layer list-metrics                            # 列出全部 13 种指标
uni-layer list-metrics --format json              # JSON 格式（方便程序解析）
uni-layer analyze bert-base-uncased               # 分析 HuggingFace 模型
uni-layer analyze model -m GradientNorm,CKA -o results.json
```

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

### v0.4.0
- [x] Diffusion 模型支持（UNet 时间步感知分析）
- [x] Mamba / SSM 架构支持
- [x] MoE 路由层分析
- [x] 残差感知 DropLayer 指标
- [x] 层间 CKA 相似度矩阵
- [x] GNN 支持（PyG MessagePassing 层）

### v0.5.0
- [x] 多模态模型分支分析（视觉编码器 + 语言解码器）
- [x] Wanda 风格重要性（权重 x 激活范数）
- [x] IG 灵敏度评分（IGU-LoRA 风格）
- [x] LLM 训练框架集成（Axolotl / LLaMA-Factory）
- [x] ONNX / TensorRT 优化提示导出

### v0.6.0（当前版本）
- [x] 安全与红队分析指标（`security/` 类别）
  - [x] AdversarialSensitivity：逐层 FGSM/PGD 扰动敏感度
  - [x] ActivationAnomalyScore：基于激活模式异常的后门检测
  - [x] MembershipInferenceRisk：逐层梯度泄露风险评分
  - [x] AttentionPathTrace：对抗性注意力流 / Prompt 注入路径分析
- [x] 压缩安全审计（剪枝/量化前后的安全性退化量化）
- [x] `LayerProfile.security_report()` 自动化漏洞摘要

### v0.7.0
- [ ] LLM 推理 KV Cache 分析
  - [ ] KV cache 冗余检测（跨层 / 跨头相似度）
  - [ ] KV cache 压缩建议（逐层预算分配）
  - [ ] Cache 信息泄露评分
- [ ] 推理框架集成（SGLang / vLLM）
  - [ ] 基于推理引擎 hook 的运行时层分析
  - [ ] 服务感知的重要性评分（延迟 vs. 质量权衡）
- [ ] 对抗性 Prompt 攻击路径分析
  - [ ] 逐层对越狱 / 注入 prompt 的脆弱性
  - [ ] 跨解码步的注意力劫持检测
- [ ] 推理攻击面分析
  - [ ] 逐层模型反演风险
  - [ ] 嵌入提取漏洞评分

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
