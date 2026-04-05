# Uni-Layer：神经网络层贡献度分析框架

**先理解你的层，再优化它们。**

[![PyPI](https://img.shields.io/pypi/v/uni-layer.svg)](https://pypi.org/project/uni-layer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-413%20passed-brightgreen.svg)]()

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
| `"full"` | 全部 26 指标 | 完整深度分析 |

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

## 支持的架构（7 大类）

| 架构 | 模型 | 层提取 |
|---|---|---|
| **Transformer** | BERT, GPT-2, LLaMA, Qwen, T5, ViT, DINOv2, Wav2Vec2 (20+) | 块级（自动检测） |
| **Mamba/SSM** | Mamba, S4, S6 | 块级（自动检测） |
| **GNN** | GCNConv, GATConv, SAGEConv (PyG) | 卷积层级 |
| **Diffusion** | UNet, DDPM, DiT | down/mid/up blocks |
| **MoE** | Mixtral, Switch Transformer, DeepSeek-V3 | 路由器 + 专家分析 |
| **多模态** | CLIP, LLaVA, Mistral-Small-3.1 | 分支级分析 |
| **CNN** | ResNet, ConvNeXt, EfficientNet | 块/层级 |

### 模型兼容性

Uni-Layer 使用**基于模式的 BFS 搜索**自动检测 transformer 块，无需针对特定模型编写代码。已在 **38 个模型家族**上通过真实 HuggingFace checkpoint 测试。

<details>
<summary><b>已测试模型家族 — 38 通过，5 部分通过（点击展开）</b></summary>

#### 完全通过（层检测 + 指标计算 + 分析报告）

| 模型家族 | 测试模型 | 块路径 |
|---|---|---|
| **Gemma 4 E2B** ⭐ (2026年4月，多模态) | `google/gemma-4-E2B` | `model.language_model.layers` |
| **Gemma 4 E4B** ⭐ (2026年4月，多模态) | `google/gemma-4-E4B` | `model.language_model.layers` |
| **Gemma 4 31B** ⭐ (2026年4月，多模态) | `google/gemma-4-31B` | `model.language_model.layers` |
| **Gemma 4 26B-A4B** ⭐ (2026年4月，多模态 MoE) | `google/gemma-4-26B-A4B` | `model.language_model.layers` |
| **Llama 4 Scout** ⭐ (2026年4月，多模态 MoE) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `language_model.model.layers` |
| **Qwen3** | `Qwen/Qwen3-0.6B` | `layers` |
| **Qwen2.5** | `Qwen/Qwen2.5-0.5B` | `layers` |
| **Qwen2** | `Qwen/Qwen2-0.5B` | `layers` |
| **Gemma 3** | `google/gemma-3-1b-pt` | `layers` |
| **Gemma 2** | `google/gemma-2-2b` | `layers` |
| **Gemma** | `google/gemma-7b` | `layers` |
| **LLaMA** | `tiny-random-LlamaForCausalLM` | `layers` |
| **TinyLlama** | `TinyLlama-1.1B-Chat-v1.0` | `layers` |
| **Mistral** | `tiny-random-MistralForCausalLM` | `layers` |
| **Mistral-3.1** (多模态) | `Mistral-Small-3.1-24B` | `language_model.layers` |
| **Mixtral** (MoE) | `tiny-random-MixtralForCausalLM` | `layers` |
| **Yi-1.5** | `01-ai/Yi-1.5-6B` | `layers` |
| **InternLM-2.5** | `internlm/internlm2_5-1_8b` | `model.layers` |
| **GLM-4** | `THUDM/glm-4-9b` | `transformer.encoder.layers` |
| **StarCoder2** | `bigcode/starcoder2-3b` | `layers` |
| **Falcon-3** | `tiiuae/Falcon3-1B-Base` | `layers` |
| **Nemotron** | `nvidia/Nemotron-Mini-4B-Instruct` | `layers` |
| **OLMo-2** | `allenai/OLMo-2-0425-1B` | `layers` |
| **StableLM** | `stabilityai/stablelm-3b-4e1t` | `layers` |
| **Pythia** | `EleutherAI/pythia-70m` | `layers` |
| **GPT-2 / Neo / J** | tiny-random 系列 | `h` |
| **GPT-NeoX** | tiny-random 系列 | `layers` |
| **BLOOM** | tiny-random 系列 | `h` |
| **Falcon** (经典) | tiny-random 系列 | `h` |
| **BERT / RoBERTa / ELECTRA / DeBERTa** | tiny-random 系列 | `encoder.layer` |
| **T5 / BART** | tiny-random 系列 | `encoder.block` + `decoder.block` |
| **ALBERT** | tiny-random 系列 | 逐层提取（权重共享） |

> ⭐ = 2026年4月验证的最新发布模型。Gemma 4 需要 `transformers>=5.0.0`；Llama 4 需要 `transformers>=4.51.0`。

#### 部分通过（层检测正常，指标计算需额外依赖）

| 模型家族 | 问题 |
|---|---|
| **DeepSeek-V3 / V2** (MoE) | 需要 `transformers>=4.48`（DynamicCache API） |
| **Jamba** (Mamba+Attn 混合) | 需要 [mamba-ssm](https://github.com/state-spaces/mamba) CUDA 内核 |
| **ViT** | 需要图像 tensor 输入（非 token ID） |
| **Phi-2** | 层检测正常；指标计算需匹配配置 |

</details>

**自定义模型**：任何 `nn.Module`，只要其块容器属性名为 `layer`、`layers`、`h`、`blocks` 或 `block`，均可自动检测。

运行兼容性测试：`python tests/test_model_compatibility.py`

---

## 26 种指标（9 大类别）

| 类别 | 指标 | 衡量内容 |
|---|---|---|
| **优化** (5) | `GradientNorm`, `HessianTrace`, `FisherInformation`, `WandaImportance`, `IGSensitivity` | 损失曲面、权重x激活重要性、IG 归因 |
| **谱方法** (3) | `CKA`, `EffectiveRank`, `NTKTrace` | 表征相似性、多样性、核影响力 |
| **信息论** (2) | `ActivationEntropy`, `MutualInformation` | 信息含量与任务相关性 |
| **表征** (2) | `JacobianRank`, `BlockInfluence` | 表达能力与层冗余度 |
| **鲁棒性** (2) | `DropLayerRobustness`, `ResidualDropLayer` | 消融（含/不含残差保留） |
| **贝叶斯** (1) | `LaplacePosterior` | 参数不确定性 |
| **效率** (4) | `EfficiencyProfiler`, `WeightDistribution`, `IntrinsicDimensionality`, `QuantizationSensitivity` | FLOPs、稀疏度、流形维度、量化噪声 |
| **安全** (4) | `AdversarialSensitivity`, `ActivationAnomalyScore`, `MembershipInferenceRisk`, `AttentionPathTrace` | 对抗鲁棒性、后门检测、隐私泄露、注入攻击 |
| **架构特定** (3) | `AttentionFlow`, `MoERouterAnalysis`, `DiffusionTimestepAnalysis` | 注意力头、MoE 路由、扩散时间步 |

详见 [docs/METRICS_CN.md](docs/METRICS_CN.md)。

---

## 集成桥（7 个）

```python
from uni_layer.integrations import (
    TorchPruningBridge,        # 结构化剪枝 (Torch-Pruning)
    HuggingFacePEFTBridge,     # LoRA/Adapter (HuggingFace PEFT)
    DistillationBridge,        # 知识蒸馏层配对
    ExportHintsBridge,         # ONNX/TensorRT 量化+融合建议
    AxolotlConfigBridge,       # Axolotl YAML 配置生成
    LLaMAFactoryConfigBridge,  # LLaMA-Factory JSON 配置生成
    CompressionSafetyAudit,    # 压缩前后安全审计
)
```

### 剪枝

```python
bridge = TorchPruningBridge(model, contributions)
ratios = bridge.as_layer_pruning_ratios(target_sparsity=0.5)
```

### LoRA 微调

```python
bridge = HuggingFacePEFTBridge(model, contributions)
config = bridge.recommend_lora_config_params(metric_name='gradient_norm')
```

### 量化部署

```python
bridge = ExportHintsBridge(model, contributions)
plan = bridge.quantization_plan(target="int8", protect_ratio=0.2)
config = bridge.tensorrt_config()
```

### LLM 训练框架

```python
AxolotlConfigBridge(model, contributions).save_yaml("config.yml", base_model="meta-llama/Llama-2-7b")
LLaMAFactoryConfigBridge(model, contributions).save_json("config.json", model_name="my-model")
```

### 安全审计

```python
audit = CompressionSafetyAudit(pre_contributions, post_contributions)
report = audit.audit()  # overall_degradation, recommendations
```

---

## CLI 命令行

```bash
uni-layer info                                    # 版本、PyTorch、CUDA、指标数
uni-layer list-metrics                            # 列出全部 26 种指标
uni-layer list-metrics --format json              # JSON 格式（方便程序解析）
uni-layer analyze bert-base-uncased               # 分析 HuggingFace 模型
uni-layer analyze bert-base-uncased -p llm_fast --profile  # 预设 + 洞察报告
uni-layer analyze model -m GradientNorm,CKA -o results.json
```

---

## MCP Server（Claude Code / AI 助手集成）

Uni-Layer 提供 [MCP](https://modelcontextprotocol.io/) 服务器，可集成到 Claude Code、Codex 等 AI 编程助手中。

```bash
pip install uni-layer[mcp]
```

在 `~/.claude/settings.json` 中配置：

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

可用工具：`list_metrics`、`analyze_model`、`layer_profile`、`suggest_pruning`、`suggest_lora`、`suggest_quantization`、`security_audit`

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

### v0.6.0
- [x] 安全与红队分析指标（AdversarialSensitivity, ActivationAnomalyScore, MembershipInferenceRisk, AttentionPathTrace）
- [x] 压缩安全审计 + `LayerProfile.security_report()`

### v0.6.1
- [x] 效率指标（`efficiency/` 类别）
  - [x] EfficiencyProfiler：逐层 FLOPs、参数量、内存、计算占比
  - [x] WeightDistribution：稀疏度、范数、秩缺陷、峰度
  - [x] IntrinsicDimensionality：MLE 流形维度估计（LoRA 秩选择）
  - [x] QuantizationSensitivity：INT8/FP16 量化噪声容忍度

### v0.7.0（当前版本）
- [x] **38 个模型家族兼容性** — Gemma 4（E2B/E4B/31B/26B-A4B）、Llama 4 Scout、DeepSeek-R1-Distill、Qwen3.5（2026年4月新发布模型全部验证通过）
- [x] **MCP Server** — 7 工具 FastMCP 服务器，支持 Claude Code / AI 助手集成
- [x] **CLI 增强** — 全量开放 26 个指标，新增 `--preset`、`--profile` 参数
- [x] **Claude Code 技能** — `/layer-analyze` 编辑器内分析
- [x] Bug 修复：CausalLM 标签注入、Gemma 4 `mm_token_type_ids`、音频塔去优先级

### v0.8.0
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
