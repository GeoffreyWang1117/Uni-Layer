# 快速开始指南 (v0.6.1)

5 分钟上手 Uni-Layer，分析任意 PyTorch 模型的 26 个指标。

## 安装

```bash
pip install uni-layer
pip install uni-layer[integrations,viz,science]  # 含可选依赖
```

## 基本用法

```python
from uni_layer import LayerAnalyzer, LayerProfile

# 初始化分析器（自动检测层结构）
analyzer = LayerAnalyzer(model, task_type="classification")

# 使用预设运行分析
contributions = analyzer.compute_metrics(preset="llm_fast", data_loader=loader)

# 获取自动洞察
profile = LayerProfile(contributions, model_name="my-model")
print(profile.summary())
profile.print_report()
```

## 预设选项

```python
# 秒级 — 快速筛选
contributions = analyzer.compute_metrics(preset="llm_fast", data_loader=loader)

# 分钟级 — 详细分析
contributions = analyzer.compute_metrics(preset="llm_full", data_loader=loader)

# 全部 26 个指标
contributions = analyzer.compute_metrics(preset="full", data_loader=loader)
```

## CKA 相似度矩阵

```python
from uni_layer import CKASimilarity

cka = CKASimilarity(model)
matrix, names = cka.compute(loader, num_batches=10)
print(cka.most_similar_pairs(matrix, names, top_k=5))
print(cka.layer_uniqueness(matrix, names))
```

## 多模态模型

```python
from uni_layer import MultiModalBranchAnalyzer

mm = MultiModalBranchAnalyzer(clip_model)
print(mm.branch_names)     # ["vision", "language", "fusion"]
print(mm.branch_summary()) # 各分支的参数量和层数
```

## 安全分析

```python
from uni_layer.metrics import (
    AdversarialSensitivity, ActivationAnomalyScore,
    MembershipInferenceRisk,
)

contributions = analyzer.compute_metrics(
    metrics=[AdversarialSensitivity(), ActivationAnomalyScore(), MembershipInferenceRisk()],
    data_loader=loader,
)

report = LayerProfile(contributions).security_report()
print(report["summary"])
```

## 效率分析

```python
from uni_layer.metrics import EfficiencyProfiler, WeightDistribution, QuantizationSensitivity

# 权重分析（无需数据！）
wd = WeightDistribution()
for name, layer in analyzer.layers.items():
    result = wd.compute(model=model, layer=layer, layer_name=name, layer_idx=0, device="cpu")
    print(f"{name}: 稀疏度={result['weight_sparsity']:.3f}")
```

## 集成桥接 (7 个)

```python
from uni_layer.integrations import (
    TorchPruningBridge,       # 结构化剪枝
    HuggingFacePEFTBridge,    # LoRA/Adapter 配置
    DistillationBridge,       # 知识蒸馏
    ExportHintsBridge,        # ONNX/TensorRT 提示
    AxolotlConfigBridge,      # Axolotl YAML 生成
    LLaMAFactoryConfigBridge, # LLaMA-Factory JSON
    CompressionSafetyAudit,   # 压缩安全审计
)
```

## 支持的架构

| 架构 | 示例模型 |
|------|---------|
| Transformer | BERT, GPT-2, LLaMA, Qwen, T5, ViT |
| CNN | ResNet, ConvNeXt, EfficientNet |
| Mamba/SSM | Mamba, S4, S6 |
| GNN | GCNConv, GATConv, SAGEConv (PyG) |
| Diffusion | UNet, DDPM, DiT |
| MoE | Mixtral, Switch Transformer |
| Multi-Modal | CLIP, LLaVA |

## 命令行

```bash
uni-layer info                          # 环境信息
uni-layer list-metrics                  # 全部 26 个指标
uni-layer analyze bert-base-uncased     # 分析 HF 模型
```

## 文档导航

- [指标参考](METRICS_CN.md) — 全部 26 个指标详解
- [压缩指南](COMPRESSION_CN.md) — 剪枝/蒸馏/LoRA
- [API 参考](API.md) — 完整类/方法参考 (English)
- [架构指南](ARCHITECTURE.md) — 各架构使用示例 (English)
- [安全指南](SECURITY.md) — 红队分析工作流 (English)
