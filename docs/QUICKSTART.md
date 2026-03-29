# Quick Start Guide (v0.6.1)

Get started with Uni-Layer in 5 minutes. Analyze any PyTorch model across 26 metrics.

## Installation

```bash
pip install uni-layer

# With all optional dependencies
pip install uni-layer[integrations,viz,science]
```

## 1. Basic Analysis

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from uni_layer import LayerAnalyzer, LayerProfile

# Your model + data
model = ...  # any PyTorch model
X, y = torch.randn(500, 768), torch.randint(0, 10, (500,))
loader = DataLoader(TensorDataset(X, y), batch_size=32)

# Initialize analyzer (auto-detects layers)
analyzer = LayerAnalyzer(model, task_type="classification")

# Run analysis with a preset
contributions = analyzer.compute_metrics(preset="llm_fast", data_loader=loader)

# Get automatic insights
profile = LayerProfile(contributions, model_name="my-model")
print(profile.summary())
```

## 2. Available Presets

```python
# Seconds — quick screening
contributions = analyzer.compute_metrics(preset="llm_fast", data_loader=loader)

# Minutes — thorough analysis
contributions = analyzer.compute_metrics(preset="llm_full", data_loader=loader)

# All 26 metrics
contributions = analyzer.compute_metrics(preset="full", data_loader=loader)
```

## 3. Specific Metrics

```python
from uni_layer.metrics import GradientNorm, BlockInfluence, WandaImportance

contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), BlockInfluence(), WandaImportance()],
    data_loader=loader,
    num_batches=10,
)
```

## 4. LayerProfile Insights

```python
profile = LayerProfile(contributions, model_name="bert-base")

# Automatic analysis (no GPU needed)
profile.redundant_layers      # Layers safe to prune
profile.bottleneck_layers     # Representation bottlenecks
profile.consensus_ranking     # Multi-metric agreement ranking
profile.anomalies             # Statistically unusual layers
profile.depth_trends          # How metrics evolve with depth

# Actionable suggestions
profile.pruning_suggestion(target_ratio=0.3)  # Pruning plan
profile.lora_suggestion(base_rank=8)          # LoRA targets + ranks

# Display
profile.print_report()        # Full report
profile.print_table()         # Compact table
```

## 5. CKA Similarity Matrix

```python
from uni_layer import CKASimilarity

cka = CKASimilarity(model)
matrix, layer_names = cka.compute(loader, num_batches=10)

# Find redundant layer pairs
print(cka.most_similar_pairs(matrix, layer_names, top_k=5))
print(cka.redundant_layers(matrix, layer_names, threshold=0.95))
print(cka.layer_uniqueness(matrix, layer_names))
```

## 6. Multi-Modal Models

```python
from uni_layer import MultiModalBranchAnalyzer

mm = MultiModalBranchAnalyzer(clip_model)
print(mm.branches)          # {'vision': ..., 'language': ..., 'fusion': ...}
print(mm.branch_summary())  # params, layers per branch

# Analyze each branch separately
vision_layers = mm.get_branch_layers("vision")
```

## 7. Security Analysis

```python
from uni_layer.metrics import (
    AdversarialSensitivity, ActivationAnomalyScore,
    MembershipInferenceRisk, AttentionPathTrace,
)

# Run security metrics
contributions = analyzer.compute_metrics(
    metrics=[AdversarialSensitivity(), ActivationAnomalyScore(),
             MembershipInferenceRisk()],
    data_loader=loader,
)

# Automated vulnerability report
profile = LayerProfile(contributions)
report = profile.security_report()
print(report["summary"])
print(report["top_risks"])
```

## 8. Efficiency Profiling

```python
from uni_layer.metrics import (
    EfficiencyProfiler, WeightDistribution,
    IntrinsicDimensionality, QuantizationSensitivity,
)

# Weight analysis (no data needed!)
wd = WeightDistribution()
for name, layer in analyzer.layers.items():
    result = wd.compute(model=model, layer=layer, layer_name=name,
                        layer_idx=0, device="cpu")
    print(f"{name}: sparsity={result['weight_sparsity']:.3f}, "
          f"rank_ratio={result['weight_rank_ratio']:.3f}")

# FLOPs + quantization sensitivity
contributions = analyzer.compute_metrics(
    metrics=[EfficiencyProfiler(), QuantizationSensitivity()],
    data_loader=loader,
)
```

## 9. Integration Bridges

```python
from uni_layer.integrations import (
    TorchPruningBridge,       # Structural pruning
    HuggingFacePEFTBridge,    # LoRA/Adapter config
    DistillationBridge,       # Knowledge distillation
    ExportHintsBridge,        # ONNX/TensorRT hints
    AxolotlConfigBridge,      # Axolotl YAML generation
    LLaMAFactoryConfigBridge, # LLaMA-Factory JSON
    CompressionSafetyAudit,   # Pre/post compression security
)

# Example: generate LoRA config
peft_bridge = HuggingFacePEFTBridge(model, contributions)
print(peft_bridge.recommend_target_modules("gradient_norm", top_k=4))
print(peft_bridge.recommend_adaptive_ranks("gradient_norm", base_rank=16))

# Example: ONNX/TensorRT quantization plan
export = ExportHintsBridge(model, contributions)
print(export.quantization_plan(target="int8", protect_ratio=0.2))
print(export.tensorrt_config())

# Example: Axolotl config
axolotl = AxolotlConfigBridge(model, contributions)
axolotl.save_yaml("config.yml", base_model="meta-llama/Llama-2-7b")
```

## 10. Supported Architectures

Uni-Layer auto-detects layer structure for:

| Architecture | Examples | Layer Extraction |
|---|---|---|
| **Transformer** | BERT, GPT-2, LLaMA, Qwen, T5, ViT | Block-level (encoder.layer.N) |
| **CNN** | ResNet, ConvNeXt, EfficientNet | Block/layer level |
| **Mamba/SSM** | Mamba, S4, S6 | Block-level (auto-detected) |
| **GNN** | GCNConv, GATConv, SAGEConv (PyG) | Conv-level (MessagePassing) |
| **Diffusion** | UNet, DDPM, DiT | down/mid/up blocks |
| **MoE** | Mixtral, Switch Transformer | Router + expert analysis |
| **Multi-Modal** | CLIP, LLaVA | Per-branch analysis |

## CLI

```bash
uni-layer info                          # Environment info
uni-layer list-metrics                  # All 26 metrics
uni-layer analyze bert-base-uncased     # Analyze HF model
uni-layer analyze bert-base-uncased -m GradientNorm,BlockInfluence -o results.json
```

## Next Steps

- [Metric Reference](METRICS.md) - All 26 metrics with output keys
- [Architecture Guide](ARCHITECTURE.md) - Per-architecture examples
- [Security Guide](SECURITY.md) - Red-team analysis workflows
- [Compression Guide](COMPRESSION.md) - Pruning/distillation/LoRA
- [API Reference](API.md) - Complete class/method reference
