# Uni-Layer Documentation (v0.6.1)

**26 metrics | 9 categories | 7 integrations | 7 architectures**

## English

| Document | Description |
|----------|-------------|
| [Quick Start](QUICKSTART.md) | Get started in 5 minutes |
| [Metric Reference](METRICS.md) | All 26 metrics with output keys and usage |
| [API Reference](API.md) | Complete class and method reference |
| [Architecture Guide](ARCHITECTURE.md) | Per-architecture examples (Transformer, Mamba, GNN, Diffusion, MoE, MultiModal) |
| [Security Guide](SECURITY.md) | Red-team analysis, backdoor detection, privacy audit |
| [Compression Guide](COMPRESSION.md) | Pruning, distillation, LoRA workflows |

## Chinese

| Document | Description |
|----------|-------------|
| [Quick Start](QUICKSTART_CN.md) | 5 minutes |
| [Metric Reference](METRICS_CN.md) | 26 |
| [Compression Guide](COMPRESSION_CN.md) |  |
| [API Reference (partial)](API_CN.md) | API  |

## Metric Categories at a Glance

```
Optimization (5)     GradientNorm, HessianTrace, FisherInformation, WandaImportance, IGSensitivity
Spectral (3)         CKA, EffectiveRank, NTKTrace
Information Theory (2) ActivationEntropy, MutualInformation
Representation (2)   BlockInfluence, JacobianRank
Robustness (2)       DropLayerRobustness, ResidualDropLayer
Bayesian (1)         LaplacePosterior
Efficiency (4)       EfficiencyProfiler, WeightDistribution, IntrinsicDimensionality, QuantizationSensitivity
Security (4)         AdversarialSensitivity, ActivationAnomalyScore, MembershipInferenceRisk, AttentionPathTrace
Arch-Specific (3)    AttentionFlow, MoERouterAnalysis, DiffusionTimestepAnalysis
```

## Integration Bridges

```
TorchPruningBridge        -> Structural pruning (DepGraph)
HuggingFacePEFTBridge     -> LoRA/AdaLoRA/Adapter config
DistillationBridge        -> Knowledge distillation layer pairing
ExportHintsBridge         -> ONNX/TensorRT quantization + fusion
AxolotlConfigBridge       -> Axolotl YAML generation
LLaMAFactoryConfigBridge  -> LLaMA-Factory JSON generation
CompressionSafetyAudit    -> Pre/post compression security comparison
```
