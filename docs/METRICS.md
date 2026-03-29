# Metric Reference (v0.6.1)

Uni-Layer provides **26 metrics** across **9 theoretical categories**. Each metric is a self-contained `LayerMetric` subclass that can be used independently or combined via `LayerAnalyzer`.

## Quick Reference Table

| # | Metric | Category | Gradient? | Data? | Primary Key | Cost |
|---|--------|----------|-----------|-------|-------------|------|
| 1 | GradientNorm | Optimization | Yes | Yes | `gradient_norm` | Low |
| 2 | HessianTrace | Optimization | Yes | Yes | `hessian_trace` | High |
| 3 | FisherInformation | Optimization | Yes | Yes | `fisher_information` | Med |
| 4 | WandaImportance | Optimization | No | Yes | `wanda_score` | Low |
| 5 | IGSensitivity | Optimization | Yes | Yes | `ig_sensitivity` | High |
| 6 | CKA | Spectral | No | Yes | `cka_score` | Med |
| 7 | EffectiveRank | Spectral | No | Yes | `effective_rank` | Med |
| 8 | NTKTrace | Spectral | Yes | Yes | `ntk_trace` | High |
| 9 | ActivationEntropy | Info Theory | No | Yes | `activation_entropy` | Low |
| 10 | MutualInformation | Info Theory | No | Yes | `mutual_information` | Med |
| 11 | JacobianRank | Representation | Yes | Yes | `jacobian_rank` | High |
| 12 | BlockInfluence | Representation | No | Yes | `block_influence` | Low |
| 13 | DropLayerRobustness | Robustness | No | Yes | `droplayer_loss_increase` | Med |
| 14 | ResidualDropLayer | Robustness | No | Yes | `residual_droplayer_loss_increase` | Med |
| 15 | LaplacePosterior | Bayesian | Yes | Yes | `laplace_posterior` | High |
| 16 | EfficiencyProfiler | Efficiency | No | Yes | `flops` | Low |
| 17 | WeightDistribution | Efficiency | No | No | `weight_sparsity` | Low |
| 18 | IntrinsicDimensionality | Efficiency | No | Yes | `intrinsic_dim` | Med |
| 19 | QuantizationSensitivity | Efficiency | No | Yes | `quant_sensitivity_int8` | Med |
| 20 | AdversarialSensitivity | Security | Yes | Yes | `adv_sensitivity` | Med |
| 21 | ActivationAnomalyScore | Security | No | Yes | `activation_skewness` | Low |
| 22 | MembershipInferenceRisk | Security | Yes | Yes | `mi_risk_score` | Med |
| 23 | AttentionPathTrace | Security | No | Yes | `injection_vulnerability` | Med |
| 24 | AttentionFlow | Arch-Specific | No | Yes | `attention_entropy` | Med |
| 25 | MoERouterAnalysis | Arch-Specific | No | Yes | `routing_entropy` | Med |
| 26 | DiffusionTimestepAnalysis | Arch-Specific | No | Yes | `timestep_sensitivity` | High |

---

## Presets

```python
contributions = analyzer.compute_metrics(preset="llm_fast", data_loader=loader)
```

| Preset | Metrics | Use Case |
|--------|---------|----------|
| `llm_fast` | BlockInfluence, EffectiveRank, CKA, ActivationEntropy, AttentionFlow | Quick LLM screening (seconds) |
| `llm_full` | Above + GradientNorm, FisherInformation | Thorough LLM analysis (minutes) |
| `quick` | GradientNorm, BlockInfluence, EffectiveRank | Rapid importance check |
| `full` | All metrics including ResidualDropLayer | Complete analysis |

---

## Category 1: Optimization (5 metrics)

### GradientNorm

Layer importance from gradient magnitude.

**Output:** `gradient_norm`, `gradient_norm_std`, `gradient_norm_max`, `gradient_norm_min`

```python
m = GradientNorm(num_batches=10)
```

**Use:** General-purpose importance ranking, pruning decisions, LoRA target selection.

### HessianTrace

Loss landscape curvature via Hutchinson trace estimator.

**Output:** `hessian_trace`, `hessian_trace_std`

```python
m = HessianTrace(num_samples=5, num_batches=10)
```

**Use:** Loss landscape analysis, learning rate sensitivity, quantization-aware importance.

### FisherInformation

Empirical Fisher Information Matrix trace.

**Output:** `fisher_information`, `fisher_mean`

```python
m = FisherInformation(num_batches=10)
```

**Use:** EWC-style continual learning, Bayesian pruning.

### WandaImportance

Weight x activation norm. **No gradient required.** (Sun et al., ICLR 2024)

**Output:** `wanda_score`, `weight_norm`, `activation_norm`, `wanda_sparsity`

```python
m = WandaImportance(num_batches=10, sparsity_threshold=1e-3)
```

**Use:** Gradient-free LLM pruning at scale, unstructured pruning.

### IGSensitivity

Integrated Gradients per-layer attribution via path integral.

**Output:** `ig_sensitivity`, `ig_variance`, `ig_relative`

```python
m = IGSensitivity(num_steps=10, num_batches=5)
```

**Use:** Adaptive LoRA rank allocation (IGU-LoRA style), fine-grained attribution.

---

## Category 2: Spectral & Kernel (3 metrics)

### CKA

Centered Kernel Alignment — representation similarity with output.

**Output:** `cka_score`

```python
m = CKA(compare_to="output", kernel="linear", cka_batch_size=256)
```

**Use:** Layer-output alignment, distillation layer pairing. See also `CKASimilarity` for N x N pairwise matrix.

### EffectiveRank

Representation diversity via singular value entropy. Uses randomized SVD.

**Output:** `effective_rank`, `stable_rank`, `rank_ratio`

```python
m = EffectiveRank(num_batches=10, n_components=50)
```

**Use:** Bottleneck detection (rank drops), representation quality.

### NTKTrace

Neural Tangent Kernel trace approximation.

**Output:** `ntk_trace`, `ntk_trace_per_param`

```python
m = NTKTrace(num_batches=5)
```

**Use:** Training dynamics research, parameter influence analysis.

---

## Category 3: Information Theory (2 metrics)

### ActivationEntropy

Shannon entropy of activation distribution.

**Output:** `activation_entropy`, `activation_mean`, `activation_std`, `activation_sparsity`

```python
m = ActivationEntropy(num_batches=10, num_bins=50)
```

**Use:** Information bottleneck analysis, layer capacity estimation.

### MutualInformation

MI between activations and targets.

**Output:** `mutual_information`, `mi_max`, `mi_std`

```python
m = MutualInformation(num_batches=10)
```

**Use:** Task-relevance scoring, information flow analysis.

---

## Category 4: Representation (2 metrics)

### BlockInfluence

Layer transformation magnitude — cosine distance between input/output (ShortGPT, ACL 2025).

**Output:** `block_influence`, `block_similarity`

```python
m = BlockInfluence(num_batches=10)
```

**Use:** LLM layer pruning (low influence = safe to remove), redundancy detection.

### JacobianRank

Effective dimensionality of input-output Jacobian.

**Output:** `jacobian_rank`, `jacobian_rank_ratio`, `jacobian_condition`, `jacobian_max_sv`

```python
m = JacobianRank(num_batches=5)
```

**Use:** Training stability analysis (condition number), transformation properties.

---

## Category 5: Robustness (2 metrics)

### DropLayerRobustness

Performance degradation when layer is zeroed (standard ablation).

**Output:** `droplayer_loss_increase`, `droplayer_loss_ratio`

```python
m = DropLayerRobustness(num_batches=10, metric="loss", drop_type="zero")
```

### ResidualDropLayer

Residual-aware ablation — replaces output with input (preserves residual stream).

**Output:** `residual_droplayer_loss_increase`, `residual_droplayer_loss_ratio`, `residual_ratio`, `transform_norm_ratio`

```python
m = ResidualDropLayer(num_batches=10, metric="loss")
```

**Use:** Correct ablation for transformers/Mamba with skip connections. `residual_ratio` near 1.0 means the layer barely transforms its input.

---

## Category 6: Bayesian (1 metric)

### LaplacePosterior

Parameter uncertainty via Laplace approximation. Requires `laplace-torch`.

**Output:** `laplace_posterior`, `laplace_posterior_std`

```python
m = LaplacePosterior(num_batches=10)
```

---

## Category 7: Efficiency (4 metrics)

### EfficiencyProfiler

Per-layer FLOPs, parameters, memory footprint.

**Output:** `flops`, `param_count`, `param_memory_mb`, `activation_memory_mb`, `compute_ratio`

```python
m = EfficiencyProfiler()
```

**Use:** Hardware-aware pruning budget, deployment cost estimation.

### WeightDistribution

Weight matrix statistics. **Does not require data** (`requires_data=False`).

**Output:** `weight_sparsity`, `weight_l1_norm`, `weight_l2_norm`, `weight_rank_ratio`, `weight_outlier_ratio`, `weight_kurtosis`

```python
m = WeightDistribution(sparsity_threshold=1e-3, svd_components=50)
# No data_loader needed!
result = m.compute(model=model, layer=layer, layer_name="0", layer_idx=0, device="cpu")
```

**Use:** Pre-pruning analysis, quantization readiness check, weight health monitoring.

### IntrinsicDimensionality

MLE intrinsic dimension of activation manifold (Levina-Bickel 2004).

**Output:** `intrinsic_dim`, `intrinsic_dim_ratio`, `ambient_dim`

```python
m = IntrinsicDimensionality(num_batches=10, k=20)
```

**Use:** **Optimal LoRA rank selection** (rank ~ intrinsic dim), compression budget allocation.

> Aghajanyan et al. (ACL 2021) showed fine-tuning operates on a low intrinsic dimension subspace. This metric estimates that dimension per layer.

### QuantizationSensitivity

Simulates INT8/FP16 quantization, measures output deviation.

**Output:** `quant_sensitivity_int8`, `quant_sensitivity_fp16`, `activation_range`, `weight_dynamic_range`

```python
m = QuantizationSensitivity(num_batches=5)
```

**Use:** Mixed-precision deployment, INT8 quantization budget, ONNX/TensorRT precision selection.

---

## Category 8: Security (4 metrics)

### AdversarialSensitivity

FGSM perturbation sensitivity per layer.

**Output:** `adv_sensitivity`, `adv_amplification`, `adv_directional_change`

```python
m = AdversarialSensitivity(epsilon=0.01, num_batches=5)
```

**Use:** Adversarial robustness audit, identifying vulnerable layers.

### ActivationAnomalyScore

Backdoor detection via activation distribution statistics.

**Output:** `activation_skewness`, `activation_kurtosis`, `neuron_outlier_ratio`, `activation_bimodality`

```python
m = ActivationAnomalyScore(num_batches=10)
```

**Use:** Trojan/backdoor detection (`activation_bimodality > 0.555` is suspicious), supply chain security.

### MembershipInferenceRisk

Gradient leakage risk scoring.

**Output:** `gradient_entropy`, `gradient_snr`, `gradient_memorization`, `mi_risk_score`

```python
m = MembershipInferenceRisk(num_batches=5)
```

**Use:** Privacy audit, federated learning risk, differential privacy budget allocation.

### AttentionPathTrace

Prompt injection vulnerability analysis for transformers.

**Output:** `attention_concentration`, `attention_manipulability`, `attention_persistence`, `injection_vulnerability`

```python
m = AttentionPathTrace(num_batches=5)
```

**Use:** LLM safety audit, prompt injection defense. Non-attention layers use activation-based proxy.

---

## Category 9: Architecture-Specific (3 metrics)

### AttentionFlow

Transformer attention head diversity and entropy analysis.

**Output:** `attention_entropy`, `attention_max_weight`, `head_diversity`, `attention_distance`

```python
m = AttentionFlow(num_batches=10, analyze_heads=True)
```

### MoERouterAnalysis

MoE routing behavior: entropy, expert utilization, load balance.

**Output:** `routing_entropy`, `expert_utilization`, `load_balance_score`, `top_expert_ratio`, `expert_overlap`

```python
m = MoERouterAnalysis(num_batches=10, top_k=2)
```

Non-MoE layers return `None` for all keys.

### DiffusionTimestepAnalysis

Per-layer importance across denoising timesteps (UNet/DiT).

**Output:** `timestep_sensitivity`, `mean_activation_norm`, `early_importance`, `late_importance`, `timestep_variance`

```python
m = DiffusionTimestepAnalysis(num_timesteps=10, max_timestep=1000, num_batches=5)
```

Non-diffusion models return `None`. Use `get_diffusion_blocks()` to extract UNet blocks.

---

## Metric Selection Guide

| Goal | Recommended Metrics |
|------|-------------------|
| **LLM pruning** | BlockInfluence, ResidualDropLayer, WandaImportance, EffectiveRank |
| **LoRA rank selection** | IntrinsicDimensionality, IGSensitivity, GradientNorm |
| **Quantization planning** | QuantizationSensitivity, WeightDistribution, EfficiencyProfiler |
| **Distillation** | CKA, EffectiveRank, BlockInfluence |
| **Security audit** | AdversarialSensitivity, ActivationAnomalyScore, MembershipInferenceRisk |
| **MoE optimization** | MoERouterAnalysis, BlockInfluence, EffectiveRank |
| **Diffusion model** | DiffusionTimestepAnalysis, BlockInfluence, EffectiveRank |
| **Hardware deployment** | EfficiencyProfiler, QuantizationSensitivity, WeightDistribution |

---

## Adding Custom Metrics

```python
from uni_layer.core.base_metric import LayerMetric

class MyMetric(LayerMetric):
    def __init__(self, **kwargs):
        super().__init__(name="my_metric", category="custom",
                         requires_gradient=False, requires_data=True, **kwargs)

    def compute(self, model, layer, layer_name, layer_idx,
                data_loader=None, device="cuda", _cached_activations=None, **kwargs):
        return {"my_metric": score}
```

Register primary/output keys in `uni_layer/core/schema.py` to enable validation and ranking.
