# Available Metrics in Uni-Layer

This document provides a comprehensive overview of all layer contribution metrics available in Uni-Layer.

## Metric Categories

1. [Optimization Geometry](#optimization-geometry)
2. [Spectral & Kernel Methods](#spectral--kernel-methods)
3. [Information Theory](#information-theory)
4. [Representation Structure](#representation-structure)
5. [Robustness](#robustness)
6. [Probabilistic Bayesian](#probabilistic-bayesian)
7. [Architecture-Specific](#architecture-specific)

---

## Optimization Geometry

### GradientNorm

**Description**: Measures the magnitude of gradients flowing through a layer.

**Interpretation**: Higher values indicate more important layers that contribute more to learning.

**Usage**:
```python
from uni_layer.metrics import GradientNorm

metric = GradientNorm(
    norm_type='l2',      # 'l1', 'l2', or 'linf'
    num_batches=10,      # Number of batches to average
    aggregate='mean'     # 'mean', 'sum', or 'max'
)
```

**Returns**: `gradient_norm`, `gradient_norm_std`, `gradient_norm_max`, `gradient_norm_min`

**Computational Cost**: Low
**Requires Labels**: Yes

---

### HessianTrace

**Description**: Approximates the trace of the Hessian matrix using Hutchinson estimator.

**Interpretation**: Measures curvature of loss landscape. Higher values indicate sharper minima.

**Usage**:
```python
from uni_layer.metrics import HessianTrace

metric = HessianTrace(
    num_samples=5,    # Number of random vectors for estimation
    num_batches=5     # Number of batches
)
```

**Returns**: `hessian_trace`, `hessian_trace_std`

**Computational Cost**: High
**Requires Labels**: Yes

---

### FisherInformation

**Description**: Computes empirical Fisher Information Matrix trace.

**Interpretation**: Measures sensitivity of output distribution to parameter changes.

**Usage**:
```python
from uni_layer.metrics import FisherInformation

metric = FisherInformation(
    num_batches=10,
    empirical=True    # Use empirical Fisher
)
```

**Returns**: `fisher_information`, `fisher_mean`

**Computational Cost**: Medium
**Requires Labels**: Yes

---

## Spectral & Kernel Methods

### CKA (Centered Kernel Alignment)

**Description**: Measures similarity between layer representations and output.

**Interpretation**: Higher CKA indicates better alignment with final predictions.

**Usage**:
```python
from uni_layer.metrics import CKA

metric = CKA(
    compare_to='output',  # 'output', 'previous', or 'input'
    num_batches=10,
    kernel='linear'       # 'linear' or 'rbf'
)
```

**Returns**: `cka_score`

**Computational Cost**: Medium
**Requires Labels**: No

---

### EffectiveRank

**Description**: Computes effective rank based on singular value entropy.

**Interpretation**: Higher rank indicates more diverse, less redundant representations.

**Usage**:
```python
from uni_layer.metrics import EffectiveRank

metric = EffectiveRank(
    num_batches=10,
    epsilon=1e-10
)
```

**Returns**: `effective_rank`, `stable_rank`, `rank_ratio`

**Computational Cost**: Medium
**Requires Labels**: No

---

### NTKTrace

**Description**: Computes Neural Tangent Kernel trace approximation.

**Interpretation**: Measures layer's influence on model predictions via parameter gradients.

**Usage**:
```python
from uni_layer.metrics import NTKTrace

metric = NTKTrace(
    num_samples=100,
    num_classes=None    # Auto-detect if None
)
```

**Returns**: `ntk_trace`, `ntk_trace_per_param`

**Computational Cost**: High
**Requires Labels**: No

---

## Information Theory

### MutualInformation

**Description**: Estimates mutual information between activations and targets.

**Interpretation**: Higher MI means layer captures more task-relevant information.

**Usage**:
```python
from uni_layer.metrics import MutualInformation

metric = MutualInformation(
    num_batches=10,
    task_type='classification',  # or 'regression'
    n_neighbors=3
)
```

**Returns**: `mutual_information`, `mi_max`, `mi_std`

**Computational Cost**: Medium
**Requires Labels**: Yes

---

### ActivationEntropy

**Description**: Computes entropy of layer activation distribution.

**Interpretation**: Higher entropy indicates more diverse activations.

**Usage**:
```python
from uni_layer.metrics import ActivationEntropy

metric = ActivationEntropy(
    num_batches=10,
    num_bins=50
)
```

**Returns**: `activation_entropy`, `activation_mean`, `activation_std`, `activation_sparsity`

**Computational Cost**: Low
**Requires Labels**: No

---

## Representation Structure

### JacobianRank

**Description**: Computes rank of the layer's Jacobian matrix.

**Interpretation**: Higher rank indicates more expressive transformations.

**Usage**:
```python
from uni_layer.metrics import JacobianRank

metric = JacobianRank(
    num_samples=100,
    rank_threshold=1e-3
)
```

**Returns**: `jacobian_rank`, `jacobian_rank_ratio`, `jacobian_condition`, `jacobian_max_sv`

**Computational Cost**: High
**Requires Labels**: No

---

## Robustness

### DropLayerRobustness

**Description**: Measures performance degradation when layer is dropped.

**Interpretation**: Larger performance drop indicates more critical layer.

**Usage**:
```python
from uni_layer.metrics import DropLayerRobustness

metric = DropLayerRobustness(
    num_batches=10,
    metric='loss',      # 'loss' or 'accuracy'
    drop_type='zero'    # 'zero' or 'identity'
)
```

**Returns**: `droplayer_loss_increase`, `droplayer_loss_ratio`, `droplayer_acc_decrease`, `droplayer_acc_ratio`

**Computational Cost**: Medium
**Requires Labels**: Yes

---

## Metric Selection Guide

### For Model Compression (Pruning)
1. **GradientNorm** - Fast, reliable indicator
2. **DropLayerRobustness** - Direct measure of importance
3. **FisherInformation** - Principled importance measure

### For Knowledge Distillation
1. **CKA** - Measures representation similarity
2. **GradientNorm** - Identifies learning-critical layers
3. **EffectiveRank** - Ensures diverse knowledge transfer

### For Model Interpretability
1. **MutualInformation** - Task-relevant information
2. **ActivationEntropy** - Representation diversity
3. **CKA** - Layer similarity analysis

### For PEFT (Adapter Insertion)
1. **GradientNorm** - High gradient = high adaptation potential
2. **NTKTrace** - Measures parameter influence
3. **JacobianRank** - Layer expressiveness

---

## Computational Cost Summary

| Metric | Cost | Time (relative) | Memory |
|--------|------|-----------------|--------|
| GradientNorm | Low | 1x | Low |
| ActivationEntropy | Low | 1x | Low |
| CKA | Medium | 2x | Medium |
| EffectiveRank | Medium | 2x | Medium |
| FisherInformation | Medium | 2.5x | Medium |
| MutualInformation | Medium | 2.5x | Medium |
| DropLayerRobustness | Medium | 3x | Low |
| NTKTrace | High | 5x | High |
| HessianTrace | High | 6x | High |
| JacobianRank | High | 5x | High |

---

## Best Practices

1. **Start with fast metrics**: GradientNorm, ActivationEntropy
2. **Validate with robust metrics**: DropLayerRobustness
3. **Use multiple metrics**: Different metrics capture different aspects
4. **Consider computational budget**: Use `num_batches` to control cost
5. **Match metric to task**: Classification vs. regression metrics differ

---

## Adding Custom Metrics

You can easily add custom metrics by extending the `LayerMetric` base class:

```python
from uni_layer.core.base_metric import LayerMetric

class MyCustomMetric(LayerMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="my_custom_metric",
            category="custom",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )

    def compute(self, model, layer, layer_name, layer_idx, data_loader, device, **kwargs):
        # Your metric computation logic
        return {"my_custom_metric": value}
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details on contributing new metrics.
