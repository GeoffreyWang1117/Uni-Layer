# Model Compression Guide

Uni-Layer provides powerful compression utilities that leverage layer contribution analysis to optimize:

- **Model Pruning**: Remove redundant weights/neurons
- **Knowledge Distillation**: Transfer knowledge from large to small models
- **Parameter-Efficient Fine-Tuning (PEFT)**: Fine-tune with minimal parameters

All compression methods use layer contribution analysis to make intelligent decisions about what to compress, distill, or augment.

---

## Table of Contents

1. [Model Pruning](#model-pruning)
2. [Knowledge Distillation](#knowledge-distillation)
3. [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
4. [Best Practices](#best-practices)

---

## Model Pruning

Pruning removes less important weights or neurons to reduce model size and improve inference speed.

### Mathematical Foundation

For a neural network parameter θ, we define importance score:

- **Magnitude-based**: `I(θ) = |θ|`
- **Gradient-based**: `I(θ) = ||∂L/∂θ||₂`
- **Fisher-based**: `I(θ) = E[(∂L/∂θ)²]`

**Differential Pruning Strategy:**

```
pruning_ratio_ℓ = max_ratio · (1 - normalized_contribution_ℓ)
```

Layers with low contribution → high pruning ratio
Layers with high contribution → low pruning ratio

### Quick Start

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm
from uni_layer.compression import LayerPruner, PruningStrategy

# 1. Analyze layer contributions
analyzer = LayerAnalyzer(model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=10)],
    data_loader=data_loader
)

# 2. Create pruner
pruner = LayerPruner(
    model,
    contributions,
    strategy=PruningStrategy.GRADIENT_NORM
)

# 3. Compute differential pruning ratios
pruning_ratios = pruner.compute_layer_pruning_ratios(
    base_ratio=0.3,
    max_ratio=0.6,  # Prune up to 60% for unimportant layers
    min_ratio=0.1,  # Prune at least 10% for important layers
)

# 4. Apply pruning
pruned_model = pruner.prune_unstructured(pruning_ratios)
pruned_model = pruner.remove_pruning_masks()

# 5. Get statistics
stats = pruner.get_sparsity_stats()
print(f"Sparsity: {stats['overall_sparsity']:.2%}")

speedup = pruner.estimate_speedup()
print(f"Estimated speedup: {speedup['practical_speedup']:.2f}x")
```

### Pruning Types

#### 1. Unstructured Pruning

Removes individual weights (creates sparse matrices):

```python
pruned_model = pruner.prune_unstructured(
    pruning_ratios,
    global_pruning=False  # Use layer-specific ratios
)
```

**Pros:**
- Higher sparsity achievable
- Minimal accuracy loss

**Cons:**
- Requires sparse operations support
- Limited speedup without specialized hardware

#### 2. Structured Pruning

Removes entire neurons/channels (hardware-friendly):

```python
pruned_model = pruner.prune_structured(
    pruning_ratios,
    dim=0  # 0: output neurons, 1: input neurons
)
```

**Pros:**
- Actual speedup on standard hardware
- Smaller model size

**Cons:**
- More accuracy loss
- Lower maximum sparsity

#### 3. Gradual Pruning

Progressively increase sparsity (better accuracy retention):

```python
models = pruner.prune_gradual(
    initial_ratio=0.0,
    final_ratio=0.5,
    num_steps=10,
    structured=False
)

# Train after each pruning step
for step, pruned_model in enumerate(models):
    train(pruned_model, data_loader, epochs=1)
```

Uses cubic sparsity schedule:

```
s_t = s_f + (s_i - s_f) · (1 - t/n)³
```

### API Reference

**`LayerPruner`**

```python
LayerPruner(
    model: nn.Module,
    contributions: Dict[str, Dict[str, float]],
    strategy: PruningStrategy = PruningStrategy.GRADIENT_NORM
)
```

**Methods:**

- `compute_layer_pruning_ratios()`: Compute differential pruning ratios
- `prune_unstructured()`: Apply weight-level pruning
- `prune_structured()`: Apply neuron/channel-level pruning
- `prune_gradual()`: Gradual pruning with progressive sparsity
- `remove_pruning_masks()`: Make pruning permanent
- `get_sparsity_stats()`: Compute sparsity statistics
- `estimate_speedup()`: Estimate theoretical and practical speedup

---

## Knowledge Distillation

Transfer knowledge from a large teacher model to a smaller student model.

### Mathematical Foundation

**Standard Distillation (Hinton et al., 2015):**

```
L_KD = α·L_soft + (1-α)·L_hard
```

Where:
- `L_soft = KL(softmax(z_t/T), softmax(z_s/T)) · T²`
- `L_hard = CE(y, softmax(z_s))`
- `T`: temperature (higher → softer probabilities)

**Intermediate Layer Distillation:**

```
L_layer = β·Σ_ℓ w_ℓ · d(h_s^(ℓ), h_t^(ℓ))
```

Where `d(·,·)` is a distance metric (MSE, cosine, KL).

**Total Loss:**

```
L_total = α·L_soft + (1-α)·L_hard + β·L_layer
```

### Quick Start

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA
from uni_layer.compression import KnowledgeDistiller, DistillationConfig

# 1. Analyze teacher layer contributions
analyzer = LayerAnalyzer(teacher_model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=10), CKA(num_batches=10)],
    data_loader=data_loader
)

# 2. Configure distillation
config = DistillationConfig(
    temperature=4.0,
    alpha=0.7,  # 70% soft, 30% hard
    layer_weight=0.5,  # Weight for intermediate layers
    distance_metric="mse",
    top_k_layers=3  # Distill top 3 important layers
)

# 3. Create distiller (auto-selects layers based on CKA/GradNorm)
distiller = KnowledgeDistiller(
    teacher_model=teacher_model,
    student_model=student_model,
    contributions=contributions,
    config=config
)

# 4. Train student with distillation
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # Distillation training step
        loss_components = distiller.train_step(inputs, labels, optimizer)

        # loss_components contains: total, soft, hard, layer losses
```

### Layer Selection Strategy

The distiller automatically selects which layers to distill based on:

1. **High CKA scores** → semantically important representations
2. **High gradient norms** → actively learning features
3. **Balanced depth** → avoid only early or late layers

### Distance Metrics

Choose the appropriate distance metric for intermediate layers:

```python
config = DistillationConfig(
    distance_metric="mse"  # Options: "mse", "cosine", "kl"
)
```

- **MSE**: `||h_s - h_t||²` - Best for similar architectures
- **Cosine**: `1 - cos(h_s, h_t)` - Direction-based, scale-invariant
- **KL**: `KL(h_t || h_s)` - Probabilistic, for attention/activations

### API Reference

**`DistillationConfig`**

```python
DistillationConfig(
    temperature: float = 4.0,
    alpha: float = 0.5,
    layer_weight: float = 0.3,
    distance_metric: str = "mse",
    top_k_layers: int = 3
)
```

**`KnowledgeDistiller`**

```python
KnowledgeDistiller(
    teacher_model: nn.Module,
    student_model: nn.Module,
    contributions: Dict[str, Dict[str, float]],
    config: DistillationConfig = None
)
```

**Methods:**

- `train_step()`: Perform one distillation training step
- `compute_loss()`: Compute distillation loss components
- `get_distillation_info()`: Get configuration details

---

## Parameter-Efficient Fine-Tuning

Fine-tune large models by adding small trainable modules (adapters/LoRA) while freezing original weights.

### Mathematical Foundation

**LoRA (Low-Rank Adaptation):**

```
W' = W + ΔW = W + B·A
```

Where:
- `W ∈ ℝ^(d×k)`: frozen pre-trained weights
- `B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)`: trainable low-rank matrices
- `r << min(d,k)`: rank (typically 1-64)

**Forward pass:**

```
h = W·x + (α/r)·B·A·x
```

**Parameter efficiency:**

```
# trainable params = r(d + k) << dk
```

For `d=k=4096, r=8`: **99.8% parameter reduction**

**Adapter (Bottleneck Architecture):**

```
h' = h + f(h·W_down)·W_up
```

Where:
- `W_down ∈ ℝ^(d×r)`: down-projection
- `W_up ∈ ℝ^(r×d)`: up-projection
- `f`: non-linearity (ReLU/GELU)
- `r << d`: bottleneck dimension

### Quick Start

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, FisherInformation
from uni_layer.compression import PEFTOptimizer, AdapterConfig

# 1. Analyze layer contributions
analyzer = LayerAnalyzer(model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=10), FisherInformation(num_batches=10)],
    data_loader=data_loader
)

# 2. Configure PEFT
config = AdapterConfig(
    method="lora",  # or "adapter"
    rank=8,
    alpha=16.0,
    dropout=0.1,
    adaptive_rank=True  # Different ranks per layer
)

# 3. Create PEFT optimizer
peft_optimizer = PEFTOptimizer(
    model=model,
    contributions=contributions,
    config=config
)

# 4. Select layers (auto-selects based on importance)
selected_layers = peft_optimizer.select_layers(
    top_k=6,
    metric_name="gradient_norm",
    min_contribution=0.01
)

# 5. Compute adaptive ranks
ranks = peft_optimizer.compute_adaptive_ranks(
    selected_layers,
    base_rank=8,
    max_rank=32
)
# Important layers get higher rank (more capacity)
# Less important layers get lower rank (efficiency)

# 6. Inject LoRA layers
model_with_lora = peft_optimizer.inject_lora(selected_layers, ranks)

# 7. Get parameter efficiency
efficiency = peft_optimizer.get_parameter_efficiency()
print(f"Trainable: {efficiency['trainable_params']:,}")
print(f"Reduction: {efficiency['reduction_ratio']:.1f}x")

# 8. Fine-tune (only LoRA parameters are trainable)
trainable_params = [p for p in model_with_lora.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model_with_lora(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Adaptive Rank Selection

The rank for each layer is determined by its contribution:

```
rank_ℓ = base_rank + (max_rank - base_rank) · normalized_contribution_ℓ
```

**Intuition:**
- Important layers → higher rank → more capacity
- Less important layers → lower rank → parameter efficiency

### Methods Comparison

|Method|Parameters|Speed|Flexibility|Use Case|
|------|----------|-----|-----------|--------|
|**LoRA**|0.1-1%|Fast|High|LLMs, multi-task|
|**Adapter**|1-3%|Medium|Medium|General fine-tuning|
|**Prefix Tuning**|0.1-0.5%|Fast|Low|Language generation|

### API Reference

**`AdapterConfig`**

```python
AdapterConfig(
    method: str = "lora",  # "lora", "adapter", "prefix_tuning"
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    adaptive_rank: bool = True
)
```

**`PEFTOptimizer`**

```python
PEFTOptimizer(
    model: nn.Module,
    contributions: Dict[str, Dict[str, float]],
    config: AdapterConfig = None
)
```

**Methods:**

- `select_layers()`: Auto-select layers for PEFT
- `compute_adaptive_ranks()`: Compute rank for each layer
- `inject_lora()`: Add LoRA layers to model
- `inject_adapters()`: Add adapter layers to model
- `get_trainable_parameters()`: Count trainable params
- `get_parameter_efficiency()`: Get efficiency stats
- `get_peft_info()`: Get PEFT configuration details

---

## Best Practices

### 1. Pruning Best Practices

**Start Conservative:**
```python
# Begin with moderate pruning
pruning_ratios = pruner.compute_layer_pruning_ratios(
    base_ratio=0.2,
    max_ratio=0.4,
    min_ratio=0.05
)
```

**Use Gradual Pruning:**
```python
# Better accuracy retention
models = pruner.prune_gradual(
    initial_ratio=0.0,
    final_ratio=0.5,
    num_steps=10
)
```

**Fine-tune After Pruning:**
```python
# Recover accuracy lost from pruning
train(pruned_model, data_loader, epochs=5, lr=1e-4)
```

**Choose Pruning Type Based on Deployment:**
- CPU/mobile: Use structured pruning
- GPU with sparse ops: Can use unstructured
- Memory-constrained: Structured + model quantization

### 2. Distillation Best Practices

**Temperature Selection:**
```python
# Higher temperature for more knowledge transfer
config = DistillationConfig(temperature=6.0)  # For large gap
config = DistillationConfig(temperature=3.0)  # For small gap
```

**Balance Loss Weights:**
```python
config = DistillationConfig(
    alpha=0.8,  # More weight on soft targets for dissimilar architectures
    alpha=0.5,  # Balanced for similar architectures
)
```

**Select Enough Layers:**
```python
config = DistillationConfig(
    top_k_layers=5  # More layers for complex tasks
)
```

**Pre-train Student:**
```python
# Train student on hard targets first
train(student, data_loader, epochs=5)
# Then apply distillation
distiller = KnowledgeDistiller(teacher, student, contributions, config)
```

### 3. PEFT Best Practices

**Rank Selection:**
```python
# For general tasks
config = AdapterConfig(rank=8, adaptive_rank=True)

# For complex/specific tasks
config = AdapterConfig(rank=16, adaptive_rank=True)

# For very large models
config = AdapterConfig(rank=4, adaptive_rank=True)
```

**Layer Selection:**
```python
# Select more layers for better performance
selected_layers = peft_optimizer.select_layers(
    top_k=10,  # 10 layers for comprehensive coverage
    metric_name="gradient_norm"
)
```

**Learning Rate:**
```python
# Use higher LR for PEFT than full fine-tuning
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)  # vs 1e-5 for full FT
```

**Task Switching:**
```python
# Easy to switch between tasks
model_with_lora_task1 = peft_optimizer.inject_lora(layers, ranks)
# Save LoRA weights, swap for task 2
model_with_lora_task2 = load_lora_weights("task2_lora.pt")
```

### 4. Combined Compression

Combine multiple techniques for maximum compression:

```python
# 1. Distill into smaller model
distiller = KnowledgeDistiller(large_model, small_model, contributions)
train_with_distillation(distiller, epochs=10)

# 2. Prune distilled model
pruner = LayerPruner(small_model, contributions_small)
pruned_model = pruner.prune_structured(ratios)
fine_tune(pruned_model, epochs=5)

# 3. Use PEFT for task adaptation
peft_optimizer = PEFTOptimizer(pruned_model, contributions_pruned)
model_with_lora = peft_optimizer.inject_lora(layers, ranks)
```

**Compression Pipeline:**
```
Large Model (100M params)
    ↓ Distillation
Small Model (25M params, -75%)
    ↓ Pruning
Pruned Model (12M params, -88%)
    ↓ PEFT for new task
Fine-tuned (12M + 0.1M params, -87.9%)
```

---

## Examples

Complete examples available in `examples/`:

- `pruning_example.py`: Model pruning with differential strategies
- `distillation_example.py`: Knowledge distillation with layer selection
- `peft_example.py`: LoRA fine-tuning with adaptive ranks

Run examples:
```bash
python examples/pruning_example.py
python examples/distillation_example.py
python examples/peft_example.py
```

---

## References

**Pruning:**
- [Magnitude Pruning (Han et al., 2015)](https://arxiv.org/abs/1506.02626)
- [Gradual Magnitude Pruning (Zhu & Gupta, 2017)](https://arxiv.org/abs/1710.01878)

**Distillation:**
- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [FitNets (Romero et al., 2014)](https://arxiv.org/abs/1412.6550)

**PEFT:**
- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Adapter Layers (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751)
