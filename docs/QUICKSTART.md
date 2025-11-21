# Uni-Layer Quick Start Guide

This guide will help you get started with Uni-Layer in under 5 minutes.

## Installation

### From PyPI (recommended)

```bash
pip install uni-layer
```

### From Source

```bash
git clone https://github.com/GeoffreyWang1117/Uni-Layer.git
cd Uni-Layer
pip install -e .
```

## Basic Usage

### 1. Import Required Components

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, EffectiveRank
```

### 2. Prepare Your Model and Data

```python
import torch
from torch.utils.data import DataLoader

# Your PyTorch model
model = YourModel()

# Your data loader
train_loader = DataLoader(dataset, batch_size=32)
```

### 3. Initialize the Analyzer

```python
analyzer = LayerAnalyzer(
    model=model,
    task_type='classification',  # or 'regression', 'generation'
    device='cuda'  # or 'cpu'
)
```

### 4. Compute Layer Contributions

```python
# Define which metrics to compute
metrics = [
    GradientNorm(num_batches=10),
    CKA(num_batches=10),
    EffectiveRank(num_batches=10),
]

# Compute metrics
contributions = analyzer.compute_metrics(
    metrics=metrics,
    data_loader=train_loader,
    verbose=True
)
```

### 5. Analyze Results

```python
# Rank layers by importance
rankings = analyzer.rank_layers(contributions, "gradient_norm")

print("Top 5 most important layers:")
for layer_name, value in rankings[:5]:
    print(f"  {layer_name}: {value:.4f}")

# Get pruning strategy
pruning_strategy = analyzer.get_pruning_strategy(
    contributions,
    metric_name="gradient_norm",
    prune_ratio=0.3
)

# Get layers for distillation
distill_layers = analyzer.get_distillation_layers(
    contributions,
    metric_name="gradient_norm",
    top_k=6
)
```

### 6. Visualize (Optional)

```python
from uni_layer.visualization import plot_layer_contributions

plot_layer_contributions(
    contributions,
    metric_name="gradient_norm",
    save_path="analysis.png"
)
```

## Next Steps

- Explore the [examples/](../examples/) directory for more detailed examples
- Check out the [API Reference](API.md) for complete documentation
- Read about [available metrics](METRICS.md)
- Learn about [model-specific tips](MODEL_TIPS.md)

## Common Use Cases

### Knowledge Distillation

```python
# Identify best layers for distillation
distill_layers = analyzer.get_distillation_layers(
    contributions,
    metric_name="gradient_norm",
    top_k=6
)
```

### Model Pruning

```python
# Get layer-wise pruning ratios
pruning_strategy = analyzer.get_pruning_strategy(
    contributions,
    metric_name="gradient_norm",
    prune_ratio=0.4
)
```

### PEFT (Parameter-Efficient Fine-Tuning)

```python
# Find optimal adapter insertion points
adapter_layers = analyzer.get_peft_insertion_points(
    contributions,
    metric_name="gradient_norm",
    num_adapters=4
)
```

## Troubleshooting

### Out of Memory

If you run out of memory:
- Reduce `num_batches` in metric constructors
- Use fewer metrics at once
- Use smaller batch sizes in your data loader
- Switch to CPU with `device='cpu'`

### Slow Computation

To speed up:
- Reduce `num_batches` in metrics
- Use faster metrics (Gradient-based > Hessian-based)
- Enable GPU with `device='cuda'`
- Compute metrics in separate runs

### Metric Returns None

If a metric returns None:
- Check that the metric is compatible with your model architecture
- Ensure your data loader provides correct format (inputs, targets)
- Some metrics require labels - check metric documentation

## Getting Help

- 📖 [Full Documentation](https://uni-layer.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/GeoffreyWang1117/Uni-Layer/discussions)
- 🐛 [Issue Tracker](https://github.com/GeoffreyWang1117/Uni-Layer/issues)
