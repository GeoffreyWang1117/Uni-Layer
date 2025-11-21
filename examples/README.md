# Uni-Layer Examples

This directory contains example scripts demonstrating how to use Uni-Layer with different model architectures and use cases.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

**What it demonstrates:**
- Initializing LayerAnalyzer
- Computing multiple metrics
- Analyzing results
- Generating pruning and distillation strategies
- Creating visualizations

**Run it:**
```bash
python examples/basic_usage.py
```

**Good for:** First-time users, understanding the API

---

### 2. Transformer Analysis (`transformer_analysis.py`)

**What it demonstrates:**
- Analyzing Transformer/BERT-like models
- Comparing attention vs feedforward layers
- Identifying critical transformer blocks
- PEFT adapter placement

**Run it:**
```bash
python examples/transformer_analysis.py
```

**Good for:** NLP models, BERT, GPT, Llama

---

### 3. Vision Model Analysis (`vision_model_analysis.py`)

**What it demonstrates:**
- Analyzing CNN architectures
- Layer-type specific analysis (conv vs linear)
- DropLayer robustness testing
- Depth-based analysis
- Vision model pruning strategies

**Run it:**
```bash
python examples/vision_model_analysis.py
```

**Good for:** Computer vision models, ResNet, ViT, CNNs

---

## Running Examples with Your Own Models

### Option 1: Modify Example Scripts

Replace the model definition with your own:

```python
# Instead of the example model
# model = SimpleClassifier()

# Use your model
from your_module import YourModel
model = YourModel()

# Rest of the code stays the same
analyzer = LayerAnalyzer(model, ...)
```

### Option 2: Use as Template

Copy an example and adapt it:

```bash
cp examples/basic_usage.py my_analysis.py
# Edit my_analysis.py with your model and data
python my_analysis.py
```

## Example Output

All examples will:
1. Print layer information
2. Compute specified metrics
3. Display analysis results
4. Generate recommendations
5. (Optionally) Create visualization plots

### Sample Output:

```
==============================================================
Uni-Layer Framework - Basic Usage Example
==============================================================

[1/5] Creating dummy dataset...
[2/5] Initializing model...
[3/5] Initializing LayerAnalyzer...
✓ Initialized LayerAnalyzer with 12 layers
  Device: cuda
  Task Type: classification

[4/5] Computing layer contribution metrics...
Computing metrics: 100%|████████████████| 5/5 [00:15<00:00]

[5/5] Analysis Results:
==============================================================

model.0 (Linear):
  Gradient Norm: 0.2341
  CKA Score: 0.4523
  Effective Rank: 127.45

...

==============================================================
Layer Rankings (by Gradient Norm):
==============================================================
1. model.6: 0.5234
2. model.3: 0.4123
3. model.0: 0.2341
...

✓ Visualizations saved!
Analysis Complete!
```

## Tips for Using Examples

### Memory Management

If you run out of memory:
```python
# Reduce number of batches
metric = GradientNorm(num_batches=3)  # Instead of 10

# Use CPU
analyzer = LayerAnalyzer(model, device='cpu')

# Process fewer layers at once
contributions = analyzer.compute_metrics(
    metrics=metrics[:2],  # Process 2 metrics at a time
    ...
)
```

### Speed Optimization

For faster results:
```python
# Use fewer batches
metrics = [
    GradientNorm(num_batches=5),  # Reduce from 10
    CKA(num_batches=5),
]

# Skip expensive metrics
# Don't use: HessianTrace, JacobianRank for quick analysis
```

### Custom Data Loaders

All examples work with any PyTorch DataLoader:

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Your custom dataset
transform = transforms.Compose([...])
dataset = datasets.CIFAR10(root='./data', transform=transform)
data_loader = DataLoader(dataset, batch_size=32)

# Use with analyzer
contributions = analyzer.compute_metrics(
    metrics=metrics,
    data_loader=data_loader,  # Your data loader
    verbose=True
)
```

## Creating Your Own Examples

Template for a new example:

```python
"""
Description of what this example demonstrates.
"""

import torch
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA

def main():
    # 1. Setup model and data
    model = ...
    data_loader = ...

    # 2. Initialize analyzer
    analyzer = LayerAnalyzer(model, task_type='...')

    # 3. Define metrics
    metrics = [GradientNorm(), CKA()]

    # 4. Compute
    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader
    )

    # 5. Analyze results
    rankings = analyzer.rank_layers(contributions, "gradient_norm")
    print(rankings)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Import Errors

```bash
# Make sure uni-layer is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Uni-Layer"
```

### CUDA Out of Memory

```python
# Switch to CPU
analyzer = LayerAnalyzer(model, device='cpu')

# Or reduce batch size
data_loader = DataLoader(dataset, batch_size=16)  # Smaller batches
```

### Metrics Return None

- Check data loader format: should yield `(inputs, targets)` tuples
- Some metrics require labels - ensure your data loader provides them
- Check metric compatibility with your architecture

## Additional Resources

- [Quick Start Guide](../docs/QUICKSTART.md)
- [Metrics Documentation](../docs/METRICS.md)
- [API Reference](../docs/API.md)
- [Contributing Guide](../CONTRIBUTING.md)

## Questions?

Open an issue on [GitHub](https://github.com/GeoffreyWang1117/Uni-Layer/issues) if you have questions or run into problems!
