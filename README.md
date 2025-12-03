# Uni-Layer: A Universal Framework for Layer Contribution Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

**Uni-Layer** is a comprehensive framework for analyzing layer contributions across diverse deep learning architectures including NLP, Vision, Speech, Graph, and Recommendation models. Unlike traditional approaches that focus on single domains, Uni-Layer provides a unified API for computing 30+ layer contribution metrics across 7 major categories.

## 🎯 Key Features

- **Universal Metrics**: 30+ layer contribution metrics spanning Information Theory, Optimization Geometry, Spectral Methods, and more
- **Cross-Architecture Support**: Works with Transformers, CNNs, GNNs, RecSys, Diffusion models, and more
- **10+ Model Categories**: NLP (BERT, GPT, Llama), Vision (ViT, ResNet), Speech (Whisper), Graph (GCN, GAT), Recommendation (DeepFM), and more
- **Downstream Applications**: Validated on distillation, pruning, PEFT, and model interpretability tasks
- **Plug-and-Play API**: Easy integration with existing PyTorch models

## 📊 Supported Metrics

### 1. Information Theory
- Mutual Information (MI)
- Information Bottleneck (IB)
- Entropy-based measures

### 2. Optimization Geometry
- Gradient Norm
- Hessian Eigenspectrum
- Loss Landscape Sharpness

### 3. Probabilistic Bayesian
- Fisher Information
- Laplace Posterior Variance
- PAC-Bayes Layer Bound

### 4. Spectral & Kernel Methods
- CKA (Centered Kernel Alignment)
- CKTA (Centered Kernel Target Alignment)
- NTK (Neural Tangent Kernel) Decomposition
- Spectral Effective Rank

### 5. Representation Structure
- Representation Jacobian Rank
- Representation Diversity

### 6. Robustness
- Adversarial Layer Sensitivity
- DropLayer Robustness
- Layer Perturbation Impact

### 7. Architecture-Specific
- Attention Flow (Transformers)
- Patch Attribution (ViT)
- Token Mixing Influence
- Path Integral Influence

## 🏗️ Architecture

```
uni_layer/
├── metrics/              # Layer contribution metrics
│   ├── information_theory/
│   ├── optimization/
│   ├── spectral/
│   ├── representation/
│   ├── robustness/
│   ├── bayesian/
│   └── architecture_specific/
├── models/               # Model loaders and registry
├── benchmark/            # Benchmark evaluation pipeline
├── visualization/        # Visualization tools
└── utils/                # Utility functions
```

## 🚀 Quick Start

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, MutualInformation

# Initialize analyzer with your model
analyzer = LayerAnalyzer(model, task_type='classification')

# Compute layer contributions
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA(), MutualInformation()],
    data_loader=train_loader
)

# Visualize results
analyzer.visualize(contributions, save_path='layer_analysis.png')

# Use for downstream tasks
pruning_strategy = analyzer.get_pruning_strategy(contributions)
distillation_layers = analyzer.get_distillation_layers(contributions, top_k=6)
```

## 🗜️ Model Compression Tools

Uni-Layer provides production-ready compression utilities that leverage layer contribution analysis:

### 1. **Intelligent Pruning**
Remove redundant weights/neurons with differential strategies based on layer importance.

```python
from uni_layer.compression import LayerPruner, PruningStrategy

pruner = LayerPruner(model, contributions, strategy=PruningStrategy.GRADIENT_NORM)
pruned_model = pruner.prune_unstructured(pruning_ratios)

# Achieve 50%+ sparsity with minimal accuracy loss
stats = pruner.get_sparsity_stats()
speedup = pruner.estimate_speedup()
```

### 2. **Knowledge Distillation**
Distill large models into smaller ones with automatic layer selection.

```python
from uni_layer.compression import KnowledgeDistiller, DistillationConfig

config = DistillationConfig(temperature=4.0, alpha=0.7, top_k_layers=3)
distiller = KnowledgeDistiller(teacher, student, contributions, config)

# Auto-selects important layers based on CKA/GradNorm
for inputs, labels in data_loader:
    loss_components = distiller.train_step(inputs, labels, optimizer)
```

### 3. **Parameter-Efficient Fine-Tuning (PEFT)**
Fine-tune with LoRA/Adapters using optimal layer selection and adaptive ranks.

```python
from uni_layer.compression import PEFTOptimizer, AdapterConfig

config = AdapterConfig(method="lora", rank=8, adaptive_rank=True)
peft_optimizer = PEFTOptimizer(model, contributions, config)

# Auto-selects layers and computes adaptive ranks
model_with_lora = peft_optimizer.inject_lora(selected_layers, ranks)

# 10-100x fewer trainable parameters
efficiency = peft_optimizer.get_parameter_efficiency()
```

**See [Compression Guide](docs/COMPRESSION.md) for detailed documentation.**

## 🔬 Supported Model Categories

| Category | Models | Status |
|----------|--------|--------|
| **NLP** | BERT, RoBERTa, GPT, Llama3 | ✅ |
| **Vision** | ViT, Swin, ResNet, ConvNeXt | ✅ |
| **Speech** | Whisper, Conformer | ✅ |
| **Graph** | GCN, GraphSAGE, GAT | ✅ |
| **Multimodal** | CLIP, BLIP | ✅ |
| **Recommendation** | DeepFM, DCNv2, AutoInt, DLRM | ✅ |
| **RL** | Atari CNN, Transformer-RL | ✅ |
| **Diffusion** | UNet | ✅ |
| **MLP-Mixer** | Mixer-B/16 | ✅ |
| **Baselines** | MLP, CNN, RNN | ✅ |

## 📦 Installation

```bash
pip install uni-layer
```

Or install from source:

```bash
git clone https://github.com/GeoffreyWang1117/Uni-Layer.git
cd Uni-Layer
pip install -e .
```

## 🎓 Citation

If you use Uni-Layer in your research, please cite:

```bibtex
@article{unilayer2025,
  title={Uni-Layer: A Universal Framework for Layer Contribution Analysis Across Deep Learning Architectures},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://uni-layer.readthedocs.io)
- [Paper](https://arxiv.org/abs/xxx)
- [Examples](examples/)
- [API Reference](docs/api.md)
- [Compression Guide](docs/COMPRESSION.md)
- [中文文档](README_CN.md)

## 🙏 Acknowledgments

This project builds upon research in neural network interpretability, model compression, and representation learning.
