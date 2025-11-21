# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added

#### Core Framework
- Initial release of Uni-Layer framework
- `LayerAnalyzer` main class for analyzing model layers
- `LayerMetric` base class for implementing custom metrics
- Automatic layer extraction for various architectures (Transformer, CNN, GNN, etc.)
- Layer type identification system
- Comprehensive hook management utilities

#### Metrics - Optimization Geometry
- `GradientNorm`: Gradient magnitude measurement
- `HessianTrace`: Hessian trace approximation via Hutchinson estimator
- `FisherInformation`: Empirical Fisher Information Matrix computation

#### Metrics - Spectral & Kernel Methods
- `CKA`: Centered Kernel Alignment for representation similarity
- `EffectiveRank`: Effective rank based on singular value entropy
- `NTKTrace`: Neural Tangent Kernel trace approximation

#### Metrics - Information Theory
- `MutualInformation`: MI estimation between activations and targets
- `ActivationEntropy`: Entropy of activation distributions

#### Metrics - Representation Structure
- `JacobianRank`: Jacobian matrix rank computation

#### Metrics - Robustness
- `DropLayerRobustness`: Layer importance via ablation studies

#### Visualization
- `plot_layer_contributions`: Bar chart visualization
- `plot_contribution_heatmap`: Multi-metric heatmap
- `plot_depth_analysis`: Depth-based aggregation plots
- `plot_metric_comparison`: Scatter plots comparing metrics

#### Analysis Features
- Layer ranking by any metric
- Top-k layer selection
- Pruning strategy generation
- Knowledge distillation layer selection
- PEFT adapter insertion point identification
- Depth-based aggregation
- Summary statistics computation

#### Examples
- Basic usage example with simple MLP
- Transformer model analysis
- Vision model (CNN) analysis

#### Documentation
- Comprehensive README
- Quick Start Guide
- Metrics documentation
- Contributing guidelines
- API reference

#### Development
- Setup.py configuration
- Requirements.txt with core dependencies
- pyproject.toml for modern packaging
- .gitignore for Python projects
- MIT License
- GitHub-ready repository structure

### Supported Model Families
- BERT family (BERT, RoBERTa, ELECTRA)
- GPT family (GPT, GPT-2, GPT-Neo)
- Llama family (Llama, Mistral, Mixtral)
- ViT family (Vision Transformer)
- Swin Transformer family
- CLIP family
- ResNet family
- ConvNeXt family
- GCN family (Graph Convolutional Networks)
- GAT family (Graph Attention Networks)
- GraphSAGE family
- DeepFM family (Recommendation)
- DCN family (Deep & Cross Network)
- DLRM family (Deep Learning Recommendation Model)
- Whisper family (Speech)
- Conformer family (Speech)
- Generic MLP, CNN, RNN support

### Features
- Cross-architecture support (NLP, Vision, Speech, Graph, RecSys)
- Plug-and-play API design
- Efficient computation with configurable batch limits
- GPU and CPU support
- Progress bars for long computations
- Graceful error handling
- Comprehensive type hints
- Extensive documentation

### Performance
- Optimized tensor operations
- Memory-efficient implementations
- Configurable computation budget via `num_batches`
- Support for large models via activation checkpointing

---

## [Unreleased]

### Planned Features

#### Metrics
- Information Bottleneck (IB) metric
- Laplace Posterior Variance
- PAC-Bayes Layer Bound
- Adversarial Layer Sensitivity
- Attention Flow (Transformer-specific)
- Patch Attribution (ViT-specific)
- Token Mixing Influence
- Path Integral Influence
- Fractal Dimension

#### Model Support
- Diffusion model (UNet) specific support
- MLP-Mixer specific support
- More comprehensive RL model support
- Mamba/State Space Model support

#### Benchmark
- LayerBench: Cross-architecture benchmark dataset
- Standard evaluation protocols
- Reproducible benchmarking scripts
- Leaderboard system

#### Downstream Applications
- Automatic distillation pipeline
- Intelligent pruning strategies
- PEFT optimization toolkit
- Model interpretation reports

#### Visualization
- Interactive web-based visualizations
- Attention flow diagrams
- Layer importance animations
- Cross-model comparison plots

#### Documentation
- Video tutorials
- Jupyter notebook examples
- Research paper

---

## Version History

- **0.1.0** - Initial release (2025-01)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this changelog.
