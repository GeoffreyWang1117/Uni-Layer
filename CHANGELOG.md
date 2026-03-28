# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.0] - 2026-03-28

### Added
- **WandaImportance**: weight × activation norm metric (Sun et al., ICLR 2024)
  - `wanda_score`, `weight_norm`, `activation_norm`, `wanda_sparsity`
  - No gradient required, works on any layer with weight parameters
- **IGSensitivity**: Integrated Gradients per-layer sensitivity scoring
  - `ig_sensitivity`, `ig_variance`, `ig_relative`
  - Path integral attribution for adaptive LoRA rank allocation (IGU-LoRA style)
- **MultiModalBranchAnalyzer**: auto-detect and analyze vision/language/fusion branches
  - Supports CLIP, LLaVA, OpenCLIP naming conventions
  - `branch_summary()`, `compare_branches()`, `get_all_layers()` with branch prefixes
- **ExportHintsBridge**: ONNX/TensorRT optimization recommendations
  - `quantization_plan()`: per-layer precision (FP32/FP16/INT8) based on importance
  - `prunable_layers()`: safe-to-remove layers for inference
  - `fusion_candidates()`: operator fusion opportunities
  - `tensorrt_config()`: complete TensorRT builder hints
- **AxolotlConfigBridge**: generate Axolotl YAML config from analysis
  - Auto-select LoRA target modules by importance
  - Adaptive rank recommendation based on importance variance
- **LLaMAFactoryConfigBridge**: generate LLaMA-Factory JSON config
  - Layer freezing recommendations by importance ranking
  - Complete training config with metadata

### Changed
- Total metrics: 16 → 18 (added WandaImportance, IGSensitivity)
- Total integration bridges: 3 → 6 (added ExportHints, Axolotl, LLaMA-Factory)

## [0.4.0] - 2026-03-28

### Added
- **CKA Similarity Matrix**: `CKASimilarity` class for computing N×N pairwise CKA between all layers
  - `most_similar_pairs()`, `most_distinct_pairs()`, `redundant_layers()`, `layer_uniqueness()`
  - `LayerAnalyzer.compute_cka_matrix()` convenience method
  - Full JSON export via `to_dict()`
- **Residual-aware DropLayer** (`ResidualDropLayer`): ablation metric that preserves residual stream
  - Replaces output with input (identity skip) instead of zeroing
  - `residual_ratio`: cosine similarity between input/output (quantifies residual contribution)
  - `transform_norm_ratio`: relative magnitude of learned transform vs. residual
- **MoE Router Analysis** (`MoERouterAnalysis`): routing behavior analysis for Mixture of Experts
  - `routing_entropy`, `expert_utilization`, `load_balance_score`, `top_expert_ratio`, `expert_overlap`
  - Auto-detects gate/router submodules and number of experts
  - Gracefully skips non-MoE layers (returns None)
- **Mamba / SSM architecture support**: layer extraction and metrics for state-space models
  - `_find_transformer_blocks()` now detects Mamba block ModuleLists
  - `identify_layer_type()` returns `"ssm_block"` / `"ssm_layer"` for Mamba components
  - All activation-based metrics (BlockInfluence, EffectiveRank, CKA, etc.) verified on Mamba
- **GNN support** (PyG MessagePassing layers):
  - `_find_gnn_layers()` detects GCNConv, GATConv, SAGEConv, GINConv, etc.
  - `model_forward()` handles PyG `Data`/`Batch` objects (auto-unpacks x, edge_index, batch)
  - `identify_layer_type()` returns `"gnn_conv"`, `"gnn_attention"`, `"gnn_sage"`, etc.
  - Metrics verified on GCN and GAT models
- **Diffusion model support** (UNet timestep-aware analysis):
  - `DiffusionTimestepAnalysis` metric: per-layer importance across denoising timesteps
  - `timestep_sensitivity`, `early_importance`, `late_importance`, `mean_activation_norm`
  - `get_diffusion_blocks()`: extracts down_blocks + mid_block + up_blocks from UNet
  - Auto-detects timestep parameter name in model.forward()

### Changed
- Total metrics: 13 → 16 (added ResidualDropLayer, MoERouterAnalysis, DiffusionTimestepAnalysis)
- `"full"` preset now includes `ResidualDropLayer`
- Analyzer metric map updated with all new metrics
- 111 tests passing (up from 205+)

## [0.3.3] - 2026-03-25

### Fixed
- Updated README: added LayerProfile, presets, CLI, verified model list
- Updated Roadmap: moved unimplemented items from v0.3 to v0.4
- Updated CHANGELOG: added all missing release notes (v0.2.0 - v0.3.2)
- Synced README_CN.md with English README
- Removed duplicate pytest.ini (use pyproject.toml only)

## [0.3.2] - 2026-03-25

### Fixed
- Seq2Seq model support: auto-inject `decoder_input_ids` for T5/BART/ByT5
- `_find_transformer_blocks` returns ALL block lists (encoder + decoder)
- Verified on 20 HuggingFace models: BERT, RoBERTa, DeBERTa, DistilBERT,
  SciBERT, MiniLM, GPT-2, Pythia, BLOOM, Falcon, TinyLlama, Llama-3.2-3B,
  Qwen2.5-3B, ByT5, DINOv2, Wav2Vec2, HuBERT (all PASS)

## [0.3.1] - 2026-03-25

### Changed
- Rewrote `get_model_layers()` with `_find_transformer_blocks()` auto-detection
  - Recursively searches for transformer blocks at any nesting depth
  - Pythia: 147 sub-layers -> 24 blocks (7x faster)
  - Llama-3.2-3B: ~250 sub-layers -> 28 blocks (now completes in 7.5min)
  - Qwen2.5-3B: ~280 sub-layers -> 36 blocks (3.8min)

## [0.3.0] - 2026-03-25

### Added
- **LayerProfile**: automatic insight extraction from contributions data
  - Redundant layer detection (BlockInfluence/GradientNorm thresholds)
  - Bottleneck layer detection (EffectiveRank drop analysis)
  - Multi-metric consensus ranking (Borda count)
  - Depth trend analysis (U-shaped/increasing/decreasing/flat)
  - Statistical anomaly detection (z-score > 2)
  - Layer clustering (high/medium/low contribution)
  - Pruning suggestion with estimated speedup
  - LoRA suggestion with adaptive rank allocation
  - Natural language summary generation
  - Full JSON export via `to_dict()`
- **Metric presets**: `"llm_fast"`, `"llm_full"`, `"quick"`, `"full"`
  - `analyzer.compute_metrics(preset="llm_fast")` for one-line usage

## [0.2.1] - 2026-03-25

### Added
- **CLI**: `uni-layer --help`, `info`, `list-metrics`, `analyze`
- Console script entry point (`uni-layer` command after pip install)
- 10 CLI tests

## [0.2.0] - 2026-03-25

### Added
- 13 layer contribution metrics across 7 categories
- **BlockInfluence** metric (ShortGPT ACL 2025)
- HuggingFace model adaptation: dict/dataclass output, attention_mask, labels
- Integration bridges: TorchPruningBridge, HuggingFacePEFTBridge, DistillationBridge
- Output format schema (`uni_layer.core.schema`) with validation
- Activation/gradient caching system
- GitHub Actions CI (pytest + lint, Python 3.8-3.11)
- PyPI publish workflow (trusted publishing)
- 168 tests covering all metrics, HF adaptation, integrations, benchmarks
- Examples: ResNet, ViT, BERT layer analysis

### Fixed
- JacobianRank `hook_fn.__self__` AttributeError
- Tuple output handling in activation cache hooks

### Changed
- Consolidated packaging: pyproject.toml as single source of truth

## [0.1.0] - 2025-01

### Added
- Initial release: LayerAnalyzer, LayerMetric base class
- 10 metrics: GradientNorm, HessianTrace, FisherInformation, CKA,
  EffectiveRank, NTKTrace, MutualInformation, ActivationEntropy,
  JacobianRank, DropLayerRobustness
- Automatic layer extraction for Transformer/CNN architectures
- Visualization utilities
- Basic examples and documentation
