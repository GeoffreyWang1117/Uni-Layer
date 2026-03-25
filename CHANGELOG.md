# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
