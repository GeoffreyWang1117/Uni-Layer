# API Reference (v0.6.1)

## Core Classes

### `LayerAnalyzer`

Central entry point for all layer analysis.

```python
from uni_layer import LayerAnalyzer

analyzer = LayerAnalyzer(model, task_type="classification", device=None, criterion=None)
```

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `nn.Module` | required | PyTorch model to analyze |
| `task_type` | `str` | `"classification"` | Task type (`"classification"`, `"regression"`, `"generation"`) |
| `device` | `str` | auto | Device (`"cuda"` / `"cpu"`) |
| `criterion` | `nn.Module` | auto | Loss function (auto: CrossEntropy for classification) |

**Properties:**
- `analyzer.layers` — `OrderedDict[str, nn.Module]` of extracted layers
- `analyzer.layer_types` — `Dict[str, str]` of layer type classifications

**Methods:**

#### `compute_metrics()`

```python
contributions = analyzer.compute_metrics(
    metrics=None,           # List[LayerMetric] — manual metric selection
    data_loader=None,       # DataLoader for data-dependent metrics
    layer_names=None,       # List[str] — subset of layers (default: all)
    verbose=True,           # Show progress bars
    use_cache=True,         # Share activations across metrics
    num_batches=10,         # Batches for cache capture
    preset=None,            # "llm_fast" | "llm_full" | "quick" | "full"
)
# Returns: Dict[str, Dict[str, float]]
```

#### `compute_cka_matrix()`

```python
matrix, layer_names = analyzer.compute_cka_matrix(
    data_loader=loader, num_batches=10, verbose=True, cka_batch_size=256
)
# Returns: Tuple[np.ndarray, List[str]]
```

#### `rank_layers()`

```python
rankings = analyzer.rank_layers(contributions, metric_name="gradient_norm", ascending=False)
# Returns: List[Tuple[str, float]]
```

#### `get_top_k_layers()`

```python
top_layers = analyzer.get_top_k_layers(contributions, "gradient_norm", k=5)
# Returns: List[str]
```

#### `get_pruning_strategy()`

```python
strategy = analyzer.get_pruning_strategy(contributions, metric_name="gradient_norm", prune_ratio=0.3)
# Returns: Dict[str, float] — per-layer pruning ratios
```

#### `aggregate_by_depth()`

```python
depth_bins = analyzer.aggregate_by_depth(contributions, "gradient_norm", num_bins=5)
# Returns: Dict[str, float]
```

---

### `LayerProfile`

Automatic insight extraction from contributions data. All analyses run on CPU in milliseconds.

```python
from uni_layer import LayerProfile

profile = LayerProfile(contributions, model_name="my-model")
```

**Properties (lazy-evaluated, cached):**

| Property | Type | Description |
|----------|------|-------------|
| `redundant_layers` | `List[Dict]` | Layers with low transformation (safe to prune) |
| `bottleneck_layers` | `List[Dict]` | Layers where EffectiveRank drops > 30% |
| `consensus_ranking` | `List[Dict]` | Multi-metric Borda count ranking |
| `depth_trends` | `Dict[str, str]` | Per-metric trend (increasing/decreasing/U-shaped/flat) |
| `anomalies` | `List[Dict]` | Layers with z-score > 2 |
| `layer_clusters` | `Dict[str, List]` | High/medium/low contribution groups |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `pruning_suggestion(target_ratio=0.3)` | `Dict` | Safe-to-remove layers + estimated speedup |
| `lora_suggestion(base_rank=8, max_rank=64)` | `Dict` | Target layers + adaptive ranks |
| `security_report()` | `Dict` | Composite risk scoring per layer |
| `summary()` | `str` | One-paragraph natural language summary |
| `print_report()` | None | Full human-readable analysis |
| `print_table(metrics=None, top_k=0)` | None | Compact per-layer table |
| `to_markdown()` | `str` | Markdown table export |
| `to_dict()` | `Dict` | Full JSON-serializable report |

---

### `CKASimilarity`

Pairwise CKA similarity between all layers.

```python
from uni_layer import CKASimilarity

cka = CKASimilarity(model, layers=None, kernel="linear", cka_batch_size=256, device=None)
matrix, names = cka.compute(data_loader, num_batches=10, verbose=True)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute(data_loader, num_batches, verbose)` | `(np.ndarray, List[str])` | N x N matrix + layer names |
| `most_similar_pairs(matrix, names, top_k=5)` | `List[Tuple]` | Most similar layer pairs |
| `most_distinct_pairs(matrix, names, top_k=5)` | `List[Tuple]` | Most distinct layer pairs |
| `redundant_layers(matrix, names, threshold=0.95)` | `List[Tuple]` | Pairs above threshold |
| `layer_uniqueness(matrix, names)` | `Dict[str, float]` | Per-layer uniqueness score |
| `to_dict(matrix, names)` | `Dict` | Full JSON export |

---

### `MultiModalBranchAnalyzer`

Auto-detect and analyze vision/language/fusion branches in multi-modal models.

```python
from uni_layer import MultiModalBranchAnalyzer

mm = MultiModalBranchAnalyzer(model)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_multimodal` | `bool` | Whether 2+ branches detected |
| `branch_names` | `List[str]` | Detected branches (e.g., `["vision", "language", "fusion"]`) |
| `branches` | `Dict[str, nn.Module]` | Branch name -> module |
| `branch_layers` | `Dict[str, OrderedDict]` | Branch name -> extracted layers |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_branch_layers(branch_name)` | `OrderedDict` | Layers for one branch |
| `get_all_layers()` | `OrderedDict` | All layers with `branch/` prefix |
| `branch_summary()` | `Dict` | num_layers, num_params, param_ratio per branch |
| `compare_branches(contrib_a, contrib_b, metric)` | `Dict` | Cross-branch metric comparison |
| `to_dict()` | `Dict` | Full JSON export |

**Auto-detected naming conventions:**
- Vision: `vision_model`, `vision_tower`, `visual`, `image_encoder`
- Language: `text_model`, `language_model`, `text_encoder`, `transformer`
- Fusion: `mm_projector`, `multi_modal_projector`, `fusion`, `bridge`

---

## Integration Bridges

### `TorchPruningBridge`

```python
from uni_layer.integrations import TorchPruningBridge

bridge = TorchPruningBridge(model, contributions)
scores = bridge.as_importance_scores("gradient_norm")             # Dict[str, float]
ratios = bridge.as_layer_pruning_ratios("gradient_norm", 0.5)    # Dict[str, float]
protected = bridge.get_protected_layers(top_k=5)                  # List[str]
```

### `HuggingFacePEFTBridge`

```python
from uni_layer.integrations import HuggingFacePEFTBridge

bridge = HuggingFacePEFTBridge(model, contributions)
modules = bridge.recommend_target_modules("gradient_norm", top_k=4)     # List[str]
ranks = bridge.recommend_adaptive_ranks("gradient_norm", base_rank=16)  # Dict[str, int]
config = bridge.recommend_lora_config_params("gradient_norm")           # Dict
```

### `DistillationBridge`

```python
from uni_layer.integrations import DistillationBridge

bridge = DistillationBridge(model, contributions)
pairs = bridge.recommend_layer_pairs(top_k=6)       # List[Tuple]
weights = bridge.recommend_layer_weights()           # Dict[str, float]
```

### `ExportHintsBridge`

```python
from uni_layer.integrations import ExportHintsBridge

bridge = ExportHintsBridge(model, contributions)
plan = bridge.quantization_plan(target="int8", protect_ratio=0.2)  # Dict[str, str]
prunable = bridge.prunable_layers(threshold=0.02)                  # List[Dict]
fusion = bridge.fusion_candidates()                                # List[Dict]
config = bridge.tensorrt_config()                                  # Dict
```

### `AxolotlConfigBridge`

```python
from uni_layer.integrations import AxolotlConfigBridge

bridge = AxolotlConfigBridge(model, contributions)
config = bridge.generate_config(base_model="meta-llama/Llama-2-7b", base_rank=16)
bridge.save_yaml("axolotl_config.yml")
```

### `LLaMAFactoryConfigBridge`

```python
from uni_layer.integrations import LLaMAFactoryConfigBridge

bridge = LLaMAFactoryConfigBridge(model, contributions)
frozen = bridge.recommend_freeze_layers(freeze_ratio=0.3)
config = bridge.generate_config(model_name="my-model")
bridge.save_json("llama_factory.json")
```

### `CompressionSafetyAudit`

```python
from uni_layer.integrations import CompressionSafetyAudit

audit = CompressionSafetyAudit(pre_contributions, post_contributions)
report = audit.audit()                                    # Dict with overall_degradation
deltas = audit.per_layer_delta("adv_sensitivity")         # Dict[str, float]
degraded = audit.degraded_layers("mi_risk_score")         # List[Dict]
```

---

## Utility Functions

### Layer Extraction

```python
from uni_layer.utils.layer_utils import (
    get_model_layers,          # Auto-extract layers from any model
    identify_layer_type,       # Classify layer type (str)
    get_architecture_family,   # Classify model family (str)
    is_attention_layer,        # bool
    is_feedforward_layer,      # bool
    get_layer_parameters,      # (total_params, trainable_params)
)
```

### Model Adaptation

```python
from uni_layer.utils.model_adapter import (
    model_forward,    # Auto-detect HF args, PyG data, etc.
    extract_logits,   # Handle tensor/dict/dataclass/tuple outputs
    compute_loss,     # Handle internal loss (HF) or external criterion
)
```

### Diffusion Utilities

```python
from uni_layer.metrics import get_diffusion_blocks

blocks = get_diffusion_blocks(unet_model)
# OrderedDict: {"down_blocks.0": ..., "mid_block": ..., "up_blocks.0": ...}
```

---

## Output Format

All `compute_metrics()` results follow this schema:

```python
{
    "encoder.layer.0": {
        "layer_idx": 0,
        "layer_type": "transformer_block",
        "gradient_norm": 0.0193,
        "gradient_norm_std": 0.0045,
        "block_influence": 0.1465,
        "effective_rank": 10.54,
        # ... one entry per computed metric
    },
    "encoder.layer.1": { ... },
}
```

See [METRICS.md](METRICS.md) for all output keys per metric.
