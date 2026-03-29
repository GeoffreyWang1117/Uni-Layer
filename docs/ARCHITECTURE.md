# Architecture Support Guide (v0.6.1)

Uni-Layer auto-detects model architecture and extracts layers at the appropriate granularity. This guide shows usage for each supported architecture family.

## Supported Architectures

| Family | Layer Extraction | Verified Models | Type Label |
|--------|-----------------|-----------------|------------|
| Transformer | Block-level | BERT, GPT-2, LLaMA, Qwen, T5, ViT, DINOv2 | `transformer_block` |
| CNN | Block/layer level | ResNet, ConvNeXt, EfficientNet | `convolution` |
| Mamba/SSM | Block-level | Mamba, S4, S6 | `ssm_block` |
| GNN | Conv-level (PyG) | GCNConv, GATConv, SAGEConv, GINConv | `gnn_conv`, `gnn_attention` |
| Diffusion | down/mid/up blocks | UNet, DDPM, DiT | `diffusion_block` |
| MoE | Router + expert | Mixtral, Switch Transformer | `transformer_block` + MoE metrics |
| Multi-Modal | Per-branch | CLIP, LLaVA, OpenCLIP | Per-branch types |

---

## 1. Transformer Models

Auto-detected at block level regardless of nesting depth.

```python
from transformers import AutoModel
from uni_layer import LayerAnalyzer

model = AutoModel.from_pretrained("bert-base-uncased")
analyzer = LayerAnalyzer(model, task_type="classification")
# -> 12 layers: encoder.layer.0 ... encoder.layer.11

contributions = analyzer.compute_metrics(preset="llm_fast", data_loader=loader)
```

**Verified models (20+):**
- BERT, RoBERTa, DeBERTa-v3, DistilBERT, SciBERT, MiniLM
- GPT-2, Pythia, BLOOM, Falcon, TinyLlama, Llama-3.2, Qwen2.5
- T5, BART, ByT5 (Seq2Seq with encoder + decoder blocks)
- ViT, DINOv2, Wav2Vec2, HuBERT

**Seq2Seq note:** For encoder-decoder models, Uni-Layer extracts both encoder and decoder blocks automatically. `decoder_input_ids` is auto-injected.

---

## 2. Mamba / SSM Models

Detected by class name (`mamba`, `ssm`, `s4`, `s6`) or SSM-specific parameters (`A_log`, `dt_proj`).

```python
# Mamba models with .layers ModuleList
analyzer = LayerAnalyzer(mamba_model)
# -> layers.0, layers.1, ... (MambaBlocks)

contributions = analyzer.compute_metrics(preset="quick", data_loader=loader)
# ResidualDropLayer works here — Mamba blocks have residual connections
```

**Type labels:** `ssm_block` (compound block) or `ssm_layer` (single SSM).

**Recommended metrics:** BlockInfluence, ResidualDropLayer, EffectiveRank, ActivationEntropy.

---

## 3. GNN Models (PyG)

Requires `torch-geometric`. Detected via `MessagePassing` subclasses.

```python
from torch_geometric.nn import GCNConv, GATConv
from uni_layer import LayerAnalyzer

# GNN model with conv layers
analyzer = LayerAnalyzer(gnn_model)
# -> convs.0 (GCNConv), convs.1 (GCNConv), ...

# DataLoader yields (PyG Batch, targets) tuples
contributions = analyzer.compute_metrics(
    metrics=[BlockInfluence(), EffectiveRank(), ActivationEntropy()],
    data_loader=graph_loader,
)
```

**Type labels:** `gnn_conv` (GCN), `gnn_attention` (GAT), `gnn_sage` (GraphSAGE).

**PyG Data handling:** `model_forward()` auto-unpacks `Data`/`Batch` objects, passing `x`, `edge_index`, `batch` to the model.

---

## 4. Diffusion Models (UNet / DiT)

Detected by `down_blocks`/`mid_block`/`up_blocks` attributes or timestep parameter.

```python
from uni_layer.metrics import DiffusionTimestepAnalysis, get_diffusion_blocks

# Extract UNet blocks
blocks = get_diffusion_blocks(unet_model)
# -> down_blocks.0, down_blocks.1, mid_block, up_blocks.0, up_blocks.1

# Timestep-aware analysis
m = DiffusionTimestepAnalysis(num_timesteps=10, max_timestep=1000)
for name, block in blocks.items():
    result = m.compute(model=unet_model, layer=block, layer_name=name,
                       layer_idx=0, data_loader=loader, device="cuda")
    print(f"{name}: sensitivity={result['timestep_sensitivity']:.3f}, "
          f"early={result['early_importance']:.3f}, late={result['late_importance']:.3f}")
```

**Type labels:** `diffusion_block`, `diffusion_unet`.

**Key insight:** `early_importance` vs `late_importance` reveals which layers matter most at low vs. high noise — critical for timestep-adaptive pruning.

---

## 5. MoE (Mixture of Experts)

MoE layers are detected by `gate`/`router` submodules and `experts` ModuleList.

```python
from uni_layer.metrics import MoERouterAnalysis

m = MoERouterAnalysis(num_batches=10, top_k=2)
result = m.compute(model=moe_model, layer=moe_layer, ...)
# routing_entropy, expert_utilization, load_balance_score, top_expert_ratio
```

**Non-MoE layers return `None`** — safe to include in a sweep across all layers.

**Supported patterns:**
- `layer.gate` (Mixtral)
- `layer.router` (Switch Transformer)
- `layer.experts` (ModuleList of expert networks)
- `layer.num_experts` or inferred from `gate.out_features`

---

## 6. Multi-Modal Models

Use `MultiModalBranchAnalyzer` for models with separate vision/language/fusion components.

```python
from uni_layer import MultiModalBranchAnalyzer, LayerAnalyzer

mm = MultiModalBranchAnalyzer(clip_model)
print(mm.branch_names)     # ["vision", "language", "fusion"]
print(mm.branch_summary()) # params, layers per branch

# Analyze each branch independently
for branch_name in mm.branch_names:
    branch_layers = mm.get_branch_layers(branch_name)
    # Create a sub-analyzer or use layers directly
    print(f"{branch_name}: {len(branch_layers)} layers")

# Compare branches
comparison = mm.compare_branches(vision_contributions, language_contributions, "effective_rank")
print(f"Vision avg rank: {comparison['mean_a']:.1f}, Language: {comparison['mean_b']:.1f}")
```

**Auto-detected naming:**
- Vision: `vision_model`, `vision_tower`, `visual`, `image_encoder`
- Language: `text_model`, `language_model`, `text_encoder`
- Fusion: `mm_projector`, `multi_modal_projector`, `fusion`, `bridge`

---

## 7. CNN Models

Fallback extraction via type-based detection (nn.Conv2d, nn.Linear, etc.).

```python
import torchvision.models as models
from uni_layer import LayerAnalyzer

model = models.resnet50(pretrained=True)
analyzer = LayerAnalyzer(model)
# -> Extracts residual blocks (layer1.0, layer1.1, ..., layer4.2)

contributions = analyzer.compute_metrics(preset="quick", data_loader=loader)
```

**Recommended metrics:** EffectiveRank, GradientNorm, EfficiencyProfiler (FLOPs especially useful for CNNs).

---

## Layer Type Detection

`identify_layer_type(layer)` returns one of:

| Type String | Architectures |
|-------------|--------------|
| `transformer_block` | BERT, GPT, LLaMA, ViT blocks |
| `transformer_encoder` | nn.TransformerEncoderLayer |
| `transformer_decoder` | nn.TransformerDecoderLayer |
| `attention` | nn.MultiheadAttention |
| `ssm_block` | Mamba compound blocks |
| `ssm_layer` | Individual SSM layers |
| `gnn_conv` | GCNConv, GINConv |
| `gnn_attention` | GATConv |
| `gnn_sage` | SAGEConv |
| `diffusion_block` | UNet down/up blocks |
| `diffusion_unet` | UNet top-level |
| `linear` | nn.Linear |
| `convolution` | nn.Conv1d/2d/3d |
| `normalization` | nn.LayerNorm, nn.BatchNorm |
| `embedding` | nn.Embedding |
| `recurrent` | nn.LSTM, nn.GRU |
| `pooling` | nn.MaxPool, nn.AvgPool |
| `unknown` | Unrecognized |

## Architecture Family Detection

`get_architecture_family(model)` returns one of:

| Family | Triggers |
|--------|----------|
| `bert_family` | bert, roberta, electra, albert |
| `gpt_family` | gpt, gpt2, gpt-neo |
| `llama_family` | llama, mistral, mixtral |
| `mamba_family` | mamba, ssm, s4 |
| `diffusion_family` | unet, diffusion, ddpm, dit |
| `vit_family` | vit, vision_transformer |
| `resnet_family` | resnet, resnext |
| `gcn_family` | gcn, graphconv |
| `gat_family` | gat, graphattention |
| `gnn_family` | jknet, graphunet |
| `unknown_family` | No match |
