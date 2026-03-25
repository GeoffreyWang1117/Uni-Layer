"""
Utilities for extracting and identifying layers from different model architectures.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


def _find_transformer_blocks(model: nn.Module) -> List[Tuple[str, nn.ModuleList]]:
    """
    Search for ALL transformer block ModuleLists in a model.

    Handles arbitrary nesting and Seq2Seq models with both encoder and decoder:
    - model.encoder.layer / model.decoder.layer  (T5, BART)
    - model.model.layers                         (LLaMA, Qwen)
    - model.gpt_neox.layers                      (Pythia)
    - model.transformer.h                        (GPT-2)
    - model.encoder.layer                        (BERT)
    - model.blocks                               (ViT)

    Returns:
        List of (dotted_path, module_list) tuples. Empty list if none found.
    """
    block_attrs = ("layer", "layers", "h", "blocks", "block")
    found = []
    found_ids = set()

    # BFS up to 3 levels deep
    queue = [(name, child) for name, child in model.named_children()]
    visited = set()

    while queue:
        path, module = queue.pop(0)
        if id(module) in visited:
            continue
        visited.add(id(module))

        # Check if this module itself is a ModuleList of transformer blocks
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            first = module[0]
            if len(list(first.children())) >= 2 and id(module) not in found_ids:
                found.append((path, module))
                found_ids.add(id(module))
                continue  # Don't recurse into found block lists

        # Check known attribute names on this module
        for attr in block_attrs:
            child = getattr(module, attr, None)
            if child is not None and isinstance(child, nn.ModuleList) and len(child) >= 2:
                first = child[0]
                if len(list(first.children())) >= 2 and id(child) not in found_ids:
                    found.append((f"{path}.{attr}", child))
                    found_ids.add(id(child))

        # Go deeper (but not too deep)
        if path.count(".") < 2:
            for name, child in module.named_children():
                if id(child) not in visited:
                    queue.append((f"{path}.{name}", child))

    return found


def get_model_layers(model: nn.Module, include_types: Optional[list] = None) -> OrderedDict:
    """
    Extract all relevant layers from a model.

    For transformer models, extracts at **block level** (not individual Linear/LN).
    This works for any nesting depth:
    - model.encoder.layer        (BERT)
    - model.transformer.h        (GPT-2)
    - model.layers               (LLaMA base)
    - model.model.layers         (LLaMA CausalLM wrapper)
    - model.gpt_neox.layers      (Pythia/GPT-NeoX)
    - model.blocks               (ViT/Swin)

    For non-transformer models, falls back to type-based extraction.

    Args:
        model: PyTorch model
        include_types: List of layer types to include (for fallback only)

    Returns:
        OrderedDict of {layer_name: layer_module}
    """
    if include_types is None:
        include_types = [
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.MultiheadAttention,
            nn.Embedding,
            nn.LSTM,
            nn.GRU,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
        ]

    layers = OrderedDict()

    # First: try to find transformer blocks automatically
    found_blocks = _find_transformer_blocks(model)
    if found_blocks:
        for path, module_list in found_blocks:
            for i, block in enumerate(module_list):
                layers[f"{path}.{i}"] = block
        return layers

    # Fallback: generic type-based extraction
    for name, module in model.named_modules():
        if any(isinstance(module, layer_type) for layer_type in include_types):
            # Skip if it's a submodule of another included layer
            parent_included = False
            for existing_name in layers.keys():
                if name.startswith(existing_name + "."):
                    parent_included = True
                    break

            if not parent_included:
                layers[name] = module

    # If still empty, just get all modules with parameters
    if len(layers) == 0:
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                layers[name] = module

    return layers


def identify_layer_type(layer: nn.Module) -> str:
    """
    Identify the type of a layer.

    Args:
        layer: PyTorch module

    Returns:
        String describing the layer type
    """
    # Check for Transformer components
    if isinstance(layer, nn.TransformerEncoderLayer):
        return "transformer_encoder"
    elif isinstance(layer, nn.TransformerDecoderLayer):
        return "transformer_decoder"
    elif isinstance(layer, nn.MultiheadAttention):
        return "attention"

    # Check for common layer types by attribute
    if hasattr(layer, "self_attn") or hasattr(layer, "attn"):
        return "transformer_block"
    elif isinstance(layer, (nn.Linear, nn.LazyLinear)):
        return "linear"
    elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "convolution"
    elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        return "normalization"
    elif isinstance(layer, nn.Embedding):
        return "embedding"
    elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
        return "recurrent"
    elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
        return "pooling"
    else:
        return "unknown"


def get_layer_depth(model: nn.Module, layer_name: str) -> int:
    """
    Get the depth (hierarchy level) of a layer in the model.

    Args:
        model: PyTorch model
        layer_name: Name of the layer

    Returns:
        Depth level (0 for top-level)
    """
    return layer_name.count(".")


def get_layer_parameters(layer: nn.Module) -> Tuple[int, int]:
    """
    Get the number of parameters in a layer.

    Args:
        layer: PyTorch module

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in layer.parameters())
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    return total_params, trainable_params


def is_attention_layer(layer: nn.Module) -> bool:
    """
    Check if a layer is an attention layer.

    Args:
        layer: PyTorch module

    Returns:
        True if the layer contains attention mechanism
    """
    return (
        isinstance(layer, nn.MultiheadAttention)
        or hasattr(layer, "self_attn")
        or hasattr(layer, "attn")
        or "attention" in layer.__class__.__name__.lower()
    )


def is_feedforward_layer(layer: nn.Module) -> bool:
    """
    Check if a layer is a feedforward/MLP layer.

    Args:
        layer: PyTorch module

    Returns:
        True if the layer is a feedforward layer
    """
    return (
        isinstance(layer, nn.Linear)
        or hasattr(layer, "mlp")
        or hasattr(layer, "fc")
        or "feedforward" in layer.__class__.__name__.lower()
        or "mlp" in layer.__class__.__name__.lower()
    )


def get_architecture_family(model: nn.Module) -> str:
    """
    Identify the architecture family of a model.

    Args:
        model: PyTorch model

    Returns:
        Architecture family name
    """
    model_name = model.__class__.__name__.lower()

    # Transformer families
    if any(x in model_name for x in ["bert", "roberta", "electra", "albert"]):
        return "bert_family"
    elif any(x in model_name for x in ["gpt", "gpt2", "gpt-neo"]):
        return "gpt_family"
    elif any(x in model_name for x in ["llama", "mistral", "mixtral"]):
        return "llama_family"
    elif any(x in model_name for x in ["vit", "vision_transformer"]):
        return "vit_family"
    elif any(x in model_name for x in ["swin"]):
        return "swin_family"
    elif "clip" in model_name:
        return "clip_family"

    # CNN families
    elif any(x in model_name for x in ["resnet", "resnext"]):
        return "resnet_family"
    elif any(x in model_name for x in ["convnext"]):
        return "convnext_family"
    elif any(x in model_name for x in ["efficientnet"]):
        return "efficientnet_family"

    # GNN families
    elif any(x in model_name for x in ["gcn", "graphconv"]):
        return "gcn_family"
    elif any(x in model_name for x in ["gat", "graphattention"]):
        return "gat_family"
    elif "sage" in model_name:
        return "graphsage_family"

    # RecSys families
    elif "deepfm" in model_name:
        return "deepfm_family"
    elif "dcn" in model_name:
        return "dcn_family"
    elif "dlrm" in model_name:
        return "dlrm_family"

    # Speech families
    elif "whisper" in model_name:
        return "whisper_family"
    elif "conformer" in model_name:
        return "conformer_family"

    else:
        return "unknown_family"
