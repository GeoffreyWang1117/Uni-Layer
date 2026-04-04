"""
Utilities for adapting different model output formats (HuggingFace, custom, etc.)
to the standard tensor format expected by Uni-Layer metrics.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def extract_logits(outputs: Any) -> torch.Tensor:
    """
    Extract logits tensor from various model output formats.

    Handles:
    - Raw tensor (standard PyTorch)
    - Dict with 'logits' key (HuggingFace style)
    - Dataclass with .logits attribute (HuggingFace ModelOutput)
    - Tuple/list (first element assumed to be logits)

    Args:
        outputs: Raw model output in any format

    Returns:
        Logits tensor suitable for loss computation
    """
    if isinstance(outputs, torch.Tensor):
        return outputs

    # HuggingFace dataclass (BaseModelOutput, etc.)
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits

    # Some HF models use last_hidden_state without logits
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state

    # Dict output
    if isinstance(outputs, dict):
        for key in ("logits", "prediction_logits", "seq_relationship_logits", "start_logits"):
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        # Fallback: return first tensor value
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v

    # Tuple/list output (common in older models)
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
        return extract_logits(first)

    raise ValueError(
        f"Cannot extract logits from output type {type(outputs).__name__}. "
        f"Provide a custom criterion or ensure model returns tensors/dicts with 'logits' key."
    )


def compute_loss(
    outputs: Any,
    targets: Optional[torch.Tensor],
    criterion: Optional[nn.Module],
) -> torch.Tensor:
    """
    Compute loss from model outputs, handling various output formats.

    Args:
        outputs: Raw model output (tensor, dict, dataclass, tuple)
        targets: Target labels/values (can be None)
        criterion: Loss function (can be None for fallback)

    Returns:
        Scalar loss tensor
    """
    # Some HF models compute loss internally
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs and outputs["loss"] is not None:
        return outputs["loss"]

    logits = extract_logits(outputs)

    if criterion is not None and targets is not None:
        # Only apply classification criterion when logits are 2-D [B, num_classes].
        # For sequence models returning [B, seq_len, hidden] use mean() instead,
        # which still creates a valid gradient signal for layer importance scoring.
        if logits.dim() == 2:
            try:
                return criterion(logits, targets)
            except Exception:
                pass
        # For 3-D (last_hidden_state) or fallback: differentiate through mean
        return logits.mean()

    # Fallback: mean of logits (works for any shape)
    return logits.mean()


# Cache for inspect.signature results — avoids 7us overhead per forward call
# Keyed by class object (not id) to prevent stale entries after GC
_forward_params_cache: Dict[type, set] = {}


def _get_forward_params(model: nn.Module) -> set:
    """Get forward() parameter names with caching (52x faster than inspect each call)."""
    cls = model.__class__
    if cls not in _forward_params_cache:
        try:
            import inspect

            _forward_params_cache[cls] = set(inspect.signature(model.forward).parameters.keys())
        except (ValueError, TypeError):
            _forward_params_cache[cls] = set()
    return _forward_params_cache[cls]


def _is_pyg_data(obj: Any) -> bool:
    """Check if an object is a PyG Data/Batch without hard dependency."""
    # Check MRO class names to handle dynamic subclasses like DataBatch
    mro_names = {cls.__name__ for cls in type(obj).__mro__}
    return bool(mro_names & {"Data", "Batch", "HeteroData", "BaseData"}) and hasattr(obj, "x")


def model_forward(
    model: nn.Module,
    inputs: Any,
    targets: Optional[torch.Tensor] = None,
) -> Any:
    """
    Run model forward pass with automatic argument detection.

    Handles:
    - Standard PyTorch models (tensor input)
    - HuggingFace models (attention_mask, labels, decoder_input_ids)
    - PyG GNN models (Data/Batch objects with x, edge_index, batch)

    Args:
        model: PyTorch model
        inputs: Input tensor or PyG Data/Batch object
        targets: Optional targets (passed as 'labels' for HF models)

    Returns:
        Raw model output
    """
    # Handle PyG graph data objects
    if _is_pyg_data(inputs):
        forward_params = _get_forward_params(model)
        kwargs = {}
        # Common PyG forward signatures: forward(x, edge_index, batch=None)
        if "edge_index" in forward_params and hasattr(inputs, "edge_index"):
            kwargs["edge_index"] = inputs.edge_index
        if "batch" in forward_params and hasattr(inputs, "batch"):
            kwargs["batch"] = inputs.batch
        if "edge_attr" in forward_params and hasattr(inputs, "edge_attr"):
            kwargs["edge_attr"] = inputs.edge_attr
        return model(inputs.x, **kwargs)

    forward_params = _get_forward_params(model)

    kwargs = {}

    if "attention_mask" in forward_params:
        # Create attention mask (all ones = no masking)
        if isinstance(inputs, torch.Tensor) and inputs.dim() >= 2:
            kwargs["attention_mask"] = torch.ones(
                inputs.shape[:2], dtype=torch.long, device=inputs.device
            )

    if "labels" in forward_params and targets is not None:
        # Causal/Seq2Seq LM models (Llama 4, Gemma 4, BERT-LM, etc.) expect labels of
        # shape [B, seq_len]. Injecting 1-D classification labels ([B]) causes internal
        # shape errors (e.g. 2-D boolean mask applied to 1-D shift_labels tensor).
        # Detect CausalLM/ConditionalGeneration by class name convention and skip label
        # injection — compute_loss() uses logits.mean() as the gradient signal instead.
        # Raw nn.Module models (e.g. custom HFStyleModel) always receive labels normally.
        _is_seq_input = isinstance(inputs, torch.Tensor) and inputs.dim() >= 2
        _is_class_target = targets.dim() == 1
        _cls_name = type(model).__name__
        _is_lm_head = any(
            s in _cls_name
            for s in ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel", "CausalLM")
        )
        if not (_is_seq_input and _is_class_target and _is_lm_head):
            kwargs["labels"] = targets

    # Seq2Seq models (T5, BART, etc.) require decoder_input_ids
    if "decoder_input_ids" in forward_params and "decoder_input_ids" not in kwargs:
        kwargs["decoder_input_ids"] = inputs[:, :1]  # Minimal decoder input

    # Gemma 4 multimodal models require mm_token_type_ids during training.
    # Pass all-zeros (= text tokens, no image tokens) so the model runs text-only.
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type == "gemma4" and isinstance(inputs, torch.Tensor) and inputs.dim() >= 2:
        kwargs["mm_token_type_ids"] = torch.zeros(
            inputs.shape[:2], dtype=torch.long, device=inputs.device
        )

    if kwargs:
        return model(inputs, **kwargs)
    return model(inputs)
