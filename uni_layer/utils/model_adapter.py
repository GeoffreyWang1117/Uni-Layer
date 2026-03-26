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
        return criterion(logits, targets)

    # Fallback: mean of logits (works for any shape)
    return logits.mean()


# Cache for inspect.signature results — avoids 7us overhead per forward call
_forward_params_cache: Dict[int, set] = {}


def _get_forward_params(model: nn.Module) -> set:
    """Get forward() parameter names with caching (52x faster than inspect each call)."""
    key = id(model.__class__)
    if key not in _forward_params_cache:
        try:
            import inspect

            _forward_params_cache[key] = set(inspect.signature(model.forward).parameters.keys())
        except (ValueError, TypeError):
            _forward_params_cache[key] = set()
    return _forward_params_cache[key]


def model_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> Any:
    """
    Run model forward pass with automatic HuggingFace argument detection.

    Handles models that need attention_mask, labels, or other kwargs.

    Args:
        model: PyTorch model
        inputs: Input tensor
        targets: Optional targets (passed as 'labels' for HF models)

    Returns:
        Raw model output
    """
    forward_params = _get_forward_params(model)

    kwargs = {}

    if "attention_mask" in forward_params:
        # Create attention mask (all ones = no masking)
        if inputs.dim() >= 2:
            kwargs["attention_mask"] = torch.ones(
                inputs.shape[:2], dtype=torch.long, device=inputs.device
            )

    if "labels" in forward_params and targets is not None:
        kwargs["labels"] = targets

    # Seq2Seq models (T5, BART, etc.) require decoder_input_ids
    if "decoder_input_ids" in forward_params and "decoder_input_ids" not in kwargs:
        kwargs["decoder_input_ids"] = inputs[:, :1]  # Minimal decoder input

    if kwargs:
        return model(inputs, **kwargs)
    return model(inputs)
