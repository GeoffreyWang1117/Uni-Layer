"""
Utilities for registering hooks to extract activations and gradients.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any


class ActivationHook:
    """Hook to capture layer activations"""

    def __init__(self):
        self.activations = []

    def __call__(self, module, input, output):
        self.activations.append(output.detach().cpu())

    def clear(self):
        self.activations = []


class GradientHook:
    """Hook to capture layer gradients"""

    def __init__(self):
        self.gradients = []

    def __call__(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients.append(grad_output[0].detach().cpu())

    def clear(self):
        self.gradients = []


def register_hooks(
    model: nn.Module,
    layers: Dict[str, nn.Module],
    hook_type: str = "forward"
) -> Dict[str, Any]:
    """
    Register hooks on specified layers.

    Args:
        model: PyTorch model
        layers: Dictionary of {layer_name: layer_module}
        hook_type: Type of hook ('forward', 'backward', or 'both')

    Returns:
        Dictionary of {layer_name: hook_handle}
    """
    handles = {}

    for layer_name, layer in layers.items():
        if hook_type in ["forward", "both"]:
            hook = ActivationHook()
            handle = layer.register_forward_hook(hook)
            handles[f"{layer_name}_forward"] = (handle, hook)

        if hook_type in ["backward", "both"]:
            hook = GradientHook()
            handle = layer.register_full_backward_hook(hook)
            handles[f"{layer_name}_backward"] = (handle, hook)

    return handles


def remove_hooks(handles: Dict[str, Any]):
    """
    Remove registered hooks.

    Args:
        handles: Dictionary of hook handles from register_hooks()
    """
    for handle, hook in handles.values():
        handle.remove()
        if hasattr(hook, "clear"):
            hook.clear()
