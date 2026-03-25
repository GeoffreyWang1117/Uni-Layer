"""
Activation and gradient caching system for LayerAnalyzer.

The core performance problem: without caching, M metrics x L layers = M*L forward passes.
With caching, we do 1 forward pass (or 1 forward+backward) and share results across metrics.

This module provides:
- ActivationCache: One forward pass captures all layer activations
- GradientCache: One backward pass captures all layer gradients
- CachedDataLoader: Wraps a DataLoader to avoid re-iterating over data
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from uni_layer.utils.model_adapter import compute_loss, model_forward


class ActivationCache:
    """
    Captures activations from all layers in a single forward pass.

    Instead of each metric registering its own hook and running its own forward
    pass, the analyzer runs ONE forward pass with hooks on all layers, then
    distributes the cached activations to each metric.

    Usage:
        cache = ActivationCache(model, layers)
        cache.capture(data_loader, num_batches=10, device='cuda')

        # Each metric reads from cache instead of running forward pass
        acts = cache.get(layer_name)  # List[Tensor]
    """

    def __init__(self, model: nn.Module, layers: Dict[str, nn.Module]):
        self.model = model
        self.layers = layers
        self._activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layers}
        self._handles = []

    def capture(
        self,
        data_loader: Any,
        num_batches: int = 10,
        device: str = "cuda",
    ) -> "ActivationCache":
        """Run one forward pass and cache all layer activations."""
        self.clear()
        self._register_hooks()

        try:
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    inputs = inputs.to(device)
                    model_forward(self.model, inputs)
        finally:
            self._remove_hooks()

        return self

    def get(self, layer_name: str) -> List[torch.Tensor]:
        """Get cached activations for a layer."""
        return self._activations.get(layer_name, [])

    def get_concatenated(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get concatenated activation tensor for a layer."""
        acts = self.get(layer_name)
        if acts:
            return torch.cat(acts, dim=0)
        return None

    def has(self, layer_name: str) -> bool:
        return bool(self._activations.get(layer_name))

    def clear(self):
        for name in self._activations:
            self._activations[name] = []

    def _register_hooks(self):
        for name, layer in self.layers.items():

            def make_hook(n):
                def hook_fn(module, input, output):
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    self._activations[n].append(output.detach().cpu())

                return hook_fn

            handle = layer.register_forward_hook(make_hook(name))
            self._handles.append(handle)

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class GradientCache:
    """
    Captures gradients from all layers in a single backward pass.

    Same principle as ActivationCache but for gradient-based metrics.
    One backward pass -> all layer gradients cached -> shared across metrics.
    """

    def __init__(self, model: nn.Module, layers: Dict[str, nn.Module]):
        self.model = model
        self.layers = layers
        self._gradients: Dict[str, List[torch.Tensor]] = {name: [] for name in layers}
        self._handles = []

    def capture(
        self,
        data_loader: Any,
        criterion: nn.Module,
        num_batches: int = 10,
        device: str = "cuda",
    ) -> "GradientCache":
        """Run one forward+backward pass and cache all layer gradients."""
        self.clear()
        self._register_hooks()

        try:
            self.model.train()
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)

                self.model.zero_grad()
                outputs = model_forward(self.model, inputs, targets)
                loss = compute_loss(outputs, targets, criterion)
                loss.backward()
        finally:
            self._remove_hooks()

        return self

    def get(self, layer_name: str) -> List[torch.Tensor]:
        return self._gradients.get(layer_name, [])

    def has(self, layer_name: str) -> bool:
        return bool(self._gradients.get(layer_name))

    def clear(self):
        for name in self._gradients:
            self._gradients[name] = []

    def _register_hooks(self):
        for name, layer in self.layers.items():

            def make_hook(n):
                def hook_fn(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        self._gradients[n].append(grad_output[0].detach().cpu())

                return hook_fn

            handle = layer.register_full_backward_hook(make_hook(name))
            self._handles.append(handle)

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class BatchCache:
    """
    Caches data batches so multiple metrics don't re-iterate the DataLoader.

    Problem: DataLoader iteration order may differ across calls, and some
    DataLoaders can only be iterated once (generators). This caches the
    first N batches for reuse.
    """

    def __init__(self, data_loader: Any, num_batches: int = 10, device: str = "cuda"):
        self.batches = []
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            self.batches.append(batch)
        self.device = device

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
