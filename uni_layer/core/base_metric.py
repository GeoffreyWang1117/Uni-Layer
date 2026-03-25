"""
Base class for all layer contribution metrics.
All metrics inherit from this class and implement the compute() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from uni_layer.utils.model_adapter import extract_logits, compute_loss, model_forward


class LayerMetric(ABC):
    """
    Abstract base class for layer contribution metrics.

    All layer contribution metrics should inherit from this class and implement
    the compute() method. This ensures a unified API across all metrics.

    Attributes:
        name (str): Name of the metric
        category (str): Category of the metric (e.g., 'information_theory', 'optimization')
        requires_gradient (bool): Whether the metric requires gradient computation
        requires_data (bool): Whether the metric requires data samples
        batch_size (int): Batch size for metric computation
    """

    def __init__(
        self,
        name: str,
        category: str,
        requires_gradient: bool = False,
        requires_data: bool = True,
        batch_size: int = 32,
        **kwargs
    ):
        self.name = name
        self.category = category
        self.requires_gradient = requires_gradient
        self.requires_data = requires_data
        self.batch_size = batch_size
        self.config = kwargs

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute the layer contribution metric.

        Args:
            model: The neural network model
            layer: The specific layer to analyze
            layer_name: Name of the layer
            layer_idx: Index of the layer in the model
            data_loader: DataLoader for computing data-dependent metrics
            device: Device to run computation on
            **kwargs: Additional metric-specific arguments

        Returns:
            Dictionary containing metric values
        """
        pass

    def _get_layer_activations(
        self,
        model: nn.Module,
        layer: nn.Module,
        data_loader: Any,
        num_batches: int = 10,
        device: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """
        Helper method to extract layer activations.

        Args:
            model: The neural network model
            layer: The specific layer to analyze
            data_loader: DataLoader for input data
            num_batches: Number of batches to process
            device: Device to run on (auto-detects from model if None)

        Returns:
            List of activation tensors
        """
        if device is None:
            device = next(model.parameters()).device

        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            activations.append(output.detach().cpu())

        handle = layer.register_forward_hook(hook_fn)

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break

                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    inputs = inputs.to(device)

                    model_forward(model, inputs)
        finally:
            handle.remove()

        return activations

    def _get_layer_gradients(
        self,
        model: nn.Module,
        layer: nn.Module,
        data_loader: Any,
        criterion: nn.Module,
        num_batches: int = 10,
        device: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """
        Helper method to extract layer gradients.

        Args:
            model: The neural network model
            layer: The specific layer to analyze
            data_loader: DataLoader for input data
            criterion: Loss function
            num_batches: Number of batches to process
            device: Device to run on (auto-detects from model if None)

        Returns:
            List of gradient tensors
        """
        if device is None:
            device = next(model.parameters()).device

        gradients = []

        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach().cpu())

        handle = layer.register_full_backward_hook(hook_fn)

        try:
            model.train()
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

                model.zero_grad()
                outputs = model_forward(model, inputs, targets)
                loss = compute_loss(outputs, targets, criterion)
                loss.backward()
        finally:
            handle.remove()

        return gradients

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"
