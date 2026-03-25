"""
Gradient Norm metric for measuring layer contribution based on gradient magnitudes.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import compute_loss, model_forward


class GradientNorm(LayerMetric):
    """
    Compute gradient norm for a layer.

    This metric measures the magnitude of gradients flowing through a layer,
    which indicates how much the layer contributes to learning.

    Higher gradient norms typically indicate:
    - More important layers for the task
    - Layers actively being optimized
    - Potential training instability if too high

    Args:
        norm_type: Type of norm to compute ('l1', 'l2', 'linf')
        num_batches: Number of batches to average over
        aggregate: How to aggregate across parameters ('mean', 'sum', 'max')
    """

    def __init__(
        self,
        norm_type: str = "l2",
        num_batches: int = 10,
        aggregate: str = "mean",
        **kwargs
    ):
        super().__init__(
            name="gradient_norm",
            category="optimization",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )
        self.norm_type = norm_type
        self.num_batches = num_batches
        self.aggregate = aggregate

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute gradient norm for the layer.

        Returns:
            Dictionary with gradient norm statistics
        """
        gradient_norms = []

        # Collect gradients over multiple batches
        model.train()
        for i, batch in enumerate(data_loader):
            if i >= self.num_batches:
                break

            # Parse batch
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            # Move to device
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)

            # Forward pass
            model.zero_grad()
            outputs = model_forward(model, inputs, targets)
            loss = compute_loss(outputs, targets, criterion)
            loss.backward()

            # Collect gradient norms for this layer
            layer_grad_norms = []
            for param in layer.parameters():
                if param.grad is not None:
                    if self.norm_type == "l1":
                        norm = param.grad.abs().sum().item()
                    elif self.norm_type == "l2":
                        norm = param.grad.norm(2).item()
                    elif self.norm_type == "linf":
                        norm = param.grad.abs().max().item()
                    else:
                        norm = param.grad.norm(2).item()

                    layer_grad_norms.append(norm)

            # Aggregate across parameters
            if layer_grad_norms:
                if self.aggregate == "mean":
                    batch_norm = np.mean(layer_grad_norms)
                elif self.aggregate == "sum":
                    batch_norm = np.sum(layer_grad_norms)
                elif self.aggregate == "max":
                    batch_norm = np.max(layer_grad_norms)
                else:
                    batch_norm = np.mean(layer_grad_norms)

                gradient_norms.append(batch_norm)

        # Compute statistics
        if gradient_norms:
            return {
                "gradient_norm": float(np.mean(gradient_norms)),
                "gradient_norm_std": float(np.std(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "gradient_norm_min": float(np.min(gradient_norms)),
            }
        else:
            return {
                "gradient_norm": 0.0,
                "gradient_norm_std": 0.0,
                "gradient_norm_max": 0.0,
                "gradient_norm_min": 0.0,
            }
