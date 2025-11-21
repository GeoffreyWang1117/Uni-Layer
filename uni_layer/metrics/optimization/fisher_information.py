"""
Fisher Information metric for measuring layer importance.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class FisherInformation(LayerMetric):
    """
    Compute Fisher Information Matrix trace for a layer.

    Fisher Information measures the sensitivity of the model's output distribution
    to changes in layer parameters. Higher Fisher Information indicates more
    important layers.

    This implementation computes the empirical Fisher Information:
    F = E[∇log p(y|x) ∇log p(y|x)^T]

    Args:
        num_batches: Number of batches to average over
        empirical: Whether to use empirical Fisher (True) or true Fisher (False)
    """

    def __init__(
        self,
        num_batches: int = 10,
        empirical: bool = True,
        **kwargs
    ):
        super().__init__(
            name="fisher_information",
            category="optimization",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )
        self.num_batches = num_batches
        self.empirical = empirical

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
        Compute Fisher Information for the layer.

        Returns:
            Dictionary with Fisher Information trace
        """
        fisher_diag = {}
        for name, param in layer.named_parameters():
            fisher_diag[name] = torch.zeros_like(param)

        model.train()
        num_samples = 0

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
            outputs = model(inputs)

            if self.empirical and targets is not None:
                # Empirical Fisher: use actual labels
                if criterion is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = outputs.mean()
            else:
                # True Fisher: sample from model's output distribution
                if outputs.dim() == 2:  # Classification
                    probs = torch.softmax(outputs, dim=1)
                    sampled_targets = torch.multinomial(probs, 1).squeeze()
                    loss = nn.functional.cross_entropy(outputs, sampled_targets)
                else:
                    loss = outputs.mean()

            # Backward pass
            loss.backward()

            # Accumulate squared gradients (diagonal Fisher approximation)
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] += param.grad.pow(2).detach()

            num_samples += inputs.size(0)

        # Average Fisher Information
        for name in fisher_diag:
            fisher_diag[name] /= num_samples

        # Compute trace (sum of diagonal elements)
        fisher_trace = sum(f.sum().item() for f in fisher_diag.values())

        return {
            "fisher_information": float(fisher_trace),
            "fisher_mean": float(fisher_trace / sum(f.numel() for f in fisher_diag.values()))
            if fisher_diag else 0.0,
        }
