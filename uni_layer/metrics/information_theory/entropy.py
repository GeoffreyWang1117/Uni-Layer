"""
Activation Entropy metric for measuring layer information content.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class ActivationEntropy(LayerMetric):
    """
    Compute entropy of layer activations.

    Activation entropy measures the diversity and information content of
    layer representations. Higher entropy indicates more diverse activations.

    Args:
        num_batches: Number of batches to use
        num_bins: Number of bins for discretization
    """

    def __init__(
        self,
        num_batches: int = 10,
        num_bins: int = 50,
        **kwargs
    ):
        super().__init__(
            name="activation_entropy",
            category="information_theory",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )
        self.num_batches = num_batches
        self.num_bins = num_bins

    def _compute_entropy(self, X: torch.Tensor) -> float:
        """
        Compute entropy of activation values.

        Args:
            X: Activation tensor

        Returns:
            Entropy value
        """
        # Flatten
        X = X.flatten()

        # Create histogram
        hist, _ = np.histogram(X.numpy(), bins=self.num_bins, density=True)

        # Normalize to get probability distribution
        hist = hist / (hist.sum() + 1e-10)

        # Compute entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))

        return entropy

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
        Compute activation entropy for the layer.

        Returns:
            Dictionary with entropy metrics
        """
        activations = self._get_layer_activations(
            model=model,
            layer=layer,
            data_loader=data_loader,
            num_batches=self.num_batches
        )

        if activations:
            X = torch.cat(activations, dim=0)

            # Compute entropy
            entropy = self._compute_entropy(X)

            # Also compute activation statistics
            mean_activation = X.mean().item()
            std_activation = X.std().item()
            sparsity = (X.abs() < 1e-3).float().mean().item()

            return {
                "activation_entropy": float(entropy),
                "activation_mean": float(mean_activation),
                "activation_std": float(std_activation),
                "activation_sparsity": float(sparsity),
            }
        else:
            return {
                "activation_entropy": 0.0,
                "activation_mean": 0.0,
                "activation_std": 0.0,
                "activation_sparsity": 0.0,
            }
