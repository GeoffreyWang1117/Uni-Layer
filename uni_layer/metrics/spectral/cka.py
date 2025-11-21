"""
Centered Kernel Alignment (CKA) metric for comparing layer representations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class CKA(LayerMetric):
    """
    Compute Centered Kernel Alignment (CKA) score for a layer.

    CKA measures the similarity between layer representations and the output.
    Higher CKA scores indicate that the layer's representations are more aligned
    with the final predictions, suggesting higher importance.

    This metric compares the layer's activations with:
    1. The final layer's activations (default)
    2. The previous layer's activations (for measuring redundancy)

    Args:
        compare_to: What to compare against ('output', 'previous', 'input')
        num_batches: Number of batches to use for computation
        kernel: Kernel type ('linear', 'rbf')
    """

    def __init__(
        self,
        compare_to: str = "output",
        num_batches: int = 10,
        kernel: str = "linear",
        **kwargs
    ):
        super().__init__(
            name="cka",
            category="spectral",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )
        self.compare_to = compare_to
        self.num_batches = num_batches
        self.kernel = kernel

    def _center_kernel(self, K: torch.Tensor) -> torch.Tensor:
        """Center the kernel matrix"""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def _linear_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear kernel"""
        return X @ Y.T

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Compute RBF kernel"""
        XX = (X * X).sum(dim=1).unsqueeze(1)
        YY = (Y * Y).sum(dim=1).unsqueeze(0)
        XY = X @ Y.T
        dist = XX + YY - 2 * XY
        return torch.exp(-dist / (2 * sigma ** 2))

    def _compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute CKA between two sets of representations.

        Args:
            X: Representations from layer 1 (n_samples x d1)
            Y: Representations from layer 2 (n_samples x d2)

        Returns:
            CKA score
        """
        # Flatten if needed
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)
        if Y.dim() > 2:
            Y = Y.reshape(Y.size(0), -1)

        # Compute kernels
        if self.kernel == "linear":
            K_X = self._linear_kernel(X, X)
            K_Y = self._linear_kernel(Y, Y)
        else:  # rbf
            K_X = self._rbf_kernel(X, X)
            K_Y = self._rbf_kernel(Y, Y)

        # Center kernels
        K_X = self._center_kernel(K_X)
        K_Y = self._center_kernel(K_Y)

        # Compute CKA
        numerator = (K_X * K_Y).sum()
        denominator = torch.sqrt((K_X * K_X).sum() * (K_Y * K_Y).sum())

        if denominator == 0:
            return 0.0

        return (numerator / denominator).item()

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
        Compute CKA for the layer.

        Returns:
            Dictionary with CKA scores
        """
        # Collect activations from this layer
        layer_activations = []

        def hook_fn(module, input, output):
            layer_activations.append(output.detach())

        handle = layer.register_forward_hook(hook_fn)

        # Collect activations from comparison target
        target_activations = []
        target_handle = None

        # Find comparison layer
        if self.compare_to == "output":
            # Get the last layer with activations
            all_modules = list(model.modules())
            target_layer = all_modules[-1]

            def target_hook_fn(module, input, output):
                target_activations.append(output.detach())

            target_handle = target_layer.register_forward_hook(target_hook_fn)

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= self.num_batches:
                        break

                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    inputs = inputs.to(device)
                    model(inputs)

        finally:
            handle.remove()
            if target_handle is not None:
                target_handle.remove()

        # Concatenate activations
        if layer_activations and target_activations:
            X = torch.cat([act.cpu() for act in layer_activations], dim=0)
            Y = torch.cat([act.cpu() for act in target_activations], dim=0)

            cka_score = self._compute_cka(X, Y)

            return {
                "cka_score": float(cka_score),
            }
        else:
            return {
                "cka_score": 0.0,
            }
