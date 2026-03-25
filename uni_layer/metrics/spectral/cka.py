"""
Centered Kernel Alignment (CKA) metric for comparing layer representations.

Uses minibatch CKA for large sample counts to avoid O(N^2) memory usage.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.fast_math import minibatch_cka
from uni_layer.utils.model_adapter import model_forward


class CKA(LayerMetric):
    """
    Compute Centered Kernel Alignment (CKA) score for a layer.

    CKA measures the similarity between layer representations and the output.
    Higher CKA scores indicate that the layer's representations are more aligned
    with the final predictions, suggesting higher importance.

    For large sample counts (>256), uses minibatch CKA (Nguyen et al., 2021)
    which reduces memory from O(N^2) to O(batch_size^2).

    Args:
        compare_to: What to compare against ('output', 'previous', 'input')
        num_batches: Number of batches to use for computation
        kernel: Kernel type ('linear', 'rbf')
        cka_batch_size: Batch size for minibatch CKA (controls memory usage)
    """

    def __init__(
        self,
        compare_to: str = "output",
        num_batches: int = 10,
        kernel: str = "linear",
        cka_batch_size: int = 256,
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
        self.cka_batch_size = cka_batch_size

    def _center_kernel(self, K: torch.Tensor) -> torch.Tensor:
        """Center the kernel matrix"""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def _compute_cka_standard(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Standard CKA for small sample counts (N <= cka_batch_size)."""
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)
        if Y.dim() > 2:
            Y = Y.reshape(Y.size(0), -1)

        if self.kernel == "linear":
            K_X = X @ X.T
            K_Y = Y @ Y.T
        else:
            K_X = self._rbf_kernel(X, X)
            K_Y = self._rbf_kernel(Y, Y)

        K_X = self._center_kernel(K_X)
        K_Y = self._center_kernel(K_Y)

        numerator = (K_X * K_Y).sum()
        denominator = torch.sqrt((K_X * K_X).sum() * (K_Y * K_Y).sum())

        if denominator == 0:
            return 0.0

        return (numerator / denominator).item()

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Compute RBF kernel"""
        XX = (X * X).sum(dim=1).unsqueeze(1)
        YY = (Y * Y).sum(dim=1).unsqueeze(0)
        XY = X @ Y.T
        dist = XX + YY - 2 * XY
        return torch.exp(-dist / (2 * sigma ** 2))

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        _cached_activations: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute CKA for the layer.

        Uses minibatch CKA when sample count exceeds cka_batch_size
        to keep memory usage bounded.

        Returns:
            Dictionary with CKA scores
        """
        layer_activations = []
        target_activations = []

        # Use cached activations for this layer if available
        if _cached_activations:
            layer_activations = _cached_activations
            # Still need to capture target (output) activations
            target_handle = None
            if self.compare_to == "output":
                all_modules = list(model.modules())
                target_layer = all_modules[-1]

                def target_hook_fn(module, input, output):
                    target_activations.append(output.detach().cpu())

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
                            model_forward(model, inputs)
                finally:
                    if target_handle:
                        target_handle.remove()
        else:
            # No cache: capture both layers in one forward pass
            def hook_fn(module, input, output):
                layer_activations.append(output.detach().cpu())

            handle = layer.register_forward_hook(hook_fn)
            target_handle = None

            if self.compare_to == "output":
                all_modules = list(model.modules())
                target_layer = all_modules[-1]

                def target_hook_fn(module, input, output):
                    target_activations.append(output.detach().cpu())

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
                        model_forward(model, inputs)
            finally:
                handle.remove()
                if target_handle is not None:
                    target_handle.remove()

        # Compute CKA
        if layer_activations and target_activations:
            X = torch.cat(layer_activations, dim=0)
            Y = torch.cat(target_activations, dim=0)

            if X.dim() > 2:
                X = X.reshape(X.size(0), -1)
            if Y.dim() > 2:
                Y = Y.reshape(Y.size(0), -1)

            # Choose method based on sample count
            N = X.size(0)
            if N > self.cka_batch_size and self.kernel == "linear":
                cka_score = minibatch_cka(X, Y, batch_size=self.cka_batch_size)
            else:
                cka_score = self._compute_cka_standard(X, Y)

            return {"cka_score": float(cka_score)}
        else:
            return {"cka_score": 0.0}
