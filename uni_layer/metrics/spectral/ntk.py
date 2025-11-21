"""
Neural Tangent Kernel (NTK) trace metric for layer contribution.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class NTKTrace(LayerMetric):
    """
    Compute Neural Tangent Kernel (NTK) trace for a layer.

    The NTK measures how layer parameters influence the model's predictions.
    Higher NTK trace indicates greater influence on the output.

    This implementation computes an approximation of the layer's contribution
    to the full NTK using Jacobian-based methods.

    Args:
        num_samples: Number of samples to use for NTK computation
        num_classes: Number of output classes (for classification)
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            name="ntk_trace",
            category="spectral",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )
        self.num_samples = num_samples
        self.num_classes = num_classes

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
        Compute NTK trace for the layer.

        Returns:
            Dictionary with NTK trace approximation
        """
        # Collect layer parameters
        params = [p for p in layer.parameters() if p.requires_grad]
        if not params:
            return {"ntk_trace": 0.0}

        jacobians = []

        model.eval()
        num_processed = 0

        for batch in data_loader:
            if num_processed >= self.num_samples:
                break

            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            batch_size = min(inputs.size(0), self.num_samples - num_processed)
            inputs = inputs[:batch_size]

            # Compute Jacobian for each sample
            for i in range(batch_size):
                model.zero_grad()
                x = inputs[i:i+1]

                # Forward pass
                output = model(x)

                # For multi-class, we need to compute Jacobian for each output
                if output.dim() > 1 and output.size(1) > 1:
                    # Take mean over output dimensions for efficiency
                    output = output.mean(dim=1)

                # Compute gradients w.r.t. layer parameters
                layer_jacobian = []
                output.backward(retain_graph=False)

                for param in params:
                    if param.grad is not None:
                        layer_jacobian.append(param.grad.detach().flatten())

                if layer_jacobian:
                    jacobians.append(torch.cat(layer_jacobian))

            num_processed += batch_size

        if jacobians:
            # Stack Jacobians: (num_samples, num_params)
            J = torch.stack(jacobians)

            # Compute NTK = J @ J^T
            # Trace(NTK) = Trace(J @ J^T) = ||J||_F^2
            ntk_trace = (J ** 2).sum().item()

            # Normalize by number of samples
            ntk_trace /= len(jacobians)

            return {
                "ntk_trace": float(ntk_trace),
                "ntk_trace_per_param": float(ntk_trace / J.size(1)) if J.size(1) > 0 else 0.0,
            }
        else:
            return {
                "ntk_trace": 0.0,
                "ntk_trace_per_param": 0.0,
            }
