"""
Neural Tangent Kernel (NTK) trace metric for layer contribution.

Optimized to avoid storing the full Jacobian matrix, which is infeasible
for large models (e.g., LLaMA-7B would need >5TB GPU memory).

Instead, we compute Tr(NTK) = Tr(J @ J^T) = ||J||_F^2 incrementally:
accumulate ||grad_i||^2 for each sample without storing the full J matrix.
Memory: O(P) instead of O(S*P).
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import extract_logits, model_forward


class NTKTrace(LayerMetric):
    """
    Compute Neural Tangent Kernel (NTK) trace for a layer.

    The NTK measures how layer parameters influence the model's predictions.
    Higher NTK trace indicates greater influence on the output.

    This implementation uses an incremental Frobenius norm approach:
        Tr(NTK) = Tr(J J^T) = sum_i ||J_i||^2

    where J_i is the Jacobian row for sample i. This avoids storing the
    full (num_samples x num_params) Jacobian matrix.

    Args:
        num_samples: Number of samples to use for NTK computation
        num_classes: Number of output classes (for classification)
    """

    def __init__(self, num_samples: int = 100, num_classes: Optional[int] = None, **kwargs):
        super().__init__(
            name="ntk_trace",
            category="spectral",
            requires_gradient=True,
            requires_data=True,
            **kwargs,
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
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute NTK trace for the layer using incremental accumulation.

        Memory: O(P) where P = number of layer parameters.
        Time: O(S * forward_pass) where S = num_samples.

        Returns:
            Dictionary with NTK trace approximation
        """
        params = [p for p in layer.parameters() if p.requires_grad]
        if not params:
            return {"ntk_trace": 0.0, "ntk_trace_per_param": 0.0}

        total_params = sum(p.numel() for p in params)

        # Incremental: accumulate ||grad||^2 per sample, don't store J
        trace_sum = 0.0
        num_processed = 0

        model.eval()

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

            for i in range(batch_size):
                model.zero_grad()
                x = inputs[i : i + 1]

                output = extract_logits(model_forward(model, x))

                # Scalar output for backward
                if output.dim() > 1 and output.size(1) > 1:
                    output = output.mean(dim=1)

                output.backward(retain_graph=False)

                # Accumulate ||grad_i||^2 (this is ||J_i||^2)
                grad_norm_sq = 0.0
                for param in params:
                    if param.grad is not None:
                        grad_norm_sq += (param.grad**2).sum().item()

                trace_sum += grad_norm_sq

            num_processed += batch_size

        if num_processed > 0:
            ntk_trace = trace_sum / num_processed
            return {
                "ntk_trace": float(ntk_trace),
                "ntk_trace_per_param": float(ntk_trace / total_params) if total_params > 0 else 0.0,
            }
        else:
            return {
                "ntk_trace": 0.0,
                "ntk_trace_per_param": 0.0,
            }
