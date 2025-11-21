"""
Hessian Trace metric for measuring layer curvature and optimization landscape.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class HessianTrace(LayerMetric):
    """
    Compute Hessian trace for a layer (approximation).

    The Hessian trace measures the curvature of the loss landscape with respect
    to layer parameters. Higher trace values indicate sharper minima.

    This implementation uses the Hutchinson trace estimator for efficiency.

    Args:
        num_samples: Number of random vectors for trace estimation
        num_batches: Number of batches to average over
    """

    def __init__(
        self,
        num_samples: int = 5,
        num_batches: int = 5,
        **kwargs
    ):
        super().__init__(
            name="hessian_trace",
            category="optimization",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )
        self.num_samples = num_samples
        self.num_batches = num_batches

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
        Compute Hessian trace approximation for the layer.

        Returns:
            Dictionary with Hessian trace estimate
        """
        traces = []

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

            # Hutchinson trace estimator
            trace_estimate = 0.0

            for _ in range(self.num_samples):
                # Generate random vector
                v = {}
                for name, param in layer.named_parameters():
                    v[name] = torch.randn_like(param)

                # Compute gradient
                model.zero_grad()
                outputs = model(inputs)

                if criterion is not None and targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = outputs.mean()

                # Compute gradient-vector product
                grads = torch.autograd.grad(
                    loss,
                    layer.parameters(),
                    create_graph=True,
                    allow_unused=True
                )

                # Compute v^T H v ≈ trace
                gv = sum(
                    (grad * v[name]).sum()
                    for name, grad in zip([n for n, _ in layer.named_parameters()], grads)
                    if grad is not None
                )

                # Second derivative
                if gv is not None:
                    hvs = torch.autograd.grad(
                        gv,
                        layer.parameters(),
                        retain_graph=False,
                        allow_unused=True
                    )

                    trace_sample = sum(
                        (hv * v[name]).sum().item()
                        for name, hv in zip([n for n, _ in layer.named_parameters()], hvs)
                        if hv is not None
                    )

                    trace_estimate += trace_sample

            trace_estimate /= self.num_samples
            traces.append(trace_estimate)

        if traces:
            return {
                "hessian_trace": float(np.mean(traces)),
                "hessian_trace_std": float(np.std(traces)),
            }
        else:
            return {
                "hessian_trace": 0.0,
                "hessian_trace_std": 0.0,
            }
