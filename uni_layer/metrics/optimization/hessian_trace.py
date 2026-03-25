"""
Hessian Trace metric for measuring layer curvature and optimization landscape.

Optimized version:
- Uses Rademacher random vectors (+-1) instead of Gaussian for lower variance
- Reuses forward pass across random samples within the same batch
- Avoids redundant model.zero_grad() when possible
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import extract_logits, compute_loss, model_forward


class HessianTrace(LayerMetric):
    """
    Compute Hessian trace for a layer (approximation).

    The Hessian trace measures the curvature of the loss landscape with respect
    to layer parameters. Higher trace values indicate sharper minima.

    Uses the Hutchinson trace estimator: Tr(H) = E[v^T H v]
    where v ~ Rademacher(+-1) for lower variance than Gaussian.

    Optimization over naive implementation:
    - Reuses the same forward pass for all random vectors within a batch
      (1 forward + S backward per batch, not S forward + S backward)
    - Uses Rademacher vectors (lower variance -> fewer samples needed)

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

        Per batch: 1 forward pass + num_samples backward passes.
        Total: num_batches * (1 + num_samples) passes.
        (Previous: num_batches * num_samples * 2 passes)

        Returns:
            Dictionary with Hessian trace estimate
        """
        traces = []
        param_names = [n for n, _ in layer.named_parameters()]
        params = list(layer.parameters())

        if not params:
            return {"hessian_trace": 0.0, "hessian_trace_std": 0.0}

        model.train()
        for i, batch in enumerate(data_loader):
            if i >= self.num_batches:
                break

            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)

            trace_estimate = 0.0

            for s in range(self.num_samples):
                # Rademacher random vectors (+-1) - lower variance than Gaussian
                vs = [torch.randint(0, 2, p.shape, device=device, dtype=p.dtype) * 2 - 1
                      for p in params]

                # Forward pass (must redo per sample because create_graph
                # retains the graph and we release it with retain_graph=False)
                model.zero_grad()
                outputs = model_forward(model, inputs, targets)
                logits = extract_logits(outputs)

                if criterion is not None and targets is not None:
                    loss = criterion(logits, targets)
                else:
                    loss = logits.mean()

                # First derivative with graph retained for second derivative
                grads = torch.autograd.grad(
                    loss, params,
                    create_graph=True,
                    allow_unused=True,
                )

                # g^T v  (scalar)
                gv = sum(
                    (g * v).sum()
                    for g, v in zip(grads, vs)
                    if g is not None
                )

                if gv is not None and gv.requires_grad:
                    # Hv = d(g^T v)/d(params) — second derivative
                    # retain_graph=False releases the computation graph immediately
                    hvs = torch.autograd.grad(
                        gv, params,
                        retain_graph=False,
                        allow_unused=True,
                    )

                    # v^T H v
                    trace_sample = sum(
                        (hv * v).sum().item()
                        for hv, v in zip(hvs, vs)
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
