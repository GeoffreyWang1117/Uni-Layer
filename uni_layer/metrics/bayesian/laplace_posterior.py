"""
Laplace Posterior Variance metric for layer contribution.

Optimized version:
- Uses Rademacher random vectors (lower variance than Gaussian)
- Uses empirical Fisher diagonal as a fast alternative to full Hessian
- Properly releases computation graphs to minimize GPU memory

Mathematical basis:
    p(theta|D) ~ N(theta_MAP, H^{-1})
    Var(theta) = Tr(H^{-1}) ~ sum(1/lambda_i)

    We approximate via Hutchinson: Tr(H^{-1}) ~ E[v^T H^{-1} v]
    Using the identity: v^T H v gives curvature, and 1/(v^T H v + lambda)
    gives an estimate of the inverse trace.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import compute_loss, extract_logits, model_forward


class LaplacePosterior(LayerMetric):
    """
    Compute Laplace posterior variance for a layer.

    The Laplace approximation approximates the posterior distribution
    over parameters as a Gaussian centered at the MAP estimate:
        p(theta|D) ~ N(theta_MAP, H^{-1})

    Higher posterior variance indicates:
    - Greater uncertainty about parameters
    - Sensitivity to perturbations
    - Potential for further learning

    Lower posterior variance indicates:
    - More confident parameters
    - Stable learned features

    Two modes:
    - mode='hutchinson': Full Hessian trace via Hutchinson estimator (accurate, slow)
    - mode='fisher': Diagonal Fisher approximation (fast, approximate)

    Args:
        num_samples: Number of random vectors for Hutchinson estimation
        num_batches: Number of batches to average over
        damping: Damping factor for numerical stability
        mode: 'hutchinson' for Hessian-based, 'fisher' for diagonal Fisher (faster)
    """

    def __init__(
        self,
        num_samples: int = 5,
        num_batches: int = 5,
        damping: float = 1e-3,
        mode: str = "fisher",
        **kwargs,
    ):
        super().__init__(
            name="laplace_posterior",
            category="bayesian",
            requires_gradient=True,
            requires_data=True,
            **kwargs,
        )
        self.num_samples = num_samples
        self.num_batches = num_batches
        self.damping = damping
        self.mode = mode

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        params = [p for p in layer.parameters() if p.requires_grad]
        if not params:
            return {"laplace_posterior": 0.0, "laplace_posterior_std": 0.0}

        if self.mode == "fisher":
            return self._compute_fisher_mode(model, layer, params, data_loader, device, criterion)
        else:
            return self._compute_hutchinson_mode(
                model, layer, params, data_loader, device, criterion
            )

    def _compute_fisher_mode(
        self, model, layer, params, data_loader, device, criterion
    ) -> Dict[str, float]:
        """
        Fast mode: Use diagonal Fisher information as Hessian approximation.

        F_diag = E[(d log p / d theta)^2]
        Var(theta) ~ Tr(F^{-1}) ~ sum(1 / (F_ii + damping))

        This avoids the expensive create_graph=True double backward.
        Cost: num_batches backward passes (no second-order gradients).
        """
        # Accumulate diagonal Fisher: E[grad^2]
        fisher_diag = {i: torch.zeros_like(p) for i, p in enumerate(params)}
        num_samples = 0

        model.train()
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= self.num_batches:
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

            for i, p in enumerate(params):
                if p.grad is not None:
                    fisher_diag[i] += p.grad.data**2

            num_samples += 1

        if num_samples == 0:
            return {"laplace_posterior": 0.0, "laplace_posterior_std": 0.0}

        # Average and compute Tr(F^{-1})
        posterior_var = 0.0
        for i in fisher_diag:
            fisher_diag[i] /= num_samples
            # Tr(F^{-1}) = sum(1 / (F_ii + damping))
            posterior_var += (1.0 / (fisher_diag[i] + self.damping)).sum().item()

        return {
            "laplace_posterior": float(posterior_var),
            "laplace_posterior_std": 0.0,  # single estimate in fisher mode
        }

    def _compute_hutchinson_mode(
        self, model, layer, params, data_loader, device, criterion
    ) -> Dict[str, float]:
        """
        Accurate mode: Full Hutchinson estimator with Hessian-vector products.

        Uses Rademacher random vectors and releases computation graphs promptly.
        Cost: num_batches * num_samples * (1 forward + 2 backward) passes.
        """
        variances = []

        model.train()
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= self.num_batches:
                break

            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)

            trace_estimate = 0.0

            for _ in range(self.num_samples):
                # Rademacher random vectors
                vs = [
                    torch.randint(0, 2, p.shape, device=device, dtype=p.dtype) * 2 - 1
                    for p in params
                ]

                model.zero_grad()
                outputs = model_forward(model, inputs, targets)
                logits = extract_logits(outputs)

                if targets is not None:
                    loss = criterion(logits, targets)
                else:
                    loss = logits.mean()

                # First derivative
                grads = torch.autograd.grad(
                    loss,
                    params,
                    create_graph=True,
                    allow_unused=True,
                )

                # g^T v
                gv = sum((g * v).sum() for g, v in zip(grads, vs) if g is not None)

                if gv is not None and gv.requires_grad:
                    # Hv
                    hvs = torch.autograd.grad(
                        gv,
                        params,
                        retain_graph=False,
                        allow_unused=True,
                    )

                    # v^T H v (curvature in random direction)
                    hv_dot_v = sum(
                        (hv * v).sum().item() for hv, v in zip(hvs, vs) if hv is not None
                    )

                    # Inverse curvature as variance proxy
                    if abs(hv_dot_v) > 0:
                        var_sample = 1.0 / (abs(hv_dot_v) + self.damping)
                        trace_estimate += var_sample

            trace_estimate /= self.num_samples
            variances.append(trace_estimate)

        if variances:
            return {
                "laplace_posterior": float(np.mean(variances)),
                "laplace_posterior_std": float(np.std(variances)),
            }
        else:
            return {
                "laplace_posterior": 0.0,
                "laplace_posterior_std": 0.0,
            }
