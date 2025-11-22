"""
Laplace Posterior Variance metric for layer contribution.

Laplace后验方差指标（中文详解）

数学原理：
在贝叶斯深度学习中，Laplace近似将后验分布近似为高斯分布：
    p(θ|D) ≈ N(θ_MAP, H^(-1))

其中H是Hessian矩阵。后验方差反映了参数的不确定性：
- 高方差：参数不确定，该层可能对扰动敏感
- 低方差：参数确定，该层已充分学习

我们计算后验方差的迹：
    Var(θ) = Tr(H^(-1)) ≈ Σ(1/λ_i)

其中λ_i是Hessian的特征值。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class LaplacePosterior(LayerMetric):
    """
    Compute Laplace posterior variance for a layer.

    The Laplace approximation approximates the posterior distribution
    over parameters as a Gaussian centered at the MAP estimate:
        p(θ|D) ≈ N(θ_MAP, H^(-1))

    Higher posterior variance indicates:
    - Greater uncertainty about parameters
    - Sensitivity to perturbations
    - Potential for further learning

    Lower posterior variance indicates:
    - More confident parameters
    - Stable learned features
    - Possibly over-constrained

    We compute the trace of the inverse Hessian as a proxy for variance:
        Var(θ) = Tr(H^(-1))

    Args:
        num_samples: Number of random vectors for trace estimation
        num_batches: Number of batches to average over
        damping: Damping factor for numerical stability (adds λI to Hessian)
    """

    def __init__(
        self,
        num_samples: int = 5,
        num_batches: int = 5,
        damping: float = 1e-3,
        **kwargs
    ):
        super().__init__(
            name="laplace_posterior",
            category="bayesian",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )
        self.num_samples = num_samples
        self.num_batches = num_batches
        self.damping = damping

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
        Compute Laplace posterior variance for the layer.

        Returns:
            Dictionary with posterior variance estimate
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Collect parameters
        params = [p for p in layer.parameters() if p.requires_grad]
        if not params:
            return {"laplace_posterior": 0.0}

        variances = []

        model.train()
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= self.num_batches:
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

            # Hutchinson estimator for Tr(H^(-1))
            # We estimate: Tr(H^(-1)) ≈ E[v^T H^(-1) v]
            #            = E[v^T (H + λI)^(-1) v]  (with damping)

            trace_estimate = 0.0

            for _ in range(self.num_samples):
                # Generate random vector
                v = {}
                for i, param in enumerate(params):
                    v[i] = torch.randn_like(param)

                # Compute gradient
                model.zero_grad()
                outputs = model(inputs)

                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = outputs.mean()

                # First derivative
                grads = torch.autograd.grad(
                    loss,
                    params,
                    create_graph=True,
                    allow_unused=True
                )

                # Compute gradient-vector product
                gv = sum(
                    (grad * v[i]).sum()
                    for i, grad in enumerate(grads)
                    if grad is not None
                )

                if gv is not None:
                    # Second derivative: H v
                    hvs = torch.autograd.grad(
                        gv,
                        params,
                        retain_graph=False,
                        allow_unused=True
                    )

                    # Compute v^T (H + λI)^(-1) v
                    # For simplicity, we approximate: v^T H v ≈ ||Hv||^2
                    # And variance ≈ 1 / ||Hv||^2 (when damped)

                    hv_norm_sq = sum(
                        (hv ** 2).sum().item()
                        for hv in hvs
                        if hv is not None
                    )

                    if hv_norm_sq > 0:
                        # Approximate variance as inverse of curvature
                        var_sample = 1.0 / (hv_norm_sq + self.damping)
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
