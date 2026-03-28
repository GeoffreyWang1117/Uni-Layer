"""
Per-layer membership inference risk scoring.

Estimates how much private training data information leaks through each
layer's gradients. Layers with higher gradient information content pose
greater risk for gradient-based membership inference and model inversion
attacks.

Reference:
    Shokri et al., "Membership Inference Attacks Against Machine Learning
    Models", IEEE S&P 2017
    Nasr et al., "Comprehensive Privacy Analysis of Deep Learning", IEEE S&P 2019
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import compute_loss, model_forward


class MembershipInferenceRisk(LayerMetric):
    """
    Estimate per-layer membership inference risk from gradient leakage.

    For each layer, computes:
    - gradient_entropy: entropy of gradient distribution (higher = more info leakage)
    - gradient_snr: signal-to-noise ratio of gradients across batches
    - gradient_memorization: gradient norm variance (high = memorization indicator)
    - mi_risk_score: composite risk score (0-1, higher = riskier)

    Args:
        num_batches: Number of batches for gradient collection
    """

    def __init__(self, num_batches: int = 5, **kwargs):
        super().__init__(
            name="membership_inference",
            category="security",
            requires_gradient=True,
            requires_data=True,
            **kwargs,
        )
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
        **kwargs,
    ) -> Dict[str, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        params = [p for p in layer.parameters() if p.requires_grad]
        if not params:
            return {
                "gradient_entropy": 0.0,
                "gradient_snr": 0.0,
                "gradient_memorization": 0.0,
                "mi_risk_score": 0.0,
            }

        # Collect per-batch gradient statistics
        grad_norms = []
        all_grad_means = []
        all_grad_stds = []

        model.train()
        for i, batch in enumerate(data_loader):
            if i >= self.num_batches:
                break
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            outputs = model_forward(model, inputs, targets)
            loss = compute_loss(outputs, targets, criterion)
            loss.backward()

            # Collect gradient stats for this batch
            batch_grads = []
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach().cpu().float().flatten()
                    batch_grads.append(g)

            if batch_grads:
                cat_grads = torch.cat(batch_grads)
                grad_norms.append(cat_grads.norm().item())
                all_grad_means.append(cat_grads.mean().item())
                all_grad_stds.append(cat_grads.std().item())

        if not grad_norms:
            return {
                "gradient_entropy": 0.0,
                "gradient_snr": 0.0,
                "gradient_memorization": 0.0,
                "mi_risk_score": 0.0,
            }

        grad_norms = np.array(grad_norms)
        grad_means = np.array(all_grad_means)
        grad_stds = np.array(all_grad_stds)

        # 1. Gradient entropy: approximate via histogram of gradient norms
        #    Higher entropy = more diverse gradient info = higher leakage risk
        if len(grad_norms) >= 2:
            hist, _ = np.histogram(grad_norms, bins=min(10, len(grad_norms)))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_entropy = np.log(len(hist))
            gradient_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            gradient_entropy = 0.0

        # 2. Gradient SNR: mean / std of gradient norms across batches
        #    Higher SNR = gradients carry more signal = easier to infer membership
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms) + 1e-10
        gradient_snr = mean_norm / std_norm

        # 3. Gradient memorization: variance of per-batch gradient norms
        #    Higher variance = model responds differently to different samples
        gradient_memorization = np.var(grad_norms)

        # 4. Composite risk score (normalized to [0, 1])
        #    Combines entropy, SNR, and memorization signals
        risk_components = [
            min(1.0, gradient_entropy),
            min(1.0, gradient_snr / 10.0),  # normalize SNR
            min(1.0, gradient_memorization / (mean_norm**2 + 1e-10)),
        ]
        mi_risk_score = np.mean(risk_components)

        return {
            "gradient_entropy": float(gradient_entropy),
            "gradient_snr": float(gradient_snr),
            "gradient_memorization": float(gradient_memorization),
            "mi_risk_score": float(mi_risk_score),
        }
