"""
Integrated Gradients (IG) sensitivity scoring per layer.

Measures the attribution of each layer's parameters to the model output
using the path integral method. Instead of a single gradient snapshot,
IG integrates gradients along a path from a baseline (zero activations)
to the actual activations, providing a more faithful importance signal.

This is particularly useful for adaptive LoRA rank allocation (IGU-LoRA):
layers with higher IG sensitivity should receive higher adapter ranks.

Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
    Guo et al., "IGU-LoRA: Integrated Gradient-based Uncertainty for LoRA", 2024
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import compute_loss, model_forward


class IGSensitivity(LayerMetric):
    """
    Compute Integrated Gradients sensitivity for a layer.

    For each layer, interpolates between zero-output (baseline) and actual
    output at multiple steps, accumulating gradients to compute the path
    integral attribution.

    Metrics:
    - ig_sensitivity: mean absolute IG attribution (higher = more important)
    - ig_variance: variance of IG across parameters (uniformity of importance)
    - ig_relative: IG normalized by layer parameter count

    Args:
        num_steps: Number of interpolation steps (more = more accurate, slower)
        num_batches: Number of data batches
    """

    def __init__(
        self,
        num_steps: int = 10,
        num_batches: int = 5,
        **kwargs,
    ):
        super().__init__(
            name="ig_sensitivity",
            category="optimization",
            requires_gradient=True,
            requires_data=True,
            **kwargs,
        )
        self.num_steps = num_steps
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
        """
        Compute IG sensitivity for the layer.

        Uses activation-space IG: interpolates the layer's output between
        zero (baseline) and actual output at multiple alpha steps.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Check if layer has parameters
        params = list(layer.parameters())
        if not params:
            return {
                "ig_sensitivity": 0.0,
                "ig_variance": 0.0,
                "ig_relative": 0.0,
            }

        num_params = sum(p.numel() for p in params)
        alphas = torch.linspace(0, 1, self.num_steps + 1, device=device)[1:]  # exclude 0

        # Accumulate gradients across alpha steps and batches
        ig_accum = {
            name: torch.zeros_like(p) for name, p in layer.named_parameters() if p.requires_grad
        }

        for alpha in alphas:
            # Hook: scale layer output by alpha (interpolate from zero baseline)
            def make_scale_hook(a):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        return (output[0] * a,) + output[1:]
                    return output * a

                return hook

            handle = layer.register_forward_hook(make_scale_hook(alpha))

            try:
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

                    # Accumulate gradients
                    for name, p in layer.named_parameters():
                        if p.requires_grad and p.grad is not None and name in ig_accum:
                            ig_accum[name] += p.grad.detach().cpu() / (
                                self.num_steps * self.num_batches
                            )
            finally:
                handle.remove()

        # Compute IG: gradient_integral * (actual_param - baseline_param)
        # Baseline is zero-scaled output, so IG ≈ accumulated_grad * param_value
        ig_scores = []
        for name, p in layer.named_parameters():
            if name in ig_accum:
                ig = ig_accum[name] * p.detach().cpu()
                ig_scores.append(ig.abs())

        if not ig_scores:
            return {
                "ig_sensitivity": 0.0,
                "ig_variance": 0.0,
                "ig_relative": 0.0,
            }

        all_ig = torch.cat([s.flatten() for s in ig_scores])
        mean_ig = all_ig.mean().item()
        var_ig = all_ig.var().item()

        return {
            "ig_sensitivity": float(mean_ig),
            "ig_variance": float(var_ig),
            "ig_relative": float(mean_ig * num_params),
        }
