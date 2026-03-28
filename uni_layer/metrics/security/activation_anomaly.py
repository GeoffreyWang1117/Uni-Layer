"""
Activation anomaly detection for backdoor/trojan analysis.

Detects layers with unusual activation patterns that may indicate
embedded backdoors. Trojan triggers typically cause a subset of neurons
to fire abnormally — this manifests as skewed activation distributions,
outlier neurons, or abnormal sparsity patterns.

Reference:
    Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks
    by Activation Clustering", SafeAI Workshop, AAAI 2019
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class ActivationAnomalyScore(LayerMetric):
    """
    Detect anomalous activation patterns that may indicate backdoors.

    For each layer, computes:
    - activation_skewness: skewness of activation distribution (trojans cause skew)
    - activation_kurtosis: kurtosis (heavy tails indicate outlier neurons)
    - neuron_outlier_ratio: fraction of neurons with activation > 3 std from mean
    - activation_bimodality: bimodality coefficient (>0.555 suggests bimodal distribution)

    Args:
        num_batches: Number of batches for activation collection
    """

    def __init__(self, num_batches: int = 10, **kwargs):
        super().__init__(
            name="activation_anomaly",
            category="security",
            requires_gradient=False,
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
        _cached_activations: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        # Collect activations
        activations = []
        if _cached_activations:
            activations = _cached_activations
        else:

            def hook_fn(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    activations.append(out.detach().cpu())

            handle = layer.register_forward_hook(hook_fn)
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

        if not activations:
            return {
                "activation_skewness": 0.0,
                "activation_kurtosis": 0.0,
                "neuron_outlier_ratio": 0.0,
                "activation_bimodality": 0.0,
            }

        all_acts = torch.cat(activations, dim=0).float()
        if all_acts.dim() > 2:
            all_acts = all_acts.reshape(all_acts.size(0), -1)

        # Per-neuron statistics (mean across samples)
        neuron_means = all_acts.mean(dim=0)  # (features,)
        neuron_stds = all_acts.std(dim=0) + 1e-10

        # Global activation distribution
        flat = all_acts.flatten()
        n = flat.numel()

        if n < 4:
            return {
                "activation_skewness": 0.0,
                "activation_kurtosis": 0.0,
                "neuron_outlier_ratio": 0.0,
                "activation_bimodality": 0.0,
            }

        mean = flat.mean()
        std = flat.std() + 1e-10
        centered = flat - mean

        # Skewness: E[(x-μ)³] / σ³
        skewness = (centered.pow(3).mean() / std.pow(3)).item()

        # Kurtosis: E[(x-μ)⁴] / σ⁴ - 3 (excess kurtosis)
        kurtosis = (centered.pow(4).mean() / std.pow(4) - 3.0).item()

        # Neuron outlier ratio: neurons with |mean| > 3*global_std
        global_neuron_mean = neuron_means.mean()
        global_neuron_std = neuron_means.std() + 1e-10
        outlier_mask = (neuron_means - global_neuron_mean).abs() > 3 * global_neuron_std
        outlier_ratio = outlier_mask.float().mean().item()

        # Bimodality coefficient: (skew² + 1) / (kurtosis + 3 * (n-1)²/((n-2)(n-3)))
        # Simplified: BC = (skew² + 1) / (kurtosis_excess + 3)
        # BC > 0.555 suggests bimodal
        bc_num = skewness**2 + 1
        bc_den = kurtosis + 3
        bimodality = bc_num / bc_den if abs(bc_den) > 1e-10 else 0.0

        return {
            "activation_skewness": float(skewness),
            "activation_kurtosis": float(kurtosis),
            "neuron_outlier_ratio": float(outlier_ratio),
            "activation_bimodality": float(bimodality),
        }
