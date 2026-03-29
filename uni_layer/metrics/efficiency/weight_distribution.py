"""
Weight distribution analysis per layer.

Characterizes the statistical properties of a layer's weight matrix:
sparsity, rank deficiency, magnitude distribution, and outliers. These
properties are prerequisites for informed pruning and quantization decisions.

Reference:
    Han et al., "Learning both Weights and Connections for Efficient
    Neural Networks", NeurIPS 2015
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.fast_math import randomized_svd


class WeightDistribution(LayerMetric):
    """
    Analyze weight matrix distribution properties.

    For each parameterized layer computes:
    - weight_sparsity: fraction of near-zero weights (|w| < threshold)
    - weight_l1_norm: L1 norm (sum of absolute values)
    - weight_l2_norm: L2 norm (Frobenius)
    - weight_rank_ratio: effective rank / full rank (rank deficiency)
    - weight_outlier_ratio: fraction of weights > 3 std from mean
    - weight_kurtosis: excess kurtosis (tail heaviness)

    Layers without parameters return zero values.

    Args:
        sparsity_threshold: |w| below this is considered near-zero
        svd_components: number of singular values for rank estimation
    """

    def __init__(
        self,
        sparsity_threshold: float = 1e-3,
        svd_components: int = 50,
        **kwargs,
    ):
        super().__init__(
            name="weight_distribution",
            category="efficiency",
            requires_gradient=False,
            requires_data=False,
            **kwargs,
        )
        self.sparsity_threshold = sparsity_threshold
        self.svd_components = svd_components

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
        # Find the main weight tensor
        weight = None
        for name, param in layer.named_parameters():
            if "weight" in name and param.dim() >= 2:
                weight = param.detach().cpu().float()
                break

        if weight is None:
            return {
                "weight_sparsity": 0.0,
                "weight_l1_norm": 0.0,
                "weight_l2_norm": 0.0,
                "weight_rank_ratio": 0.0,
                "weight_outlier_ratio": 0.0,
                "weight_kurtosis": 0.0,
            }

        flat = weight.flatten()

        # Sparsity
        sparsity = (flat.abs() < self.sparsity_threshold).float().mean().item()

        # Norms
        l1_norm = flat.abs().sum().item()
        l2_norm = flat.norm().item()

        # Rank ratio via randomized SVD
        W = weight.reshape(weight.size(0), -1)
        k = min(self.svd_components, min(W.shape) - 1)
        if k >= 1:
            _, S, _ = randomized_svd(W, n_components=k)
            S = S[S > 1e-10]
            if len(S) > 0:
                p = S / (S.sum() + 1e-10)
                entropy = -(p * torch.log(p + 1e-10)).sum()
                effective_rank = torch.exp(entropy).item()
                full_rank = min(W.shape)
                rank_ratio = effective_rank / full_rank
            else:
                rank_ratio = 0.0
        else:
            rank_ratio = 0.0

        # Outlier ratio
        mean = flat.mean()
        std = flat.std() + 1e-10
        outlier_ratio = ((flat - mean).abs() > 3 * std).float().mean().item()

        # Excess kurtosis
        centered = flat - mean
        kurtosis = (centered.pow(4).mean() / std.pow(4) - 3.0).item()

        return {
            "weight_sparsity": float(sparsity),
            "weight_l1_norm": float(l1_norm),
            "weight_l2_norm": float(l2_norm),
            "weight_rank_ratio": float(rank_ratio),
            "weight_outlier_ratio": float(outlier_ratio),
            "weight_kurtosis": float(kurtosis),
        }
