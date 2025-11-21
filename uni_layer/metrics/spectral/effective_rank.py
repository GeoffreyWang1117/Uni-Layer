"""
Effective Rank metric for measuring representation diversity.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class EffectiveRank(LayerMetric):
    """
    Compute the effective rank of layer activations.

    Effective rank measures the diversity of representations learned by a layer.
    It's based on the singular values of the activation matrix:

    EffectiveRank = exp(Entropy(singular_values))

    Higher effective rank indicates:
    - More diverse representations
    - Better utilization of layer capacity
    - Less redundancy in features

    Args:
        num_batches: Number of batches to use for computation
        epsilon: Small value to avoid log(0)
    """

    def __init__(
        self,
        num_batches: int = 10,
        epsilon: float = 1e-10,
        **kwargs
    ):
        super().__init__(
            name="effective_rank",
            category="spectral",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )
        self.num_batches = num_batches
        self.epsilon = epsilon

    def _compute_effective_rank(self, X: torch.Tensor) -> float:
        """
        Compute effective rank of a matrix.

        Args:
            X: Activation matrix (n_samples x n_features)

        Returns:
            Effective rank value
        """
        # Flatten if needed
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)

        # Compute SVD
        try:
            _, S, _ = torch.svd(X)

            # Normalize singular values
            S = S / (S.sum() + self.epsilon)

            # Compute entropy
            entropy = -(S * torch.log(S + self.epsilon)).sum()

            # Effective rank
            effective_rank = torch.exp(entropy)

            return effective_rank.item()

        except Exception:
            # Fallback: use eigenvalue decomposition of covariance
            X_centered = X - X.mean(dim=0, keepdim=True)
            cov = X_centered.T @ X_centered / X.shape[0]

            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive eigenvalues

            if len(eigenvalues) == 0:
                return 0.0

            # Normalize
            eigenvalues = eigenvalues / (eigenvalues.sum() + self.epsilon)

            # Compute entropy
            entropy = -(eigenvalues * torch.log(eigenvalues + self.epsilon)).sum()

            # Effective rank
            effective_rank = torch.exp(entropy)

            return effective_rank.item()

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
        Compute effective rank for the layer.

        Returns:
            Dictionary with effective rank and related metrics
        """
        activations = self._get_layer_activations(
            model=model,
            layer=layer,
            data_loader=data_loader,
            num_batches=self.num_batches
        )

        if activations:
            # Concatenate activations
            X = torch.cat(activations, dim=0)

            # Compute effective rank
            eff_rank = self._compute_effective_rank(X)

            # Also compute stable rank (||X||_F^2 / ||X||_2^2)
            if X.dim() > 2:
                X_flat = X.reshape(X.size(0), -1)
            else:
                X_flat = X

            frobenius_norm = torch.norm(X_flat, p='fro').item()
            spectral_norm = torch.norm(X_flat, p=2).item()

            if spectral_norm > 0:
                stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2)
            else:
                stable_rank = 0.0

            return {
                "effective_rank": float(eff_rank),
                "stable_rank": float(stable_rank),
                "rank_ratio": float(eff_rank / min(X_flat.shape)) if min(X_flat.shape) > 0 else 0.0,
            }
        else:
            return {
                "effective_rank": 0.0,
                "stable_rank": 0.0,
                "rank_ratio": 0.0,
            }
