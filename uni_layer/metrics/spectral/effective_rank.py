"""
Effective Rank metric for measuring representation diversity.

Uses randomized SVD for large activation matrices, providing
O(N*D*k) compute instead of O(N*D*min(N,D)) for full SVD.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.fast_math import randomized_svd, effective_rank_from_svd


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

    For large models, uses randomized SVD (Halko et al., 2011) which is
    O(N*D*k) instead of O(N*D*min(N,D)) for full SVD.

    Args:
        num_batches: Number of batches to use for computation
        epsilon: Small value to avoid log(0)
        n_components: Number of singular values for randomized SVD
        use_randomized_svd: Whether to use randomized SVD for large matrices
    """

    def __init__(
        self,
        num_batches: int = 10,
        epsilon: float = 1e-10,
        n_components: int = 50,
        use_randomized_svd: bool = True,
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
        self.n_components = n_components
        self.use_randomized_svd = use_randomized_svd

    def _compute_effective_rank(self, X: torch.Tensor) -> float:
        """
        Compute effective rank of a matrix.

        Uses randomized SVD when the matrix is large enough to benefit.

        Args:
            X: Activation matrix (n_samples x n_features)

        Returns:
            Effective rank value
        """
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)

        n, d = X.shape
        use_rsvd = (
            self.use_randomized_svd
            and min(n, d) > 2 * self.n_components
        )

        try:
            if use_rsvd:
                _, S, _ = randomized_svd(X, n_components=self.n_components)
            else:
                _, S, _ = torch.linalg.svd(X, full_matrices=False)

            return effective_rank_from_svd(S, self.epsilon)

        except Exception:
            # Fallback: eigenvalue decomposition of covariance
            X_centered = X - X.mean(dim=0, keepdim=True)
            cov = X_centered.T @ X_centered / n

            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 0]

            if len(eigenvalues) == 0:
                return 0.0

            return effective_rank_from_svd(eigenvalues.sqrt(), self.epsilon)

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        _cached_activations: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute effective rank for the layer.

        Can use cached activations from LayerAnalyzer to avoid redundant
        forward passes.

        Returns:
            Dictionary with effective rank and related metrics
        """
        # Use cached activations if available
        if _cached_activations:
            activations = _cached_activations
        else:
            activations = self._get_layer_activations(
                model=model,
                layer=layer,
                data_loader=data_loader,
                num_batches=self.num_batches
            )

        if activations:
            X = torch.cat(activations, dim=0)

            eff_rank = self._compute_effective_rank(X)

            # Stable rank: ||X||_F^2 / ||X||_2^2
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
