"""
Intrinsic dimensionality estimation for layer representations.

Estimates the local intrinsic dimensionality (LID) of a layer's activation
manifold using the Maximum Likelihood Estimator (MLE) of Levina & Bickel.
This is the true dimensionality of the data manifold, regardless of the
ambient dimension (hidden size).

Critical for:
- LoRA rank selection: optimal rank ≈ intrinsic dimensionality
- Compression budget: layers on low-dimensional manifolds compress more
- Bottleneck detection: sharp dimension drops indicate information loss

Reference:
    Levina & Bickel, "Maximum Likelihood Estimation of Intrinsic Dimension",
    NeurIPS 2004
    Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness
    of Language Model Fine-Tuning", ACL 2021
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class IntrinsicDimensionality(LayerMetric):
    """
    Estimate intrinsic dimensionality of layer activations.

    For each layer computes:
    - intrinsic_dim: MLE estimate of local intrinsic dimensionality
    - intrinsic_dim_ratio: intrinsic_dim / ambient_dim (compression potential)
    - ambient_dim: the actual feature dimension of the layer

    Uses k-nearest-neighbor distances in activation space.

    Args:
        num_batches: Number of batches for activation collection
        k: Number of nearest neighbors for MLE estimator
    """

    def __init__(self, num_batches: int = 10, k: int = 20, **kwargs):
        super().__init__(
            name="intrinsic_dim",
            category="efficiency",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_batches = num_batches
        self.k = k

    @staticmethod
    def _mle_id(X: torch.Tensor, k: int) -> float:
        """
        MLE intrinsic dimensionality estimator (Levina-Bickel 2004).

        For each point, compute distances to k nearest neighbors,
        then estimate dimension from the log-ratio of distances.
        """
        N = X.size(0)
        if N <= k + 1:
            return 0.0

        # Pairwise distances
        dists = torch.cdist(X, X)  # (N, N)
        # Set self-distance to inf
        dists.fill_diagonal_(float("inf"))

        # k nearest neighbor distances
        knn_dists, _ = dists.topk(k, largest=False, dim=1)  # (N, k)
        knn_dists = knn_dists.clamp(min=1e-10)  # avoid log(0)

        # MLE estimator: ID = 1 / mean(log(T_k / T_j)) for j=1..k-1
        T_k = knn_dists[:, -1:]  # (N, 1) - k-th neighbor distance
        T_j = knn_dists[:, :-1]  # (N, k-1) - 1st to (k-1)-th neighbor

        log_ratios = torch.log(T_k / T_j)  # (N, k-1)
        per_point_id = (k - 1) / log_ratios.sum(dim=1)  # (N,)

        # Average across points, filtering outliers
        valid = per_point_id.isfinite() & (per_point_id > 0) & (per_point_id < X.size(1) * 2)
        if valid.sum() < 2:
            return 0.0

        return per_point_id[valid].mean().item()

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
            return {"intrinsic_dim": 0.0, "intrinsic_dim_ratio": 0.0, "ambient_dim": 0.0}

        all_acts = torch.cat(activations, dim=0).float()
        if all_acts.dim() > 2:
            all_acts = all_acts.reshape(all_acts.size(0), -1)

        N, D = all_acts.shape
        ambient_dim = D

        # Subsample if too many points (k-NN is O(N^2))
        max_samples = 500
        if N > max_samples:
            idx = torch.randperm(N)[:max_samples]
            all_acts = all_acts[idx]

        k = min(self.k, all_acts.size(0) - 2)
        if k < 2:
            return {
                "intrinsic_dim": 0.0,
                "intrinsic_dim_ratio": 0.0,
                "ambient_dim": float(ambient_dim),
            }

        intrinsic_dim = self._mle_id(all_acts, k)

        return {
            "intrinsic_dim": float(intrinsic_dim),
            "intrinsic_dim_ratio": float(intrinsic_dim / ambient_dim) if ambient_dim > 0 else 0.0,
            "ambient_dim": float(ambient_dim),
        }
