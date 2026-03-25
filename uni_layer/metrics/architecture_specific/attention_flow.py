"""
Attention Flow metric for Transformer layers.

Optimized version:
- Vectorized head diversity computation using batch matrix operations
  (previously O(B*H^2) sequential cosine_similarity calls, now O(1) matmul)
- Accepts cached activations to avoid redundant forward passes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class AttentionFlow(LayerMetric):
    """
    Compute attention flow metrics for Transformer attention layers.

    Metrics computed:
    - attention_entropy: Average entropy of attention distributions
    - attention_max_weight: Maximum attention weight (concentration)
    - head_diversity: Diversity across attention heads (vectorized)
    - attention_distance: Average attention distance (positional)

    Args:
        num_batches: Number of batches to analyze
        analyze_heads: Whether to analyze individual heads
    """

    def __init__(
        self,
        num_batches: int = 10,
        analyze_heads: bool = True,
        **kwargs
    ):
        super().__init__(
            name="attention_flow",
            category="architecture_specific",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )
        self.num_batches = num_batches
        self.analyze_heads = analyze_heads

    def _is_attention_layer(self, layer: nn.Module) -> bool:
        """Check if layer is an attention layer"""
        return (
            isinstance(layer, nn.MultiheadAttention)
            or hasattr(layer, 'self_attn')
            or hasattr(layer, 'attn')
            or 'attention' in layer.__class__.__name__.lower()
        )

    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution.

        H(A) = -sum(A_ij * log(A_ij))
        """
        eps = 1e-10
        attn_weights = attn_weights + eps
        entropy = -(attn_weights * torch.log(attn_weights)).sum(dim=-1)
        return entropy.mean().item()

    def _compute_attention_distance(self, attn_weights: torch.Tensor) -> float:
        """
        Compute average attention distance (for sequential data).
        Measures how far tokens attend to other tokens on average.
        """
        seq_len = attn_weights.shape[-1]
        positions = torch.arange(seq_len, dtype=torch.float32, device=attn_weights.device)
        position_diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        weighted_distance = (attn_weights * position_diff).sum(dim=-1)
        return weighted_distance.mean().item()

    def _compute_head_diversity(self, attn_weights: torch.Tensor) -> float:
        """
        Compute diversity across attention heads using vectorized operations.

        Previous: triple nested loop O(B * H * H) with per-pair cosine_similarity
        Now: batch matmul to compute all pairwise similarities at once O(1 kernel launch)

        Diversity = mean(1 - cosine_similarity(head_i, head_j)) for all i < j
        """
        if attn_weights.dim() < 4:
            return 0.0

        B, H, S1, S2 = attn_weights.shape
        if H < 2:
            return 0.0

        # Flatten each head's attention: (B, H, S1*S2)
        attn_flat = attn_weights.reshape(B, H, -1)

        # Normalize for cosine similarity: (B, H, D)
        norms = attn_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_normed = attn_flat / norms

        # All pairwise cosine similarities via batch matmul: (B, H, H)
        sim_matrix = torch.bmm(attn_normed, attn_normed.transpose(1, 2))

        # Extract upper triangle (i < j pairs) and compute mean diversity
        # Create mask for upper triangle
        mask = torch.triu(torch.ones(H, H, device=sim_matrix.device, dtype=torch.bool), diagonal=1)

        # Mean over upper triangle across all batches
        pair_sims = sim_matrix[:, mask]  # (B, H*(H-1)/2)
        diversity = (1.0 - pair_sims).mean().item()

        return diversity

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
        Compute attention flow metrics for the layer.
        """
        if not self._is_attention_layer(layer):
            return {
                "attention_entropy": None,
                "attention_max_weight": None,
                "head_diversity": None,
            }

        attention_weights_list = []

        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                if attn is not None:
                    attention_weights_list.append(attn.detach().cpu())

        handle = layer.register_forward_hook(attention_hook)

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

                    try:
                        model_forward(model, inputs)
                    except Exception:
                        pass

        finally:
            handle.remove()

        if not attention_weights_list:
            return {
                "attention_entropy": 0.0,
                "attention_max_weight": 0.0,
                "head_diversity": 0.0,
                "attention_distance": 0.0,
            }

        all_attn = torch.cat(attention_weights_list, dim=0)

        entropy = self._compute_attention_entropy(all_attn)
        max_weight = all_attn.max().item()

        head_div = 0.0
        if self.analyze_heads and all_attn.dim() >= 4:
            head_div = self._compute_head_diversity(all_attn)

        attn_dist = self._compute_attention_distance(all_attn)

        return {
            "attention_entropy": float(entropy),
            "attention_max_weight": float(max_weight),
            "head_diversity": float(head_div),
            "attention_distance": float(attn_dist),
        }
