"""
Attention Flow metric for Transformer layers.

Attention Flow分析（中文详解）

数学原理：
在Transformer中，注意力机制定义为：
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Attention Flow测量信息如何通过注意力层流动：

1. 注意力权重分布：
    A = softmax(QK^T / √d_k) ∈ R^(n×n)

2. 信息流量度量：
   - 平均注意力熵：H(A) = -Σ A_ij log A_ij
   - 注意力集中度：max_j Σ_i A_ij
   - 头间多样性：不同注意力头的相似度

3. 层的重要性：
   - 高熵：注意力分散，可能学习全局模式
   - 低熵：注意力集中，可能学习局部模式
   - 头多样性高：不同头学到不同模式
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np

from uni_layer.core.base_metric import LayerMetric


class AttentionFlow(LayerMetric):
    """
    Compute attention flow metrics for Transformer attention layers.

    This metric analyzes how information flows through attention mechanisms
    by examining attention weight distributions.

    Metrics computed:
    - attention_entropy: Average entropy of attention distributions
    - attention_max_weight: Maximum attention weight (concentration)
    - head_diversity: Diversity across attention heads
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

        Args:
            attn_weights: Attention weights (batch, heads, seq_len, seq_len)
                         or (batch, seq_len, seq_len)

        Returns:
            Average entropy across all attention distributions
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        attn_weights = attn_weights + eps

        # Compute entropy: H = -Σ p log p
        entropy = -(attn_weights * torch.log(attn_weights)).sum(dim=-1)

        return entropy.mean().item()

    def _compute_attention_distance(self, attn_weights: torch.Tensor) -> float:
        """
        Compute average attention distance (for sequential data).

        This measures how far tokens attend to other tokens on average.

        Args:
            attn_weights: Attention weights

        Returns:
            Average attention distance
        """
        seq_len = attn_weights.shape[-1]

        # Create position matrix
        positions = torch.arange(seq_len, dtype=torch.float32, device=attn_weights.device)
        position_diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        # Weight positions by attention
        weighted_distance = (attn_weights * position_diff).sum(dim=-1)

        return weighted_distance.mean().item()

    def _compute_head_diversity(self, attn_weights: torch.Tensor) -> float:
        """
        Compute diversity across attention heads.

        Diversity measured as average pairwise distance (1 - similarity)
        between attention patterns of different heads.

        Args:
            attn_weights: Attention weights (batch, heads, seq_len, seq_len)

        Returns:
            Average head diversity score
        """
        if attn_weights.dim() < 4:
            return 0.0  # Not multi-head

        num_heads = attn_weights.shape[1]

        if num_heads < 2:
            return 0.0

        # Flatten each head's attention pattern
        # (batch, heads, seq_len, seq_len) -> (batch, heads, seq_len^2)
        attn_flat = attn_weights.flatten(start_dim=2)

        # Compute pairwise cosine similarity between heads
        diversities = []

        for b in range(attn_flat.shape[0]):
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    head_i = attn_flat[b, i]
                    head_j = attn_flat[b, j]

                    # Cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        head_i.unsqueeze(0),
                        head_j.unsqueeze(0)
                    ).item()

                    # Diversity = 1 - similarity
                    diversity = 1.0 - similarity
                    diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.0

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

        Returns:
            Dictionary with attention flow metrics
        """
        # Check if this is an attention layer
        if not self._is_attention_layer(layer):
            return {
                "attention_entropy": None,
                "attention_max_weight": None,
                "head_diversity": None,
            }

        attention_weights_list = []

        # Hook to capture attention weights
        def attention_hook(module, input, output):
            # Different attention implementations return weights differently
            if isinstance(output, tuple) and len(output) > 1:
                # (output, attn_weights)
                attn = output[1]
                if attn is not None:
                    attention_weights_list.append(attn.detach().cpu())

        # Register hook
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

                    # For transformer models, might need attention mask
                    try:
                        if hasattr(model, 'forward') and 'attention_mask' in model.forward.__code__.co_varnames:
                            # Create dummy attention mask
                            attention_mask = torch.ones(
                                inputs.shape[0],
                                inputs.shape[1] if inputs.dim() > 1 else 1,
                                device=device
                            )
                            model(inputs, attention_mask=attention_mask)
                        else:
                            model(inputs)
                    except:
                        # Fallback
                        try:
                            model(inputs)
                        except:
                            pass

        finally:
            handle.remove()

        # Analyze collected attention weights
        if not attention_weights_list:
            return {
                "attention_entropy": 0.0,
                "attention_max_weight": 0.0,
                "head_diversity": 0.0,
                "attention_distance": 0.0,
            }

        # Concatenate all attention weights
        all_attn = torch.cat(attention_weights_list, dim=0)

        # Compute metrics
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
