"""
Block Influence (BI) metric for measuring layer redundancy.

Based on ShortGPT (ACL 2025): "Layers in Large Language Models are More
Redundant Than You Expect."

BI measures how much a layer transforms its input by computing the cosine
similarity between the layer's input and output hidden states. Low BI
(high similarity) indicates the layer is redundant and can be pruned.

    BI(layer) = 1 - cos_sim(input, output)

This metric is widely used for LLM layer pruning and works well for
any model with residual connections (Transformers, ResNets).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class BlockInfluence(LayerMetric):
    """
    Compute Block Influence (BI) score for a layer.

    BI = 1 - cosine_similarity(layer_input, layer_output)

    Higher BI means the layer transforms its input more (more important).
    Lower BI means the layer is near-identity (redundant, safe to prune).

    Args:
        num_batches: Number of batches to average over
    """

    def __init__(self, num_batches: int = 10, **kwargs):
        super().__init__(
            name="block_influence",
            category="representation",
            requires_gradient=False,
            requires_data=True,
            **kwargs
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
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute Block Influence for the layer.

        Returns:
            Dictionary with:
            - block_influence: 1 - avg cosine similarity (higher = more important)
            - block_similarity: avg cosine similarity (higher = more redundant)
        """
        inputs_list: List[torch.Tensor] = []
        outputs_list: List[torch.Tensor] = []

        def input_hook(module, inp, out):
            # Capture input
            x = inp[0] if isinstance(inp, tuple) else inp
            inputs_list.append(x.detach().cpu())
            # Capture output
            o = out[0] if isinstance(out, tuple) else out
            outputs_list.append(o.detach().cpu())

        handle = layer.register_forward_hook(input_hook)

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= self.num_batches:
                        break
                    if isinstance(batch, (tuple, list)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device)
                    model_forward(model, x)
        finally:
            handle.remove()

        if not inputs_list or not outputs_list:
            return {"block_influence": 0.0, "block_similarity": 1.0}

        # Compute cosine similarity between input and output for each batch
        similarities = []
        for inp, out in zip(inputs_list, outputs_list):
            # Handle shape mismatches (e.g., conv layers change spatial dims)
            if inp.shape != out.shape:
                return {"block_influence": 1.0, "block_similarity": 0.0}

            # Flatten to (batch, -1)
            inp_flat = inp.reshape(inp.size(0), -1).float()
            out_flat = out.reshape(out.size(0), -1).float()

            # Per-sample cosine similarity
            cos_sim = F.cosine_similarity(inp_flat, out_flat, dim=1)
            similarities.append(cos_sim.mean().item())

        avg_sim = sum(similarities) / len(similarities)
        bi = 1.0 - avg_sim

        return {
            "block_influence": float(bi),
            "block_similarity": float(avg_sim),
        }
