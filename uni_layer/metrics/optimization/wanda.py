"""
Wanda (Pruning by Weights and Activations) importance metric.

Computes per-layer importance as the product of weight magnitude and
activation norm. This simple metric is surprisingly effective for
unstructured and semi-structured pruning without any gradient computation
or weight update.

The key insight: a weight is important if it is large AND the corresponding
input activation is large. Neither alone is sufficient.

Reference:
    Sun et al., "A Simple and Effective Pruning Approach for Large Language
    Models", ICLR 2024
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class WandaImportance(LayerMetric):
    """
    Compute Wanda importance score for a layer.

    For each layer with parameters, computes:
    - wanda_score: mean(|W| * |X|) — the core Wanda metric
    - weight_norm: L2 norm of weights (magnitude-only baseline)
    - activation_norm: mean L2 norm of input activations
    - wanda_sparsity: fraction of near-zero Wanda scores (potential pruning targets)

    Works on any layer with weight parameters (Linear, Conv, etc.).
    For layers without weights (e.g., LayerNorm, activation), returns the
    activation norm only.

    Args:
        num_batches: Number of batches for activation collection
        sparsity_threshold: Threshold for counting near-zero scores
    """

    def __init__(
        self,
        num_batches: int = 10,
        sparsity_threshold: float = 1e-3,
        **kwargs,
    ):
        super().__init__(
            name="wanda",
            category="optimization",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_batches = num_batches
        self.sparsity_threshold = sparsity_threshold

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
        """
        Compute Wanda importance for the layer.

        Returns:
            Dictionary with wanda_score, weight_norm, activation_norm, wanda_sparsity
        """
        # Collect input activations
        input_activations = []

        if _cached_activations:
            input_activations = _cached_activations
        else:

            def hook_fn(module, inp, output):
                x = inp[0] if isinstance(inp, tuple) else inp
                if isinstance(x, torch.Tensor):
                    input_activations.append(x.detach().cpu())

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

        if not input_activations:
            return {
                "wanda_score": 0.0,
                "weight_norm": 0.0,
                "activation_norm": 0.0,
                "wanda_sparsity": 0.0,
            }

        # Compute activation statistics
        all_acts = torch.cat(input_activations, dim=0).float()
        if all_acts.dim() > 2:
            all_acts = all_acts.reshape(all_acts.size(0), -1)

        act_norms = all_acts.norm(dim=0)  # per-feature norm
        mean_act_norm = act_norms.mean().item()

        # Get weight matrix
        weight = None
        for name, param in layer.named_parameters():
            if "weight" in name and param.dim() >= 2:
                weight = param.detach().cpu().float()
                break

        if weight is None:
            # No weight matrix (e.g., LayerNorm, activation layer)
            return {
                "wanda_score": 0.0,
                "weight_norm": 0.0,
                "activation_norm": float(mean_act_norm),
                "wanda_sparsity": 0.0,
            }

        # Weight norm
        weight_norm = weight.norm().item()

        # Wanda score: |W| * |X|
        # For Linear (out, in): importance[i,j] = |W[i,j]| * ||X[:,j]||
        weight_flat = weight.reshape(weight.size(0), -1)  # (out, in)
        w_abs = weight_flat.abs()

        # Match dimensions: act_norms is per-input-feature
        if act_norms.shape[0] == w_abs.shape[1]:
            wanda_scores = w_abs * act_norms.unsqueeze(0)  # (out, in)
        elif act_norms.shape[0] > w_abs.shape[1]:
            # Truncate activations to match weight input dim
            wanda_scores = w_abs * act_norms[: w_abs.shape[1]].unsqueeze(0)
        else:
            # Pad activations
            padded = torch.zeros(w_abs.shape[1])
            padded[: act_norms.shape[0]] = act_norms
            wanda_scores = w_abs * padded.unsqueeze(0)

        mean_wanda = wanda_scores.mean().item()

        # Sparsity: fraction of near-zero Wanda scores
        sparsity = (wanda_scores < self.sparsity_threshold).float().mean().item()

        return {
            "wanda_score": float(mean_wanda),
            "weight_norm": float(weight_norm),
            "activation_norm": float(mean_act_norm),
            "wanda_sparsity": float(sparsity),
        }
