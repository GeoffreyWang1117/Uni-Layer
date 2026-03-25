"""
Jacobian Rank metric for measuring layer expressiveness.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class JacobianRank(LayerMetric):
    """
    Compute the rank of the layer's Jacobian matrix.

    The Jacobian rank measures how much the layer transforms its input,
    indicating its expressiveness and contribution to the model.

    Higher rank suggests:
    - More complex transformations
    - Better utilization of layer capacity
    - Less redundancy

    Args:
        num_samples: Number of samples to use
        rank_threshold: Threshold for considering singular values (relative to max)
    """

    def __init__(
        self,
        num_samples: int = 100,
        rank_threshold: float = 1e-3,
        **kwargs
    ):
        super().__init__(
            name="jacobian_rank",
            category="representation",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )
        self.num_samples = num_samples
        self.rank_threshold = rank_threshold

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
        Compute Jacobian rank for the layer.

        Returns:
            Dictionary with rank metrics
        """
        jacobians = []

        model.eval()
        num_processed = 0

        for batch in data_loader:
            if num_processed >= self.num_samples:
                break

            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            batch_size = min(inputs.size(0), self.num_samples - num_processed)
            inputs = inputs[:batch_size]

            # Get layer output
            activations = []

            def hook_fn(module, input, output):
                activations.append(output)

            handle = layer.register_forward_hook(hook_fn)

            try:
                # Forward pass
                with torch.enable_grad():
                    inputs.requires_grad_(True)
                    model_forward(model, inputs)

                    if activations:
                        output = activations[0]

                        # Compute Jacobian for each sample
                        for i in range(batch_size):
                            # Get scalar output by summing
                            scalar_output = output[i].sum()

                            # Compute gradient w.r.t. input
                            jacobian = torch.autograd.grad(
                                scalar_output,
                                inputs,
                                retain_graph=True,
                                create_graph=False
                            )[0]

                            jacobians.append(jacobian[i].detach().cpu().flatten())

            finally:
                handle.remove()

            num_processed += batch_size

        if jacobians:
            # Stack Jacobians
            J = torch.stack(jacobians)

            # Compute SVD
            try:
                _, S, _ = torch.svd(J)

                # Compute rank (number of significant singular values)
                threshold = S[0].item() * self.rank_threshold
                rank = (S > threshold).sum().item()

                # Compute condition number
                if S[-1] > 0:
                    condition_number = (S[0] / S[-1]).item()
                else:
                    condition_number = float('inf')

                return {
                    "jacobian_rank": float(rank),
                    "jacobian_rank_ratio": float(rank / len(S)),
                    "jacobian_condition": float(condition_number) if condition_number != float('inf') else 1e10,
                    "jacobian_max_sv": float(S[0].item()),
                }

            except Exception:
                return {
                    "jacobian_rank": 0.0,
                    "jacobian_rank_ratio": 0.0,
                    "jacobian_condition": 0.0,
                    "jacobian_max_sv": 0.0,
                }
        else:
            return {
                "jacobian_rank": 0.0,
                "jacobian_rank_ratio": 0.0,
                "jacobian_condition": 0.0,
                "jacobian_max_sv": 0.0,
            }
