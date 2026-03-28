"""
Residual-aware DropLayer metric.

Standard DropLayer zeroes out the entire layer output, which is overly
destructive for transformer blocks with residual connections. In these
architectures, each block computes:

    output = residual(x) + f(x)    where f(x) is the block's learned transform

Zeroing the entire output removes both the residual stream and the
learned contribution. The residual-aware version instead makes the
block pass through only the residual (identity), effectively measuring
the importance of f(x) alone.

This provides a more accurate importance signal for transformer-style
architectures (BERT, GPT, ViT, etc.) where residual connections carry
significant information.

Reference:
    Men et al., "ShortGPT: Layers in Large Language Models are More
    Redundant Than You Expect", ACL 2025
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import extract_logits, model_forward


class ResidualDropLayer(LayerMetric):
    """
    Residual-aware layer ablation metric.

    For each layer, replaces its output with its input (identity/skip),
    effectively removing only the layer's learned transformation while
    preserving the residual stream. This is the correct ablation for
    transformer blocks and any architecture with skip connections.

    Also computes the residual contribution ratio: how much of the
    layer's output comes from the learned transform vs. the residual.

    Args:
        num_batches: Number of batches to evaluate
        metric: Performance metric ('loss', 'accuracy')
    """

    _baseline_cache: Dict[int, Dict] = {}

    def __init__(
        self,
        num_batches: int = 10,
        metric: str = "loss",
        **kwargs,
    ):
        super().__init__(
            name="residual_droplayer",
            category="robustness",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_batches = num_batches
        self.metric = metric

    def _evaluate_baseline(
        self,
        model: nn.Module,
        data_loader: Any,
        device: str,
        criterion: nn.Module,
    ) -> Dict[str, List[float]]:
        """Evaluate model without any layer dropped (baseline)."""
        model_id = id(model)

        if model_id in ResidualDropLayer._baseline_cache:
            return ResidualDropLayer._baseline_cache[model_id]

        baseline_losses = []
        baseline_accs = []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.num_batches:
                    break
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model_forward(model, inputs)
                logits = extract_logits(outputs)
                loss = criterion(logits, targets)
                baseline_losses.append(loss.item())

                if self.metric == "accuracy":
                    preds = logits.argmax(dim=1)
                    acc = (preds == targets).float().mean().item()
                    baseline_accs.append(acc)

        result = {"losses": baseline_losses, "accs": baseline_accs}
        ResidualDropLayer._baseline_cache[model_id] = result
        return result

    @classmethod
    def clear_baseline_cache(cls):
        """Clear cached baseline results."""
        cls._baseline_cache.clear()

    def _measure_residual_contribution(
        self,
        model: nn.Module,
        layer: nn.Module,
        data_loader: Any,
        device: str,
    ) -> Dict[str, float]:
        """
        Measure how much the layer transforms its input vs. passing it through.

        Returns cosine similarity between input and output (high = mostly residual)
        and the relative norm of the learned contribution.
        """
        input_outputs = []

        def capture_hook(module, inp, out):
            x_in = inp[0] if isinstance(inp, tuple) else inp
            x_out = out[0] if isinstance(out, tuple) else out
            input_outputs.append((x_in.detach(), x_out.detach()))

        handle = layer.register_forward_hook(capture_hook)
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

        if not input_outputs:
            return {"residual_ratio": 0.0, "transform_norm_ratio": 0.0}

        cos_sims = []
        transform_ratios = []

        for x_in, x_out in input_outputs:
            # Flatten to (batch, features)
            x_in_flat = x_in.reshape(x_in.size(0), -1).float()
            x_out_flat = x_out.reshape(x_out.size(0), -1).float()

            # Only compute if shapes match (residual connection exists)
            if x_in_flat.shape == x_out_flat.shape:
                # Cosine similarity: high = layer barely transforms input
                cos = torch.nn.functional.cosine_similarity(x_in_flat, x_out_flat, dim=1)
                cos_sims.append(cos.mean().item())

                # Transform contribution: ||f(x)|| / ||x + f(x)||
                transform = x_out_flat - x_in_flat
                transform_norm = transform.norm(dim=1)
                output_norm = x_out_flat.norm(dim=1) + 1e-10
                transform_ratios.append((transform_norm / output_norm).mean().item())
            else:
                # Shape mismatch: no residual connection, layer fully transforms
                cos_sims.append(0.0)
                transform_ratios.append(1.0)

        return {
            "residual_ratio": float(np.mean(cos_sims)),
            "transform_norm_ratio": float(np.mean(transform_ratios)),
        }

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        _baseline_results: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute residual-aware DropLayer metric.

        The hook replaces the layer's output with its input, effectively
        skipping only the learned transformation while keeping the
        residual stream intact.

        Returns:
            Dictionary with:
            - residual_droplayer_loss_increase: loss increase when skipping layer
            - residual_droplayer_loss_ratio: dropped_loss / baseline_loss
            - residual_ratio: cosine similarity between input and output (1.0 = pure residual)
            - transform_norm_ratio: ||f(x)|| / ||output|| (0.0 = pure residual)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Get baseline
        if _baseline_results is not None:
            baseline = _baseline_results
        else:
            baseline = self._evaluate_baseline(model, data_loader, device, criterion)

        baseline_losses = baseline["losses"]
        baseline_accs = baseline["accs"]

        if not baseline_losses:
            return {
                "residual_droplayer_loss_increase": 0.0,
                "residual_droplayer_loss_ratio": 1.0,
                "residual_ratio": 0.0,
                "transform_norm_ratio": 0.0,
            }

        # Measure residual contribution
        residual_info = self._measure_residual_contribution(model, layer, data_loader, device)

        # Evaluate with layer skipped (output = input, identity pass-through)
        dropped_losses = []
        dropped_accs = []

        def skip_hook(module, inp, output):
            """Replace output with input (skip the learned transform)."""
            x_in = inp[0] if isinstance(inp, tuple) else inp
            x_out = output[0] if isinstance(output, tuple) else output

            # If shapes match, replace with input (residual skip)
            if x_in.shape == x_out.shape:
                if isinstance(output, tuple):
                    return (x_in,) + output[1:]
                return x_in
            else:
                # Shapes don't match (e.g., projection layer): zero output
                if isinstance(output, tuple):
                    return (torch.zeros_like(x_out),) + output[1:]
                return torch.zeros_like(output)

        handle = layer.register_forward_hook(skip_hook)

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= self.num_batches:
                        break
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        continue

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    try:
                        outputs = model_forward(model, inputs)
                        logits = extract_logits(outputs)
                        loss = criterion(logits, targets)
                        dropped_losses.append(loss.item())

                        if self.metric == "accuracy":
                            preds = logits.argmax(dim=1)
                            acc = (preds == targets).float().mean().item()
                            dropped_accs.append(acc)
                    except Exception:
                        dropped_losses.append(float("inf"))
                        if self.metric == "accuracy":
                            dropped_accs.append(0.0)
        finally:
            handle.remove()

        # Compute results
        if baseline_losses and dropped_losses:
            baseline_loss = np.mean(baseline_losses)
            dropped_loss = np.mean(dropped_losses)
            loss_increase = dropped_loss - baseline_loss

            result = {
                "residual_droplayer_loss_increase": float(loss_increase),
                "residual_droplayer_loss_ratio": (
                    float(dropped_loss / baseline_loss) if baseline_loss > 0 else 1.0
                ),
                "residual_ratio": residual_info["residual_ratio"],
                "transform_norm_ratio": residual_info["transform_norm_ratio"],
            }

            if self.metric == "accuracy" and baseline_accs and dropped_accs:
                baseline_acc = np.mean(baseline_accs)
                dropped_acc = np.mean(dropped_accs)
                result.update(
                    {
                        "residual_droplayer_acc_decrease": float(baseline_acc - dropped_acc),
                        "residual_droplayer_acc_ratio": (
                            float(dropped_acc / baseline_acc) if baseline_acc > 0 else 0.0
                        ),
                    }
                )

            return result
        else:
            return {
                "residual_droplayer_loss_increase": 0.0,
                "residual_droplayer_loss_ratio": 1.0,
                "residual_ratio": residual_info["residual_ratio"],
                "transform_norm_ratio": residual_info["transform_norm_ratio"],
            }
