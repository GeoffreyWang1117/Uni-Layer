"""
Per-layer efficiency profiling: FLOPs, parameters, memory.

Computes computational cost metrics for each layer without running
inference. Uses analytical formulas for known layer types (Linear, Conv,
Attention) and falls back to parameter-count estimation for custom layers.

Metrics:
- flops: Floating-point operations for one forward pass
- param_count: Number of parameters
- param_memory_mb: Parameter memory in megabytes (FP32)
- activation_memory_mb: Peak activation memory per sample (estimated)
- compute_ratio: This layer's FLOPs / total model FLOPs

Reference:
    Molchanov et al., "Importance Estimation for Neural Network Pruning", CVPR 2019
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class EfficiencyProfiler(LayerMetric):
    """
    Profile per-layer computational efficiency.

    For each layer computes:
    - flops: FLOPs for one forward pass (analytical for known types)
    - param_count: total parameters
    - param_memory_mb: parameter memory in MB (FP32)
    - activation_memory_mb: activation memory per sample in MB
    - compute_ratio: fraction of total model FLOPs

    Args:
        num_batches: Number of batches for activation size estimation
    """

    # Class-level cache for total model FLOPs
    _total_flops_cache: Dict[int, float] = {}

    def __init__(self, num_batches: int = 1, **kwargs):
        super().__init__(
            name="efficiency",
            category="efficiency",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_batches = num_batches

    @staticmethod
    def _linear_flops(module: nn.Linear, input_shape: Optional[tuple] = None) -> int:
        """FLOPs for Linear: 2 * in * out (multiply-add)."""
        flops = 2 * module.in_features * module.out_features
        if module.bias is not None:
            flops += module.out_features
        return flops

    @staticmethod
    def _conv_flops(module, input_shape: Optional[tuple] = None) -> int:
        """FLOPs for Conv1d/2d/3d."""
        if isinstance(module, nn.Conv1d):
            kernel_ops = module.kernel_size[0]
            output_length = 1  # approximate
            if input_shape and len(input_shape) >= 3:
                L_in = input_shape[2]
                output_length = (
                    L_in + 2 * module.padding[0] - module.kernel_size[0]
                ) // module.stride[0] + 1
            flops = (
                2
                * module.in_channels
                * module.out_channels
                * kernel_ops
                * output_length
                // module.groups
            )
        elif isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            H_out, W_out = 1, 1
            if input_shape and len(input_shape) >= 4:
                H_in, W_in = input_shape[2], input_shape[3]
                H_out = (H_in + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[
                    0
                ] + 1
                W_out = (W_in + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[
                    1
                ] + 1
            flops = (
                2
                * module.in_channels
                * module.out_channels
                * kernel_ops
                * H_out
                * W_out
                // module.groups
            )
        elif isinstance(module, nn.Conv3d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
            flops = 2 * module.in_channels * module.out_channels * kernel_ops // module.groups
        else:
            flops = 0
        if hasattr(module, "bias") and module.bias is not None:
            flops += module.out_channels
        return flops

    @staticmethod
    def _attention_flops(module: nn.MultiheadAttention, seq_len: int = 128) -> int:
        """FLOPs for MultiheadAttention: QKV projections + attention + output."""
        d = module.embed_dim
        # QKV projections: 3 * 2 * seq * d * d
        proj_flops = 3 * 2 * seq_len * d * d
        # Attention scores: seq * seq * d
        attn_flops = 2 * seq_len * seq_len * d
        # Output projection: 2 * seq * d * d
        out_flops = 2 * seq_len * d * d
        return proj_flops + attn_flops + out_flops

    def _estimate_flops(self, layer: nn.Module, input_shape: Optional[tuple] = None) -> int:
        """Estimate FLOPs for a layer (recursive for compound layers)."""
        total = 0

        if isinstance(layer, nn.Linear):
            return self._linear_flops(layer, input_shape)
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return self._conv_flops(layer, input_shape)
        elif isinstance(layer, nn.MultiheadAttention):
            seq_len = input_shape[1] if input_shape and len(input_shape) >= 3 else 128
            return self._attention_flops(layer, seq_len)

        # Recursive: sum children
        for child in layer.children():
            total += self._estimate_flops(child, input_shape)

        # If no children but has parameters, estimate from param count
        if total == 0:
            params = sum(p.numel() for p in layer.parameters())
            total = 2 * params  # rough: 1 multiply + 1 add per param

        return total

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
        # Parameter count and memory
        param_count = sum(p.numel() for p in layer.parameters())
        param_memory_mb = param_count * 4 / (1024 * 1024)  # FP32

        # Capture activation shape for FLOPs estimation
        act_shape = None
        input_shape = None
        act_memory = 0.0

        def hook_fn(module, inp, output):
            nonlocal act_shape, input_shape
            x_in = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x_in, torch.Tensor):
                input_shape = tuple(x_in.shape)
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                act_shape = tuple(out.shape)

        handle = layer.register_forward_hook(hook_fn)
        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= 1:  # only need 1 batch for shapes
                        break
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    inputs = inputs.to(device)
                    model_forward(model, inputs)
        finally:
            handle.remove()

        # FLOPs
        flops = self._estimate_flops(layer, input_shape)

        # Activation memory (per sample)
        if act_shape and len(act_shape) >= 2:
            elements_per_sample = 1
            for dim in act_shape[1:]:
                elements_per_sample *= dim
            act_memory = elements_per_sample * 4 / (1024 * 1024)  # FP32 MB

        # Compute ratio (vs total model)
        model_id = id(model)
        if model_id not in EfficiencyProfiler._total_flops_cache:
            total = self._estimate_flops(model, input_shape)
            EfficiencyProfiler._total_flops_cache[model_id] = max(total, 1)
        total_flops = EfficiencyProfiler._total_flops_cache[model_id]
        compute_ratio = flops / total_flops if total_flops > 0 else 0.0

        return {
            "flops": float(flops),
            "param_count": float(param_count),
            "param_memory_mb": float(param_memory_mb),
            "activation_memory_mb": float(act_memory),
            "compute_ratio": float(compute_ratio),
        }

    @classmethod
    def clear_cache(cls):
        cls._total_flops_cache.clear()
