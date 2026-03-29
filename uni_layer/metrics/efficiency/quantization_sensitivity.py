"""
Per-layer quantization sensitivity analysis.

Simulates quantization (FP16, INT8) on each layer's weights and
activations, measuring the output deviation. Layers with high sensitivity
should stay at higher precision; layers with low sensitivity can be
aggressively quantized.

Reference:
    Wu et al., "Mixed Precision Quantization of ConvNets via Differentiable
    Neural Architecture Search", 2018
    Yao et al., "HAWQ-V2: Hessian Aware trace-Weighted Quantization", NeurIPS 2020
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


def _simulate_quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate symmetric uniform quantization to given bit-width."""
    if bits >= 32 or tensor.numel() == 0:
        return tensor
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / qmax if tensor.abs().max() > 0 else torch.tensor(1.0)
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return quantized * scale


class QuantizationSensitivity(LayerMetric):
    """
    Measure per-layer sensitivity to weight and activation quantization.

    For each layer, simulates quantization at FP16 and INT8, then measures
    the relative output change (sensitivity). Lower sensitivity = safer to quantize.

    Metrics:
    - quant_sensitivity_int8: relative output change under INT8 weight quantization
    - quant_sensitivity_fp16: relative output change under FP16 weight quantization
    - activation_range: dynamic range of activations (max - min)
    - weight_dynamic_range: ratio of max to min non-zero weight magnitude

    Args:
        num_batches: Number of batches for evaluation
    """

    def __init__(self, num_batches: int = 5, **kwargs):
        super().__init__(
            name="quantization_sensitivity",
            category="efficiency",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
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
        **kwargs,
    ) -> Dict[str, float]:
        # Find weight tensor
        weight_param = None
        for name, param in layer.named_parameters():
            if "weight" in name and param.dim() >= 2:
                weight_param = param
                break

        if weight_param is None:
            return {
                "quant_sensitivity_int8": 0.0,
                "quant_sensitivity_fp16": 0.0,
                "activation_range": 0.0,
                "weight_dynamic_range": 0.0,
            }

        original_weight = weight_param.data.clone()

        # Collect clean outputs and activation stats
        clean_outputs = []
        act_mins = []
        act_maxs = []

        def collect_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                clean_outputs.append(out.detach().cpu().float())
                act_mins.append(out.min().item())
                act_maxs.append(out.max().item())

        handle = layer.register_forward_hook(collect_hook)
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

        if not clean_outputs:
            return {
                "quant_sensitivity_int8": 0.0,
                "quant_sensitivity_fp16": 0.0,
                "activation_range": 0.0,
                "weight_dynamic_range": 0.0,
            }

        # Activation range
        activation_range = max(act_maxs) - min(act_mins) if act_mins else 0.0

        # Weight dynamic range
        w_abs = original_weight.abs().flatten()
        w_nonzero = w_abs[w_abs > 1e-10]
        if len(w_nonzero) > 0:
            weight_dynamic_range = (w_nonzero.max() / w_nonzero.min()).item()
        else:
            weight_dynamic_range = 0.0

        # Test INT8 quantization
        sensitivity_int8 = self._measure_quant_sensitivity(
            model, layer, weight_param, original_weight, data_loader, device, bits=8
        )

        # Test FP16 quantization
        sensitivity_fp16 = self._measure_quant_sensitivity(
            model, layer, weight_param, original_weight, data_loader, device, bits=16
        )

        # Restore original weights
        weight_param.data.copy_(original_weight)

        return {
            "quant_sensitivity_int8": float(sensitivity_int8),
            "quant_sensitivity_fp16": float(sensitivity_fp16),
            "activation_range": float(activation_range),
            "weight_dynamic_range": float(weight_dynamic_range),
        }

    def _measure_quant_sensitivity(
        self,
        model: nn.Module,
        layer: nn.Module,
        weight_param: nn.Parameter,
        original_weight: torch.Tensor,
        data_loader: Any,
        device: str,
        bits: int,
    ) -> float:
        """Quantize weights, measure output deviation, restore."""
        # Quantize
        if bits == 16:
            quantized = original_weight.half().float()
        else:
            quantized = _simulate_quantize(original_weight.cpu(), bits).to(original_weight.device)

        weight_param.data.copy_(quantized.to(weight_param.device))

        # Collect quantized outputs
        quant_outputs = []

        def q_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                quant_outputs.append(out.detach().cpu().float())

        handle = layer.register_forward_hook(q_hook)
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

        # Restore
        weight_param.data.copy_(original_weight)

        # Compute relative deviation
        if not quant_outputs:
            return 0.0

        # Re-collect clean outputs for comparison
        clean_outputs = []

        def c_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                clean_outputs.append(out.detach().cpu().float())

        handle = layer.register_forward_hook(c_hook)
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

        if len(clean_outputs) != len(quant_outputs):
            return 0.0

        total_diff = 0.0
        total_norm = 0.0
        for clean, quant in zip(clean_outputs, quant_outputs):
            diff = (clean - quant).flatten().norm().item()
            norm = clean.flatten().norm().item() + 1e-10
            total_diff += diff
            total_norm += norm

        return total_diff / total_norm if total_norm > 0 else 0.0
