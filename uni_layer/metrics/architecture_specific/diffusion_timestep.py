"""
Timestep-aware layer analysis for Diffusion models (UNet / DiT).

Diffusion models process inputs at varying noise levels (timesteps).
A layer that is critical at high noise (t~T) may be redundant at low
noise (t~0). This metric captures per-layer importance as a function
of timestep, enabling:

- Timestep-adaptive pruning (remove layers only at certain noise levels)
- Understanding which UNet stages matter most at each denoising phase
- Identifying layers whose importance is timestep-invariant vs. timestep-dependent

Supports:
- UNet architectures (Stable Diffusion, DDPM) with down_blocks/mid_block/up_blocks
- DiT (Diffusion Transformer) with transformer blocks
- Custom diffusion models with a `timestep` or `t` forward argument

Reference:
    Castells et al., "LD-Pruner: Efficient Pruning of Latent Diffusion Models
    using Task-Agnostic Insights", CVPR 2024
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric


class DiffusionTimestepAnalysis(LayerMetric):
    """
    Analyze layer importance across diffusion timesteps.

    For each layer, measures activation magnitude at multiple timesteps
    and computes:
    - timestep_sensitivity: how much the layer's activation changes across timesteps
    - mean_activation_norm: average activation norm across all timesteps
    - early_importance: relative activation norm at low noise (t near 0)
    - late_importance: relative activation norm at high noise (t near T)
    - timestep_variance: variance of activation norm across timesteps

    Args:
        num_timesteps: Number of timestep samples to evaluate
        max_timestep: Maximum timestep value (T)
        num_batches: Number of batches per timestep
    """

    def __init__(
        self,
        num_timesteps: int = 10,
        max_timestep: int = 1000,
        num_batches: int = 5,
        **kwargs,
    ):
        super().__init__(
            name="diffusion_timestep",
            category="architecture_specific",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_timesteps = num_timesteps
        self.max_timestep = max_timestep
        self.num_batches = num_batches

    def _get_timestep_param_name(self, model: nn.Module) -> Optional[str]:
        """Detect the timestep parameter name in model.forward()."""
        try:
            import inspect

            params = list(inspect.signature(model.forward).parameters.keys())
            for candidate in ("timestep", "timesteps", "t", "time", "noise_level"):
                if candidate in params:
                    return candidate
        except (ValueError, TypeError):
            pass
        return None

    def _is_diffusion_model(self, model: nn.Module) -> bool:
        """Check if model is a diffusion model."""
        # Check for timestep parameter
        if self._get_timestep_param_name(model) is not None:
            return True
        # Check for UNet-style structure
        if hasattr(model, "down_blocks") or hasattr(model, "up_blocks"):
            return True
        if hasattr(model, "mid_block"):
            return True
        class_lower = model.__class__.__name__.lower()
        if any(k in class_lower for k in ("unet", "diffusion", "ddpm", "dit")):
            return True
        return False

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Compute timestep-aware metrics for a layer.

        For non-diffusion models, returns None values.
        For diffusion models, evaluates the layer at multiple timesteps
        and measures how its behavior changes.
        """
        null_result = {
            "timestep_sensitivity": None,
            "mean_activation_norm": None,
            "early_importance": None,
            "late_importance": None,
            "timestep_variance": None,
        }

        if not self._is_diffusion_model(model):
            return null_result

        timestep_param = self._get_timestep_param_name(model)
        if timestep_param is None:
            return null_result

        # Sample timesteps uniformly
        timesteps = torch.linspace(0, self.max_timestep - 1, self.num_timesteps).long()

        # Collect activation norms per timestep
        activation_norms_per_t = []

        for t_val in timesteps:
            norms = []

            def hook_fn(module, inp, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                if isinstance(out, torch.Tensor):
                    norms.append(out.detach().float().norm().item())

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

                        # Create timestep tensor matching batch size
                        batch_size = inputs.shape[0]
                        t_tensor = torch.full(
                            (batch_size,), t_val.item(), dtype=torch.long, device=device
                        )

                        try:
                            model(inputs, **{timestep_param: t_tensor})
                        except Exception:
                            # Some models may not accept certain timestep formats
                            try:
                                model(inputs, **{timestep_param: t_val.item()})
                            except Exception:
                                break
            finally:
                handle.remove()

            if norms:
                activation_norms_per_t.append(np.mean(norms))

        if len(activation_norms_per_t) < 2:
            return null_result

        norms_array = np.array(activation_norms_per_t)
        mean_norm = norms_array.mean()

        # Timestep sensitivity: coefficient of variation across timesteps
        if mean_norm > 1e-10:
            sensitivity = norms_array.std() / mean_norm
        else:
            sensitivity = 0.0

        # Early importance (first 30% of timesteps) vs late importance (last 30%)
        n = len(norms_array)
        early_cut = max(1, int(0.3 * n))
        late_cut = max(1, int(0.3 * n))

        early_norm = norms_array[:early_cut].mean()
        late_norm = norms_array[-late_cut:].mean()

        # Normalize to relative importance
        total = early_norm + late_norm + 1e-10
        early_importance = early_norm / total
        late_importance = late_norm / total

        return {
            "timestep_sensitivity": float(sensitivity),
            "mean_activation_norm": float(mean_norm),
            "early_importance": float(early_importance),
            "late_importance": float(late_importance),
            "timestep_variance": float(norms_array.var()),
        }


def get_diffusion_blocks(model: nn.Module) -> OrderedDict:
    """
    Extract blocks from a UNet / DiT diffusion model.

    Handles common structures:
    - UNet: down_blocks + mid_block + up_blocks
    - DiT: transformer.blocks or layers
    - Custom: any model with identifiable block structure

    Returns:
        OrderedDict of {block_name: module}
    """
    layers = OrderedDict()

    # UNet-style: down_blocks, mid_block, up_blocks
    if hasattr(model, "down_blocks"):
        down = model.down_blocks
        if isinstance(down, nn.ModuleList):
            for i, block in enumerate(down):
                layers[f"down_blocks.{i}"] = block
        elif isinstance(down, nn.Module):
            layers["down_blocks"] = down

    if hasattr(model, "mid_block") and model.mid_block is not None:
        layers["mid_block"] = model.mid_block

    if hasattr(model, "up_blocks"):
        up = model.up_blocks
        if isinstance(up, nn.ModuleList):
            for i, block in enumerate(up):
                layers[f"up_blocks.{i}"] = block
        elif isinstance(up, nn.Module):
            layers["up_blocks"] = up

    if layers:
        return layers

    # DiT-style: look for transformer blocks
    for name, module in model.named_children():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            first = module[0]
            if len(list(first.children())) >= 2:
                for i, block in enumerate(module):
                    layers[f"{name}.{i}"] = block
                return layers

    return layers
