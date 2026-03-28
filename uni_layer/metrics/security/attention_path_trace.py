"""
Attention path tracing for prompt injection vulnerability analysis.

Traces how attention flows through transformer layers to identify paths
where adversarial tokens can hijack the model's focus. Layers with high
attention manipulability are vulnerable to prompt injection attacks.

Metrics:
- How concentrated attention is on specific positions (hijack potential)
- How much a single token can shift the attention distribution
- Cross-layer attention persistence (does an injected focus propagate?)

Reference:
    Perez & Ribeiro, "Ignore This Title and HackAPrompt", NeurIPS 2023
    Greshake et al., "Not What You've Signed Up For: Compromising Real-World
    LLM-Integrated Applications with Indirect Prompt Injection", AISec 2023
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class AttentionPathTrace(LayerMetric):
    """
    Trace attention vulnerability to prompt injection / hijacking.

    For each transformer layer, computes:
    - attention_concentration: max attention weight per head (hijack potential)
    - attention_manipulability: how much a single position dominates attention
    - attention_persistence: consistency of attention pattern across samples
    - injection_vulnerability: composite score (0-1, higher = more vulnerable)

    Non-attention layers return None values (graceful skip).

    Args:
        num_batches: Number of batches to analyze
    """

    def __init__(self, num_batches: int = 5, **kwargs):
        super().__init__(
            name="attention_path_trace",
            category="security",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_batches = num_batches

    def _find_attention_module(self, layer: nn.Module) -> Optional[nn.Module]:
        """Find the attention submodule within a transformer block."""
        for attr in ("self_attn", "attn", "attention", "self_attention"):
            module = getattr(layer, attr, None)
            if module is not None and isinstance(module, nn.Module):
                return module
        if isinstance(layer, nn.MultiheadAttention):
            return layer
        return None

    def _has_attention(self, layer: nn.Module) -> bool:
        return self._find_attention_module(layer) is not None

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
    ) -> Dict[str, Optional[float]]:
        null_result = {
            "attention_concentration": None,
            "attention_manipulability": None,
            "attention_persistence": None,
            "injection_vulnerability": None,
        }

        if not self._has_attention(layer):
            return null_result

        # Capture attention weights
        attn_weights_list = []
        attn_module = self._find_attention_module(layer)

        def attn_hook(module, inp, output):
            # Many attention modules return (output, attn_weights) or
            # store weights in module.attn_weights
            if isinstance(output, tuple) and len(output) >= 2:
                weights = output[1]
                if isinstance(weights, torch.Tensor) and weights.dim() >= 3:
                    attn_weights_list.append(weights.detach().cpu())

        # Also try to capture via output_attentions pattern
        original_output_attentions = (
            getattr(model.config, "output_attentions", None) if hasattr(model, "config") else None
        )
        if hasattr(model, "config"):
            try:
                model.config.output_attentions = True
            except (AttributeError, TypeError):
                pass

        handle = attn_module.register_forward_hook(attn_hook) if attn_module else None

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
            if handle:
                handle.remove()
            if original_output_attentions is not None and hasattr(model, "config"):
                try:
                    model.config.output_attentions = original_output_attentions
                except (AttributeError, TypeError):
                    pass

        if not attn_weights_list:
            # Fallback: use activation-based proxy metrics
            return self._compute_proxy_metrics(
                model, layer, data_loader, device, _cached_activations
            )

        # Analyze attention weights: shape (batch, heads, seq, seq)
        all_weights = torch.cat(attn_weights_list, dim=0)  # (B, H, S, S)

        # 1. Attention concentration: max weight per (batch, head) pair
        max_weights = all_weights.amax(dim=-1).mean().item()  # mean max across positions

        # 2. Attention manipulability: how much one position dominates
        # Measured by the Gini coefficient of attention distribution
        attn_flat = all_weights.reshape(-1, all_weights.size(-1))  # (B*H*S, S)
        sorted_attn, _ = attn_flat.sort(dim=-1)
        n = sorted_attn.size(-1)
        index = torch.arange(1, n + 1, dtype=torch.float32)
        gini = (
            2 * (index * sorted_attn).sum(dim=-1) / (n * sorted_attn.sum(dim=-1) + 1e-10)
            - (n + 1) / n
        )
        manipulability = gini.mean().item()

        # 3. Attention persistence: how consistent is attention across samples
        # Low variance = persistent pattern = easier to exploit
        mean_attn = all_weights.mean(dim=0)  # (H, S, S)
        variance = ((all_weights - mean_attn.unsqueeze(0)) ** 2).mean().item()
        persistence = 1.0 / (1.0 + variance)  # high persistence when low variance

        # 4. Composite injection vulnerability score
        vulnerability = max_weights * 0.4 + manipulability * 0.3 + persistence * 0.3

        return {
            "attention_concentration": float(max_weights),
            "attention_manipulability": float(manipulability),
            "attention_persistence": float(persistence),
            "injection_vulnerability": float(min(1.0, vulnerability)),
        }

    def _compute_proxy_metrics(
        self,
        model: nn.Module,
        layer: nn.Module,
        data_loader: Any,
        device: str,
        _cached_activations: Optional[List[torch.Tensor]],
    ) -> Dict[str, Optional[float]]:
        """Fallback: compute proxy vulnerability metrics from activations."""
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
            return {
                "attention_concentration": None,
                "attention_manipulability": None,
                "attention_persistence": None,
                "injection_vulnerability": None,
            }

        all_acts = torch.cat(activations, dim=0).float()
        if all_acts.dim() > 2:
            all_acts = all_acts.reshape(all_acts.size(0), -1)

        # Proxy: activation concentration as attention substitute
        act_norms = all_acts.norm(dim=0)
        total_norm = act_norms.sum() + 1e-10
        probs = act_norms / total_norm
        concentration = probs.max().item()

        # Proxy: variance across samples
        variance = all_acts.var(dim=0).mean().item()
        persistence = 1.0 / (1.0 + variance)

        return {
            "attention_concentration": float(concentration),
            "attention_manipulability": None,
            "attention_persistence": float(persistence),
            "injection_vulnerability": float(min(1.0, concentration * 0.5 + persistence * 0.5)),
        }
