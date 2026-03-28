"""
MoE (Mixture of Experts) Router Analysis metric.

Analyzes the routing behavior of MoE layers to assess:
- Expert utilization: how evenly experts are used across inputs
- Load balance: whether routing is skewed toward few experts
- Routing entropy: information-theoretic diversity of routing decisions
- Expert overlap: how often the same experts are co-selected

These metrics are critical for understanding MoE efficiency and for
identifying layers where expert consolidation or pruning is viable.

Supports common MoE implementations:
- HuggingFace Mixtral (MixtralSparseMoeBlock)
- HuggingFace Switch Transformers (SwitchTransformersSparseMLP)
- Custom MoE with a `gate` or `router` submodule

Reference:
    Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity", JMLR 2022
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import model_forward


class MoERouterAnalysis(LayerMetric):
    """
    Analyze routing behavior of MoE (Mixture of Experts) layers.

    For each MoE layer, captures the router's gating decisions and computes:
    - routing_entropy: Shannon entropy of routing probabilities (higher = more diverse)
    - expert_utilization: fraction of experts receiving > 1% of tokens
    - load_balance_score: coefficient of variation of expert loads (0 = perfect balance)
    - top_expert_ratio: fraction of tokens routed to the most popular expert
    - expert_overlap: mean pairwise co-selection rate among top-k experts

    Non-MoE layers return None for all metrics (graceful skip).

    Args:
        num_batches: Number of batches to analyze
        top_k: Number of experts selected per token (for overlap analysis)
    """

    def __init__(self, num_batches: int = 10, top_k: int = 2, **kwargs):
        super().__init__(
            name="moe_router",
            category="architecture_specific",
            requires_gradient=False,
            requires_data=True,
            **kwargs,
        )
        self.num_batches = num_batches
        self.top_k = top_k

    def _find_gate(self, layer: nn.Module) -> Optional[nn.Module]:
        """
        Find the gating/router submodule in an MoE layer.

        Handles common naming conventions:
        - layer.gate (Mixtral, custom)
        - layer.router (Switch Transformers)
        - layer.gate_proj
        - layer itself if it has 'gate' or 'router' in name
        """
        for attr_name in ("gate", "router", "gate_proj", "routing"):
            if hasattr(layer, attr_name):
                gate = getattr(layer, attr_name)
                if isinstance(gate, nn.Module):
                    return gate

        # Check if the layer itself is a gate (Linear producing expert logits)
        layer_name = layer.__class__.__name__.lower()
        if "gate" in layer_name or "router" in layer_name:
            return layer

        return None

    def _is_moe_layer(self, layer: nn.Module) -> bool:
        """Check if a layer is an MoE layer."""
        layer_name = layer.__class__.__name__.lower()
        if any(kw in layer_name for kw in ("moe", "mixture", "sparse")):
            return True
        if hasattr(layer, "experts") or hasattr(layer, "local_experts"):
            return True
        if self._find_gate(layer) is not None and (
            hasattr(layer, "experts") or hasattr(layer, "num_experts")
        ):
            return True
        return False

    def _get_num_experts(self, layer: nn.Module) -> Optional[int]:
        """Infer the number of experts from the layer."""
        if hasattr(layer, "num_experts"):
            return layer.num_experts
        if hasattr(layer, "num_local_experts"):
            return layer.num_local_experts
        if hasattr(layer, "experts"):
            experts = layer.experts
            if isinstance(experts, (nn.ModuleList, list)):
                return len(experts)
        # Infer from gate output dimension
        gate = self._find_gate(layer)
        if gate is not None and isinstance(gate, nn.Linear):
            return gate.out_features
        return None

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
        """
        Compute MoE routing metrics for a layer.

        For non-MoE layers, returns None values (graceful skip).
        For MoE layers, captures gate outputs and computes routing statistics.

        Returns:
            Dictionary with routing_entropy, expert_utilization,
            load_balance_score, top_expert_ratio, expert_overlap
        """
        null_result = {
            "routing_entropy": None,
            "expert_utilization": None,
            "load_balance_score": None,
            "top_expert_ratio": None,
            "expert_overlap": None,
        }

        # Skip non-MoE layers
        if not self._is_moe_layer(layer):
            return null_result

        gate = self._find_gate(layer)
        num_experts = self._get_num_experts(layer)

        if gate is None or num_experts is None or num_experts < 2:
            return null_result

        # Capture gate outputs
        gate_outputs = []

        def gate_hook(module, inp, output):
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            if isinstance(logits, torch.Tensor):
                gate_outputs.append(logits.detach().cpu())

        handle = gate.register_forward_hook(gate_hook)

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

        if not gate_outputs:
            return null_result

        # Concatenate all gate outputs: (total_tokens, num_experts)
        all_logits = torch.cat([g.reshape(-1, g.size(-1)) for g in gate_outputs], dim=0)

        # Convert to routing probabilities
        routing_probs = torch.softmax(all_logits, dim=-1)  # (T, E)
        T, E = routing_probs.shape

        # 1. Routing entropy: mean per-token entropy
        per_token_entropy = -(routing_probs * torch.log(routing_probs + 1e-10)).sum(dim=-1)
        max_entropy = np.log(E)
        routing_entropy = per_token_entropy.mean().item() / max_entropy if max_entropy > 0 else 0.0

        # 2. Expert utilization: fraction of experts receiving > 1% of tokens
        expert_loads = routing_probs.mean(dim=0)  # (E,)
        utilization = (expert_loads > 0.01).float().mean().item()

        # 3. Load balance score: coefficient of variation of expert loads
        # 0 = perfectly balanced, higher = more skewed
        load_cv = (expert_loads.std() / (expert_loads.mean() + 1e-10)).item()

        # 4. Top expert ratio: fraction of probability mass for the most popular expert
        top_expert_ratio = expert_loads.max().item()

        # 5. Expert overlap: mean pairwise co-selection rate among top-k
        top_k = min(self.top_k, E)
        if top_k >= 2:
            top_experts = routing_probs.topk(top_k, dim=-1).indices  # (T, top_k)
            # For each pair of top-k positions, check if they share experts across tokens
            overlap_count = 0
            pair_count = 0
            for a in range(top_k):
                for b in range(a + 1, top_k):
                    # How often do different top-k slots select the same expert?
                    overlap_count += (top_experts[:, a] == top_experts[:, b]).float().sum().item()
                    pair_count += T
            expert_overlap = overlap_count / pair_count if pair_count > 0 else 0.0
        else:
            expert_overlap = 0.0

        return {
            "routing_entropy": float(routing_entropy),
            "expert_utilization": float(utilization),
            "load_balance_score": float(load_cv),
            "top_expert_ratio": float(top_expert_ratio),
            "expert_overlap": float(expert_overlap),
        }
