"""
Per-layer adversarial sensitivity via FGSM perturbation.

Measures how much each layer's activations change when the input is
perturbed by a small adversarial step (FGSM). Layers with high
adversarial sensitivity are attack surfaces — small input perturbations
cause disproportionately large internal changes.

Reference:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import compute_loss, model_forward


class AdversarialSensitivity(LayerMetric):
    """
    Measure per-layer sensitivity to adversarial input perturbations.

    For each layer, computes:
    - adv_sensitivity: relative activation change under FGSM perturbation
    - adv_amplification: ratio of activation change to input perturbation magnitude
    - adv_directional_change: cosine distance between clean and adversarial activations

    Args:
        epsilon: FGSM perturbation magnitude
        num_batches: Number of batches to evaluate
    """

    def __init__(self, epsilon: float = 0.01, num_batches: int = 5, **kwargs):
        super().__init__(
            name="adversarial_sensitivity",
            category="security",
            requires_gradient=True,
            requires_data=True,
            **kwargs,
        )
        self.epsilon = epsilon
        self.num_batches = num_batches

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        sensitivities = []
        amplifications = []
        directional_changes = []

        for i, batch in enumerate(data_loader):
            if i >= self.num_batches:
                break
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                continue

            inputs = inputs.to(device).requires_grad_(True)
            targets = targets.to(device)

            # Phase 1: Clean forward — capture clean activations
            clean_acts = []

            def clean_hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                clean_acts.append(out.detach())

            handle = layer.register_forward_hook(clean_hook)
            try:
                model.eval()
                outputs = model_forward(model, inputs, targets)
                loss = compute_loss(outputs, targets, criterion)
            finally:
                handle.remove()

            if not clean_acts:
                continue

            # Phase 2: FGSM perturbation
            model.zero_grad()
            if inputs.grad is not None:
                inputs.grad.zero_()
            loss.backward()

            if inputs.grad is None:
                continue

            perturbation = self.epsilon * inputs.grad.sign()
            adv_inputs = (inputs + perturbation).detach()

            # Phase 3: Adversarial forward — capture adversarial activations
            adv_acts = []

            def adv_hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                adv_acts.append(out.detach())

            handle = layer.register_forward_hook(adv_hook)
            try:
                model.eval()
                with torch.no_grad():
                    model_forward(model, adv_inputs)
            finally:
                handle.remove()

            if not adv_acts:
                continue

            clean = clean_acts[0].float()
            adv = adv_acts[0].float()

            # Compute metrics
            clean_flat = clean.reshape(clean.size(0), -1)
            adv_flat = adv.reshape(adv.size(0), -1)

            # Relative activation change
            diff_norm = (adv_flat - clean_flat).norm(dim=1)
            clean_norm = clean_flat.norm(dim=1) + 1e-10
            rel_change = (diff_norm / clean_norm).mean().item()
            sensitivities.append(rel_change)

            # Amplification: activation change / input perturbation
            input_pert_norm = (
                perturbation.reshape(perturbation.size(0), -1).norm(dim=1).mean().item()
            )
            if input_pert_norm > 1e-10:
                amplifications.append(diff_norm.mean().item() / input_pert_norm)

            # Directional change (1 - cosine similarity)
            cos_sim = nn.functional.cosine_similarity(clean_flat, adv_flat, dim=1)
            directional_changes.append((1.0 - cos_sim.mean().item()))

        if not sensitivities:
            return {
                "adv_sensitivity": 0.0,
                "adv_amplification": 0.0,
                "adv_directional_change": 0.0,
            }

        return {
            "adv_sensitivity": float(np.mean(sensitivities)),
            "adv_amplification": float(np.mean(amplifications)) if amplifications else 0.0,
            "adv_directional_change": float(np.mean(directional_changes)),
        }
