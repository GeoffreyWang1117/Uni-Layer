"""
Knowledge distillation helper based on layer contribution analysis.

Auto-selects distillation layers and implements intermediate layer distillation
with temperature scaling.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""

    temperature: float = 4.0  # Softmax temperature for soft targets
    alpha: float = 0.5  # Weight for distillation loss (vs. hard target loss)
    layer_weight: float = 0.3  # Weight for intermediate layer loss
    distance_metric: str = "mse"  # Distance metric: mse, cosine, kl
    top_k_layers: int = 3  # Number of intermediate layers to distill


class KnowledgeDistiller:
    """
    Knowledge distillation with automatic layer selection.

    Mathematical Foundation:
    ========================

    Standard Knowledge Distillation (Hinton et al., 2015):
    L_KD = α·L_soft + (1-α)·L_hard

    where:
    L_soft = KL(softmax(z_s/T), softmax(z_t/T)) · T²
    L_hard = CE(y, softmax(z_s))

    T: temperature (higher → softer probabilities)
    z_s: student logits
    z_t: teacher logits
    y: ground truth labels

    Intermediate Layer Distillation:
    -------------------------------
    L_layer = β·Σ_ℓ w_ℓ · d(h_s^(ℓ), h_t^(ℓ))

    where:
    - h_s^(ℓ), h_t^(ℓ): student and teacher activations at layer ℓ
    - d(·,·): distance metric (MSE, cosine, etc.)
    - w_ℓ: layer weight based on contribution analysis

    Layer Selection Strategy:
    ------------------------
    Select layers with:
    1. High CKA scores → important representations
    2. High gradient norms → active learning
    3. Intermediate depth → balanced abstraction level

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        contributions: Layer contribution analysis from teacher
        config: Distillation configuration
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        contributions: Dict[str, Dict[str, float]],
        config: DistillationConfig = None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.contributions = contributions
        self.config = config or DistillationConfig()

        # Storage for intermediate activations
        self.teacher_activations = {}
        self.student_activations = {}
        self.hooks = []

        # Auto-select distillation layers
        self.distillation_layers = self._select_distillation_layers()

    def _select_distillation_layers(self) -> List[Tuple[str, str]]:
        """
        Automatically select which layers to distill based on contributions.

        Selection criteria:
        1. High CKA score → semantically important representations
        2. High gradient norm → actively learning features
        3. Balanced depth → avoid only early or late layers

        Returns:
            List of (teacher_layer, student_layer) pairs
        """
        # Compute composite importance score
        layer_scores = {}

        for layer_name, metrics in self.contributions.items():
            score = 0.0

            # Prefer layers with high CKA (if available)
            if "cka_score" in metrics:
                score += 2.0 * metrics["cka_score"]

            # Prefer layers with high gradient norm
            if "gradient_norm" in metrics:
                # Normalize by max
                max_grad = max(m.get("gradient_norm", 0) for m in self.contributions.values())
                if max_grad > 0:
                    score += metrics["gradient_norm"] / max_grad

            # Prefer layers with high effective rank (rich representations)
            if "effective_rank" in metrics:
                max_rank = max(m.get("effective_rank", 0) for m in self.contributions.values())
                if max_rank > 0:
                    score += metrics["effective_rank"] / max_rank

            layer_scores[layer_name] = score

        # Sort by score and select top-k
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)

        top_layers = [name for name, _ in sorted_layers[: self.config.top_k_layers]]

        # Try to match teacher layers to student layers
        # Simple heuristic: match by relative depth
        layer_pairs = []
        teacher_depth = len(list(self.teacher.modules()))
        student_depth = len(list(self.student.modules()))

        teacher_layers = list(self.teacher.named_modules())
        student_layers = list(self.student.named_modules())

        for teacher_layer_name in top_layers:
            # Find teacher layer index
            teacher_idx = None
            for idx, (name, _) in enumerate(teacher_layers):
                if name == teacher_layer_name:
                    teacher_idx = idx
                    break

            if teacher_idx is None:
                continue

            # Map to corresponding student layer by relative position
            relative_pos = teacher_idx / teacher_depth
            student_idx = int(relative_pos * student_depth)
            student_idx = min(student_idx, len(student_layers) - 1)

            student_layer_name = student_layers[student_idx][0]
            layer_pairs.append((teacher_layer_name, student_layer_name))

        return layer_pairs

    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations"""

        def get_activation(name, storage):
            def hook(module, input, output):
                storage[name] = output.detach()

            return hook

        # Register hooks for teacher
        for teacher_name, _ in self.distillation_layers:
            for name, module in self.teacher.named_modules():
                if name == teacher_name:
                    hook = module.register_forward_hook(
                        get_activation(teacher_name, self.teacher_activations)
                    )
                    self.hooks.append(hook)

        # Register hooks for student
        for _, student_name in self.distillation_layers:
            for name, module in self.student.named_modules():
                if name == student_name:
                    hook = module.register_forward_hook(
                        get_activation(student_name, self.student_activations)
                    )
                    self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.teacher_activations.clear()
        self.student_activations.clear()

    def _compute_soft_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft target distillation loss.

        L_soft = KL(P_t || P_s) · T²

        where P_t, P_s are softmax with temperature T

        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits

        Returns:
            Soft distillation loss
        """
        T = self.config.temperature

        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence loss
        loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        return loss

    def _compute_layer_loss(self) -> torch.Tensor:
        """
        Compute intermediate layer distillation loss.

        L_layer = Σ_ℓ w_ℓ · d(h_s^(ℓ), h_t^(ℓ))

        Supports multiple distance metrics:
        - MSE: ||h_s - h_t||²
        - Cosine: 1 - cos(h_s, h_t)
        - KL: KL(h_t || h_s)

        Returns:
            Intermediate layer loss
        """
        total_loss = 0.0
        num_layers = len(self.distillation_layers)

        for teacher_name, student_name in self.distillation_layers:
            if teacher_name not in self.teacher_activations:
                continue
            if student_name not in self.student_activations:
                continue

            teacher_act = self.teacher_activations[teacher_name]
            student_act = self.student_activations[student_name]

            # Handle dimension mismatch with projection
            if teacher_act.shape != student_act.shape:
                student_act = self._project_activation(student_act, teacher_act.shape)

            # Compute distance based on metric
            if self.config.distance_metric == "mse":
                loss = F.mse_loss(student_act, teacher_act)
            elif self.config.distance_metric == "cosine":
                # Flatten activations
                teacher_flat = teacher_act.view(teacher_act.size(0), -1)
                student_flat = student_act.view(student_act.size(0), -1)

                # Cosine similarity
                cos_sim = F.cosine_similarity(teacher_flat, student_flat, dim=1)
                loss = (1 - cos_sim).mean()
            elif self.config.distance_metric == "kl":
                # Normalize to probability distributions
                teacher_prob = F.softmax(teacher_act.view(teacher_act.size(0), -1), dim=1)
                student_log_prob = F.log_softmax(student_act.view(student_act.size(0), -1), dim=1)

                loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
            else:
                raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")

            total_loss += loss

        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)

    def _project_activation(
        self, student_act: torch.Tensor, target_shape: torch.Size
    ) -> torch.Tensor:
        """
        Project student activation to match teacher shape.

        Uses adaptive pooling or linear projection as needed.

        Args:
            student_act: Student activation
            target_shape: Target shape (teacher activation shape)

        Returns:
            Projected activation
        """
        if student_act.shape == target_shape:
            return student_act

        # For convolutional features (B, C, H, W)
        if len(target_shape) == 4:
            B, C, H, W = target_shape
            # Adaptive pooling for spatial dimensions
            projected = F.adaptive_avg_pool2d(student_act, (H, W))

            # Handle channel dimension mismatch
            if projected.size(1) != C:
                # Simple channel projection (could use learnable projection)
                projected = F.interpolate(projected, size=(C, H, W), mode="nearest")

            return projected

        # For sequential features (B, L, D)
        elif len(target_shape) == 3:
            B, L, D = target_shape
            # Interpolate sequence length
            if student_act.size(1) != L:
                student_act = F.interpolate(
                    student_act.transpose(1, 2), size=L, mode="linear", align_corners=False
                ).transpose(1, 2)

            # Interpolate feature dimension
            if student_act.size(2) != D:
                student_act = F.interpolate(
                    student_act.transpose(1, 2), size=D, mode="linear", align_corners=False
                ).transpose(1, 2)

            return student_act

        # For flat features (B, D)
        elif len(target_shape) == 2:
            B, D = target_shape
            if student_act.size(1) != D:
                # Linear interpolation
                student_act = F.interpolate(
                    student_act.unsqueeze(1), size=D, mode="linear", align_corners=False
                ).squeeze(1)

            return student_act

        return student_act

    def compute_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total distillation loss.

        L_total = α·L_soft + (1-α)·L_hard + β·L_layer

        Args:
            student_logits: Student model output
            teacher_logits: Teacher model output
            labels: Ground truth labels

        Returns:
            Total loss and loss components dictionary
        """
        # Soft distillation loss
        loss_soft = self._compute_soft_loss(student_logits, teacher_logits)

        # Hard target loss (cross-entropy with true labels)
        loss_hard = F.cross_entropy(student_logits, labels)

        # Intermediate layer loss
        loss_layer = self._compute_layer_loss()

        # Combine losses
        total_loss = (
            self.config.alpha * loss_soft
            + (1 - self.config.alpha) * loss_hard
            + self.config.layer_weight * loss_layer
        )

        loss_components = {
            "total": total_loss.item(),
            "soft": loss_soft.item(),
            "hard": loss_hard.item(),
            "layer": loss_layer.item(),
        }

        return total_loss, loss_components

    def train_step(
        self, inputs: torch.Tensor, labels: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one training step with distillation.

        Args:
            inputs: Input batch
            labels: Ground truth labels
            optimizer: Optimizer for student model

        Returns:
            Dictionary of loss components
        """
        # Register hooks if not already registered
        if not self.hooks:
            self._register_hooks()

        # Set teacher to eval mode
        self.teacher.eval()
        self.student.train()

        # Forward pass through teacher (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        # Forward pass through student
        student_logits = self.student(inputs)

        # Compute distillation loss
        loss, loss_components = self.compute_loss(student_logits, teacher_logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_components

    def get_distillation_info(self) -> Dict:
        """
        Get information about distillation configuration.

        Returns:
            Dictionary with distillation details
        """
        return {
            "temperature": self.config.temperature,
            "alpha": self.config.alpha,
            "layer_weight": self.config.layer_weight,
            "distance_metric": self.config.distance_metric,
            "num_distillation_layers": len(self.distillation_layers),
            "distillation_layers": self.distillation_layers,
        }

    def __del__(self):
        """Clean up hooks when object is destroyed"""
        self._remove_hooks()
