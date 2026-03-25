"""
ResNet Layer Contribution Analysis with Uni-Layer.

Analyzes which layers in a ResNet model contribute most to predictions,
useful for guiding pruning, distillation, or fine-tuning decisions.

Requirements:
    pip install uni-layer torchvision
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Optional: use real torchvision ResNet if available
try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from uni_layer import LayerAnalyzer
from uni_layer.metrics import (
    GradientNorm,
    CKA,
    EffectiveRank,
    FisherInformation,
    DropLayerRobustness,
)


def build_resnet():
    """Load a ResNet-18 model."""
    if HAS_TORCHVISION:
        model = resnet18(weights=None, num_classes=10)
    else:
        # Minimal residual block for demonstration
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
    return model


def make_dummy_data(num_samples=128, img_size=32):
    """Create synthetic image classification data."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=32, shuffle=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet().to(device)
    data_loader = make_dummy_data()

    # --- Layer Contribution Analysis ---
    analyzer = LayerAnalyzer(model, task_type="classification")

    metrics = [
        GradientNorm(num_batches=3),
        CKA(num_batches=3),
        EffectiveRank(num_batches=3),
        FisherInformation(num_batches=3),
    ]

    print("Running layer contribution analysis on ResNet-18...")
    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        device=device,
    )

    # --- Results ---
    print("\n=== Layer Contribution Scores ===")
    print(f"{'Layer':<40} {'GradNorm':>10} {'CKA':>8} {'EffRank':>10} {'Fisher':>10}")
    print("-" * 82)
    for layer_name, scores in contributions.items():
        gn = scores.get("gradient_norm", 0.0)
        cka = scores.get("cka_score", 0.0)
        er = scores.get("effective_rank", 0.0)
        fi = scores.get("fisher_information", 0.0)
        if gn or cka or er or fi:
            print(f"{layer_name:<40} {gn:>10.4f} {cka:>8.4f} {er:>10.2f} {fi:>10.6f}")

    # --- Rank layers for pruning ---
    ranked = analyzer.rank_layers(metric_name="gradient_norm")
    print("\n=== Layers ranked by gradient norm (pruning candidates) ===")
    for i, (name, score) in enumerate(ranked[:5]):
        print(f"  {i+1}. {name}: {score:.4f}")

    bottom = ranked[-5:]
    print("\nLeast important layers (prune first):")
    for name, score in reversed(bottom):
        print(f"  - {name}: {score:.4f}")


if __name__ == "__main__":
    main()
