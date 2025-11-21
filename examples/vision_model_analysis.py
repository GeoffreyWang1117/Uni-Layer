"""
Example of analyzing Vision models (CNN/ViT) with Uni-Layer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import (
    GradientNorm,
    EffectiveRank,
    DropLayerRobustness,
)


# Simple CNN for image classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x


def main():
    print("=" * 60)
    print("Uni-Layer: Vision Model (CNN) Analysis")
    print("=" * 60)

    # Create dummy image classification data
    print("\n[1/4] Creating dummy image dataset...")
    num_samples = 500
    image_size = 32
    num_classes = 10

    # Random images (CIFAR-10 style)
    X = torch.randn(num_samples, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize CNN model
    print("[2/4] Initializing CNN model...")
    model = SimpleCNN(num_classes=num_classes)

    # Initialize LayerAnalyzer
    print("[3/4] Initializing LayerAnalyzer...")
    analyzer = LayerAnalyzer(
        model=model,
        task_type="classification",
        device="cpu"
    )

    print(f"\nFound {len(analyzer.layers)} layers:")
    layer_type_counts = {}
    for layer_type in analyzer.layer_types.values():
        layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0) + 1

    for layer_type, count in layer_type_counts.items():
        print(f"  • {layer_type}: {count} layers")

    # Compute metrics
    print("\n[4/4] Computing metrics for CNN layers...")

    metrics = [
        GradientNorm(num_batches=5),
        EffectiveRank(num_batches=5),
        DropLayerRobustness(num_batches=3, metric="accuracy"),
    ]

    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        verbose=True
    )

    # Analyze by layer type
    print("\n" + "=" * 60)
    print("Analysis by Layer Type:")
    print("=" * 60)

    conv_layers = {}
    linear_layers = {}
    norm_layers = {}

    for layer_name, metrics in contributions.items():
        layer_type = metrics.get("layer_type", "unknown")
        if layer_type == "convolution":
            conv_layers[layer_name] = metrics
        elif layer_type == "linear":
            linear_layers[layer_name] = metrics
        elif layer_type == "normalization":
            norm_layers[layer_name] = metrics

    print(f"\n🔷 Convolutional Layers: {len(conv_layers)}")
    if conv_layers:
        avg_grad = sum(m.get("gradient_norm", 0) for m in conv_layers.values()) / len(conv_layers)
        avg_rank = sum(m.get("effective_rank", 0) for m in conv_layers.values()) / len(conv_layers)
        print(f"   Avg Gradient Norm: {avg_grad:.4f}")
        print(f"   Avg Effective Rank: {avg_rank:.2f}")

    print(f"\n🔶 Linear Layers: {len(linear_layers)}")
    if linear_layers:
        avg_grad = sum(m.get("gradient_norm", 0) for m in linear_layers.values()) / len(linear_layers)
        avg_rank = sum(m.get("effective_rank", 0) for m in linear_layers.values()) / len(linear_layers)
        print(f"   Avg Gradient Norm: {avg_grad:.4f}")
        print(f"   Avg Effective Rank: {avg_rank:.2f}")

    # Identify critical layers
    print("\n" + "=" * 60)
    print("Most Critical Layers (by DropLayer impact):")
    print("=" * 60)

    if any("droplayer_loss_increase" in m for m in contributions.values()):
        droplayer_rankings = analyzer.rank_layers(contributions, "droplayer_loss_increase")

        for i, (layer_name, value) in enumerate(droplayer_rankings[:5], 1):
            layer_type = contributions[layer_name].get("layer_type", "unknown")
            print(f"{i}. {layer_name}")
            print(f"   Type: {layer_type}, Loss Increase: {value:.4f}")

    # Depth analysis
    print("\n" + "=" * 60)
    print("Layer Importance by Depth:")
    print("=" * 60)

    depth_stats = analyzer.aggregate_by_depth(
        contributions,
        metric_name="gradient_norm",
        num_bins=4
    )

    for depth_bin, avg_value in depth_stats.items():
        print(f"  {depth_bin}: {avg_value:.4f}")

    # Pruning recommendations
    print("\n" + "=" * 60)
    print("Pruning Recommendations (40% overall):")
    print("=" * 60)

    pruning_strategy = analyzer.get_pruning_strategy(
        contributions,
        metric_name="gradient_norm",
        prune_ratio=0.4
    )

    conv_prune_ratios = [(name, ratio) for name, ratio in pruning_strategy.items()
                          if contributions[name].get("layer_type") == "convolution"]

    print("\nConvolutional layer pruning ratios:")
    for name, ratio in sorted(conv_prune_ratios, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {ratio:.2%}")

    print("\n✅ Vision model analysis complete!")


if __name__ == "__main__":
    main()
