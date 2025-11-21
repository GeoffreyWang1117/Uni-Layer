"""
Basic usage example of Uni-Layer framework.

This script demonstrates how to:
1. Initialize the LayerAnalyzer
2. Compute layer contribution metrics
3. Visualize the results
4. Use insights for downstream tasks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import Uni-Layer components
from uni_layer import LayerAnalyzer
from uni_layer.metrics import (
    GradientNorm,
    CKA,
    EffectiveRank,
    MutualInformation,
    ActivationEntropy,
    DropLayerRobustness,
)
from uni_layer.visualization import (
    plot_layer_contributions,
    plot_contribution_heatmap,
    plot_depth_analysis,
)


# Define a simple model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], num_classes=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def main():
    print("=" * 60)
    print("Uni-Layer Framework - Basic Usage Example")
    print("=" * 60)

    # Create dummy data
    print("\n[1/5] Creating dummy dataset...")
    num_samples = 1000
    input_dim = 784
    num_classes = 10

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    print("[2/5] Initializing model...")
    model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)

    # Initialize LayerAnalyzer
    print("[3/5] Initializing LayerAnalyzer...")
    analyzer = LayerAnalyzer(
        model=model,
        task_type="classification",
        device="cpu"  # Use "cuda" if GPU available
    )

    # Compute metrics
    print("\n[4/5] Computing layer contribution metrics...")
    print("This may take a few minutes...\n")

    metrics = [
        GradientNorm(num_batches=5),
        CKA(num_batches=5),
        EffectiveRank(num_batches=5),
        ActivationEntropy(num_batches=5),
        DropLayerRobustness(num_batches=5),
    ]

    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        verbose=True
    )

    # Print results
    print("\n[5/5] Analysis Results:")
    print("=" * 60)

    for layer_name, layer_metrics in contributions.items():
        print(f"\n{layer_name}:")
        print(f"  Type: {layer_metrics.get('layer_type', 'unknown')}")
        print(f"  Gradient Norm: {layer_metrics.get('gradient_norm', 'N/A'):.4f}")
        print(f"  CKA Score: {layer_metrics.get('cka_score', 'N/A'):.4f}")
        print(f"  Effective Rank: {layer_metrics.get('effective_rank', 'N/A'):.2f}")
        print(f"  Activation Entropy: {layer_metrics.get('activation_entropy', 'N/A'):.4f}")

    # Get layer rankings
    print("\n" + "=" * 60)
    print("Layer Rankings (by Gradient Norm):")
    print("=" * 60)

    rankings = analyzer.rank_layers(contributions, "gradient_norm")
    for i, (layer_name, value) in enumerate(rankings[:5], 1):
        print(f"{i}. {layer_name}: {value:.4f}")

    # Get top layers for distillation
    print("\n" + "=" * 60)
    print("Recommended Layers for Knowledge Distillation:")
    print("=" * 60)

    distill_layers = analyzer.get_distillation_layers(
        contributions,
        metric_name="gradient_norm",
        top_k=3
    )
    for layer_name in distill_layers:
        print(f"  • {layer_name}")

    # Get pruning strategy
    print("\n" + "=" * 60)
    print("Layer-wise Pruning Strategy (30% overall):")
    print("=" * 60)

    pruning_strategy = analyzer.get_pruning_strategy(
        contributions,
        metric_name="gradient_norm",
        prune_ratio=0.3
    )

    for layer_name, prune_ratio in list(pruning_strategy.items())[:5]:
        print(f"  {layer_name}: {prune_ratio:.2%}")

    # Visualize results
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    try:
        # Plot gradient norm
        plot_layer_contributions(
            contributions,
            metric_name="gradient_norm",
            save_path="gradient_norm_analysis.png"
        )

        # Plot heatmap
        plot_contribution_heatmap(
            contributions,
            metrics=["gradient_norm", "cka_score", "effective_rank", "activation_entropy"],
            save_path="contribution_heatmap.png"
        )

        # Plot depth analysis
        plot_depth_analysis(
            contributions,
            metric_name="gradient_norm",
            num_bins=3,
            save_path="depth_analysis.png"
        )

        print("✓ Visualizations saved!")

    except Exception as e:
        print(f"⚠ Visualization error (this is OK if matplotlib is not available): {e}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
