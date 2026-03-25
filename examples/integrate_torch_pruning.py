"""
Example: Using Uni-Layer with Torch-Pruning for metrics-driven structural pruning.

This example shows how Uni-Layer's layer contribution analysis can improve
Torch-Pruning's pruning decisions: instead of uniform or magnitude-based
pruning, we use principled metrics (gradient norm, Fisher information, etc.)
to determine how much to prune each layer.

Requirements:
    pip install uni-layer torch-pruning

Result: Important layers are preserved; unimportant layers are pruned aggressively.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Step 1: Define or load your model ---

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    model = SimpleClassifier()
    model.eval()

    # Dummy data for analysis
    X = torch.randn(200, 512)
    y = torch.randint(0, 10, (200,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # --- Step 2: Analyze with Uni-Layer ---
    from uni_layer import LayerAnalyzer
    from uni_layer.metrics import GradientNorm, FisherInformation

    analyzer = LayerAnalyzer(model, task_type='classification')
    contributions = analyzer.compute_metrics(
        metrics=[GradientNorm(), FisherInformation()],
        data_loader=loader,
    )

    # View the analysis
    print("\n=== Layer Contribution Analysis ===")
    for name, metrics in contributions.items():
        gn = metrics.get('gradient_norm', 'N/A')
        fi = metrics.get('fisher_information', 'N/A')
        print(f"  {name}: grad_norm={gn:.4f}, fisher={fi:.4f}"
              if isinstance(gn, float) else f"  {name}: {metrics.get('layer_type')}")

    # --- Step 3: Bridge to Torch-Pruning ---
    from uni_layer.integrations import TorchPruningBridge

    bridge = TorchPruningBridge(model, contributions)

    # Get per-layer pruning ratios
    print("\n=== Pruning Recommendations ===")
    print(bridge.summary('gradient_norm'))

    # Protected layers (too important to prune)
    protected = bridge.get_protected_layers('gradient_norm', top_k=2)
    print(f"\nProtected layers: {protected}")

    # Get pruning ratios for Torch-Pruning
    pruning_ratios = bridge.as_layer_pruning_ratios(
        metric_name='gradient_norm',
        target_sparsity=0.4,
        max_ratio=0.7,
    )
    print(f"\nPruning ratios (module -> ratio):")
    for module, ratio in pruning_ratios.items():
        print(f"  {module.__class__.__name__}: {ratio:.2%}")

    # --- Step 4: Apply with Torch-Pruning (if installed) ---
    try:
        import torch_pruning as tp

        example_inputs = torch.randn(1, 512)
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(),
            pruning_ratio_dict=pruning_ratios,
        )
        pruner.step()

        # Compare model sizes
        original_params = sum(p.numel() for p in SimpleClassifier().parameters())
        pruned_params = sum(p.numel() for p in model.parameters())
        print(f"\nOriginal params: {original_params:,}")
        print(f"Pruned params:   {pruned_params:,}")
        print(f"Reduction:       {1 - pruned_params/original_params:.1%}")

    except ImportError:
        print("\n[Torch-Pruning not installed]")
        print("Install with: pip install torch-pruning")
        print("The pruning_ratios dict above can be passed directly to")
        print("tp.pruner.MetaPruner(pruning_ratio_dict=pruning_ratios)")


if __name__ == "__main__":
    main()
