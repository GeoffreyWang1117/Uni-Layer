"""
Example: Model pruning based on layer contributions.

This script demonstrates how to use the LayerPruner to automatically
prune neural networks based on layer importance analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, FisherInformation
from uni_layer.compression import LayerPruner, PruningStrategy


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main():
    print("=" * 70)
    print("Model Pruning Example - Based on Layer Contributions")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ========== Step 1: Create Model and Data ==========
    print("\n[1/6] Creating model and dataset...")

    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    # Create dummy dataset
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # ========== Step 2: Analyze Layer Contributions ==========
    print("\n[2/6] Analyzing layer contributions...")

    analyzer = LayerAnalyzer(model, task_type="classification", device=device)

    contributions = analyzer.compute_metrics(
        metrics=[
            GradientNorm(num_batches=10),
            FisherInformation(num_batches=10),
        ],
        data_loader=data_loader,
        verbose=False
    )

    print("✓ Layer analysis complete")

    # Display contributions
    print("\nLayer Contributions:")
    for layer_name, metrics in contributions.items():
        if 'gradient_norm' in metrics:
            print(f"  {layer_name:20s}: GradNorm={metrics['gradient_norm']:.4f}")

    # ========== Step 3: Evaluate Original Model ==========
    print("\n[3/6] Evaluating original model...")

    original_accuracy = evaluate_model(model, data_loader, device)
    print(f"✓ Original accuracy: {original_accuracy:.2%}")

    # ========== Step 4: Compute Pruning Strategy ==========
    print("\n[4/6] Computing pruning strategy...")

    pruner = LayerPruner(
        model,
        contributions,
        strategy=PruningStrategy.GRADIENT_NORM
    )

    # Compute differential pruning ratios
    pruning_ratios = pruner.compute_layer_pruning_ratios(
        base_ratio=0.3,
        max_ratio=0.6,  # Low-contribution layers: prune up to 60%
        min_ratio=0.1,  # High-contribution layers: prune at least 10%
        metric_name="gradient_norm"
    )

    print("\nPruning Ratios (based on layer importance):")
    for layer_name, ratio in pruning_ratios.items():
        print(f"  {layer_name:20s}: {ratio:.1%}")

    # ========== Step 5: Apply Pruning ==========
    print("\n[5/6] Applying pruning...")

    # Unstructured pruning
    print("\n  a) Unstructured pruning (weight-level)...")
    pruned_model_unstructured = pruner.prune_unstructured(pruning_ratios)
    pruned_model_unstructured = pruner.remove_pruning_masks()

    # Get sparsity stats
    stats = pruner.get_sparsity_stats()
    print(f"     ✓ Overall sparsity: {stats['overall_sparsity']:.2%}")
    print(f"     ✓ Zero parameters: {stats['zero_params']:,} / {stats['total_params']:,}")

    # Evaluate pruned model
    pruned_accuracy_unstructured = evaluate_model(pruned_model_unstructured, data_loader, device)
    print(f"     ✓ Accuracy after pruning: {pruned_accuracy_unstructured:.2%}")
    print(f"     ✓ Accuracy drop: {(original_accuracy - pruned_accuracy_unstructured):.2%}")

    # Structured pruning
    print("\n  b) Structured pruning (neuron-level)...")
    pruner2 = LayerPruner(model, contributions, strategy=PruningStrategy.GRADIENT_NORM)
    pruned_model_structured = pruner2.prune_structured(pruning_ratios, dim=0)
    pruned_model_structured = pruner2.remove_pruning_masks()

    stats2 = pruner2.get_sparsity_stats()
    print(f"     ✓ Overall sparsity: {stats2['overall_sparsity']:.2%}")

    pruned_accuracy_structured = evaluate_model(pruned_model_structured, data_loader, device)
    print(f"     ✓ Accuracy after pruning: {pruned_accuracy_structured:.2%}")
    print(f"     ✓ Accuracy drop: {(original_accuracy - pruned_accuracy_structured):.2%}")

    # ========== Step 6: Estimate Speedup ==========
    print("\n[6/6] Estimating speedup...")

    speedup = pruner.estimate_speedup()
    print(f"  Theoretical speedup: {speedup['theoretical_speedup']:.2f}x")
    print(f"  Practical speedup (estimated): {speedup['practical_speedup']:.2f}x")
    print(f"  FLOPS reduction: {speedup['flops_reduction']:.2%}")
    print(f"  Memory reduction: {speedup['memory_reduction']:.2%}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("Pruning Summary")
    print("=" * 70)

    print(f"\nOriginal Model:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Accuracy: {original_accuracy:.2%}")

    print(f"\nPruned Model (Unstructured):")
    print(f"  Sparsity: {stats['overall_sparsity']:.2%}")
    print(f"  Accuracy: {pruned_accuracy_unstructured:.2%}")
    print(f"  Accuracy preserved: {(pruned_accuracy_unstructured / original_accuracy):.2%}")

    print(f"\nPruned Model (Structured):")
    print(f"  Sparsity: {stats2['overall_sparsity']:.2%}")
    print(f"  Accuracy: {pruned_accuracy_structured:.2%}")
    print(f"  Accuracy preserved: {(pruned_accuracy_structured / original_accuracy):.2%}")

    print("\n💡 Tips:")
    print("  • Structured pruning is hardware-friendly (actual speedup)")
    print("  • Unstructured pruning requires sparse operations support")
    print("  • Fine-tune after pruning to recover accuracy")
    print("  • Use gradual pruning for better accuracy retention")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
