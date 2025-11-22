"""
Example: Running benchmarks across multiple models.

This script demonstrates how to use the BenchmarkRunner to
systematically evaluate layer contributions across different models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer.benchmark import BenchmarkRunner
from uni_layer.metrics import GradientNorm, CKA, EffectiveRank


# Define some example models
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


def main():
    print("="*60)
    print("Uni-Layer Benchmark Example")
    print("="*60)

    # Create dummy data
    print("\n[1/4] Creating datasets...")
    X_train = torch.randn(500, 784)
    y_train = torch.randint(0, 10, (500,))
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=32)

    # Create models
    print("[2/4] Initializing models...")
    small_model = SmallMLP()
    large_model = LargeMLP()

    # Create benchmark runner
    print("[3/4] Setting up benchmark...")
    runner = BenchmarkRunner(save_dir="benchmarks", run_name="mlp_comparison")

    # Add models
    runner.add_model(
        "small_mlp",
        small_model,
        data_loader,
        task_type="classification",
        device="cpu"
    )

    runner.add_model(
        "large_mlp",
        large_model,
        data_loader,
        task_type="classification",
        device="cpu"
    )

    # Run benchmark
    print("\n[4/4] Running benchmark...")
    print("This will take a few minutes...\n")

    metrics = [
        GradientNorm(num_batches=5),
        CKA(num_batches=5),
        EffectiveRank(num_batches=5),
    ]

    results = runner.run(metrics=metrics, verbose=True)

    # Save results
    print("\nSaving results...")
    runner.save_results(format="json")

    # Generate report
    print("\nGenerating report...")
    report = runner.generate_report()
    print("\n" + report)

    # Compare models
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)

    for metric_name in ["gradient_norm", "cka_score", "effective_rank"]:
        print(f"\n{metric_name}:")
        comparison = runner.compare_models(metric_name)

        for model_name, avg_value in comparison.items():
            print(f"  {model_name}: {avg_value:.4f}")

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)
    print(f"\nResults saved to: {runner.run_dir}")


if __name__ == "__main__":
    main()
