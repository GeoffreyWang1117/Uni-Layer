"""
Example: Generating automated reports.

This script demonstrates how to create comprehensive HTML/Markdown
reports from layer contribution analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, EffectiveRank, ActivationEntropy
from uni_layer.utils.report import ReportGenerator


def main():
    print("="*60)
    print("Automated Report Generation Example")
    print("="*60)

    # Create a simple model
    print("\n[1/4] Creating model...")
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
    )

    # Create data
    print("[2/4] Creating dataset...")
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32)

    # Analyze
    print("[3/4] Running layer analysis...")
    analyzer = LayerAnalyzer(model, task_type="classification", device="cpu")

    contributions = analyzer.compute_metrics(
        metrics=[
            GradientNorm(num_batches=5),
            CKA(num_batches=5),
            EffectiveRank(num_batches=5),
            ActivationEntropy(num_batches=5),
        ],
        data_loader=data_loader,
        verbose=True
    )

    # Generate reports
    print("\n[4/4] Generating reports...")

    generator = ReportGenerator(contributions, model_name="Example-MLP")

    # HTML report
    html_path = generator.generate_html("layer_analysis_report.html")
    print(f"✓ HTML report: {html_path}")

    # Markdown report
    md_path = generator.generate_markdown("layer_analysis_report.md")
    print(f"✓ Markdown report: {md_path}")

    print("\n" + "="*60)
    print("Reports Generated Successfully!")
    print("="*60)
    print(f"\nView HTML report: file://{html_path}")
    print(f"View Markdown report: {md_path}")


if __name__ == "__main__":
    main()
