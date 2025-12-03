"""
Example: Parameter-Efficient Fine-Tuning (PEFT) with optimal layer selection.

This script demonstrates how to use the PEFTOptimizer to add LoRA or Adapter
layers to a model, with adaptive rank/capacity based on layer contribution analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, FisherInformation, EffectiveRank
from uni_layer.compression import PEFTOptimizer, AdapterConfig


# Define a large pre-trained model (simulated)
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(784, 512)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(8)
        ])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


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
    print("Parameter-Efficient Fine-Tuning (PEFT) Example")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ========== Step 1: Create Pre-trained Model ==========
    print("\n[1/7] Loading pre-trained model...")

    model = LargeModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✓ Model loaded with {total_params:,} parameters")

    # Create dataset for fine-tuning
    X_train = torch.randn(1000, 784)
    y_train = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_test = torch.randn(200, 784)
    y_test = torch.randint(0, 10, (200,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # ========== Step 2: Analyze Layer Contributions ==========
    print("\n[2/7] Analyzing layer contributions...")

    analyzer = LayerAnalyzer(model, task_type="classification", device=device)

    contributions = analyzer.compute_metrics(
        metrics=[
            GradientNorm(num_batches=10),
            FisherInformation(num_batches=10),
            EffectiveRank(num_batches=10),
        ],
        data_loader=train_loader,
        verbose=False
    )

    print("✓ Layer analysis complete")

    # Display layer contributions
    print("\nLayer Importance (Gradient Norm):")
    grad_scores = [
        (name, m.get('gradient_norm', 0))
        for name, m in contributions.items()
        if 'gradient_norm' in m
    ]
    grad_scores.sort(key=lambda x: x[1], reverse=True)

    for name, score in grad_scores[:5]:
        print(f"  {name:30s}: {score:.4f}")

    # ========== Step 3: Setup PEFT Optimizer ==========
    print("\n[3/7] Setting up PEFT optimizer...")

    peft_config = AdapterConfig(
        method="lora",
        rank=8,
        alpha=16.0,
        dropout=0.1,
        adaptive_rank=True
    )

    peft_optimizer = PEFTOptimizer(
        model=model,
        contributions=contributions,
        config=peft_config
    )

    # ========== Step 4: Select Layers and Compute Ranks ==========
    print("\n[4/7] Selecting layers for PEFT...")

    # Select top-k important layers
    selected_layers = peft_optimizer.select_layers(
        top_k=6,
        metric_name="gradient_norm",
        min_contribution=0.01
    )

    print(f"✓ Selected {len(selected_layers)} layers for PEFT:")
    for layer in selected_layers:
        print(f"  - {layer}")

    # Compute adaptive ranks
    ranks = peft_optimizer.compute_adaptive_ranks(
        selected_layers,
        base_rank=8,
        max_rank=32,
        metric_name="gradient_norm"
    )

    print("\n✓ Adaptive ranks computed:")
    for layer, rank in ranks.items():
        contribution = contributions[layer].get('gradient_norm', 0)
        print(f"  {layer:30s}: rank={rank:2d} (contrib={contribution:.4f})")

    # ========== Step 5: Inject LoRA Layers ==========
    print("\n[5/7] Injecting LoRA layers...")

    model_with_lora = peft_optimizer.inject_lora(
        selected_layers=selected_layers,
        ranks=ranks
    )

    # Get parameter efficiency stats
    efficiency = peft_optimizer.get_parameter_efficiency()

    print(f"✓ LoRA layers injected")
    print(f"  Total parameters: {efficiency['total_params']:,}")
    print(f"  Trainable parameters: {efficiency['trainable_params']:,}")
    print(f"  Frozen parameters: {efficiency['frozen_params']:,}")
    print(f"  Parameter efficiency: {efficiency['efficiency']:.2%}")
    print(f"  Reduction ratio: {efficiency['reduction_ratio']:.2f}x")

    # ========== Step 6: Fine-tune with PEFT ==========
    print("\n[6/7] Fine-tuning with PEFT...")

    # Only optimize trainable (LoRA) parameters
    trainable_params = [p for p in model_with_lora.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.001)

    print(f"  Training {len(trainable_params)} parameter groups...")

    model_with_lora.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_with_lora(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            accuracy = evaluate_model(model_with_lora, test_loader, device)
            print(f"  Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.4f}, "
                  f"Accuracy: {accuracy:.2%}")

    final_accuracy = evaluate_model(model_with_lora, test_loader, device)
    print(f"✓ Fine-tuning complete, final accuracy: {final_accuracy:.2%}")

    # ========== Step 7: Compare with Full Fine-tuning ==========
    print("\n[7/7] Comparing with full fine-tuning (baseline)...")

    # Full fine-tuning (all parameters)
    baseline_model = LargeModel().to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

    baseline_model.train()
    for epoch in range(10):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            baseline_optimizer.zero_grad()
            outputs = baseline_model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            baseline_optimizer.step()

    baseline_accuracy = evaluate_model(baseline_model, test_loader, device)
    print(f"✓ Baseline accuracy: {baseline_accuracy:.2%}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("PEFT Summary")
    print("=" * 70)

    print(f"\nOriginal Model:")
    print(f"  Total parameters: {total_params:,}")

    print(f"\nFull Fine-tuning (Baseline):")
    print(f"  Trainable parameters: {total_params:,} (100%)")
    print(f"  Final accuracy: {baseline_accuracy:.2%}")

    print(f"\nPEFT with LoRA:")
    print(f"  Trainable parameters: {efficiency['trainable_params']:,} "
          f"({efficiency['efficiency']:.2%})")
    print(f"  Parameter reduction: {efficiency['reduction_ratio']:.1f}x")
    print(f"  Final accuracy: {final_accuracy:.2%}")

    print(f"\n📊 Accuracy comparison:")
    print(f"  PEFT vs Full FT: {(final_accuracy / baseline_accuracy):.2%} of full accuracy")
    print(f"  Parameter efficiency: {efficiency['reduction_ratio']:.1f}x fewer trainable params")

    # Display PEFT info
    peft_info = peft_optimizer.get_peft_info()
    print(f"\n💡 PEFT Configuration:")
    print(f"  Method: {peft_info['method'].upper()}")
    print(f"  Layers augmented: {peft_info['num_layers']}")
    print(f"  Average rank: {peft_info['avg_rank']:.1f}")
    print(f"  Adaptive ranks: {list(peft_info['ranks'].values())}")

    print("\n💡 Advantages of PEFT:")
    print("  • 10-100x fewer trainable parameters")
    print("  • Faster training and lower memory usage")
    print("  • Easy to swap different task adaptations")
    print("  • Comparable accuracy to full fine-tuning")
    print("  • Layer-adaptive ranks optimize capacity allocation")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
