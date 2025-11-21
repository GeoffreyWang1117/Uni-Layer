"""
Example of analyzing Transformer models with Uni-Layer.

This script shows how to analyze BERT-like transformer models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import (
    GradientNorm,
    CKA,
    EffectiveRank,
    FisherInformation,
)


# Simple Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward
        ff_output = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, num_layers=6, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        # Pool: take mean over sequence
        x = x.mean(dim=1)

        return self.classifier(x)


def main():
    print("=" * 60)
    print("Uni-Layer: Transformer Model Analysis")
    print("=" * 60)

    # Create dummy sequence classification data
    print("\n[1/4] Creating dummy dataset...")
    num_samples = 500
    seq_length = 128
    vocab_size = 10000
    num_classes = 2

    # Random token sequences
    X = torch.randint(0, vocab_size, (num_samples, seq_length))
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize Transformer model
    print("[2/4] Initializing Transformer model...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_classes=num_classes
    )

    # Initialize LayerAnalyzer
    print("[3/4] Initializing LayerAnalyzer...")
    analyzer = LayerAnalyzer(
        model=model,
        task_type="classification",
        device="cpu"
    )

    print(f"\nFound {len(analyzer.layers)} layers:")
    for name, layer_type in list(analyzer.layer_types.items())[:10]:
        print(f"  • {name}: {layer_type}")

    # Compute metrics
    print("\n[4/4] Computing metrics for Transformer layers...")

    metrics = [
        GradientNorm(num_batches=3),
        CKA(num_batches=3),
        EffectiveRank(num_batches=3),
        FisherInformation(num_batches=3),
    ]

    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        verbose=True
    )

    # Analyze results
    print("\n" + "=" * 60)
    print("Transformer Layer Analysis:")
    print("=" * 60)

    # Group layers by type
    attention_layers = {}
    feedforward_layers = {}

    for layer_name, metrics in contributions.items():
        if "self_attn" in layer_name:
            attention_layers[layer_name] = metrics
        elif "linear" in layer_name:
            feedforward_layers[layer_name] = metrics

    print(f"\n📊 Attention Layers: {len(attention_layers)}")
    print(f"📊 Feedforward Layers: {len(feedforward_layers)}")

    # Compare attention vs feedforward importance
    if attention_layers and feedforward_layers:
        attn_grad_norms = [m.get("gradient_norm", 0) for m in attention_layers.values()]
        ff_grad_norms = [m.get("gradient_norm", 0) for m in feedforward_layers.values()]

        print(f"\nAverage Gradient Norm:")
        print(f"  Attention layers: {sum(attn_grad_norms)/len(attn_grad_norms):.4f}")
        print(f"  Feedforward layers: {sum(ff_grad_norms)/len(ff_grad_norms):.4f}")

    # Identify critical transformer blocks
    print("\n" + "=" * 60)
    print("Most Important Transformer Blocks:")
    print("=" * 60)

    rankings = analyzer.rank_layers(contributions, "gradient_norm")
    for i, (layer_name, value) in enumerate(rankings[:5], 1):
        layer_type = contributions[layer_name].get("layer_type", "unknown")
        print(f"{i}. {layer_name}")
        print(f"   Type: {layer_type}, Gradient Norm: {value:.4f}")

    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)

    peft_layers = analyzer.get_peft_insertion_points(
        contributions,
        metric_name="gradient_norm",
        num_adapters=3
    )

    print("\n🎯 Recommended adapter insertion points for PEFT:")
    for layer_name in peft_layers:
        print(f"  • {layer_name}")

    print("\n✅ Transformer analysis complete!")


if __name__ == "__main__":
    main()
