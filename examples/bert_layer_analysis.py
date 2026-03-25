"""
BERT Layer Contribution Analysis with Uni-Layer.

Analyzes which transformer layers in a BERT-like model contribute most
to text classification, useful for layer pruning and LoRA target selection.

Requirements:
    pip install uni-layer
    pip install transformers  # optional, for real BERT
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
    MutualInformation,
    AttentionFlow,
)


class BertSelfAttention(nn.Module):
    """Simplified BERT self-attention."""

    def __init__(self, hidden_size=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, S, D = x.shape
        q = self.query(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class BertLayer(nn.Module):
    """Simplified BERT transformer layer."""

    def __init__(self, hidden_size=256, num_heads=4, intermediate_size=1024):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm1(x + self.attention(x))
        h = self.norm2(h + self.output(self.act(self.intermediate(h))))
        return h


class MiniBERT(nn.Module):
    """Minimal BERT-like model for text classification."""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=6,
                 num_heads=4, max_seq_len=64, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        # Pool CLS token
        cls = torch.tanh(self.pooler(x[:, 0]))
        return self.classifier(cls)


def make_dummy_text_data(num_samples=256, seq_len=32, vocab_size=1000, num_classes=4):
    """Create synthetic tokenized text data."""
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(input_ids, labels), batch_size=32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniBERT(num_layers=6, num_heads=4).to(device)
    data_loader = make_dummy_text_data()

    # --- Analysis ---
    analyzer = LayerAnalyzer(model, task_type="classification")

    metrics = [
        GradientNorm(num_batches=3),
        CKA(num_batches=3),
        EffectiveRank(num_batches=3),
        FisherInformation(num_batches=3),
    ]

    print("Running layer contribution analysis on MiniBERT (6 layers, 4 heads)...")
    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        device=device,
    )

    # --- Per-layer results ---
    print("\n=== BERT Layer Contributions ===")
    print(f"{'Layer':<40} {'GradNorm':>10} {'CKA':>8} {'EffRank':>10} {'Fisher':>10}")
    print("-" * 82)
    for name, scores in contributions.items():
        gn = scores.get("gradient_norm", 0.0)
        cka = scores.get("cka_score", 0.0)
        er = scores.get("effective_rank", 0.0)
        fi = scores.get("fisher_information", 0.0)
        if gn or cka or er or fi:
            print(f"{name:<40} {gn:>10.4f} {cka:>8.4f} {er:>10.2f} {fi:>10.6f}")

    # --- Layer ranking for LoRA target selection ---
    ranked = analyzer.rank_layers("gradient_norm")
    print("\n=== Recommended LoRA targets (highest gradient norm) ===")
    for i, (name, score) in enumerate(ranked[:8]):
        print(f"  {i+1}. {name}: {score:.4f}")

    # --- Mutual information analysis ---
    print("\n=== Mutual Information per layer ===")
    mi_metric = MutualInformation(num_batches=3)
    for i, layer in enumerate(model.layers):
        result = mi_metric.compute(
            model=model, layer=layer,
            layer_name=f"layers.{i}", layer_idx=i,
            data_loader=data_loader, device=device,
        )
        mi = result.get("mutual_information", 0.0)
        print(f"  Layer {i}: MI = {mi:.4f}")

    # --- Integration example: LoRA configuration ---
    print("\n=== PEFT Integration ===")
    try:
        from uni_layer.integrations import HuggingFacePEFTBridge

        bridge = HuggingFacePEFTBridge(model, contributions)
        lora_params = bridge.recommend_lora_config_params(
            metric_name="gradient_norm", base_rank=8,
        )
        print("Recommended LoRA config:")
        for k, v in lora_params.items():
            print(f"  {k}: {v}")
    except ImportError:
        print("(Install peft for full integration)")


if __name__ == "__main__":
    main()
