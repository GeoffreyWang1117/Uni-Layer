"""
Vision Transformer (ViT) Layer Contribution Analysis with Uni-Layer.

Analyzes transformer blocks and attention heads in a ViT model to
understand which layers are most important for predictions.

Requirements:
    pip install uni-layer
    pip install timm  # optional, for pre-trained ViT
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import (
    GradientNorm,
    CKA,
    EffectiveRank,
    ActivationEntropy,
    AttentionFlow,
)


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=True)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    """Minimal Vision Transformer for demonstration."""

    def __init__(self, img_size=32, patch_size=8, embed_dim=128,
                 depth=6, num_heads=4, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)


def make_dummy_data(num_samples=128, img_size=32):
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniViT(depth=6, num_heads=4).to(device)
    data_loader = make_dummy_data()

    # --- Analysis ---
    analyzer = LayerAnalyzer(model, task_type="classification")

    metrics = [
        GradientNorm(num_batches=3),
        CKA(num_batches=3),
        EffectiveRank(num_batches=3),
        ActivationEntropy(num_batches=3),
    ]

    print("Running layer contribution analysis on MiniViT (6 blocks, 4 heads)...")
    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        device=device,
    )

    # --- Per-block results ---
    print("\n=== Transformer Block Contributions ===")
    print(f"{'Block':<35} {'GradNorm':>10} {'CKA':>8} {'EffRank':>10} {'Entropy':>10}")
    print("-" * 77)
    for name, scores in contributions.items():
        gn = scores.get("gradient_norm", 0.0)
        cka = scores.get("cka_score", 0.0)
        er = scores.get("effective_rank", 0.0)
        ent = scores.get("activation_entropy", 0.0)
        if gn or cka or er or ent:
            print(f"{name:<35} {gn:>10.4f} {cka:>8.4f} {er:>10.2f} {ent:>10.4f}")

    # --- Attention head analysis ---
    print("\n=== Attention Head Analysis ===")
    attn_metric = AttentionFlow(num_batches=3)
    for i, block in enumerate(model.blocks):
        result = attn_metric.compute(
            model=model, layer=block.attn,
            layer_name=f"blocks.{i}.attn", layer_idx=i,
            data_loader=data_loader, device=device,
        )
        ent = result.get("attention_entropy", 0.0)
        div = result.get("head_diversity", 0.0)
        maxw = result.get("attention_max_weight", 0.0)
        print(f"  Block {i}: entropy={ent:.4f}, head_diversity={div:.4f}, max_weight={maxw:.4f}")

    # --- Layer ranking ---
    ranked = analyzer.rank_layers("gradient_norm")
    print("\n=== Top-5 most important layers ===")
    for i, (name, score) in enumerate(ranked[:5]):
        print(f"  {i+1}. {name}: {score:.4f}")


if __name__ == "__main__":
    main()
