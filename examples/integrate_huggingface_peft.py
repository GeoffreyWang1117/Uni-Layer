"""
Example: Using Uni-Layer with HuggingFace PEFT for optimal LoRA placement.

Instead of applying LoRA to all attention layers uniformly, Uni-Layer identifies
which layers contribute most to your task, then recommends:
1. Which modules to target (important layers get LoRA)
2. What rank each adapter should have (adaptive rank allocation)

Requirements:
    pip install uni-layer peft transformers

Result: Better performance with fewer trainable parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransformerBlock(nn.Module):
    """Simplified transformer block for demonstration."""
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, dim=256, depth=6, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


def main():
    model = SimpleTransformer(dim=256, depth=6)

    # Dummy data
    X = torch.randn(200, 16, 256)  # (batch, seq_len, dim)
    y = torch.randint(0, 10, (200,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # --- Step 1: Analyze with Uni-Layer ---
    from uni_layer import LayerAnalyzer
    from uni_layer.metrics import GradientNorm, FisherInformation

    analyzer = LayerAnalyzer(model, task_type='classification')
    contributions = analyzer.compute_metrics(
        metrics=[GradientNorm(), FisherInformation()],
        data_loader=loader,
    )

    # --- Step 2: Get PEFT recommendations ---
    from uni_layer.integrations import HuggingFacePEFTBridge

    bridge = HuggingFacePEFTBridge(model, contributions)

    # Which modules should get LoRA?
    target_modules = bridge.recommend_target_modules(
        metric_name='gradient_norm',
        top_k=8,
        module_types=["Linear"],
    )
    print("=== Recommended LoRA Targets ===")
    for name in target_modules:
        print(f"  {name}")

    # Adaptive rank per layer
    ranks = bridge.recommend_adaptive_ranks(
        metric_name='gradient_norm',
        base_rank=4,
        max_rank=32,
        target_modules=target_modules,
    )
    print("\n=== Adaptive Rank Allocation ===")
    for name, rank in ranks.items():
        print(f"  {name}: rank={rank}")

    # Ready-to-use config
    config_params = bridge.recommend_lora_config_params(
        metric_name='gradient_norm',
        top_k=8,
    )
    print(f"\n=== LoraConfig Parameters ===")
    for k, v in config_params.items():
        print(f"  {k}: {v}")

    # Full summary
    print(f"\n{bridge.summary('gradient_norm')}")

    # --- Step 3: Apply with HuggingFace PEFT (if installed) ---
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(**config_params)
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

    except ImportError:
        print("\n[HuggingFace PEFT not installed]")
        print("Install with: pip install peft")
        print("The config_params dict above can be passed directly to LoraConfig(**params)")


if __name__ == "__main__":
    main()
