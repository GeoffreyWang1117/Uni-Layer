"""
Tests for Mamba/SSM architecture support (v0.4.0 Feature 4).
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMambaBlock(nn.Module):
    """Minimal Mamba-like block for testing (no real SSM, just structure)."""

    def __init__(self, dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dt_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # SSM parameters (simplified)
        self.A_log = nn.Parameter(torch.randn(dim))
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq, dim)
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        # Simplified "SSM" - just linear + activation
        x_conv = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        x_ssm = x_conv * torch.sigmoid(self.dt_proj(x_conv))
        x_out = x_ssm * torch.selu(z)
        return residual + self.out_proj(x_out)


class MambaModel(nn.Module):
    """Simplified Mamba model for testing."""

    def __init__(self, dim=32, num_layers=4, vocab_size=100, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([SimpleMambaBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # pool
        return self.head(x)


class MambaModelWithBackbone(nn.Module):
    """Mamba model with backbone nesting."""

    def __init__(self, dim=32, num_layers=3, num_classes=5):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([SimpleMambaBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        for layer in self.backbone.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))


@pytest.fixture
def mamba_model():
    return MambaModel(dim=32, num_layers=4, vocab_size=100, num_classes=5)


@pytest.fixture
def mamba_backbone_model():
    return MambaModelWithBackbone(dim=32, num_layers=3, num_classes=5)


@pytest.fixture
def sample_data():
    X = torch.randint(0, 100, (64, 16))  # (batch, seq_len)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def tensor_data():
    """Float tensor data for models expecting tensor input."""
    X = torch.randn(64, 16, 32)  # (batch, seq, dim)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


class TestMambaLayerDetection:
    def test_identify_mamba_block(self):
        from uni_layer.utils.layer_utils import identify_layer_type

        block = SimpleMambaBlock(dim=32)
        layer_type = identify_layer_type(block)
        assert layer_type == "ssm_block"

    def test_identify_ssm_layer_by_params(self):
        """Layers with A_log and dt_proj should be detected as SSM."""
        from uni_layer.utils.layer_utils import identify_layer_type

        block = SimpleMambaBlock(dim=32)
        assert identify_layer_type(block) == "ssm_block"

    def test_get_model_layers_mamba(self, mamba_model):
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(mamba_model)
        assert len(layers) == 4  # 4 MambaBlocks
        for name, module in layers.items():
            assert isinstance(module, SimpleMambaBlock)

    def test_get_model_layers_nested_mamba(self, mamba_backbone_model):
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(mamba_backbone_model)
        assert len(layers) == 3

    def test_architecture_family(self):
        from uni_layer.utils.layer_utils import get_architecture_family

        class MambaForCausalLM(nn.Module):
            pass

        model = MambaForCausalLM()
        assert get_architecture_family(model) == "mamba_family"


class TestMambaMetrics:
    def test_block_influence(self, mamba_model, sample_data):
        """BlockInfluence should work on Mamba blocks."""
        from uni_layer.metrics import BlockInfluence

        m = BlockInfluence(num_batches=2)
        layer = mamba_model.layers[0]

        result = m.compute(
            model=mamba_model,
            layer=layer,
            layer_name="layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert "block_influence" in result
        assert isinstance(result["block_influence"], float)

    def test_effective_rank(self, mamba_model, sample_data):
        """EffectiveRank should work on Mamba blocks."""
        from uni_layer.metrics import EffectiveRank

        m = EffectiveRank(num_batches=2)
        layer = mamba_model.layers[0]

        result = m.compute(
            model=mamba_model,
            layer=layer,
            layer_name="layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert "effective_rank" in result
        assert isinstance(result["effective_rank"], float)

    def test_activation_entropy(self, mamba_model, sample_data):
        """ActivationEntropy should work on Mamba blocks."""
        from uni_layer.metrics import ActivationEntropy

        m = ActivationEntropy(num_batches=2)
        layer = mamba_model.layers[1]

        result = m.compute(
            model=mamba_model,
            layer=layer,
            layer_name="layers.1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )

        assert "activation_entropy" in result
        assert isinstance(result["activation_entropy"], float)

    def test_residual_droplayer(self, mamba_model, sample_data):
        """ResidualDropLayer should detect residual connections in Mamba blocks."""
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=2)

        result = m.compute(
            model=mamba_model,
            layer=mamba_model.layers[0],
            layer_name="layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )

        assert "residual_droplayer_loss_increase" in result
        # Mamba blocks have residual connections, so residual_ratio should be > 0
        assert result["residual_ratio"] > 0.0

    def test_analyzer_preset(self, mamba_model, sample_data):
        """LayerAnalyzer should work with Mamba models using presets."""
        from uni_layer.core.analyzer import LayerAnalyzer

        analyzer = LayerAnalyzer(mamba_model, task_type="classification", device="cpu")
        assert len(analyzer.layers) == 4

        contributions = analyzer.compute_metrics(
            preset="quick", data_loader=sample_data, num_batches=2, verbose=False
        )

        assert len(contributions) == 4
        for layer_name, metrics in contributions.items():
            assert "layer_type" in metrics
            assert metrics["layer_type"] == "ssm_block"

    def test_cka_similarity_matrix(self, mamba_model, sample_data):
        """CKA similarity matrix should work on Mamba models."""
        from uni_layer.core.cka_similarity import CKASimilarity

        layers = OrderedDict()
        for i, block in enumerate(mamba_model.layers):
            layers[f"layers.{i}"] = block

        cka = CKASimilarity(mamba_model, layers=layers, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=2, verbose=False)

        assert matrix.shape == (4, 4)
        assert len(names) == 4
