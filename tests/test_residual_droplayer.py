"""
Tests for Residual-aware DropLayer metric (v0.4.0 Feature 2).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    """Simple residual block for testing."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.linear(x))


class ResidualModel(nn.Module):
    """Model with residual connections for testing."""

    def __init__(self, dim=10, num_blocks=3, num_classes=5):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@pytest.fixture
def residual_model():
    return ResidualModel(dim=10, num_blocks=3, num_classes=5)


@pytest.fixture
def simple_model():
    """Non-residual model (shape-changing layers)."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5),
    )


@pytest.fixture
def sample_data():
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


class TestResidualDropLayer:
    def test_import(self):
        from uni_layer.metrics import ResidualDropLayer

        assert ResidualDropLayer is not None

    def test_initialization(self):
        from uni_layer.metrics import ResidualDropLayer

        m = ResidualDropLayer()
        assert m.name == "residual_droplayer"
        assert m.category == "robustness"
        assert m.requires_gradient is False
        assert m.requires_data is True

    def test_compute_residual_model(self, residual_model, sample_data, criterion):
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=3)

        layer = residual_model.blocks[0]
        result = m.compute(
            model=residual_model,
            layer=layer,
            layer_name="blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert "residual_droplayer_loss_increase" in result
        assert "residual_droplayer_loss_ratio" in result
        assert "residual_ratio" in result
        assert "transform_norm_ratio" in result
        assert isinstance(result["residual_droplayer_loss_increase"], float)
        assert isinstance(result["residual_ratio"], float)

    def test_residual_ratio_range(self, residual_model, sample_data, criterion):
        """Residual ratio (cosine sim) should be in [-1, 1]."""
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=3)

        result = m.compute(
            model=residual_model,
            layer=residual_model.blocks[0],
            layer_name="blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert -1.0 - 1e-6 <= result["residual_ratio"] <= 1.0 + 1e-6

    def test_transform_norm_ratio_range(self, residual_model, sample_data, criterion):
        """Transform norm ratio should be non-negative."""
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=3)

        result = m.compute(
            model=residual_model,
            layer=residual_model.blocks[1],
            layer_name="blocks.1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert result["transform_norm_ratio"] >= 0.0

    def test_compute_nonresidual_model(self, simple_model, sample_data, criterion):
        """Non-residual model: shape mismatch -> falls back to zero ablation."""
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=3)

        result = m.compute(
            model=simple_model,
            layer=simple_model[0],  # Linear(10, 20) - shape changes
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert "residual_droplayer_loss_increase" in result
        # For shape-mismatch layers, residual_ratio should be 0.0
        assert result["residual_ratio"] == 0.0
        assert result["transform_norm_ratio"] == 1.0

    def test_baseline_cache(self, residual_model, sample_data, criterion):
        """Baseline should be computed once and cached."""
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=3)

        # First call computes baseline
        m.compute(
            model=residual_model,
            layer=residual_model.blocks[0],
            layer_name="blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert id(residual_model) in ResidualDropLayer._baseline_cache

        # Second call reuses cache
        m.compute(
            model=residual_model,
            layer=residual_model.blocks[1],
            layer_name="blocks.1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

    def test_clear_cache(self):
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer._baseline_cache[12345] = {"losses": [], "accs": []}
        ResidualDropLayer.clear_baseline_cache()
        assert len(ResidualDropLayer._baseline_cache) == 0

    def test_with_accuracy_metric(self, residual_model, sample_data, criterion):
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=3, metric="accuracy")

        result = m.compute(
            model=residual_model,
            layer=residual_model.blocks[0],
            layer_name="blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert "residual_droplayer_acc_decrease" in result
        assert "residual_droplayer_acc_ratio" in result

    def test_multiple_blocks_different_importance(self, residual_model, sample_data, criterion):
        """Different blocks should have different importance scores."""
        from uni_layer.metrics import ResidualDropLayer

        ResidualDropLayer.clear_baseline_cache()
        m = ResidualDropLayer(num_batches=5)

        results = []
        for i, block in enumerate(residual_model.blocks):
            result = m.compute(
                model=residual_model,
                layer=block,
                layer_name=f"blocks.{i}",
                layer_idx=i,
                data_loader=sample_data,
                device="cpu",
                criterion=criterion,
            )
            results.append(result)

        # All should have valid results
        for r in results:
            assert "residual_droplayer_loss_increase" in r
            assert "residual_ratio" in r

    def test_schema_registration(self):
        from uni_layer.core.schema import METRIC_OUTPUT_KEYS, METRIC_PRIMARY_KEYS

        assert "residual_droplayer" in METRIC_PRIMARY_KEYS
        assert "residual_droplayer" in METRIC_OUTPUT_KEYS
        assert METRIC_PRIMARY_KEYS["residual_droplayer"] == "residual_droplayer_loss_increase"
