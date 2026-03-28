"""
Tests for Diffusion Model Support (v0.4.0 Feature 6).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleResBlock(nn.Module):
    """Simplified residual block (like diffusion UNet ResBlock)."""

    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, dim)

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = torch.relu(h)
        h = self.conv(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        return x + h


class SimpleUNet(nn.Module):
    """Minimal UNet-like diffusion model for testing."""

    def __init__(self, in_ch=3, dim=32, time_dim=64, num_classes=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_blocks = nn.ModuleList(
            [
                SimpleResBlock(dim, time_dim),
                SimpleResBlock(dim, time_dim),
            ]
        )
        self.mid_block = SimpleResBlock(dim, time_dim)
        self.up_blocks = nn.ModuleList(
            [
                SimpleResBlock(dim, time_dim),
                SimpleResBlock(dim, time_dim),
            ]
        )

        self.input_conv = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.output_conv = nn.Conv2d(dim, in_ch, 3, padding=1)

    def forward(self, x, timestep):
        # timestep: (B,) long tensor
        t = timestep.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_mlp(t)

        x = self.input_conv(x)

        # Down
        for block in self.down_blocks:
            x = block(x, t_emb)

        # Mid
        x = self.mid_block(x, t_emb)

        # Up
        for block in self.up_blocks:
            x = block(x, t_emb)

        return self.output_conv(x)


@pytest.fixture
def unet_model():
    return SimpleUNet(in_ch=3, dim=32, time_dim=64)


@pytest.fixture
def sample_data():
    X = torch.randn(32, 3, 16, 16)
    y = torch.randn(32, 3, 16, 16)  # target noise
    return DataLoader(TensorDataset(X, y), batch_size=8)


@pytest.fixture
def non_diffusion_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


class TestDiffusionTimestepAnalysis:
    def test_import(self):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        assert DiffusionTimestepAnalysis is not None

    def test_initialization(self):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis()
        assert m.name == "diffusion_timestep"
        assert m.category == "architecture_specific"
        assert m.requires_gradient is False

    def test_is_diffusion_model(self, unet_model, non_diffusion_model):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis()
        assert m._is_diffusion_model(unet_model)
        assert not m._is_diffusion_model(non_diffusion_model)

    def test_get_timestep_param(self, unet_model):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis()
        param = m._get_timestep_param_name(unet_model)
        assert param == "timestep"

    def test_non_diffusion_returns_none(self, non_diffusion_model, sample_data):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis(num_batches=2)
        result = m.compute(
            model=non_diffusion_model,
            layer=non_diffusion_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["timestep_sensitivity"] is None

    def test_compute_unet_block(self, unet_model, sample_data):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis(num_timesteps=5, num_batches=2)
        block = unet_model.down_blocks[0]

        result = m.compute(
            model=unet_model,
            layer=block,
            layer_name="down_blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["timestep_sensitivity"] is not None
        assert result["mean_activation_norm"] is not None
        assert result["early_importance"] is not None
        assert result["late_importance"] is not None
        assert result["timestep_variance"] is not None

    def test_timestep_sensitivity_nonneg(self, unet_model, sample_data):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis(num_timesteps=5, num_batches=2)

        result = m.compute(
            model=unet_model,
            layer=unet_model.down_blocks[0],
            layer_name="down_blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["timestep_sensitivity"] >= 0.0

    def test_importance_sums_approximately_one(self, unet_model, sample_data):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis(num_timesteps=5, num_batches=2)

        result = m.compute(
            model=unet_model,
            layer=unet_model.mid_block,
            layer_name="mid_block",
            layer_idx=2,
            data_loader=sample_data,
            device="cpu",
        )

        # early + late should approximately sum to 1.0
        total = result["early_importance"] + result["late_importance"]
        assert abs(total - 1.0) < 0.01

    def test_multiple_blocks(self, unet_model, sample_data):
        from uni_layer.metrics import DiffusionTimestepAnalysis

        m = DiffusionTimestepAnalysis(num_timesteps=3, num_batches=2)

        blocks = [
            ("down_blocks.0", unet_model.down_blocks[0]),
            ("down_blocks.1", unet_model.down_blocks[1]),
            ("mid_block", unet_model.mid_block),
            ("up_blocks.0", unet_model.up_blocks[0]),
            ("up_blocks.1", unet_model.up_blocks[1]),
        ]

        for name, block in blocks:
            result = m.compute(
                model=unet_model,
                layer=block,
                layer_name=name,
                layer_idx=0,
                data_loader=sample_data,
                device="cpu",
            )
            assert result["timestep_sensitivity"] is not None, f"Failed for {name}"


class TestGetDiffusionBlocks:
    def test_unet_blocks(self, unet_model):
        from uni_layer.metrics import get_diffusion_blocks

        blocks = get_diffusion_blocks(unet_model)
        assert len(blocks) == 5  # 2 down + 1 mid + 2 up
        assert "down_blocks.0" in blocks
        assert "down_blocks.1" in blocks
        assert "mid_block" in blocks
        assert "up_blocks.0" in blocks
        assert "up_blocks.1" in blocks

    def test_non_unet_empty(self, non_diffusion_model):
        from uni_layer.metrics import get_diffusion_blocks

        blocks = get_diffusion_blocks(non_diffusion_model)
        assert len(blocks) == 0


class TestDiffusionLayerUtils:
    def test_identify_unet(self):
        from uni_layer.utils.layer_utils import get_architecture_family

        class UNet2DConditionModel(nn.Module):
            pass

        model = UNet2DConditionModel()
        assert get_architecture_family(model) == "diffusion_family"

    def test_schema_registration(self):
        from uni_layer.core.schema import METRIC_OUTPUT_KEYS, METRIC_PRIMARY_KEYS

        assert "diffusion_timestep" in METRIC_PRIMARY_KEYS
        assert "diffusion_timestep" in METRIC_OUTPUT_KEYS
        assert METRIC_PRIMARY_KEYS["diffusion_timestep"] == "timestep_sensitivity"
