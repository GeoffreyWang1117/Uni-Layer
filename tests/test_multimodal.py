"""Tests for Multi-modal branch analysis (v0.5.0 Feature 3)."""

import pytest
import torch
import torch.nn as nn


class MockVisionEncoder(nn.Module):
    def __init__(self, dim=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockTextEncoder(nn.Module):
    def __init__(self, dim=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockFusionModule(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class MockCLIPModel(nn.Module):
    """CLIP-like multi-modal model."""

    def __init__(self, dim=32):
        super().__init__()
        self.vision_model = MockVisionEncoder(dim, num_layers=3)
        self.text_model = MockTextEncoder(dim, num_layers=4)
        self.mm_projector = MockFusionModule(dim)

    def forward(self, image, text):
        v = self.vision_model(image)
        t = self.text_model(text)
        return self.mm_projector(v), t


class MockLLaVAModel(nn.Module):
    """LLaVA-like model."""

    def __init__(self, dim=32):
        super().__init__()
        self.vision_tower = MockVisionEncoder(dim, num_layers=2)
        self.language_model = MockTextEncoder(dim, num_layers=5)
        self.multi_modal_projector = MockFusionModule(dim)

    def forward(self, image, text):
        v = self.vision_tower(image)
        v_proj = self.multi_modal_projector(v)
        return self.language_model(v_proj + text)


class MockSingleModalModel(nn.Module):
    """Non-multi-modal model."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(32, 32)

    def forward(self, x):
        return self.encoder(x)


@pytest.fixture
def clip_model():
    return MockCLIPModel(dim=32)


@pytest.fixture
def llava_model():
    return MockLLaVAModel(dim=32)


@pytest.fixture
def single_model():
    return MockSingleModalModel()


class TestMultiModalBranchAnalyzer:
    def test_import(self):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        assert MultiModalBranchAnalyzer is not None

    def test_top_level_import(self):
        from uni_layer import MultiModalBranchAnalyzer

        assert MultiModalBranchAnalyzer is not None

    def test_clip_detection(self, clip_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(clip_model)
        assert mm.is_multimodal
        assert "vision" in mm.branch_names
        assert "language" in mm.branch_names
        assert "fusion" in mm.branch_names

    def test_llava_detection(self, llava_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(llava_model)
        assert mm.is_multimodal
        assert "vision" in mm.branch_names
        assert "language" in mm.branch_names

    def test_single_modal_not_multimodal(self, single_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(single_model)
        assert not mm.is_multimodal

    def test_get_branch_layers(self, clip_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(clip_model)
        vision_layers = mm.get_branch_layers("vision")
        language_layers = mm.get_branch_layers("language")

        assert len(vision_layers) == 3
        assert len(language_layers) == 4

    def test_get_all_layers(self, clip_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(clip_model)
        all_layers = mm.get_all_layers()

        # vision (3) + language (4) + fusion layers
        assert len(all_layers) >= 7
        # Check prefixed naming
        has_vision = any(k.startswith("vision/") for k in all_layers)
        has_language = any(k.startswith("language/") for k in all_layers)
        assert has_vision
        assert has_language

    def test_branch_summary(self, clip_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(clip_model)
        summary = mm.branch_summary()

        assert "vision" in summary
        assert "language" in summary
        assert summary["vision"]["num_layers"] == 3
        assert summary["language"]["num_layers"] == 4
        assert summary["vision"]["num_params"] > 0
        assert 0.0 < summary["vision"]["param_ratio"] < 1.0

    def test_compare_branches(self, clip_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(clip_model)

        # Mock contributions
        contrib_a = {
            "layer.0": {"effective_rank": 5.0},
            "layer.1": {"effective_rank": 3.0},
        }
        contrib_b = {
            "layer.0": {"effective_rank": 8.0},
            "layer.1": {"effective_rank": 6.0},
        }

        result = mm.compare_branches(contrib_a, contrib_b, "effective_rank")
        assert "mean_a" in result
        assert "mean_b" in result
        assert "ratio" in result
        assert result["mean_a"] == 4.0
        assert result["mean_b"] == 7.0

    def test_to_dict(self, clip_model):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(clip_model)
        d = mm.to_dict()

        assert d["is_multimodal"] is True
        assert "vision" in d["branches"]
        assert "summary" in d
        assert "layer_counts" in d

    def test_empty_branch(self):
        from uni_layer.core.multimodal import MultiModalBranchAnalyzer

        mm = MultiModalBranchAnalyzer(MockSingleModalModel())
        layers = mm.get_branch_layers("nonexistent")
        assert len(layers) == 0
