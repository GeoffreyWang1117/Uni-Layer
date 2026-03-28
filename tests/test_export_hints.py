"""Tests for ONNX/TensorRT export hints (v0.5.0 Feature 4)."""

import pytest
import torch.nn as nn


@pytest.fixture
def model_and_contributions():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5),
    )
    contributions = {
        "0": {"layer_idx": 0, "layer_type": "linear", "gradient_norm": 0.5, "block_influence": 0.1},
        "1": {
            "layer_idx": 1,
            "layer_type": "unknown",
            "gradient_norm": 0.01,
            "block_influence": 0.005,
        },
        "2": {
            "layer_idx": 2,
            "layer_type": "linear",
            "gradient_norm": 0.3,
            "block_influence": 0.08,
        },
        "3": {
            "layer_idx": 3,
            "layer_type": "unknown",
            "gradient_norm": 0.01,
            "block_influence": 0.003,
        },
        "4": {"layer_idx": 4, "layer_type": "linear", "gradient_norm": 0.8, "block_influence": 0.2},
    }
    return model, contributions


class TestExportHintsBridge:
    def test_import(self):
        from uni_layer.integrations import ExportHintsBridge

        assert ExportHintsBridge is not None

    def test_quantization_plan_fp16(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        plan = bridge.quantization_plan(
            metric_name="gradient_norm", target="fp16", protect_ratio=0.2
        )

        assert len(plan) == 5
        # Top layer (gradient_norm=0.8) should be protected as fp32
        assert plan["4"] == "fp32"
        # Low-importance layers should be fp16
        assert all(v in ("fp32", "fp16") for v in plan.values())

    def test_quantization_plan_int8(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        plan = bridge.quantization_plan(target="int8", protect_ratio=0.4)
        assert any(v == "int8" for v in plan.values())
        assert any(v == "fp32" for v in plan.values())

    def test_quantization_with_threshold(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        plan = bridge.quantization_plan(
            metric_name="gradient_norm", target="fp16", sensitivity_threshold=0.4
        )
        # Layers with gradient_norm >= 0.4 should be fp32
        assert plan["0"] == "fp32"  # 0.5
        assert plan["4"] == "fp32"  # 0.8
        assert plan["1"] == "fp16"  # 0.01

    def test_prunable_layers(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        prunable = bridge.prunable_layers(metric_name="block_influence", threshold=0.02)
        assert len(prunable) >= 1
        for p in prunable:
            assert "layer_name" in p
            assert "score" in p
            assert p["score"] < 0.02

    def test_prunable_max_ratio(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        prunable = bridge.prunable_layers(threshold=1.0, max_ratio=0.2)
        assert len(prunable) <= max(1, int(5 * 0.2))

    def test_fusion_candidates(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        candidates = bridge.fusion_candidates()
        assert isinstance(candidates, list)
        for c in candidates:
            assert "layers" in c
            assert "pattern" in c
            assert "description" in c

    def test_fusion_transformer_blocks(self):
        from uni_layer.integrations import ExportHintsBridge

        model = nn.Identity()
        contrib = {
            "block.0": {"layer_idx": 0, "layer_type": "transformer_block", "gradient_norm": 0.5},
            "block.1": {"layer_idx": 1, "layer_type": "transformer_block", "gradient_norm": 0.3},
        }
        bridge = ExportHintsBridge(model, contrib)
        candidates = bridge.fusion_candidates()
        patterns = [c["pattern"] for c in candidates]
        assert "transformer_block" in patterns

    def test_tensorrt_config(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        config = bridge.tensorrt_config()
        assert "precision_mode" in config
        assert "workspace_size_mb" in config
        assert "layer_precisions" in config
        assert "prunable_layers" in config
        assert "num_layers" in config
        assert config["num_layers"] == 5

    def test_to_dict(self, model_and_contributions):
        from uni_layer.integrations import ExportHintsBridge

        model, contrib = model_and_contributions
        bridge = ExportHintsBridge(model, contrib)

        d = bridge.to_dict()
        assert "quantization_plan" in d
        assert "prunable_layers" in d
        assert "fusion_candidates" in d
        assert "tensorrt_config" in d
