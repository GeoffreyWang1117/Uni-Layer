"""Tests for LLM training framework integration (v0.5.0 Feature 5)."""

import json
import os
import tempfile

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
        "layer.0.q_proj": {"layer_idx": 0, "layer_type": "linear", "gradient_norm": 0.5},
        "layer.0.v_proj": {"layer_idx": 1, "layer_type": "linear", "gradient_norm": 0.4},
        "layer.1.q_proj": {"layer_idx": 2, "layer_type": "linear", "gradient_norm": 0.1},
        "layer.1.v_proj": {"layer_idx": 3, "layer_type": "linear", "gradient_norm": 0.08},
        "layer.2.q_proj": {"layer_idx": 4, "layer_type": "linear", "gradient_norm": 0.8},
    }
    return model, contributions


class TestAxolotlConfigBridge:
    def test_import(self):
        from uni_layer.integrations import AxolotlConfigBridge

        assert AxolotlConfigBridge is not None

    def test_recommend_target_modules(self, model_and_contributions):
        from uni_layer.integrations import AxolotlConfigBridge

        model, contrib = model_and_contributions
        bridge = AxolotlConfigBridge(model, contrib)

        modules = bridge.recommend_target_modules(metric_name="gradient_norm")
        assert isinstance(modules, list)
        assert len(modules) > 0
        # Should find q_proj and v_proj patterns
        assert "q_proj" in modules or "v_proj" in modules

    def test_recommend_lora_rank(self, model_and_contributions):
        from uni_layer.integrations import AxolotlConfigBridge

        model, contrib = model_and_contributions
        bridge = AxolotlConfigBridge(model, contrib)

        rank = bridge.recommend_lora_rank(base_rank=16)
        assert 4 <= rank <= 64

    def test_generate_config(self, model_and_contributions):
        from uni_layer.integrations import AxolotlConfigBridge

        model, contrib = model_and_contributions
        bridge = AxolotlConfigBridge(model, contrib)

        config = bridge.generate_config(base_model="meta-llama/Llama-2-7b")
        assert config["base_model"] == "meta-llama/Llama-2-7b"
        assert config["adapter"] == "lora"
        assert "lora_r" in config
        assert "lora_alpha" in config
        assert "lora_target_modules" in config
        assert "_uni_layer_metadata" in config

    def test_save_yaml(self, model_and_contributions):
        from uni_layer.integrations import AxolotlConfigBridge

        model, contrib = model_and_contributions
        bridge = AxolotlConfigBridge(model, contrib)

        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
            path = f.name

        try:
            bridge.save_yaml(path, base_model="test-model")
            assert os.path.exists(path)
            with open(path) as f:
                import yaml

                loaded = yaml.safe_load(f)
            assert loaded["base_model"] == "test-model"
            assert "lora_r" in loaded
        finally:
            os.unlink(path)


class TestLLaMAFactoryConfigBridge:
    def test_import(self):
        from uni_layer.integrations import LLaMAFactoryConfigBridge

        assert LLaMAFactoryConfigBridge is not None

    def test_recommend_freeze_layers(self, model_and_contributions):
        from uni_layer.integrations import LLaMAFactoryConfigBridge

        model, contrib = model_and_contributions
        bridge = LLaMAFactoryConfigBridge(model, contrib)

        frozen = bridge.recommend_freeze_layers(freeze_ratio=0.4)
        assert isinstance(frozen, list)
        assert len(frozen) >= 1
        # Lowest importance layers should be frozen
        assert "layer.1.v_proj" in frozen  # gradient_norm=0.08

    def test_generate_config(self, model_and_contributions):
        from uni_layer.integrations import LLaMAFactoryConfigBridge

        model, contrib = model_and_contributions
        bridge = LLaMAFactoryConfigBridge(model, contrib)

        config = bridge.generate_config(model_name="test-model")
        assert config["model_name_or_path"] == "test-model"
        assert config["finetuning_type"] == "lora"
        assert "freeze_layers" in config
        assert "num_frozen_layers" in config
        assert config["num_frozen_layers"] == len(config["freeze_layers"])

    def test_save_json(self, model_and_contributions):
        from uni_layer.integrations import LLaMAFactoryConfigBridge

        model, contrib = model_and_contributions
        bridge = LLaMAFactoryConfigBridge(model, contrib)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bridge.save_json(path, model_name="test-model")
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["model_name_or_path"] == "test-model"
        finally:
            os.unlink(path)

    def test_freeze_ratio_zero(self, model_and_contributions):
        from uni_layer.integrations import LLaMAFactoryConfigBridge

        model, contrib = model_and_contributions
        bridge = LLaMAFactoryConfigBridge(model, contrib)

        # Even with 0 ratio, should freeze at least 1 (max(1, ...))
        frozen = bridge.recommend_freeze_layers(freeze_ratio=0.0)
        assert len(frozen) >= 1
