"""Tests for the MCP server tools."""

import json

import pytest


class TestMCPServerImport:
    def test_import_mcp_server(self):
        from uni_layer.mcp_server import mcp

        assert mcp.name == "uni-layer"

    def test_list_metrics_tool(self):
        from uni_layer.mcp_server import list_metrics

        result = list_metrics()
        assert "Optimization" in result
        assert "Security" in result
        assert "gradient_norm" in result
        assert "attention_path_trace" in result

    def test_list_metrics_filter(self):
        from uni_layer.mcp_server import list_metrics

        result = list_metrics(category="Security")
        assert "Security" in result
        assert "adversarial_sensitivity" in result
        # Should not contain other categories
        assert "## Optimization" not in result

    def test_list_metrics_unknown_category(self):
        from uni_layer.mcp_server import list_metrics

        result = list_metrics(category="nonexistent")
        assert "Unknown category" in result

    def test_layer_profile_without_analysis(self):
        from uni_layer.mcp_server import _cache, layer_profile

        _cache.clear()
        result = layer_profile()
        assert "Error" in result
        assert "analyze_model" in result

    def test_suggest_pruning_without_analysis(self):
        from uni_layer.mcp_server import _cache, suggest_pruning

        _cache.clear()
        result = suggest_pruning()
        assert "Error" in result

    def test_suggest_lora_without_analysis(self):
        from uni_layer.mcp_server import _cache, suggest_lora

        _cache.clear()
        result = suggest_lora()
        assert "Error" in result

    def test_security_audit_without_analysis(self):
        from uni_layer.mcp_server import _cache, security_audit

        _cache.clear()
        result = security_audit()
        assert "Error" in result

    def test_layer_profile_with_cached_data(self):
        """Test layer_profile with mock cached contributions."""
        from uni_layer.mcp_server import _cache, layer_profile

        _cache["last_model_name"] = "test-model"
        _cache["last_contributions"] = {
            "layer.0": {
                "layer_idx": 0,
                "layer_type": "transformer_block",
                "gradient_norm": 0.05,
                "block_influence": 0.1,
                "effective_rank": 10.0,
            },
            "layer.1": {
                "layer_idx": 1,
                "layer_type": "transformer_block",
                "gradient_norm": 0.02,
                "block_influence": 0.05,
                "effective_rank": 8.0,
            },
            "layer.2": {
                "layer_idx": 2,
                "layer_type": "transformer_block",
                "gradient_norm": 0.08,
                "block_influence": 0.15,
                "effective_rank": 12.0,
            },
        }

        result = layer_profile()
        data = json.loads(result)
        assert "summary" in data
        assert "redundant_layers" in data
        assert "pruning_suggestion" in data
        assert "lora_suggestion" in data

        _cache.clear()
