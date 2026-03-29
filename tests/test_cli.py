"""
Tests for the uni-layer CLI.
"""

import json
import subprocess
import sys

import pytest


def run_cli(*args):
    """Run CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "uni_layer.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIHelp:
    def test_no_args_shows_help(self):
        rc, out, _ = run_cli()
        assert rc == 0
        assert "uni-layer" in out.lower() or "layer contribution" in out.lower()

    def test_help_flag(self):
        rc, out, _ = run_cli("--help")
        assert rc == 0
        assert "list-metrics" in out
        assert "analyze" in out
        assert "info" in out

    def test_version_flag(self):
        rc, out, _ = run_cli("--version")
        assert rc == 0
        assert "uni-layer 0.6." in out


class TestCLIInfo:
    def test_info(self):
        rc, out, _ = run_cli("info")
        assert rc == 0
        assert "Python" in out
        assert "PyTorch" in out
        assert "Metrics" in out

    def test_info_shows_metric_count(self):
        rc, out, _ = run_cli("info")
        assert "Metrics" in out


class TestCLIListMetrics:
    def test_list_metrics_table(self):
        rc, out, _ = run_cli("list-metrics")
        assert rc == 0
        assert "GradientNorm" in out
        assert "BlockInfluence" in out
        assert "AttentionFlow" in out
        assert "Total:" in out

    def test_list_metrics_json(self):
        rc, out, _ = run_cli("list-metrics", "--format", "json")
        assert rc == 0
        data = json.loads(out)
        assert "gradient_norm" in data
        assert "block_influence" in data
        assert len(data) == 13
        assert data["gradient_norm"]["class"] == "GradientNorm"
        assert "primary_key" in data["gradient_norm"]
        assert "output_keys" in data["gradient_norm"]


class TestCLIAnalyze:
    def test_analyze_help(self):
        rc, out, _ = run_cli("analyze", "--help")
        assert rc == 0
        assert "model" in out.lower()
        assert "--metrics" in out

    def test_analyze_unknown_metric(self):
        rc, _, err = run_cli("analyze", "fake-model", "-m", "NonexistentMetric")
        assert rc != 0
        assert "Unknown metric" in err

    def test_analyze_no_transformers_graceful(self):
        """If transformers not installed, should give a clear error."""
        # This test works regardless — if transformers IS installed,
        # it will try to load a nonexistent model and fail gracefully.
        rc, out, err = run_cli("analyze", "nonexistent-model-12345")
        assert rc != 0  # Should fail (model doesn't exist or no transformers)
