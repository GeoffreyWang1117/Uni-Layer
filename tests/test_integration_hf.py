"""
Integration tests with real HuggingFace models (tiny variants for CI).

These tests download tiny models (~100K params) from HuggingFace Hub
and run the full Uni-Layer pipeline: compute_metrics -> LayerProfile -> to_dict.
They verify real-world compatibility, not just synthetic mocks.

Models tested:
  - sshleifer/tiny-gpt2                       (GPT-2 decoder, ~100K)
  - hf-internal-testing/tiny-random-BertModel  (BERT encoder, ~88K)
  - hf-internal-testing/tiny-random-t5         (T5 seq2seq, ~121K)

Requires: pip install transformers
"""

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer, LayerProfile
from uni_layer.core.schema import validate_contributions
from uni_layer.metrics import (
    CKA,
    ActivationEntropy,
    BlockInfluence,
    EffectiveRank,
    GradientNorm,
)

transformers = pytest.importorskip("transformers", reason="transformers not installed")

METRICS = [
    GradientNorm(num_batches=1),
    BlockInfluence(num_batches=1),
    EffectiveRank(num_batches=1),
    CKA(num_batches=1),
    ActivationEntropy(num_batches=1),
]


def _make_loader(tokenizer, n=16, max_len=16):
    texts = ["The quick brown fox jumps over the lazy dog."] * n
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(enc["input_ids"], labels), batch_size=8)


def _run_full_pipeline(model_name, model_cls, device="cpu"):
    """Run compute_metrics -> LayerProfile -> to_dict and return profile dict."""
    model = model_cls.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = _make_loader(tokenizer)
    analyzer = LayerAnalyzer(model, task_type="classification", device=device)
    results = analyzer.compute_metrics(
        metrics=METRICS,
        data_loader=loader,
        verbose=False,
        num_batches=1,
    )
    profile = LayerProfile(results, model_name=model_name)
    return results, profile


class TestTinyGPT2:
    """Integration test with sshleifer/tiny-gpt2 (GPT-2 decoder)."""

    MODEL = "sshleifer/tiny-gpt2"

    def test_full_pipeline(self):
        results, profile = _run_full_pipeline(self.MODEL, transformers.AutoModelForCausalLM)

        # Basic structure
        assert len(results) >= 2
        validate_contributions(results)

        # Layer type detection
        first = list(results.values())[0]
        assert first["layer_type"] == "transformer_block"

        # Metrics present
        assert "gradient_norm" in first
        assert "block_influence" in first
        assert "effective_rank" in first

        # Profile serializable
        d = profile.to_dict()
        json.dumps(d)
        assert d["num_layers"] >= 2
        assert len(d["analysis"]["consensus_ranking"]) >= 2
        assert d["summary"]

    def test_preset_works(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(self.MODEL)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        loader = _make_loader(tokenizer)

        analyzer = LayerAnalyzer(model, task_type="classification", device="cpu")
        results = analyzer.compute_metrics(
            data_loader=loader,
            preset="quick",
            num_batches=1,
            verbose=False,
        )
        assert len(results) >= 2
        validate_contributions(results)


class TestTinyBERT:
    """Integration test with hf-internal-testing/tiny-random-BertModel (BERT encoder)."""

    MODEL = "hf-internal-testing/tiny-random-BertModel"

    def test_full_pipeline(self):
        results, profile = _run_full_pipeline(self.MODEL, transformers.AutoModel)

        assert len(results) >= 2
        validate_contributions(results)

        # Profile features
        assert isinstance(profile.redundant_layers, list)
        assert isinstance(profile.consensus_ranking, list)
        assert isinstance(profile.depth_trends, dict)

        d = profile.to_dict()
        json.dumps(d)


class TestTinyT5:
    """Integration test with hf-internal-testing/tiny-random-t5 (T5 seq2seq)."""

    MODEL = "hf-internal-testing/tiny-random-t5"

    def test_full_pipeline(self):
        results, profile = _run_full_pipeline(
            self.MODEL,
            transformers.AutoModelForSeq2SeqLM,
        )

        # T5 should have both encoder and decoder blocks
        assert len(results) >= 4
        validate_contributions(results)

        d = profile.to_dict()
        json.dumps(d)
        assert d["num_layers"] >= 4

    def test_profile_report(self):
        _, profile = _run_full_pipeline(
            self.MODEL,
            transformers.AutoModelForSeq2SeqLM,
        )
        report = profile.print_report()
        assert "Layer Metrics" in report
        assert "Suggestions" in report

    def test_profile_markdown(self):
        _, profile = _run_full_pipeline(
            self.MODEL,
            transformers.AutoModelForSeq2SeqLM,
        )
        md = profile.to_markdown()
        assert "| Layer |" in md
        assert "|---|" in md
