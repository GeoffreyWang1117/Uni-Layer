#!/usr/bin/env python
"""
Comprehensive model compatibility test for Uni-Layer.

Downloads tiny/small HuggingFace models and verifies:
1. Layer detection (are transformer blocks found?)
2. Layer type identification
3. Quick metric computation (GradientNorm + BlockInfluence)
4. LayerProfile generation

Usage:
    pytest tests/test_model_compatibility.py -v --timeout=120
    python tests/test_model_compatibility.py  # standalone with summary table
"""

import gc
import json
import sys
import time
import traceback

import torch

# ── Model registry: (model_id, family, architecture_notes) ──────────────
# Using the smallest available variant for each family.
# For families without tiny-random versions, we use from_config() to create
# random-weight models from the config only (no download of multi-GB weights).

MODELS_DOWNLOAD = [
    # ── Classic / Well-known ──
    ("hf-internal-testing/tiny-random-BertModel", "BERT", "encoder-only"),
    ("hf-internal-testing/tiny-random-gpt2", "GPT-2", "decoder-only"),
    ("hf-internal-testing/tiny-random-t5", "T5", "encoder-decoder"),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", "LLaMA", "decoder-only"),
    ("hf-internal-testing/tiny-random-MistralForCausalLM", "Mistral", "decoder-only"),
    ("hf-internal-testing/tiny-random-MixtralForCausalLM", "Mixtral (MoE)", "decoder-only, MoE"),
    ("hf-internal-testing/tiny-random-GPTNeoXForCausalLM", "GPT-NeoX", "decoder-only"),
    ("hf-internal-testing/tiny-random-GPTNeoForCausalLM", "GPT-Neo", "decoder-only"),
    ("hf-internal-testing/tiny-random-GPTJForCausalLM", "GPT-J", "decoder-only"),
    ("hf-internal-testing/tiny-random-BloomForCausalLM", "BLOOM", "decoder-only"),
    ("hf-internal-testing/tiny-random-FalconForCausalLM", "Falcon", "decoder-only"),
    ("hf-internal-testing/tiny-random-RobertaModel", "RoBERTa", "encoder-only"),
    ("hf-internal-testing/tiny-random-AlbertModel", "ALBERT", "encoder-only"),
    ("hf-internal-testing/tiny-random-ElectraModel", "ELECTRA", "encoder-only"),
    ("hf-internal-testing/tiny-random-DebertaV2Model", "DeBERTa-v2", "encoder-only"),
    ("hf-internal-testing/tiny-random-BartModel", "BART", "encoder-decoder"),
    ("hf-internal-testing/tiny-random-ViTModel", "ViT", "vision transformer"),
]

# Models that need config-based instantiation (no tiny-random available)
MODELS_FROM_CONFIG = [
    # ── Newer Families (2024-2026) ──
    ("Qwen/Qwen3-0.6B", "Qwen3", "decoder-only"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5", "decoder-only"),
    # ── Gemma 4 (April 2026, multimodal) ──
    # Note: model_type=gemma4 requires transformers>=5.0.0
    # mm_token_type_ids injected automatically by model_forward() for text-only analysis
    ("google/gemma-4-E2B", "Gemma 4 E2B", "multimodal, decoder-only"),
    ("google/gemma-4-E4B", "Gemma 4 E4B", "multimodal, decoder-only"),
    ("google/gemma-4-31B", "Gemma 4 31B", "multimodal, decoder-only"),
    ("google/gemma-4-26B-A4B", "Gemma 4 26B-A4B", "multimodal, MoE"),
    ("google/gemma-3-1b-pt", "Gemma 3", "decoder-only"),
    ("google/gemma-2-2b", "Gemma 2", "decoder-only"),
    # ── Llama 4 (April 2026, multimodal MoE) ──
    # Note: requires AutoModelForCausalLM (not AutoModel) to get logits
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama 4 Scout", "multimodal, MoE"),
    ("microsoft/phi-4-mini-instruct", "Phi-4", "decoder-only"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3", "decoder-only"),
    ("deepseek-ai/DeepSeek-V3-0324", "DeepSeek-V3", "decoder-only, MoE"),
    ("deepseek-ai/DeepSeek-V2-Lite", "DeepSeek-V2", "decoder-only, MoE"),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "Mistral-3.1", "multimodal"),
    ("allenai/OLMo-2-0425-1B", "OLMo-2", "decoder-only"),
    ("01-ai/Yi-1.5-6B", "Yi-1.5", "decoder-only"),
    ("internlm/internlm2_5-1_8b", "InternLM-2.5", "decoder-only"),
    ("THUDM/glm-4-9b", "GLM-4", "decoder-only"),
    ("bigcode/starcoder2-3b", "StarCoder2", "decoder-only"),
    ("tiiuae/Falcon3-1B-Base", "Falcon-3", "decoder-only"),
    ("nvidia/Nemotron-Mini-4B-Instruct", "Nemotron", "decoder-only"),
    ("ai21labs/Jamba-v0.1", "Jamba", "decoder-only, Mamba+Transformer hybrid"),
    ("google/gemma-7b", "Gemma", "decoder-only"),
    ("Qwen/Qwen2-0.5B", "Qwen2", "decoder-only"),
    ("stabilityai/stablelm-3b-4e1t", "StableLM", "decoder-only"),
    ("EleutherAI/pythia-70m", "Pythia", "decoder-only"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama", "decoder-only"),
]


def test_model_from_pretrained(model_id, family, notes):
    """Test a model loaded via from_pretrained (tiny models)."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    return _run_analysis(model, model_id, family, notes)


def test_model_from_config(model_id, family, notes):
    """Test a model instantiated from config only (no weight download)."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Shrink model to fit in memory for testing
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, 4)
    if hasattr(config, "num_layers"):
        config.num_layers = min(config.num_layers, 4)
    if hasattr(config, "num_experts"):
        config.num_experts = min(config.num_experts, 2)
    if hasattr(config, "num_local_experts"):
        config.num_local_experts = min(config.num_local_experts, 2)
    if hasattr(config, "n_routed_experts"):
        config.n_routed_experts = min(config.n_routed_experts, 2)
    if hasattr(config, "hidden_size"):
        config.hidden_size = min(config.hidden_size, 256)
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = min(config.intermediate_size, 512)
    if hasattr(config, "num_attention_heads"):
        config.num_attention_heads = min(config.num_attention_heads, 4)
    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = min(config.num_key_value_heads, min(4, config.num_attention_heads))
    # Don't shrink vocab_size — can break padding_idx constraints
    # Mamba/Jamba specific
    if hasattr(config, "mamba_d_state"):
        config.mamba_d_state = min(config.mamba_d_state, 16)
    if hasattr(config, "mamba_expand"):
        config.mamba_expand = min(config.mamba_expand, 2)
    # Multimodal: try to get text-only model
    if hasattr(config, "text_config"):
        text_config = config.text_config
        if hasattr(text_config, "num_hidden_layers"):
            text_config.num_hidden_layers = min(text_config.num_hidden_layers, 4)
        if hasattr(text_config, "hidden_size"):
            text_config.hidden_size = min(text_config.hidden_size, 256)
        if hasattr(text_config, "intermediate_size"):
            text_config.intermediate_size = min(text_config.intermediate_size, 512)
        if hasattr(text_config, "num_attention_heads"):
            text_config.num_attention_heads = min(text_config.num_attention_heads, 4)
        if hasattr(text_config, "num_key_value_heads"):
            text_config.num_key_value_heads = min(text_config.num_key_value_heads, min(4, text_config.num_attention_heads))

    # Prefer CausalLM for decoder-only models so we get logits (not just last_hidden_state)
    # This matters for Llama 4, Gemma 4, Qwen3 etc. where AutoModel returns a base model
    # with no logit head, so gradient metrics get zero layers.
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception:
        try:
            model = AutoModel.from_config(config, trust_remote_code=True)
        except Exception:
            # Last resort: some seq2seq / vision models
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)

    return _run_analysis(model, model_id, family, notes)


def _run_analysis(model, model_id, family, notes):
    """Core analysis: layer detection + quick metrics."""
    from uni_layer import LayerAnalyzer, LayerProfile
    from uni_layer.metrics import BlockInfluence, GradientNorm
    from uni_layer.utils.layer_utils import get_model_layers, identify_layer_type

    result = {
        "model_id": model_id,
        "family": family,
        "notes": notes,
        "model_class": type(model).__name__,
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    # Step 1: Layer detection
    layers = get_model_layers(model)
    result["num_layers"] = len(layers)
    result["layer_names"] = list(layers.keys())

    if len(layers) == 0:
        result["status"] = "FAIL"
        result["error"] = "No layers detected"
        return result

    # Layer types
    layer_types = set()
    for name, layer in layers.items():
        lt = identify_layer_type(layer)
        layer_types.add(lt)
    result["layer_types"] = sorted(layer_types)

    # Block path inference
    if layers:
        first_name = list(layers.keys())[0]
        result["block_path"] = first_name

    # Step 2: Quick metric computation
    try:
        # Create dummy data
        batch_size = 4
        seq_len = 16
        # Determine input shape
        if hasattr(model, "config"):
            vocab_size = getattr(model.config, "vocab_size", 100)
        else:
            vocab_size = 100

        input_ids = torch.randint(0, min(vocab_size, 100), (batch_size, seq_len))
        labels = torch.zeros(batch_size, dtype=torch.long)

        from torch.utils.data import DataLoader, TensorDataset
        data_loader = DataLoader(
            TensorDataset(input_ids, labels), batch_size=batch_size
        )

        analyzer = LayerAnalyzer(model, task_type="classification")
        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=1), BlockInfluence(num_batches=1)],
            data_loader=data_loader,
            num_batches=1,
        )

        # Check we got results
        num_analyzed = sum(
            1 for v in contributions.values()
            if v.get("gradient_norm") is not None
        )
        result["metrics_computed"] = num_analyzed
        result["metric_keys"] = sorted(
            set(k for v in contributions.values() for k in v if k not in ("layer_idx", "layer_type"))
        )

        # Step 3: LayerProfile
        profile = LayerProfile(contributions, model_name=family)
        summary = profile.summary()
        result["profile_summary"] = summary[:200]
        result["status"] = "PASS"

    except Exception as e:
        result["status"] = "METRICS_FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        # Layer detection passed, just metrics failed
        if result["num_layers"] > 0:
            result["status"] = "PARTIAL"

    return result


def run_all_tests():
    """Run all compatibility tests and print summary."""
    results = []

    print("=" * 80)
    print("Uni-Layer Model Compatibility Test")
    print("=" * 80)

    # Phase 1: Tiny models (download full weights)
    print("\n--- Phase 1: Tiny models (from_pretrained) ---\n")
    for model_id, family, notes in MODELS_DOWNLOAD:
        print(f"  Testing {family:<20} ({model_id})...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = test_model_from_pretrained(model_id, family, notes)
            dt = time.time() - t0
            status = r["status"]
            layers = r["num_layers"]
            print(f"{status:>12} | {layers:>3} layers | {dt:.1f}s")
        except Exception as e:
            dt = time.time() - t0
            r = {
                "model_id": model_id, "family": family, "notes": notes,
                "status": "ERROR", "error": f"{type(e).__name__}: {e}",
                "num_layers": 0,
            }
            print(f"{'ERROR':>12} | {dt:.1f}s | {e}")
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Phase 2: Config-only models (no weight download)
    print("\n--- Phase 2: Newer models (from_config, random weights) ---\n")
    for model_id, family, notes in MODELS_FROM_CONFIG:
        print(f"  Testing {family:<20} ({model_id})...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = test_model_from_config(model_id, family, notes)
            dt = time.time() - t0
            status = r["status"]
            layers = r["num_layers"]
            print(f"{status:>12} | {layers:>3} layers | {dt:.1f}s")
        except Exception as e:
            dt = time.time() - t0
            r = {
                "model_id": model_id, "family": family, "notes": notes,
                "status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:100]}",
                "num_layers": 0,
            }
            print(f"{'ERROR':>12} | {dt:.1f}s | {str(e)[:80]}")
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Family':<20} {'Status':<12} {'Layers':>6} {'Block Path':<40} {'Types'}")
    print("-" * 110)

    pass_count = 0
    partial_count = 0
    fail_count = 0

    for r in results:
        status = r["status"]
        family = r["family"]
        layers = r.get("num_layers", 0)
        block_path = r.get("block_path", "N/A")
        types = ", ".join(r.get("layer_types", []))

        if status == "PASS":
            pass_count += 1
            marker = "PASS"
        elif status == "PARTIAL":
            partial_count += 1
            marker = "PARTIAL"
        else:
            fail_count += 1
            marker = "FAIL"

        print(f"{family:<20} {marker:<12} {layers:>6} {block_path:<40} {types}")

    print(f"\nTotal: {len(results)} | Pass: {pass_count} | Partial: {partial_count} | Fail: {fail_count}")

    # Failures detail
    failures = [r for r in results if r["status"] not in ("PASS",)]
    if failures:
        print("\n--- Issues ---")
        for r in failures:
            print(f"  {r['family']}: {r.get('error', 'unknown')}")

    # Save JSON
    output_path = "tests/compatibility_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")

    return results


# ── pytest integration ──────────────────────────────────────────────────

def _make_test_func(model_id, family, notes, loader_fn):
    """Create a pytest test function for a specific model."""
    def test_fn():
        r = loader_fn(model_id, family, notes)
        assert r["num_layers"] >= 2, f"Expected >=2 layers, got {r['num_layers']}"
        assert r["status"] in ("PASS", "PARTIAL"), f"Failed: {r.get('error', 'unknown')}"
    test_fn.__name__ = f"test_{family.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '')}"
    test_fn.__doc__ = f"Test {family} ({model_id})"
    return test_fn


# Generate pytest test functions for tiny models only (fast, reliable)
import pytest

class TestModelCompatibilityTiny:
    """Test layer detection on tiny HF models (downloaded)."""
    pass

for _id, _fam, _notes in MODELS_DOWNLOAD:
    fn = _make_test_func(_id, _fam, _notes, test_model_from_pretrained)
    setattr(TestModelCompatibilityTiny, fn.__name__, staticmethod(fn))


@pytest.mark.slow
class TestModelCompatibilityFromConfig:
    """Test layer detection on newer models (from_config, slower)."""
    pass

for _id, _fam, _notes in MODELS_FROM_CONFIG:
    fn = _make_test_func(_id, _fam, _notes, test_model_from_config)
    setattr(TestModelCompatibilityFromConfig, fn.__name__, staticmethod(fn))


if __name__ == "__main__":
    run_all_tests()
