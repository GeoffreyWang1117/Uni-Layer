# Model Profiles

Pre-computed layer analysis profiles for popular HuggingFace models.

Each JSON file contains the output of `LayerProfile.to_dict()` — including
consensus ranking, depth trends, anomalies, pruning/LoRA suggestions, and
a natural language summary.

## Available Profiles

| Model | Architecture | Layers | File |
|---|---|---|---|
| bert-base-uncased | BERT Encoder | 12 | [bert-base-uncased.json](bert-base-uncased.json) |
| distilbert-base-uncased | DistilBERT | 6 | [distilbert-base-uncased.json](distilbert-base-uncased.json) |
| gpt2 | GPT-2 Decoder | 12 | [gpt2.json](gpt2.json) |
| bigscience/bloom-560m | BLOOM | 24 | [bigscience_bloom-560m.json](bigscience_bloom-560m.json) |
| tiiuae/falcon-rw-1b | Falcon | 24 | [tiiuae_falcon-rw-1b.json](tiiuae_falcon-rw-1b.json) |
| google/byt5-small | ByT5 Seq2Seq | 36 | [google_byt5-small.json](google_byt5-small.json) |
| facebook/dinov2-base | DINOv2 Vision | 12 | [facebook_dinov2-base.json](facebook_dinov2-base.json) |
| facebook/wav2vec2-base | Wav2Vec2 Speech | 19 | [facebook_wav2vec2-base.json](facebook_wav2vec2-base.json) |

## Using Profiles

```python
import json
from uni_layer.core.profile import LayerProfile

# Load pre-computed profile
with open("benchmarks/model_profiles/bert-base-uncased.json") as f:
    data = json.load(f)

print(data["summary"])
print(data["analysis"]["suggestions"]["pruning"])
print(data["analysis"]["depth_trends"]["block_influence"])
```

## Regenerating

```bash
# Profiles are generated with llm_fast preset (5 metrics, 2 batches)
# See benchmarks/generate_profiles.py for the script
```

## Metrics Used

All profiles are computed with: GradientNorm, BlockInfluence, EffectiveRank, CKA, ActivationEntropy.
