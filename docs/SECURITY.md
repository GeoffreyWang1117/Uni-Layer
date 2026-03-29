# Security Analysis Guide (v0.6.1)

Uni-Layer provides 4 security metrics and a compression safety audit for model vulnerability assessment. This guide covers red-team analysis workflows.

## Overview

| Metric | Threat Model | What It Detects |
|--------|-------------|-----------------|
| AdversarialSensitivity | Adversarial examples | Layers that amplify input perturbations |
| ActivationAnomalyScore | Backdoor/Trojan | Unusual activation patterns from embedded triggers |
| MembershipInferenceRisk | Privacy leakage | Gradient-based information extraction risk |
| AttentionPathTrace | Prompt injection | Attention hijacking vulnerability in LLMs |

## Quick Start

```python
from uni_layer import LayerAnalyzer, LayerProfile
from uni_layer.metrics import (
    AdversarialSensitivity,
    ActivationAnomalyScore,
    MembershipInferenceRisk,
    AttentionPathTrace,
)

analyzer = LayerAnalyzer(model, task_type="classification")

# Run all security metrics
contributions = analyzer.compute_metrics(
    metrics=[
        AdversarialSensitivity(epsilon=0.01),
        ActivationAnomalyScore(),
        MembershipInferenceRisk(),
        AttentionPathTrace(),
    ],
    data_loader=loader,
    num_batches=10,
)

# Automated vulnerability report
profile = LayerProfile(contributions, model_name="my-model")
report = profile.security_report()

print(report["summary"])
# "Security analysis across 12 layers (adversarial, anomaly, privacy, injection).
#  Average risk: 0.234. Highest risk layer: encoder.layer.11 (0.456)."

print(report["top_risks"][:3])
# Top 3 riskiest layers with composite scores
```

---

## 1. Adversarial Sensitivity

Measures how much each layer's activations change under FGSM perturbation.

```python
m = AdversarialSensitivity(epsilon=0.01, num_batches=5)
```

**Interpretation:**

| Key | High Value Means |
|-----|-----------------|
| `adv_sensitivity` | Layer amplifies adversarial perturbations (attack surface) |
| `adv_amplification` | Small input changes cause large internal shifts |
| `adv_directional_change` | Adversarial inputs push activations in different directions |

**Red-team use:**
- Identify which layers are most vulnerable to adversarial attacks
- Prioritize adversarial training on high-sensitivity layers
- Verify that defensive distillation reduced sensitivity

---

## 2. Activation Anomaly (Backdoor Detection)

Detects layers with unusual activation distributions that may indicate embedded trojans.

```python
m = ActivationAnomalyScore(num_batches=10)
```

**Interpretation:**

| Key | Suspicious When |
|-----|----------------|
| `activation_skewness` | |skew| > 2 (heavily asymmetric distribution) |
| `activation_kurtosis` | kurtosis > 5 (heavy tails, outlier neurons) |
| `neuron_outlier_ratio` | > 0.05 (5%+ neurons firing abnormally) |
| `activation_bimodality` | > 0.555 (bimodal distribution = potential trigger) |

**Red-team workflow:**
```python
for layer_name, metrics in contributions.items():
    bimodal = metrics.get("activation_bimodality", 0)
    outliers = metrics.get("neuron_outlier_ratio", 0)
    if bimodal > 0.555 or outliers > 0.05:
        print(f"WARNING: {layer_name} — bimodality={bimodal:.3f}, outliers={outliers:.3f}")
```

---

## 3. Membership Inference Risk

Estimates how much private training data information leaks through gradients.

```python
m = MembershipInferenceRisk(num_batches=5)
```

**Interpretation:**

| Key | High Value Means |
|-----|-----------------|
| `gradient_entropy` | Diverse gradient information (high leakage potential) |
| `gradient_snr` | Strong gradient signal (easier to infer membership) |
| `gradient_memorization` | High variance = model memorizes individual samples |
| `mi_risk_score` | Composite: 0 = safe, 1 = high risk |

**Privacy use cases:**
- Federated learning: identify layers that leak most information
- Differential privacy: allocate noise budget proportionally to `mi_risk_score`
- Audit model before deployment for privacy compliance

---

## 4. Attention Path Trace (Prompt Injection)

Analyzes attention patterns for prompt injection vulnerability in transformers.

```python
m = AttentionPathTrace(num_batches=5)
```

**Interpretation:**

| Key | High Value Means |
|-----|-----------------|
| `attention_concentration` | Few positions dominate attention (hijack potential) |
| `attention_manipulability` | Attention distribution is skewed (Gini coefficient) |
| `attention_persistence` | Consistent attention pattern (predictable = exploitable) |
| `injection_vulnerability` | Composite: 0 = resistant, 1 = highly vulnerable |

**Note:** Non-attention layers use activation-based proxy metrics. Returns `None` for `attention_manipulability` on non-attention layers.

---

## 5. Compression Safety Audit

Compare security metrics before and after model compression (pruning, quantization, distillation).

```python
from uni_layer.integrations import CompressionSafetyAudit

# Run security metrics on original model
pre_contributions = analyzer.compute_metrics(
    metrics=[AdversarialSensitivity(), MembershipInferenceRisk()],
    data_loader=loader,
)

# ... compress model ...

# Run same metrics on compressed model
post_contributions = analyzer.compute_metrics(
    metrics=[AdversarialSensitivity(), MembershipInferenceRisk()],
    data_loader=loader,
)

# Audit
audit = CompressionSafetyAudit(pre_contributions, post_contributions)
report = audit.audit()

print(f"Overall degradation: {report['overall_degradation']:.3f}")
print(f"Recommendations: {report['recommendations']}")

# Per-metric details
for metric, summary in report["metric_summaries"].items():
    print(f"  {metric}: {summary['num_degraded_layers']}/{summary['num_total_layers']} layers worse")

# Most affected layers
for layer in report["most_affected_layers"][:5]:
    print(f"  {layer['layer_name']}: total degradation = {layer['total_degradation']:.4f}")
```

**Risk levels:**
- `overall_degradation > 0.5`: HIGH RISK — compression significantly increased vulnerability
- `overall_degradation > 0.2`: MODERATE RISK — some metrics degraded
- `overall_degradation <= 0.2`: LOW RISK — minimal impact

---

## 6. Security Report (`LayerProfile`)

Generates an automated summary from any security metrics present in contributions.

```python
report = profile.security_report()
```

**Output structure:**
```python
{
    "categories_found": ["adversarial", "anomaly", "privacy", "injection"],
    "layer_risks": {
        "encoder.layer.0": {
            "composite_risk": 0.23,
            "max_risk": 0.45,
            "num_risk_signals": 4,
        },
        ...
    },
    "top_risks": [
        {"layer": "encoder.layer.11", "composite_risk": 0.46, "max_risk": 0.72, ...},
        ...
    ],
    "summary": "Security analysis across 12 layers ..."
}
```

---

## Complete Red-Team Workflow

```python
from uni_layer import LayerAnalyzer, LayerProfile
from uni_layer.metrics import *

# 1. Full analysis
analyzer = LayerAnalyzer(model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[
        # Standard importance
        GradientNorm(), BlockInfluence(), EffectiveRank(),
        # Security
        AdversarialSensitivity(epsilon=0.01),
        ActivationAnomalyScore(),
        MembershipInferenceRisk(),
        AttentionPathTrace(),
        # Efficiency (for deployment context)
        EfficiencyProfiler(),
        QuantizationSensitivity(),
    ],
    data_loader=loader,
)

# 2. Security report
profile = LayerProfile(contributions)
security = profile.security_report()
print(security["summary"])

# 3. Identify attack surfaces
for risk in security["top_risks"]:
    layer = risk["layer"]
    print(f"\n--- {layer} (risk: {risk['composite_risk']:.3f}) ---")
    m = contributions[layer]
    if m.get("adv_sensitivity"):
        print(f"  Adversarial amplification: {m['adv_amplification']:.2f}x")
    if m.get("neuron_outlier_ratio", 0) > 0.05:
        print(f"  Anomalous neurons: {m['neuron_outlier_ratio']:.1%}")
    if m.get("mi_risk_score", 0) > 0.5:
        print(f"  Privacy risk: {m['mi_risk_score']:.3f}")

# 4. Export for downstream tools
import json
with open("security_audit.json", "w") as f:
    json.dump(security, f, indent=2)
```
