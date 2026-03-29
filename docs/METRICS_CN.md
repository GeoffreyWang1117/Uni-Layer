# 指标参考手册 (v0.6.1)

Uni-Layer 提供 **26 个指标**，覆盖 **9 个理论类别**。

## 快速索引

| # | 指标 | 类别 | 需梯度 | 需数据 | 主键 | 开销 |
|---|------|------|--------|--------|------|------|
| 1 | GradientNorm | 优化 | Yes | Yes | `gradient_norm` | 低 |
| 2 | HessianTrace | 优化 | Yes | Yes | `hessian_trace` | 高 |
| 3 | FisherInformation | 优化 | Yes | Yes | `fisher_information` | 中 |
| 4 | WandaImportance | 优化 | No | Yes | `wanda_score` | 低 |
| 5 | IGSensitivity | 优化 | Yes | Yes | `ig_sensitivity` | 高 |
| 6 | CKA | 谱方法 | No | Yes | `cka_score` | 中 |
| 7 | EffectiveRank | 谱方法 | No | Yes | `effective_rank` | 中 |
| 8 | NTKTrace | 谱方法 | Yes | Yes | `ntk_trace` | 高 |
| 9 | ActivationEntropy | 信息论 | No | Yes | `activation_entropy` | 低 |
| 10 | MutualInformation | 信息论 | No | Yes | `mutual_information` | 中 |
| 11 | JacobianRank | 表示 | Yes | Yes | `jacobian_rank` | 高 |
| 12 | BlockInfluence | 表示 | No | Yes | `block_influence` | 低 |
| 13 | DropLayerRobustness | 鲁棒性 | No | Yes | `droplayer_loss_increase` | 中 |
| 14 | ResidualDropLayer | 鲁棒性 | No | Yes | `residual_droplayer_loss_increase` | 中 |
| 15 | LaplacePosterior | 贝叶斯 | Yes | Yes | `laplace_posterior` | 高 |
| 16 | EfficiencyProfiler | 效率 | No | Yes | `flops` | 低 |
| 17 | WeightDistribution | 效率 | No | No | `weight_sparsity` | 低 |
| 18 | IntrinsicDimensionality | 效率 | No | Yes | `intrinsic_dim` | 中 |
| 19 | QuantizationSensitivity | 效率 | No | Yes | `quant_sensitivity_int8` | 中 |
| 20 | AdversarialSensitivity | 安全 | Yes | Yes | `adv_sensitivity` | 中 |
| 21 | ActivationAnomalyScore | 安全 | No | Yes | `activation_skewness` | 低 |
| 22 | MembershipInferenceRisk | 安全 | Yes | Yes | `mi_risk_score` | 中 |
| 23 | AttentionPathTrace | 安全 | No | Yes | `injection_vulnerability` | 中 |
| 24 | AttentionFlow | 架构特定 | No | Yes | `attention_entropy` | 中 |
| 25 | MoERouterAnalysis | 架构特定 | No | Yes | `routing_entropy` | 中 |
| 26 | DiffusionTimestepAnalysis | 架构特定 | No | Yes | `timestep_sensitivity` | 高 |

## 预设

| 预设 | 包含指标 | 适用场景 |
|------|---------|---------|
| `llm_fast` | BlockInfluence, EffectiveRank, CKA, ActivationEntropy, AttentionFlow | LLM 快速筛选（秒级） |
| `llm_full` | 上述 + GradientNorm, FisherInformation | LLM 详细分析（分钟级） |
| `quick` | GradientNorm, BlockInfluence, EffectiveRank | 快速重要性检查 |
| `full` | 全部 26 个指标 | 完整分析 |

---

## 类别 1: 优化 (5 个指标)

### GradientNorm
梯度幅度衡量层重要性。输出: `gradient_norm`, `gradient_norm_std`, `gradient_norm_max`, `gradient_norm_min`

### HessianTrace
Hutchinson 迹估计器计算损失曲面曲率。输出: `hessian_trace`, `hessian_trace_std`

### FisherInformation
经验 Fisher 信息矩阵迹。输出: `fisher_information`, `fisher_mean`

### WandaImportance
权重幅度 x 激活范数，**无需梯度**（Sun et al., ICLR 2024）。输出: `wanda_score`, `weight_norm`, `activation_norm`, `wanda_sparsity`

### IGSensitivity
积分梯度逐层归因（路径积分方法）。输出: `ig_sensitivity`, `ig_variance`, `ig_relative`

## 类别 2: 谱与核方法 (3 个指标)

### CKA
中心核对齐 — 层与输出的表示相似度。输出: `cka_score`

### EffectiveRank
基于奇异值熵的表示多样性。输出: `effective_rank`, `stable_rank`, `rank_ratio`

### NTKTrace
神经切线核迹近似。输出: `ntk_trace`, `ntk_trace_per_param`

## 类别 3: 信息论 (2 个指标)

### ActivationEntropy
激活分布的 Shannon 熵。输出: `activation_entropy`, `activation_mean`, `activation_std`, `activation_sparsity`

### MutualInformation
激活与目标间的互信息。输出: `mutual_information`, `mi_max`, `mi_std`

## 类别 4: 表示 (2 个指标)

### BlockInfluence
层变换幅度 — 输入/输出余弦距离（ShortGPT, ACL 2025）。输出: `block_influence`, `block_similarity`

### JacobianRank
输入-输出 Jacobian 矩阵的有效维度。输出: `jacobian_rank`, `jacobian_rank_ratio`, `jacobian_condition`, `jacobian_max_sv`

## 类别 5: 鲁棒性 (2 个指标)

### DropLayerRobustness
将层输出置零后的性能下降。输出: `droplayer_loss_increase`, `droplayer_loss_ratio`

### ResidualDropLayer
残差感知消融 — 用输入替换输出（保留残差流）。输出: `residual_droplayer_loss_increase`, `residual_droplayer_loss_ratio`, `residual_ratio`, `transform_norm_ratio`

## 类别 6: 贝叶斯 (1 个指标)

### LaplacePosterior
Laplace 近似估计参数不确定性。输出: `laplace_posterior`, `laplace_posterior_std`

## 类别 7: 效率 (4 个指标)

### EfficiencyProfiler
逐层 FLOPs、参数量、内存。输出: `flops`, `param_count`, `param_memory_mb`, `activation_memory_mb`, `compute_ratio`

### WeightDistribution
权重矩阵统计特性（**无需数据**）。输出: `weight_sparsity`, `weight_l1_norm`, `weight_l2_norm`, `weight_rank_ratio`, `weight_outlier_ratio`, `weight_kurtosis`

### IntrinsicDimensionality
激活流形的 MLE 内在维度估计（Levina-Bickel 2004）。输出: `intrinsic_dim`, `intrinsic_dim_ratio`, `ambient_dim`

> 关键洞察：LoRA 的最优秩 ~ 内在维度。此指标直接为每层提供最优秩建议。

### QuantizationSensitivity
模拟 INT8/FP16 量化，测量输出偏差。输出: `quant_sensitivity_int8`, `quant_sensitivity_fp16`, `activation_range`, `weight_dynamic_range`

## 类别 8: 安全 (4 个指标)

### AdversarialSensitivity
FGSM 扰动敏感度。输出: `adv_sensitivity`, `adv_amplification`, `adv_directional_change`

### ActivationAnomalyScore
基于激活分布异常的后门检测。输出: `activation_skewness`, `activation_kurtosis`, `neuron_outlier_ratio`, `activation_bimodality`

### MembershipInferenceRisk
梯度泄露风险评分。输出: `gradient_entropy`, `gradient_snr`, `gradient_memorization`, `mi_risk_score`

### AttentionPathTrace
Prompt 注入漏洞分析。输出: `attention_concentration`, `attention_manipulability`, `attention_persistence`, `injection_vulnerability`

## 类别 9: 架构特定 (3 个指标)

### AttentionFlow
Transformer 注意力头多样性和熵分析。输出: `attention_entropy`, `attention_max_weight`, `head_diversity`, `attention_distance`

### MoERouterAnalysis
MoE 路由行为分析。输出: `routing_entropy`, `expert_utilization`, `load_balance_score`, `top_expert_ratio`, `expert_overlap`

### DiffusionTimestepAnalysis
扩散模型逐时间步重要性分析。输出: `timestep_sensitivity`, `mean_activation_norm`, `early_importance`, `late_importance`, `timestep_variance`

---

## 指标选择指南

| 目标 | 推荐指标 |
|------|---------|
| **LLM 剪枝** | BlockInfluence, ResidualDropLayer, WandaImportance |
| **LoRA 秩选择** | IntrinsicDimensionality, IGSensitivity, GradientNorm |
| **量化规划** | QuantizationSensitivity, WeightDistribution, EfficiencyProfiler |
| **蒸馏** | CKA, EffectiveRank, BlockInfluence |
| **安全审计** | AdversarialSensitivity, ActivationAnomalyScore, MembershipInferenceRisk |
| **硬件部署** | EfficiencyProfiler, QuantizationSensitivity, WeightDistribution |
