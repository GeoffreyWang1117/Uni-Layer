# 模型压缩指南

Uni-Layer 提供了强大的压缩工具，利用层贡献度分析来优化：

- **模型剪枝（Pruning）**：移除冗余权重/神经元
- **知识蒸馏（Knowledge Distillation）**：将大模型知识转移到小模型
- **参数高效微调（PEFT）**：使用最少参数进行微调

所有压缩方法都利用层贡献度分析来智能决策应该压缩、蒸馏或增强什么内容。

---

## 目录

1. [模型剪枝](#模型剪枝)
2. [知识蒸馏](#知识蒸馏)
3. [参数高效微调](#参数高效微调)
4. [最佳实践](#最佳实践)

---

## 模型剪枝

剪枝通过移除不重要的权重或神经元来减少模型大小并提高推理速度。

### 数学基础

对于神经网络参数 θ，我们定义重要性分数：

- **基于幅度**: `I(θ) = |θ|`
- **基于梯度**: `I(θ) = ||∂L/∂θ||₂`
- **基于Fisher信息**: `I(θ) = E[(∂L/∂θ)²]`

**差异化剪枝策略：**

```
剪枝比例_ℓ = 最大比例 · (1 - 归一化贡献度_ℓ)
```

贡献度低的层 → 高剪枝比例
贡献度高的层 → 低剪枝比例

### 快速开始

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm
from uni_layer.compression import LayerPruner, PruningStrategy

# 1. 分析层贡献度
analyzer = LayerAnalyzer(model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=10)],
    data_loader=data_loader
)

# 2. 创建剪枝器
pruner = LayerPruner(
    model,
    contributions,
    strategy=PruningStrategy.GRADIENT_NORM
)

# 3. 计算差异化剪枝比例
pruning_ratios = pruner.compute_layer_pruning_ratios(
    base_ratio=0.3,
    max_ratio=0.6,  # 不重要层最多剪枝60%
    min_ratio=0.1,  # 重要层最少剪枝10%
)

# 4. 应用剪枝
pruned_model = pruner.prune_unstructured(pruning_ratios)
pruned_model = pruner.remove_pruning_masks()

# 5. 获取统计信息
stats = pruner.get_sparsity_stats()
print(f"稀疏度: {stats['overall_sparsity']:.2%}")

speedup = pruner.estimate_speedup()
print(f"预估加速: {speedup['practical_speedup']:.2f}x")
```

### 剪枝类型

#### 1. 非结构化剪枝

移除单个权重（创建稀疏矩阵）：

```python
pruned_model = pruner.prune_unstructured(
    pruning_ratios,
    global_pruning=False  # 使用层特定比例
)
```

**优点：**
- 可达到更高的稀疏度
- 精度损失最小

**缺点：**
- 需要稀疏操作支持
- 无专用硬件加速有限

#### 2. 结构化剪枝

移除整个神经元/通道（硬件友好）：

```python
pruned_model = pruner.prune_structured(
    pruning_ratios,
    dim=0  # 0: 输出神经元, 1: 输入神经元
)
```

**优点：**
- 在标准硬件上真实加速
- 模型尺寸更小

**缺点：**
- 精度损失更多
- 最大稀疏度较低

#### 3. 渐进式剪枝

逐步增加稀疏度（更好的精度保持）：

```python
models = pruner.prune_gradual(
    initial_ratio=0.0,
    final_ratio=0.5,
    num_steps=10,
    structured=False
)

# 每个剪枝步骤后训练
for step, pruned_model in enumerate(models):
    train(pruned_model, data_loader, epochs=1)
```

使用三次稀疏度计划：

```
s_t = s_f + (s_i - s_f) · (1 - t/n)³
```

### API 参考

**`LayerPruner`**

```python
LayerPruner(
    model: nn.Module,
    contributions: Dict[str, Dict[str, float]],
    strategy: PruningStrategy = PruningStrategy.GRADIENT_NORM
)
```

**方法：**

- `compute_layer_pruning_ratios()`: 计算差异化剪枝比例
- `prune_unstructured()`: 应用权重级剪枝
- `prune_structured()`: 应用神经元/通道级剪枝
- `prune_gradual()`: 渐进式剪枝，逐步增加稀疏度
- `remove_pruning_masks()`: 使剪枝永久化
- `get_sparsity_stats()`: 计算稀疏度统计
- `estimate_speedup()`: 估计理论和实际加速

---

## 知识蒸馏

将大型教师模型的知识转移到较小的学生模型。

### 数学基础

**标准蒸馏（Hinton et al., 2015）：**

```
L_KD = α·L_soft + (1-α)·L_hard
```

其中：
- `L_soft = KL(softmax(z_t/T), softmax(z_s/T)) · T²`
- `L_hard = CE(y, softmax(z_s))`
- `T`: 温度（越高 → 越软的概率）

**中间层蒸馏：**

```
L_layer = β·Σ_ℓ w_ℓ · d(h_s^(ℓ), h_t^(ℓ))
```

其中 `d(·,·)` 是距离度量（MSE、余弦、KL）。

**总损失：**

```
L_total = α·L_soft + (1-α)·L_hard + β·L_layer
```

### 快速开始

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA
from uni_layer.compression import KnowledgeDistiller, DistillationConfig

# 1. 分析教师模型层贡献度
analyzer = LayerAnalyzer(teacher_model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=10), CKA(num_batches=10)],
    data_loader=data_loader
)

# 2. 配置蒸馏参数
config = DistillationConfig(
    temperature=4.0,
    alpha=0.7,  # 70%软目标，30%硬目标
    layer_weight=0.5,  # 中间层权重
    distance_metric="mse",
    top_k_layers=3  # 蒸馏前3个重要层
)

# 3. 创建蒸馏器（基于CKA/GradNorm自动选择层）
distiller = KnowledgeDistiller(
    teacher_model=teacher_model,
    student_model=student_model,
    contributions=contributions,
    config=config
)

# 4. 使用蒸馏训练学生模型
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # 蒸馏训练步骤
        loss_components = distiller.train_step(inputs, labels, optimizer)

        # loss_components 包含: total, soft, hard, layer losses
```

### 层选择策略

蒸馏器基于以下标准自动选择要蒸馏的层：

1. **高CKA分数** → 语义上重要的表示
2. **高梯度范数** → 积极学习的特征
3. **平衡深度** → 避免仅选择早期或晚期层

### 距离度量

为中间层选择合适的距离度量：

```python
config = DistillationConfig(
    distance_metric="mse"  # 选项: "mse", "cosine", "kl"
)
```

- **MSE**: `||h_s - h_t||²` - 适用于相似架构
- **余弦**: `1 - cos(h_s, h_t)` - 基于方向，尺度不变
- **KL**: `KL(h_t || h_s)` - 概率性，适用于注意力/激活

### API 参考

**`DistillationConfig`**

```python
DistillationConfig(
    temperature: float = 4.0,
    alpha: float = 0.5,
    layer_weight: float = 0.3,
    distance_metric: str = "mse",
    top_k_layers: int = 3
)
```

**`KnowledgeDistiller`**

```python
KnowledgeDistiller(
    teacher_model: nn.Module,
    student_model: nn.Module,
    contributions: Dict[str, Dict[str, float]],
    config: DistillationConfig = None
)
```

**方法：**

- `train_step()`: 执行一次蒸馏训练步骤
- `compute_loss()`: 计算蒸馏损失分量
- `get_distillation_info()`: 获取配置详情

---

## 参数高效微调

通过添加小型可训练模块（适配器/LoRA）来微调大型模型，同时冻结原始权重。

### 数学基础

**LoRA（低秩适应）：**

```
W' = W + ΔW = W + B·A
```

其中：
- `W ∈ ℝ^(d×k)`: 冻结的预训练权重
- `B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)`: 可训练的低秩矩阵
- `r << min(d,k)`: 秩（通常为1-64）

**前向传播：**

```
h = W·x + (α/r)·B·A·x
```

**参数效率：**

```
# 可训练参数 = r(d + k) << dk
```

对于 `d=k=4096, r=8`：**参数减少99.8%**

**适配器（瓶颈架构）：**

```
h' = h + f(h·W_down)·W_up
```

其中：
- `W_down ∈ ℝ^(d×r)`: 降维投影
- `W_up ∈ ℝ^(r×d)`: 升维投影
- `f`: 非线性（ReLU/GELU）
- `r << d`: 瓶颈维度

### 快速开始

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, FisherInformation
from uni_layer.compression import PEFTOptimizer, AdapterConfig

# 1. 分析层贡献度
analyzer = LayerAnalyzer(model, task_type="classification")
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=10), FisherInformation(num_batches=10)],
    data_loader=data_loader
)

# 2. 配置PEFT
config = AdapterConfig(
    method="lora",  # 或 "adapter"
    rank=8,
    alpha=16.0,
    dropout=0.1,
    adaptive_rank=True  # 每层不同的秩
)

# 3. 创建PEFT优化器
peft_optimizer = PEFTOptimizer(
    model=model,
    contributions=contributions,
    config=config
)

# 4. 选择层（基于重要性自动选择）
selected_layers = peft_optimizer.select_layers(
    top_k=6,
    metric_name="gradient_norm",
    min_contribution=0.01
)

# 5. 计算自适应秩
ranks = peft_optimizer.compute_adaptive_ranks(
    selected_layers,
    base_rank=8,
    max_rank=32
)
# 重要层获得更高的秩（更大容量）
# 不重要层获得更低的秩（参数效率）

# 6. 注入LoRA层
model_with_lora = peft_optimizer.inject_lora(selected_layers, ranks)

# 7. 获取参数效率
efficiency = peft_optimizer.get_parameter_efficiency()
print(f"可训练参数: {efficiency['trainable_params']:,}")
print(f"参数缩减: {efficiency['reduction_ratio']:.1f}x")

# 8. 微调（仅LoRA参数可训练）
trainable_params = [p for p in model_with_lora.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model_with_lora(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
```

### 自适应秩选择

每层的秩由其贡献度决定：

```
秩_ℓ = 基础秩 + (最大秩 - 基础秩) · 归一化贡献度_ℓ
```

**直观理解：**
- 重要层 → 更高秩 → 更大容量
- 不重要层 → 更低秩 → 参数效率

### 方法比较

|方法|参数量|速度|灵活性|使用场景|
|------|----------|-----|-----------|--------|
|**LoRA**|0.1-1%|快|高|大语言模型，多任务|
|**适配器**|1-3%|中等|中等|通用微调|
|**前缀调优**|0.1-0.5%|快|低|语言生成|

### API 参考

**`AdapterConfig`**

```python
AdapterConfig(
    method: str = "lora",  # "lora", "adapter", "prefix_tuning"
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    adaptive_rank: bool = True
)
```

**`PEFTOptimizer`**

```python
PEFTOptimizer(
    model: nn.Module,
    contributions: Dict[str, Dict[str, float]],
    config: AdapterConfig = None
)
```

**方法：**

- `select_layers()`: 自动选择PEFT层
- `compute_adaptive_ranks()`: 计算每层的秩
- `inject_lora()`: 向模型添加LoRA层
- `inject_adapters()`: 向模型添加适配器层
- `get_trainable_parameters()`: 统计可训练参数
- `get_parameter_efficiency()`: 获取效率统计
- `get_peft_info()`: 获取PEFT配置详情

---

## 最佳实践

### 1. 剪枝最佳实践

**从保守开始：**
```python
# 从适度剪枝开始
pruning_ratios = pruner.compute_layer_pruning_ratios(
    base_ratio=0.2,
    max_ratio=0.4,
    min_ratio=0.05
)
```

**使用渐进式剪枝：**
```python
# 更好的精度保持
models = pruner.prune_gradual(
    initial_ratio=0.0,
    final_ratio=0.5,
    num_steps=10
)
```

**剪枝后微调：**
```python
# 恢复剪枝损失的精度
train(pruned_model, data_loader, epochs=5, lr=1e-4)
```

**根据部署选择剪枝类型：**
- CPU/移动设备：使用结构化剪枝
- 支持稀疏操作的GPU：可使用非结构化
- 内存受限：结构化剪枝 + 模型量化

### 2. 蒸馏最佳实践

**温度选择：**
```python
# 更高的温度用于更多知识转移
config = DistillationConfig(temperature=6.0)  # 差距大时
config = DistillationConfig(temperature=3.0)  # 差距小时
```

**平衡损失权重：**
```python
config = DistillationConfig(
    alpha=0.8,  # 架构差异大时，更多权重给软目标
    alpha=0.5,  # 架构相似时平衡
)
```

**选择足够的层：**
```python
config = DistillationConfig(
    top_k_layers=5  # 复杂任务需要更多层
)
```

**预训练学生模型：**
```python
# 先在硬目标上训练学生
train(student, data_loader, epochs=5)
# 然后应用蒸馏
distiller = KnowledgeDistiller(teacher, student, contributions, config)
```

### 3. PEFT最佳实践

**秩选择：**
```python
# 通用任务
config = AdapterConfig(rank=8, adaptive_rank=True)

# 复杂/特定任务
config = AdapterConfig(rank=16, adaptive_rank=True)

# 非常大的模型
config = AdapterConfig(rank=4, adaptive_rank=True)
```

**层选择：**
```python
# 选择更多层以获得更好性能
selected_layers = peft_optimizer.select_layers(
    top_k=10,  # 10层以全面覆盖
    metric_name="gradient_norm"
)
```

**学习率：**
```python
# PEFT使用比全量微调更高的学习率
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)  # vs 全量微调的1e-5
```

**任务切换：**
```python
# 轻松在任务间切换
model_with_lora_task1 = peft_optimizer.inject_lora(layers, ranks)
# 保存LoRA权重，为任务2切换
model_with_lora_task2 = load_lora_weights("task2_lora.pt")
```

### 4. 组合压缩

组合多种技术以实现最大压缩：

```python
# 1. 蒸馏到更小模型
distiller = KnowledgeDistiller(large_model, small_model, contributions)
train_with_distillation(distiller, epochs=10)

# 2. 剪枝蒸馏后的模型
pruner = LayerPruner(small_model, contributions_small)
pruned_model = pruner.prune_structured(ratios)
fine_tune(pruned_model, epochs=5)

# 3. 使用PEFT进行任务适配
peft_optimizer = PEFTOptimizer(pruned_model, contributions_pruned)
model_with_lora = peft_optimizer.inject_lora(layers, ranks)
```

**压缩流水线：**
```
大模型 (100M参数)
    ↓ 蒸馏
小模型 (25M参数, -75%)
    ↓ 剪枝
剪枝模型 (12M参数, -88%)
    ↓ PEFT用于新任务
微调模型 (12M + 0.1M参数, -87.9%)
```

---

## 示例

完整示例位于 `examples/` 目录：

- `pruning_example.py`: 使用差异化策略的模型剪枝
- `distillation_example.py`: 带层选择的知识蒸馏
- `peft_example.py`: 带自适应秩的LoRA微调

运行示例：
```bash
python examples/pruning_example.py
python examples/distillation_example.py
python examples/peft_example.py
```

---

## 参考文献

**剪枝：**
- [幅度剪枝 (Han et al., 2015)](https://arxiv.org/abs/1506.02626)
- [渐进式幅度剪枝 (Zhu & Gupta, 2017)](https://arxiv.org/abs/1710.01878)

**蒸馏：**
- [知识蒸馏 (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [FitNets (Romero et al., 2014)](https://arxiv.org/abs/1412.6550)

**PEFT：**
- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [适配器层 (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751)
