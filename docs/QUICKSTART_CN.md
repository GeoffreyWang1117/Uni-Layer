# Uni-Layer 快速开始指南

本指南将帮助您在5分钟内上手Uni-Layer框架。

## 📦 安装

### 方式一：从PyPI安装（推荐）

```bash
pip install uni-layer
```

### 方式二：从源码安装

```bash
git clone https://github.com/GeoffreyWang1117/Uni-Layer.git
cd Uni-Layer
pip install -e .
```

### 依赖要求

- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0（可视化需要）

## 🚀 基础用法

### 步骤1：导入必要组件

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, EffectiveRank
```

### 步骤2：准备模型和数据

```python
import torch
from torch.utils.data import DataLoader

# 您的PyTorch模型
model = YourModel()

# 您的数据加载器
train_loader = DataLoader(dataset, batch_size=32)
```

### 步骤3：初始化分析器

```python
analyzer = LayerAnalyzer(
    model=model,
    task_type='classification',  # 或 'regression', 'generation'
    device='cuda'  # 或 'cpu'
)

# 查看检测到的层
print(f"检测到 {len(analyzer.layers)} 个层")
for name, layer_type in list(analyzer.layer_types.items())[:5]:
    print(f"  {name}: {layer_type}")
```

### 步骤4：计算层贡献度

```python
# 定义要计算的指标
metrics = [
    GradientNorm(num_batches=10),    # 梯度范数
    CKA(num_batches=10),             # 表征相似性
    EffectiveRank(num_batches=10),   # 有效秩
]

# 计算指标
contributions = analyzer.compute_metrics(
    metrics=metrics,
    data_loader=train_loader,
    verbose=True  # 显示进度条
)
```

### 步骤5：分析结果

```python
# 按重要性排序层
rankings = analyzer.rank_layers(contributions, "gradient_norm")

print("前5个最重要的层：")
for layer_name, value in rankings[:5]:
    print(f"  {layer_name}: {value:.4f}")

# 生成剪枝策略
pruning_strategy = analyzer.get_pruning_strategy(
    contributions,
    metric_name="gradient_norm",
    prune_ratio=0.3  # 30%的剪枝率
)

# 获取知识蒸馏层
distill_layers = analyzer.get_distillation_layers(
    contributions,
    metric_name="gradient_norm",
    top_k=6  # 选择前6层
)
```

### 步骤6：可视化（可选）

```python
from uni_layer.visualization import plot_layer_contributions

# 绘制层贡献度柱状图
plot_layer_contributions(
    contributions,
    metric_name="gradient_norm",
    save_path="analysis.png"
)

# 绘制多指标热力图
from uni_layer.visualization import plot_contribution_heatmap

plot_contribution_heatmap(
    contributions,
    metrics=["gradient_norm", "cka_score", "effective_rank"],
    save_path="heatmap.png"
)
```

## 📚 完整示例

### 示例1：简单MLP分类

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA

# 定义模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

# 创建数据
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32)

# 分析
model = SimpleMLP()
analyzer = LayerAnalyzer(model, task_type='classification', device='cpu')

contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA()],
    data_loader=data_loader
)

# 查看结果
for layer_name, metrics in contributions.items():
    print(f"{layer_name}:")
    print(f"  梯度范数: {metrics.get('gradient_norm', 'N/A'):.4f}")
    print(f"  CKA得分: {metrics.get('cka_score', 'N/A'):.4f}")
```

### 示例2：Transformer模型分析

```python
from transformers import BertModel, BertTokenizer

# 加载预训练模型
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备数据
texts = ["这是一个示例文本" for _ in range(100)]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
data_loader = DataLoader(dataset, batch_size=16)

# 分析
analyzer = LayerAnalyzer(model, task_type='classification')

# 只分析Transformer层（跳过embedding等）
transformer_layers = [
    name for name, layer_type in analyzer.layer_types.items()
    if 'transformer' in layer_type or 'attention' in name
]

contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), EffectiveRank()],
    data_loader=data_loader,
    layer_names=transformer_layers
)
```

## 💡 常见用例

### 用例1：知识蒸馏层选择

```python
# 计算层贡献度
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA()],
    data_loader=train_loader
)

# 方法1：基于梯度范数选择学习活跃的层
distill_layers_grad = analyzer.get_distillation_layers(
    contributions,
    metric_name="gradient_norm",
    top_k=6
)

# 方法2：基于CKA选择表征丰富的层
distill_layers_cka = analyzer.get_distillation_layers(
    contributions,
    metric_name="cka_score",
    top_k=6
)

print("推荐蒸馏层（基于梯度）:", distill_layers_grad)
print("推荐蒸馏层（基于CKA）:", distill_layers_cka)
```

### 用例2：智能剪枝策略

```python
from uni_layer.metrics import DropLayerRobustness

# 计算层重要性
contributions = analyzer.compute_metrics(
    metrics=[
        GradientNorm(),
        DropLayerRobustness(num_batches=5)
    ],
    data_loader=train_loader
)

# 生成差异化剪枝策略
pruning_strategy = analyzer.get_pruning_strategy(
    contributions,
    metric_name="gradient_norm",
    prune_ratio=0.4  # 总体40%剪枝率
)

# 显示每层的剪枝比例
for layer_name, ratio in pruning_strategy.items():
    drop_impact = contributions[layer_name].get('droplayer_loss_increase', 0)
    print(f"{layer_name}:")
    print(f"  剪枝比例: {ratio:.1%}")
    print(f"  Drop影响: {drop_impact:.4f}")
```

### 用例3：PEFT适配器位置选择

```python
# 对于参数高效微调，选择梯度大的层插入适配器
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), NTKTrace()],
    data_loader=train_loader
)

# 选择适配器插入点
adapter_positions = analyzer.get_peft_insertion_points(
    contributions,
    metric_name="gradient_norm",
    num_adapters=4
)

print("推荐的适配器插入位置：")
for i, layer_name in enumerate(adapter_positions, 1):
    grad_norm = contributions[layer_name]['gradient_norm']
    print(f"{i}. {layer_name} (梯度范数: {grad_norm:.4f})")
```

### 用例4：模型深度分析

```python
from uni_layer.metrics import MutualInformation, ActivationEntropy

# 分析信息流动
contributions = analyzer.compute_metrics(
    metrics=[
        MutualInformation(task_type='classification'),
        ActivationEntropy()
    ],
    data_loader=train_loader
)

# 按深度聚合
depth_mi = analyzer.aggregate_by_depth(
    contributions,
    metric_name="mutual_information",
    num_bins=5  # 分为5个深度区间
)

depth_entropy = analyzer.aggregate_by_depth(
    contributions,
    metric_name="activation_entropy",
    num_bins=5
)

print("各深度段的互信息：")
for depth, mi in depth_mi.items():
    entropy = depth_entropy[depth]
    print(f"{depth}: MI={mi:.4f}, 熵={entropy:.4f}")
```

## ⚙️ 高级配置

### 控制计算预算

```python
# 快速分析：减少批次数
quick_metrics = [
    GradientNorm(num_batches=3),  # 默认10
    CKA(num_batches=3)
]

# 精确分析：增加批次数
precise_metrics = [
    GradientNorm(num_batches=20),
    CKA(num_batches=20)
]
```

### 只分析特定层

```python
# 只分析线性层
linear_layers = [
    name for name, layer in analyzer.layers.items()
    if isinstance(layer, nn.Linear)
]

contributions = analyzer.compute_metrics(
    metrics=[GradientNorm()],
    data_loader=train_loader,
    layer_names=linear_layers  # 指定层
)
```

### 自定义损失函数

```python
# 使用自定义损失函数
custom_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

analyzer = LayerAnalyzer(
    model=model,
    task_type='classification',
    criterion=custom_criterion  # 自定义损失
)
```

## 🐛 常见问题

### 问题1：内存不足（OOM）

**解决方案**：

```python
# 1. 减少批次数
metric = GradientNorm(num_batches=3)  # 从10减到3

# 2. 使用CPU
analyzer = LayerAnalyzer(model, device='cpu')

# 3. 减少数据加载器的batch_size
data_loader = DataLoader(dataset, batch_size=8)  # 从32减到8

# 4. 分批计算指标
metrics_batch1 = [GradientNorm(), CKA()]
results1 = analyzer.compute_metrics(metrics_batch1, data_loader)

metrics_batch2 = [EffectiveRank(), MutualInformation()]
results2 = analyzer.compute_metrics(metrics_batch2, data_loader)

# 合并结果
for layer_name in results1:
    results1[layer_name].update(results2[layer_name])
```

### 问题2：计算太慢

**解决方案**：

```python
# 1. 只使用快速指标
fast_metrics = [
    GradientNorm(),      # 1x速度
    ActivationEntropy()  # 1x速度
]

# 2. 避免使用慢速指标
# 慢: HessianTrace, NTKTrace, JacobianRank

# 3. 启用GPU加速
analyzer = LayerAnalyzer(model, device='cuda')

# 4. 减少样本数
metric = NTKTrace(num_samples=50)  # 从100减到50
```

### 问题3：指标返回None

**原因与解决**：

```python
# 原因1：数据格式不正确
# 数据加载器应返回 (inputs, targets) 元组
for batch in data_loader:
    inputs, targets = batch  # 确保这样解包成功
    break

# 原因2：缺少标签但指标需要标签
# 检查指标是否需要标签
print(metric.requires_data)  # 应为True
print(metric.requires_gradient)  # 某些指标需要梯度

# 原因3：层类型不兼容
# 某些指标可能不适用于特定层类型（如Embedding层）
```

### 问题4：无法检测到层

**解决方案**：

```python
# 1. 手动指定层
from collections import OrderedDict

manual_layers = OrderedDict()
manual_layers['layer1'] = model.layer1
manual_layers['layer2'] = model.layer2

# 直接设置
analyzer.layers = manual_layers

# 2. 查看模型结构
print(model)

# 3. 查看自动检测的层
for name, layer in analyzer.layers.items():
    print(f"{name}: {type(layer)}")
```

## 📊 性能优化建议

### 1. 渐进式分析

```python
# 第一步：快速筛选（30秒）
quick = analyzer.compute_metrics(
    metrics=[GradientNorm(num_batches=3)],
    data_loader=train_loader
)

# 第二步：重点分析（2分钟）
important = analyzer.get_top_k_layers(quick, "gradient_norm", k=10)
medium = analyzer.compute_metrics(
    metrics=[CKA(), EffectiveRank()],
    data_loader=train_loader,
    layer_names=important
)

# 第三步：深度研究（可选，5分钟）
critical = analyzer.get_top_k_layers(medium, "cka_score", k=3)
deep = analyzer.compute_metrics(
    metrics=[HessianTrace(num_batches=3)],
    data_loader=train_loader,
    layer_names=critical
)
```

### 2. 缓存结果

```python
import pickle

# 保存结果
with open('contributions.pkl', 'wb') as f:
    pickle.dump(contributions, f)

# 加载结果
with open('contributions.pkl', 'rb') as f:
    contributions = pickle.load(f)
```

### 3. 并行计算（高级）

```python
# 注意：某些指标可能不支持多进程
from torch.multiprocessing import Pool

# 这是伪代码，实际实现需要更多处理
def analyze_layer(layer_name):
    # 在独立进程中分析单个层
    pass
```

## 📖 下一步

- 阅读[指标详细文档](METRICS_CN.md)了解各指标的数学原理
- 查看[examples/](../examples/)目录获取更多示例
- 参考[API文档](API_CN.md)了解完整API
- 加入[讨论区](https://github.com/GeoffreyWang1117/Uni-Layer/discussions)提问

## 🆘 获取帮助

- 📖 [完整文档](https://uni-layer.readthedocs.io)
- 💬 [GitHub讨论](https://github.com/GeoffreyWang1117/Uni-Layer/discussions)
- 🐛 [问题追踪](https://github.com/GeoffreyWang1117/Uni-Layer/issues)

---

**祝您使用愉快！如有问题欢迎反馈。**
