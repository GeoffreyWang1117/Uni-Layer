# Uni-Layer API参考文档（中文版）

本文档提供Uni-Layer框架所有公开API的详细说明。

## 目录

1. [LayerAnalyzer - 核心分析器](#layeranalyzer)
2. [LayerMetric - 指标基类](#layermetric)
3. [优化指标](#优化指标)
4. [谱方法指标](#谱方法指标)
5. [信息论指标](#信息论指标)
6. [表征指标](#表征指标)
7. [鲁棒性指标](#鲁棒性指标)
8. [可视化工具](#可视化工具)
9. [工具函数](#工具函数)

---

## LayerAnalyzer

核心分析器类，用于计算和分析层贡献度。

### 初始化

```python
LayerAnalyzer(
    model: nn.Module,
    task_type: str = "classification",
    device: Optional[str] = None,
    criterion: Optional[nn.Module] = None
)
```

**参数：**
- `model` (nn.Module): 要分析的PyTorch模型
- `task_type` (str): 任务类型
  - `"classification"`: 分类任务（默认）
  - `"regression"`: 回归任务
  - `"generation"`: 生成任务
- `device` (str, 可选): 计算设备，默认自动检测
  - `"cuda"`: 使用GPU
  - `"cpu"`: 使用CPU
- `criterion` (nn.Module, 可选): 损失函数，默认根据task_type自动设置

**返回：**
- LayerAnalyzer实例

**示例：**
```python
from uni_layer import LayerAnalyzer

analyzer = LayerAnalyzer(
    model=my_model,
    task_type="classification",
    device="cuda"
)

print(f"检测到 {len(analyzer.layers)} 个层")
```

---

### 方法：compute_metrics()

计算多个层贡献度指标。

```python
compute_metrics(
    metrics: List[LayerMetric],
    data_loader: Optional[DataLoader] = None,
    layer_names: Optional[List[str]] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Dict[str, float]]
```

**参数：**
- `metrics` (List[LayerMetric]): 要计算的指标列表
- `data_loader` (DataLoader, 可选): 数据加载器
  - 必须提供给需要数据的指标
  - 应返回 `(inputs, targets)` 元组
- `layer_names` (List[str], 可选): 要分析的层名称列表
  - 默认分析所有层
- `verbose` (bool): 是否显示进度条，默认True
- `**kwargs`: 传递给指标的额外参数

**返回：**
- 字典，格式：`{layer_name: {metric_name: value}}`

**示例：**
```python
from uni_layer.metrics import GradientNorm, CKA

contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA()],
    data_loader=train_loader,
    verbose=True
)

# 访问结果
print(contributions["layer1"]["gradient_norm"])
```

---

### 方法：rank_layers()

根据指定指标对层进行排序。

```python
rank_layers(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    ascending: bool = False
) -> List[Tuple[str, float]]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 用于排序的指标名称
- `ascending` (bool): 是否升序排列
  - `False`: 降序（默认，最重要的在前）
  - `True`: 升序

**返回：**
- 列表，每个元素是 `(layer_name, metric_value)` 元组

**示例：**
```python
# 按梯度范数降序排列（最重要的在前）
rankings = analyzer.rank_layers(
    contributions,
    metric_name="gradient_norm",
    ascending=False
)

print("前3个最重要的层：")
for layer_name, value in rankings[:3]:
    print(f"{layer_name}: {value:.4f}")
```

---

### 方法：get_top_k_layers()

获取Top-K最重要的层。

```python
get_top_k_layers(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    k: int = 5,
    ascending: bool = False
) -> List[str]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 用于排序的指标
- `k` (int): 返回前k个层
- `ascending` (bool): 是否选择最小值

**返回：**
- 层名称列表

**示例：**
```python
# 获取梯度范数最大的5层
top_layers = analyzer.get_top_k_layers(
    contributions,
    metric_name="gradient_norm",
    k=5
)

print("Top-5重要层:", top_layers)
```

---

### 方法：get_pruning_strategy()

生成层级剪枝策略。

```python
get_pruning_strategy(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str = "gradient_norm",
    prune_ratio: float = 0.3
) -> Dict[str, float]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 基于哪个指标生成策略
- `prune_ratio` (float): 总体剪枝率 [0, 1]

**返回：**
- 字典，格式：`{layer_name: layer_prune_ratio}`

**策略说明：**
- 重要层（高指标值）：分配较低剪枝率
- 不重要层（低指标值）：分配较高剪枝率
- 剪枝率上限为90%

**示例：**
```python
strategy = analyzer.get_pruning_strategy(
    contributions,
    metric_name="gradient_norm",
    prune_ratio=0.3  # 总体30%剪枝
)

for layer, ratio in strategy.items():
    print(f"{layer}: 剪枝{ratio:.1%}")
```

---

### 方法：get_distillation_layers()

选择知识蒸馏层。

```python
get_distillation_layers(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str = "gradient_norm",
    top_k: int = 6
) -> List[str]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 选择依据的指标
  - `"gradient_norm"`: 选择学习活跃的层
  - `"cka_score"`: 选择表征丰富的层
- `top_k` (int): 选择层数

**返回：**
- 层名称列表

**示例：**
```python
# 基于梯度范数选择
distill_layers = analyzer.get_distillation_layers(
    contributions,
    metric_name="gradient_norm",
    top_k=6
)

print("蒸馏层:", distill_layers)
```

---

### 方法：get_peft_insertion_points()

识别PEFT适配器插入点。

```python
get_peft_insertion_points(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str = "gradient_norm",
    num_adapters: int = 4
) -> List[str]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 选择依据
- `num_adapters` (int): 适配器数量

**返回：**
- 层名称列表

**建议：**
- 高梯度层适合插入适配器
- 避免插入到低秩层（已退化）

**示例：**
```python
adapter_layers = analyzer.get_peft_insertion_points(
    contributions,
    metric_name="gradient_norm",
    num_adapters=4
)

print("适配器插入位置:", adapter_layers)
```

---

### 方法：aggregate_by_depth()

按深度聚合层贡献度。

```python
aggregate_by_depth(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    num_bins: int = 5
) -> Dict[str, float]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 要聚合的指标
- `num_bins` (int): 深度区间数

**返回：**
- 字典，格式：`{depth_bin: avg_value}`

**示例：**
```python
depth_analysis = analyzer.aggregate_by_depth(
    contributions,
    metric_name="gradient_norm",
    num_bins=3  # 早期/中期/后期
)

for depth, value in depth_analysis.items():
    print(f"{depth}: {value:.4f}")
```

---

### 方法：get_summary_statistics()

获取指标的统计摘要。

```python
get_summary_statistics(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str
) -> Dict[str, float]
```

**参数：**
- `contributions`: compute_metrics()的返回值
- `metric_name` (str): 要统计的指标

**返回：**
- 统计字典，包含：
  - `mean`: 均值
  - `std`: 标准差
  - `min`: 最小值
  - `max`: 最大值
  - `median`: 中位数
  - `q25`: 25%分位数
  - `q75`: 75%分位数

**示例：**
```python
stats = analyzer.get_summary_statistics(
    contributions,
    "gradient_norm"
)

print(f"均值: {stats['mean']:.4f}")
print(f"标准差: {stats['std']:.4f}")
```

---

## LayerMetric

所有指标的抽象基类。

### 基类属性

```python
class LayerMetric:
    name: str                  # 指标名称
    category: str              # 指标类别
    requires_gradient: bool    # 是否需要梯度
    requires_data: bool        # 是否需要数据
    batch_size: int           # 批次大小
    config: dict              # 其他配置
```

### 抽象方法：compute()

所有子类必须实现此方法。

```python
@abstractmethod
def compute(
    self,
    model: nn.Module,
    layer: nn.Module,
    layer_name: str,
    layer_idx: int,
    data_loader: Optional[Any] = None,
    device: str = "cuda",
    **kwargs
) -> Dict[str, float]:
    pass
```

**参数：**
- `model`: 完整模型
- `layer`: 当前分析的层
- `layer_name`: 层名称
- `layer_idx`: 层索引
- `data_loader`: 数据加载器
- `device`: 计算设备
- `**kwargs`: 额外参数

**返回：**
- 指标值字典

---

## 优化指标

### GradientNorm - 梯度范数

```python
GradientNorm(
    norm_type: str = 'l2',
    num_batches: int = 10,
    aggregate: str = 'mean'
)
```

**参数：**
- `norm_type`: 范数类型
  - `"l1"`: L1范数（绝对值和）
  - `"l2"`: L2范数（欧几里得范数）
  - `"linf"`: L∞范数（最大值）
- `num_batches`: 批次数
- `aggregate`: 聚合方式（mean/sum/max）

**返回指标：**
- `gradient_norm`: 梯度范数
- `gradient_norm_std`: 标准差
- `gradient_norm_max`: 最大值
- `gradient_norm_min`: 最小值

**数学公式：**
$$\text{GradNorm} = \left\| \frac{\partial \mathcal{L}}{\partial \theta} \right\|_p$$

---

### HessianTrace - Hessian迹

```python
HessianTrace(
    num_samples: int = 5,
    num_batches: int = 5
)
```

**参数：**
- `num_samples`: Hutchinson估计器的随机向量数
- `num_batches`: 批次数

**返回指标：**
- `hessian_trace`: Hessian迹估计值
- `hessian_trace_std`: 标准差

**数学公式：**
$$\text{Tr}(H) \approx \frac{1}{K} \sum_{k=1}^K v_k^T H v_k$$

---

### FisherInformation - Fisher信息

```python
FisherInformation(
    num_batches: int = 10,
    empirical: bool = True
)
```

**参数：**
- `num_batches`: 批次数
- `empirical`: 是否使用经验Fisher

**返回指标：**
- `fisher_information`: Fisher信息迹
- `fisher_mean`: 归一化Fisher信息

**数学公式：**
$$F(\theta) = \mathbb{E}\left[\nabla_\theta \log p(y|x,\theta) \nabla_\theta \log p(y|x,\theta)^T\right]$$

---

## 谱方法指标

### CKA - 核对齐

```python
CKA(
    compare_to: str = 'output',
    num_batches: int = 10,
    kernel: str = 'linear'
)
```

**参数：**
- `compare_to`: 比较对象
  - `"output"`: 与输出层比较
  - `"previous"`: 与前一层比较
  - `"input"`: 与输入比较
- `num_batches`: 批次数
- `kernel`: 核函数（linear/rbf）

**返回指标：**
- `cka_score`: CKA得分 [0, 1]

**数学公式：**
$$\text{CKA}(X, Y) = \frac{\text{HSIC}(K_X, K_Y)}{\sqrt{\text{HSIC}(K_X, K_X) \cdot \text{HSIC}(K_Y, K_Y)}}$$

---

### EffectiveRank - 有效秩

```python
EffectiveRank(
    num_batches: int = 10,
    epsilon: float = 1e-10
)
```

**参数：**
- `num_batches`: 批次数
- `epsilon`: 数值稳定性参数

**返回指标：**
- `effective_rank`: 有效秩
- `stable_rank`: 稳定秩
- `rank_ratio`: 秩比率

**数学公式：**
$$\text{EffectiveRank}(X) = \exp\left(-\sum_i p_i \log p_i\right)$$

其中 $p_i = \sigma_i / \sum_j \sigma_j$

---

### NTKTrace - NTK迹

```python
NTKTrace(
    num_samples: int = 100,
    num_classes: Optional[int] = None
)
```

**参数：**
- `num_samples`: 样本数
- `num_classes`: 类别数（自动检测）

**返回指标：**
- `ntk_trace`: NTK迹
- `ntk_trace_per_param`: 归一化NTK迹

**数学公式：**
$$\text{NTKTrace} = \sum_i \left\| \frac{\partial f(x_i)}{\partial \theta} \right\|_2^2$$

---

## 信息论指标

### MutualInformation - 互信息

```python
MutualInformation(
    num_batches: int = 10,
    task_type: str = 'classification',
    n_neighbors: int = 3
)
```

**参数：**
- `num_batches`: 批次数
- `task_type`: 任务类型（classification/regression）
- `n_neighbors`: k-NN估计器邻居数

**返回指标：**
- `mutual_information`: 平均MI
- `mi_max`: 最大MI
- `mi_std`: MI标准差

**数学公式：**
$$I(Z; Y) = H(Y) - H(Y|Z)$$

---

### ActivationEntropy - 激活熵

```python
ActivationEntropy(
    num_batches: int = 10,
    num_bins: int = 50
)
```

**参数：**
- `num_batches`: 批次数
- `num_bins`: 直方图箱数

**返回指标：**
- `activation_entropy`: 激活熵
- `activation_mean`: 激活均值
- `activation_std`: 激活标准差
- `activation_sparsity`: 稀疏度

**数学公式：**
$$H(Z) = -\sum_i p_i \log p_i$$

---

## 表征指标

### JacobianRank - Jacobian秩

```python
JacobianRank(
    num_samples: int = 100,
    rank_threshold: float = 1e-3
)
```

**参数：**
- `num_samples`: 样本数
- `rank_threshold`: 秩阈值

**返回指标：**
- `jacobian_rank`: Jacobian秩
- `jacobian_rank_ratio`: 秩比率
- `jacobian_condition`: 条件数
- `jacobian_max_sv`: 最大奇异值

**数学公式：**
$$\text{Rank}(J) = \#\{\sigma_i : \sigma_i > \tau \cdot \sigma_{\max}\}$$

---

## 鲁棒性指标

### DropLayerRobustness - DropLayer鲁棒性

```python
DropLayerRobustness(
    num_batches: int = 10,
    metric: str = 'loss',
    drop_type: str = 'zero'
)
```

**参数：**
- `num_batches`: 批次数
- `metric`: 度量方式（loss/accuracy）
- `drop_type`: Drop策略（zero/identity）

**返回指标：**
- `droplayer_loss_increase`: 损失增加量
- `droplayer_loss_ratio`: 损失比率
- `droplayer_acc_decrease`: 准确率下降（如果metric='accuracy'）
- `droplayer_acc_ratio`: 准确率比率

**数学公式：**
$$\Delta\mathcal{L} = \mathcal{L}_{\text{drop}} - \mathcal{L}_{\text{base}}$$

---

## 可视化工具

### plot_layer_contributions() - 层贡献度柱状图

```python
plot_layer_contributions(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None
)
```

**参数：**
- `contributions`: 贡献度字典
- `metric_name`: 要绘制的指标
- `save_path`: 保存路径
- `figsize`: 图像大小
- `title`: 自定义标题

---

### plot_contribution_heatmap() - 多指标热力图

```python
plot_contribution_heatmap(
    contributions: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    cmap: str = 'RdYlGn'
)
```

**参数：**
- `contributions`: 贡献度字典
- `metrics`: 指标列表
- `save_path`: 保存路径
- `figsize`: 图像大小
- `cmap`: 颜色映射

---

### plot_depth_analysis() - 深度分析图

```python
plot_depth_analysis(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    num_bins: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
)
```

**参数：**
- `contributions`: 贡献度字典
- `metric_name`: 指标名称
- `num_bins`: 深度区间数
- `save_path`: 保存路径
- `figsize`: 图像大小

---

### plot_metric_comparison() - 指标对比图

```python
plot_metric_comparison(
    contributions: Dict[str, Dict[str, float]],
    metric1: str,
    metric2: str,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
)
```

**参数：**
- `contributions`: 贡献度字典
- `metric1`: 第一个指标（x轴）
- `metric2`: 第二个指标（y轴）
- `save_path`: 保存路径
- `figsize`: 图像大小

---

## 工具函数

### get_model_layers() - 提取模型层

```python
from uni_layer.utils import get_model_layers

layers = get_model_layers(
    model: nn.Module,
    include_types: Optional[List] = None
) -> OrderedDict
```

**参数：**
- `model`: PyTorch模型
- `include_types`: 包含的层类型列表

**返回：**
- 有序字典：`{layer_name: layer_module}`

---

### identify_layer_type() - 识别层类型

```python
from uni_layer.utils import identify_layer_type

layer_type = identify_layer_type(layer: nn.Module) -> str
```

**参数：**
- `layer`: 层模块

**返回：**
- 层类型字符串（如"transformer_block"、"linear"等）

---

## 完整使用示例

```python
import torch
from torch.utils.data import DataLoader
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, EffectiveRank
from uni_layer.visualization import plot_contribution_heatmap

# 1. 初始化分析器
analyzer = LayerAnalyzer(
    model=model,
    task_type="classification",
    device="cuda"
)

# 2. 计算指标
contributions = analyzer.compute_metrics(
    metrics=[
        GradientNorm(num_batches=10),
        CKA(num_batches=10),
        EffectiveRank(num_batches=10)
    ],
    data_loader=train_loader,
    verbose=True
)

# 3. 分析结果
rankings = analyzer.rank_layers(contributions, "gradient_norm")
print("Top-5层:", [name for name, _ in rankings[:5]])

# 4. 生成策略
pruning = analyzer.get_pruning_strategy(contributions, prune_ratio=0.3)
distill = analyzer.get_distillation_layers(contributions, top_k=6)

# 5. 可视化
plot_contribution_heatmap(
    contributions,
    metrics=["gradient_norm", "cka_score", "effective_rank"],
    save_path="heatmap.png"
)
```

---

## 获取帮助

如有疑问，请：
- 查看[示例代码](../examples/)
- 阅读[快速开始指南](QUICKSTART_CN.md)
- 访问[GitHub讨论区](https://github.com/GeoffreyWang1117/Uni-Layer/discussions)
