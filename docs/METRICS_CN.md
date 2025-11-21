# Uni-Layer 层贡献度指标详解

本文档详细介绍Uni-Layer框架中所有层贡献度指标的数学原理、计算方法和使用场景。

## 目录

1. [优化几何指标](#优化几何指标)
2. [谱与核方法指标](#谱与核方法指标)
3. [信息论指标](#信息论指标)
4. [表征结构指标](#表征结构指标)
5. [鲁棒性指标](#鲁棒性指标)
6. [指标选择指南](#指标选择指南)

---

## 优化几何指标

### 1. 梯度范数（Gradient Norm）

#### 📐 数学原理

梯度范数测量损失函数对层参数的梯度大小，反映该层对模型优化的贡献程度。

对于第 $\ell$ 层，参数为 $\theta^{(\ell)}$，损失函数为 $\mathcal{L}$，梯度范数定义为：

$$
\text{GradNorm}^{(\ell)} = \left\| \frac{\partial \mathcal{L}}{\partial \theta^{(\ell)}} \right\|_p
$$

其中 $\|\cdot\|_p$ 表示 $L_p$ 范数：
- **L1范数**：$\|g\|_1 = \sum_i |g_i|$
- **L2范数**：$\|g\|_2 = \sqrt{\sum_i g_i^2}$ （最常用）
- **L∞范数**：$\|g\|_\infty = \max_i |g_i|$

#### 🔍 物理意义

- **高梯度范数**：表示该层参数正在积极调整，对当前任务的学习至关重要
- **低梯度范数**：表示该层已接近最优或对任务贡献较小，可能是剪枝候选
- **梯度消失**：梯度范数接近0，表明该层学习困难

#### 💻 使用方法

```python
from uni_layer.metrics import GradientNorm

metric = GradientNorm(
    norm_type='l2',      # 范数类型：'l1', 'l2', 'linf'
    num_batches=10,      # 计算梯度的批次数
    aggregate='mean'     # 聚合方式：'mean', 'sum', 'max'
)
```

#### 📊 返回值

- `gradient_norm`: 梯度范数均值
- `gradient_norm_std`: 梯度范数标准差
- `gradient_norm_max`: 最大梯度范数
- `gradient_norm_min`: 最小梯度范数

#### ⚡ 计算复杂度

- **时间复杂度**：$O(B \times P)$，其中 $B$ 为批次数，$P$ 为参数量
- **空间复杂度**：$O(P)$
- **相对速度**：快（1x基准）

---

### 2. Hessian迹（Hessian Trace）

#### 📐 数学原理

Hessian矩阵的迹测量损失曲面的曲率，反映局部最优的锐度（sharpness）。

Hessian矩阵定义为：

$$
H^{(\ell)} = \frac{\partial^2 \mathcal{L}}{\partial \theta^{(\ell)} \partial \theta^{(\ell)T}}
$$

直接计算Hessian矩阵的迹 $\text{Tr}(H)$ 计算量巨大（$O(P^2)$），我们使用**Hutchinson随机迹估计器**：

$$
\text{Tr}(H) \approx \frac{1}{K} \sum_{k=1}^{K} v_k^T H v_k
$$

其中 $v_k \sim \mathcal{N}(0, I)$ 是随机高斯向量。

利用自动微分，可以将 $v^T H v$ 计算为：

$$
v^T H v = v^T \nabla_\theta \left( \nabla_\theta \mathcal{L} \cdot v \right)
$$

#### 🔍 物理意义

- **高Hessian迹**：损失曲面陡峭（sharp minimum），模型对参数扰动敏感，泛化性能可能较差
- **低Hessian迹**：损失曲面平坦（flat minimum），模型鲁棒性好，泛化性能通常更优
- **层级差异**：不同层的曲率差异揭示了层的功能特化程度

#### 💻 使用方法

```python
from uni_layer.metrics import HessianTrace

metric = HessianTrace(
    num_samples=5,    # Hutchinson估计器的随机向量数量
    num_batches=5     # 计算的批次数（计算量大，建议较少）
)
```

#### 📊 返回值

- `hessian_trace`: Hessian迹的估计值
- `hessian_trace_std`: 估计值的标准差（反映估计质量）

#### ⚡ 计算复杂度

- **时间复杂度**：$O(B \times K \times P)$
- **空间复杂度**：$O(P)$
- **相对速度**：慢（6x基准）

#### 📚 相关研究

- Yao et al. (2020): "PYHESSIAN: Neural Networks Through the Lens of the Hessian"
- Keskar et al. (2017): "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"

---

### 3. Fisher信息（Fisher Information）

#### 📐 数学原理

Fisher信息矩阵（FIM）测量模型输出分布对参数变化的敏感度，是贝叶斯推断和自然梯度下降的核心。

对于参数 $\theta$，Fisher信息矩阵定义为：

$$
F(\theta) = \mathbb{E}_{p(y|x,\theta)} \left[ \nabla_\theta \log p(y|x,\theta) \nabla_\theta \log p(y|x,\theta)^T \right]
$$

我们计算**经验Fisher信息**（Empirical Fisher），使用训练数据的真实标签：

$$
\hat{F}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p(y_i|x_i,\theta) \nabla_\theta \log p(y_i|x_i,\theta)^T
$$

对于计算效率，我们只计算**对角Fisher**（Fisher diagonal）：

$$
\text{diag}(\hat{F}^{(\ell)}) = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\partial \log p(y_i|x_i)}{\partial \theta^{(\ell)}} \right)^2
$$

Fisher信息迹：

$$
\text{FisherTrace}^{(\ell)} = \sum_{j=1}^{P^{(\ell)}} \hat{F}_{jj}^{(\ell)}
$$

#### 🔍 物理意义

- **高Fisher信息**：参数对输出分布影响大，该层对模型预测至关重要
- **低Fisher信息**：参数改变对输出影响小，该层可能存在冗余
- **应用**：弹性权重整合（EWC）、持续学习、神经架构搜索

#### 💻 使用方法

```python
from uni_layer.metrics import FisherInformation

metric = FisherInformation(
    num_batches=10,
    empirical=True    # True: 经验Fisher, False: 真实Fisher
)
```

#### 📊 返回值

- `fisher_information`: Fisher信息迹
- `fisher_mean`: 平均Fisher信息（归一化到每个参数）

#### ⚡ 计算复杂度

- **时间复杂度**：$O(B \times P)$
- **空间复杂度**：$O(P)$
- **相对速度**：中等（2.5x基准）

#### 📚 相关研究

- Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks" (EWC)
- Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"

---

## 谱与核方法指标

### 4. CKA（Centered Kernel Alignment）

#### 📐 数学原理

CKA是一种表征相似性度量，用于比较两个神经网络层的表征是否捕获相似的信息。

给定两层的激活矩阵 $X \in \mathbb{R}^{n \times d_1}$ 和 $Y \in \mathbb{R}^{n \times d_2}$（$n$ 个样本），CKA定义为：

$$
\text{CKA}(X, Y) = \frac{\text{HSIC}(K_X, K_Y)}{\sqrt{\text{HSIC}(K_X, K_X) \cdot \text{HSIC}(K_Y, K_Y)}}
$$

其中 HSIC（Hilbert-Schmidt Independence Criterion）为：

$$
\text{HSIC}(K_X, K_Y) = \frac{1}{(n-1)^2} \text{Tr}(\tilde{K}_X \tilde{K}_Y)
$$

**核矩阵**：
- 线性核：$K_X = XX^T$
- RBF核：$K_X[i,j] = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$

**中心化核矩阵**：

$$
\tilde{K} = HKH, \quad H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T
$$

CKA的值域为 $[0, 1]$：
- **CKA = 1**：表征完全对齐（线性相关）
- **CKA = 0**：表征完全独立
- **CKA > 0.8**：高度相似（可能存在冗余）

#### 🔍 物理意义

- **层间相似性**：比较不同层的功能特化程度
- **模型比较**：评估不同初始化、架构、训练方法的影响
- **知识蒸馏**：高CKA的层适合作为蒸馏目标
- **剪枝指导**：高度相似的连续层可能存在冗余

#### 💻 使用方法

```python
from uni_layer.metrics import CKA

metric = CKA(
    compare_to='output',  # 比较对象：'output', 'previous', 'input'
    num_batches=10,
    kernel='linear'       # 核函数：'linear', 'rbf'
)
```

#### 📊 返回值

- `cka_score`: CKA相似度得分 [0, 1]

#### ⚡ 计算复杂度

- **时间复杂度**：$O(n^2 d)$，其中 $n$ 为样本数，$d$ 为特征维度
- **空间复杂度**：$O(n^2)$
- **相对速度**：中等（2x基准）

#### 📚 相关研究

- Kornblith et al. (2019): "Similarity of Neural Network Representations Revisited" (NeurIPS)
- Nguyen et al. (2021): "Do Wide and Deep Networks Learn the Same Things?"

---

### 5. 有效秩（Effective Rank）

#### 📐 数学原理

有效秩测量表征矩阵的"真实"维度，反映特征的多样性和冗余程度。

对于激活矩阵 $X \in \mathbb{R}^{n \times d}$，进行奇异值分解：

$$
X = U\Sigma V^T, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)
$$

归一化奇异值：

$$
p_i = \frac{\sigma_i}{\sum_{j=1}^{r} \sigma_j}
$$

**有效秩**定义为奇异值分布的指数熵：

$$
\text{EffectiveRank}(X) = \exp\left(-\sum_{i=1}^{r} p_i \log p_i\right)
$$

**稳定秩**（Stable Rank）是另一种常用度量：

$$
\text{StableRank}(X) = \frac{\|X\|_F^2}{\|X\|_2^2} = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sigma_1^2}
$$

**秩比率**：

$$
\text{RankRatio}(X) = \frac{\text{EffectiveRank}(X)}{\min(n, d)}
$$

值域为 $[0, 1]$，接近1表示充分利用了表征空间。

#### 🔍 物理意义

- **高有效秩**：特征多样，冗余少，表征能力强
- **低有效秩**：特征重复，存在退化（rank collapse）
- **训练动态**：有效秩随训练可能先增后减，反映学习过程中的特征选择
- **容量利用**：秩比率反映层的容量利用效率

#### 💻 使用方法

```python
from uni_layer.metrics import EffectiveRank

metric = EffectiveRank(
    num_batches=10,
    epsilon=1e-10    # 数值稳定性参数
)
```

#### 📊 返回值

- `effective_rank`: 有效秩
- `stable_rank`: 稳定秩
- `rank_ratio`: 秩比率 [0, 1]

#### ⚡ 计算复杂度

- **时间复杂度**：$O(n \times d \times \min(n, d))$ （SVD）
- **空间复杂度**：$O(\min(n, d))$
- **相对速度**：中等（2x基准）

#### 📚 相关研究

- Roy & Vetterli (2007): "The effective rank: A measure of effective dimensionality"
- Vershynin (2018): "High-Dimensional Probability"

---

### 6. NTK迹（Neural Tangent Kernel Trace）

#### 📐 数学原理

神经切线核（NTK）理论在无限宽度极限下分析神经网络的训练动态。NTK迹衡量层参数对模型输出的影响。

对于输出 $f(x; \theta)$，NTK定义为：

$$
\Theta(x, x') = \nabla_\theta f(x; \theta)^T \nabla_\theta f(x'; \theta)
$$

对于第 $\ell$ 层，**层级NTK贡献**为：

$$
\Theta^{(\ell)}(x, x') = \frac{\partial f(x)}{\partial \theta^{(\ell)}}^T \frac{\partial f(x')}{\partial \theta^{(\ell)}}
$$

**NTK迹**（对角元素之和）：

$$
\text{NTKTrace}^{(\ell)} = \sum_{i=1}^{n} \Theta^{(\ell)}(x_i, x_i) = \sum_{i=1}^{n} \left\| \frac{\partial f(x_i)}{\partial \theta^{(\ell)}} \right\|_2^2
$$

实际计算中，我们使用Jacobian矩阵 $J \in \mathbb{R}^{n \times p}$：

$$
J[i, j] = \frac{\partial f(x_i)}{\partial \theta_j^{(\ell)}}
$$

则：

$$
\text{NTKTrace}^{(\ell)} = \text{Tr}(JJ^T) = \|J\|_F^2
$$

#### 🔍 物理意义

- **高NTK迹**：参数对输出影响大，训练时该层变化剧烈
- **低NTK迹**：参数对输出影响小，层可能已"冻结"
- **懒惰训练**：NTK不变时，网络在懒惰训练区（lazy training regime）
- **特征学习**：NTK变化时，网络在主动学习新特征

#### 💻 使用方法

```python
from uni_layer.metrics import NTKTrace

metric = NTKTrace(
    num_samples=100,      # 计算Jacobian的样本数
    num_classes=None      # 输出类别数（自动检测）
)
```

#### 📊 返回值

- `ntk_trace`: NTK迹
- `ntk_trace_per_param`: 归一化到每个参数的NTK迹

#### ⚡ 计算复杂度

- **时间复杂度**：$O(n \times p \times c)$，其中 $c$ 为输出维度
- **空间复杂度**：$O(n \times p)$
- **相对速度**：慢（5x基准）

#### 📚 相关研究

- Jacot et al. (2018): "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
- Arora et al. (2019): "On Exact Computation with an Infinitely Wide Neural Net"

---

## 信息论指标

### 7. 互信息（Mutual Information）

#### 📐 数学原理

互信息（MI）测量两个随机变量之间的统计依赖性，在深度学习中用于量化层激活包含多少关于目标标签的信息。

对于层激活 $Z^{(\ell)}$ 和目标标签 $Y$，互信息定义为：

$$
I(Z^{(\ell)}; Y) = H(Y) - H(Y|Z^{(\ell)})
$$

或等价地：

$$
I(Z^{(\ell)}; Y) = H(Z^{(\ell)}) + H(Y) - H(Z^{(\ell)}, Y)
$$

其中熵定义为：

$$
H(Y) = -\sum_{y} p(y) \log p(y)
$$

$$
H(Y|Z) = -\sum_{z,y} p(z,y) \log p(y|z)
$$

**实际估计**：由于高维连续变量的MI难以精确计算，我们使用**k-最近邻（k-NN）估计器**：

$$
\hat{I}(Z; Y) \approx \psi(k) - \frac{1}{N}\sum_{i=1}^{N} \psi(n_{y_i}(i)) + \psi(N) - \psi(N_y)
$$

其中：
- $\psi$ 是digamma函数
- $n_{y_i}(i)$ 是与样本 $i$ 同类且在 $k$-近邻中的样本数
- $N_y$ 是类别 $y$ 的样本数

对于**分类任务**，使用 `mutual_info_classif`；对于**回归任务**，使用 `mutual_info_regression`。

#### 🔍 物理意义

- **高互信息**：层包含大量任务相关信息，对分类/回归至关重要
- **低互信息**：层可能只学到了输入的统计特性，未捕获任务特定模式
- **信息瓶颈**：深层网络通过压缩去除冗余信息（Tishby的信息瓶颈理论）
- **训练动态**：MI先增后减，反映"拟合-压缩"两阶段

#### 💻 使用方法

```python
from uni_layer.metrics import MutualInformation

metric = MutualInformation(
    num_batches=10,
    task_type='classification',  # 'classification' 或 'regression'
    n_neighbors=3                # k-NN估计器的邻居数
)
```

#### 📊 返回值

- `mutual_information`: 平均互信息
- `mi_max`: 最大互信息（跨特征维度）
- `mi_std`: 互信息标准差

#### ⚡ 计算复杂度

- **时间复杂度**：$O(n \times d \times k \times \log n)$
- **空间复杂度**：$O(n \times d)$
- **相对速度**：中等（2.5x基准）

#### 📚 相关研究

- Tishby & Zaslavsky (2015): "Deep Learning and the Information Bottleneck Principle"
- Saxe et al. (2019): "On the Information Bottleneck Theory of Deep Learning"

---

### 8. 激活熵（Activation Entropy）

#### 📐 数学原理

激活熵测量层表征的随机性和多样性。

对于激活值的分布 $p(z)$，**微分熵**定义为：

$$
H(Z) = -\int p(z) \log p(z) \, dz
$$

对于离散化的激活值（使用直方图）：

$$
\hat{H}(Z) = -\sum_{i=1}^{B} p_i \log p_i
$$

其中 $B$ 是直方图的箱数（bins），$p_i$ 是第 $i$ 个箱的概率。

**归一化熵**：

$$
\hat{H}_{\text{norm}}(Z) = \frac{\hat{H}(Z)}{\log B}
$$

值域 $[0, 1]$，1表示均匀分布（最大熵）。

#### 🔍 物理意义

- **高熵**：激活分布均匀，特征多样性好
- **低熵**：激活集中在少数模式，可能过拟合或特征崩溃
- **稀疏性**：接近零的激活多时，熵低且稀疏性高
- **死亡神经元**：熵接近0，表示ReLU等激活函数导致的神经元死亡

#### 💻 使用方法

```python
from uni_layer.metrics import ActivationEntropy

metric = ActivationEntropy(
    num_batches=10,
    num_bins=50    # 直方图箱数（影响熵估计精度）
)
```

#### 📊 返回值

- `activation_entropy`: 激活熵
- `activation_mean`: 激活均值
- `activation_std`: 激活标准差
- `activation_sparsity`: 稀疏度（接近0的激活比例）

#### ⚡ 计算复杂度

- **时间复杂度**：$O(n \times d)$
- **空间复杂度**：$O(n \times d)$
- **相对速度**：快（1x基准）

#### 📚 相关研究

- Cover & Thomas (2006): "Elements of Information Theory"
- Shwartz-Ziv & Tishby (2017): "Opening the Black Box of Deep Neural Networks via Information"

---

## 表征结构指标

### 9. Jacobian秩（Jacobian Rank）

#### 📐 数学原理

Jacobian矩阵的秩测量层变换的表达能力和非线性程度。

对于输入 $x$ 和层输出 $z = f^{(\ell)}(x)$，Jacobian矩阵定义为：

$$
J^{(\ell)} = \frac{\partial f^{(\ell)}(x)}{\partial x} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
$$

对于批量输入 $X = [x_1, \ldots, x_n]$，我们计算每个样本的Jacobian并堆叠：

$$
\mathcal{J} = \begin{bmatrix} J_1 \\ J_2 \\ \vdots \\ J_n \end{bmatrix} \in \mathbb{R}^{(n \times d_{\text{out}}) \times d_{\text{in}}}
$$

通过奇异值分解：

$$
\mathcal{J} = U\Sigma V^T, \quad \Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)
$$

**秩**定义为显著奇异值的数量：

$$
\text{Rank}(\mathcal{J}) = \#\{\sigma_i : \sigma_i > \tau \cdot \sigma_{\max}\}
$$

其中 $\tau$ 是阈值（如 $10^{-3}$）。

**条件数**（衡量数值稳定性）：

$$
\kappa(\mathcal{J}) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

**秩比率**：

$$
\text{RankRatio}(\mathcal{J}) = \frac{\text{Rank}(\mathcal{J})}{\min(n \times d_{\text{out}}, d_{\text{in}})}
$$

#### 🔍 物理意义

- **高秩**：层进行复杂的非线性变换，表达能力强
- **低秩**：层近似线性变换，可能退化或过度正则化
- **秩缺陷**：表示层学习了低维流形，可能有利于泛化
- **条件数**：高条件数表示梯度不稳定，训练困难

#### 💻 使用方法

```python
from uni_layer.metrics import JacobianRank

metric = JacobianRank(
    num_samples=100,        # 计算Jacobian的样本数
    rank_threshold=1e-3     # 奇异值阈值（相对于最大值）
)
```

#### 📊 返回值

- `jacobian_rank`: Jacobian秩
- `jacobian_rank_ratio`: 秩比率 [0, 1]
- `jacobian_condition`: 条件数
- `jacobian_max_sv`: 最大奇异值

#### ⚡ 计算复杂度

- **时间复杂度**：$O(n \times d_{\text{out}} \times d_{\text{in}}^2)$
- **空间复杂度**：$O(n \times d_{\text{out}} \times d_{\text{in}})$
- **相对速度**：慢（5x基准）

#### 📚 相关研究

- Jacot et al. (2020): "Implicit regularization of random feature models"
- Golub & Van Loan (2013): "Matrix Computations"

---

## 鲁棒性指标

### 10. DropLayer鲁棒性（DropLayer Robustness）

#### 📐 数学原理

DropLayer通过消融研究（ablation study）直接测量层的重要性。

**基准性能**（完整模型）：

$$
\mathcal{L}_{\text{base}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathcal{L}(f(x; \theta), y)]
$$

**Drop第 $\ell$ 层后的性能**：

令 $f^{-\ell}$ 表示移除或零化第 $\ell$ 层输出的模型：

$$
\mathcal{L}_{\text{drop}}^{(\ell)} = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathcal{L}(f^{-\ell}(x; \theta), y)]
$$

**性能退化**（损失增加）：

$$
\Delta\mathcal{L}^{(\ell)} = \mathcal{L}_{\text{drop}}^{(\ell)} - \mathcal{L}_{\text{base}}
$$

**性能比率**：

$$
R^{(\ell)} = \frac{\mathcal{L}_{\text{drop}}^{(\ell)}}{\mathcal{L}_{\text{base}}}
$$

对于分类任务，也可测量**准确率下降**：

$$
\Delta\text{Acc}^{(\ell)} = \text{Acc}_{\text{base}} - \text{Acc}_{\text{drop}}^{(\ell)}
$$

**Drop策略**：
- **零化**（Zero）：$z^{(\ell)} \leftarrow 0$
- **恒等映射**（Identity）：$z^{(\ell)} \leftarrow z^{(\ell-1)}$（跳过该层）

#### 🔍 物理意义

- **大性能退化**：该层对模型至关重要，是核心功能层
- **小性能退化**：该层可能冗余，是剪枝候选
- **性能提升**（负退化）：该层可能过拟合或引入噪声
- **崩溃**：某些关键层（如首层、尾层）Drop后模型完全失效

#### 💻 使用方法

```python
from uni_layer.metrics import DropLayerRobustness

metric = DropLayerRobustness(
    num_batches=10,
    metric='loss',        # 'loss' 或 'accuracy'
    drop_type='zero'      # 'zero' 或 'identity'
)
```

#### 📊 返回值

- `droplayer_loss_increase`: 损失增加量
- `droplayer_loss_ratio`: 损失比率
- `droplayer_acc_decrease`: 准确率下降（如果metric='accuracy'）
- `droplayer_acc_ratio`: 准确率比率

#### ⚡ 计算复杂度

- **时间复杂度**：$O(2 \times B \times T)$，其中 $T$ 为前向传播时间
- **空间复杂度**：$O(1)$（相对于激活）
- **相对速度**：中等（3x基准，需两次前向传播）

#### 📚 相关研究

- Morcos et al. (2018): "Insights on representational similarity in neural networks with canonical correlation"
- Frankle & Carbin (2019): "The Lottery Ticket Hypothesis"

---

## 指标选择指南

### 按应用场景选择

#### 🔧 模型压缩（剪枝）

**推荐指标组合**：
1. **GradientNorm** - 快速识别低贡献层
2. **DropLayerRobustness** - 直接验证层的必要性
3. **FisherInformation** - 基于理论的重要性度量

**策略**：
- 低GradientNorm + 小DropLayer退化 → 安全剪枝
- 低FisherInformation → 参数冗余，可大幅剪枝

#### 📚 知识蒸馏

**推荐指标组合**：
1. **CKA** - 选择与教师模型高度对齐的层
2. **GradientNorm** - 选择学习活跃的层
3. **EffectiveRank** - 确保表征多样性

**策略**：
- 高CKA层：适合特征蒸馏
- 中间层高GradientNorm：适合中间层监督

#### 🎯 PEFT（适配器插入）

**推荐指标组合**：
1. **GradientNorm** - 高梯度层适合插入适配器
2. **NTKTrace** - 参数影响力大的层
3. **JacobianRank** - 表达能力强的层

**策略**：
- 插入到高GradientNorm层以最大化微调效果
- 避免插入到低秩层（已退化）

#### 🔍 模型可解释性

**推荐指标组合**：
1. **MutualInformation** - 任务相关信息含量
2. **ActivationEntropy** - 表征多样性
3. **CKA** - 层间功能相似性

**策略**：
- 绘制MI曲线理解信息流动
- 比较不同模型的CKA矩阵

### 按计算预算选择

#### ⚡ 快速分析（< 1分钟）

仅使用：
- **GradientNorm**（1x速度）
- **ActivationEntropy**（1x速度）

#### 🕐 平衡分析（1-5分钟）

使用：
- **GradientNorm**
- **CKA**（2x速度）
- **EffectiveRank**（2x速度）
- **DropLayerRobustness**（3x速度）

#### 🐢 全面分析（> 5分钟）

使用全部指标，包括：
- **HessianTrace**（6x速度）
- **NTKTrace**（5x速度）
- **JacobianRank**（5x速度）

### 按模型类型选择

#### 🤖 Transformer模型

**必选**：
- **GradientNorm** - 识别关键注意力层
- **CKA** - 比较attention vs FFN层

**可选**：
- **MutualInformation** - 分析信息瓶颈
- **AttentionFlow**（即将支持）- Transformer专用

#### 👁️ 视觉模型（CNN/ViT）

**必选**：
- **GradientNorm**
- **EffectiveRank** - 检测特征多样性

**可选**：
- **DropLayerRobustness** - 识别关键卷积层
- **PatchAttribution**（即将支持）- ViT专用

#### 📊 图神经网络

**必选**：
- **GradientNorm**
- **JacobianRank** - 检测过平滑问题

**可选**：
- **ActivationEntropy** - 节点表征多样性

### 计算复杂度对比表

| 指标 | 时间复杂度 | 空间复杂度 | 相对速度 | 是否需要标签 |
|------|-----------|-----------|---------|------------|
| GradientNorm | $O(BP)$ | $O(P)$ | 1x | ✅ |
| ActivationEntropy | $O(nd)$ | $O(nd)$ | 1x | ❌ |
| CKA | $O(n^2d)$ | $O(n^2)$ | 2x | ❌ |
| EffectiveRank | $O(nd\min(n,d))$ | $O(\min(n,d))$ | 2x | ❌ |
| FisherInformation | $O(BP)$ | $O(P)$ | 2.5x | ✅ |
| MutualInformation | $O(ndk\log n)$ | $O(nd)$ | 2.5x | ✅ |
| DropLayerRobustness | $O(2BT)$ | $O(1)$ | 3x | ✅ |
| NTKTrace | $O(npc)$ | $O(np)$ | 5x | ❌ |
| JacobianRank | $O(nd_{\text{out}}d_{\text{in}}^2)$ | $O(nd_{\text{out}}d_{\text{in}})$ | 5x | ❌ |
| HessianTrace | $O(BKP)$ | $O(P)$ | 6x | ✅ |

**符号说明**：
- $B$: 批次数, $P$: 参数量, $n$: 样本数, $d$: 特征维度
- $k$: k-NN邻居数, $T$: 前向传播时间, $c$: 输出类别数

---

## 最佳实践

### 1. 渐进式分析策略

```python
# 第一步：快速筛选（1分钟）
quick_metrics = [GradientNorm(num_batches=5)]
quick_results = analyzer.compute_metrics(quick_metrics, data_loader)

# 第二步：重点分析（5分钟）
important_layers = analyzer.get_top_k_layers(quick_results, "gradient_norm", k=10)
detailed_metrics = [CKA(), EffectiveRank(), MutualInformation()]
detailed_results = analyzer.compute_metrics(
    detailed_metrics,
    data_loader,
    layer_names=important_layers
)

# 第三步：深入研究（可选）
critical_layers = analyzer.get_top_k_layers(detailed_results, "cka_score", k=3)
deep_metrics = [HessianTrace(num_batches=3), NTKTrace(num_samples=50)]
deep_results = analyzer.compute_metrics(
    deep_metrics,
    data_loader,
    layer_names=critical_layers
)
```

### 2. 多指标融合

不同指标捕获不同视角，建议融合多个指标：

```python
# 加权融合
weights = {'gradient_norm': 0.4, 'cka_score': 0.3, 'fisher_information': 0.3}
combined_score = {}

for layer_name in contributions.keys():
    score = sum(
        weights[metric] * contributions[layer_name].get(metric, 0)
        for metric in weights.keys()
    )
    combined_score[layer_name] = score
```

### 3. 统计显著性检验

使用多批次计算并检验统计显著性：

```python
# 增加批次数以获得置信区间
metric = GradientNorm(num_batches=20)
results = analyzer.compute_metrics([metric], data_loader)

# 检查标准差
for layer_name, metrics in results.items():
    mean = metrics['gradient_norm']
    std = metrics['gradient_norm_std']
    print(f"{layer_name}: {mean:.4f} ± {std:.4f}")
```

---

## 参考文献

1. **Gradient-based methods**: Molchanov et al. (2017), "Variational Dropout Sparsifies Deep Neural Networks"
2. **Hessian analysis**: Yao et al. (2020), "PYHESSIAN"
3. **Fisher Information**: Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting"
4. **CKA**: Kornblith et al. (2019), "Similarity of Neural Network Representations Revisited"
5. **NTK**: Jacot et al. (2018), "Neural Tangent Kernel"
6. **Information Theory**: Tishby & Zaslavsky (2015), "Deep Learning and the Information Bottleneck"
7. **Effective Rank**: Roy & Vetterli (2007), "The effective rank"

---

## 附录：数学符号表

| 符号 | 含义 |
|-----|------|
| $\ell$ | 层索引 |
| $\theta^{(\ell)}$ | 第$\ell$层参数 |
| $z^{(\ell)}$ | 第$\ell$层激活 |
| $\mathcal{L}$ | 损失函数 |
| $H(\cdot)$ | 熵 |
| $I(\cdot;\cdot)$ | 互信息 |
| $\|\cdot\|_p$ | $L_p$范数 |
| $\text{Tr}(\cdot)$ | 矩阵的迹 |
| $\mathbb{E}[\cdot]$ | 期望 |
| $\nabla_\theta$ | 对$\theta$的梯度 |
| $\sigma_i$ | 第$i$个奇异值 |
| $\kappa$ | 条件数 |

---

如有疑问或建议，欢迎提Issue或PR！
