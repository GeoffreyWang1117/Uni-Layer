# Uni-Layer：跨NLP、视觉、语音、图网络和推荐系统的通用层贡献度分析框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 项目概述

**Uni-Layer** 是一个全面的层贡献度分析框架，旨在跨越多种深度学习架构（包括NLP、计算机视觉、语音、图神经网络和推荐系统）统一分析神经网络各层的功能、影响力及其在模型压缩与知识蒸馏中的价值。

不同于传统仅在单一领域研究层贡献度的工作，Uni-Layer提供了一套通用的API，可计算30+种层贡献度指标，涵盖7大类别的理论基础。

## 🎯 核心特性

- **通用指标库**：30+ 层贡献度指标，跨越信息论、优化几何、谱方法等多个理论基础
- **跨架构支持**：适用于Transformer、CNN、GNN、推荐系统、扩散模型等各类架构
- **10+ 模型类别**：支持NLP（BERT、GPT、Llama）、视觉（ViT、ResNet）、语音（Whisper）、图（GCN、GAT）、推荐（DeepFM）等
- **下游任务验证**：在蒸馏、剪枝、PEFT、模型解释等任务中验证有效性
- **即插即用**：简单易用的API，可快速集成到现有PyTorch模型中

## 📊 支持的指标类别

### 1. 信息论 (Information Theory)
- **互信息（Mutual Information）**：测量层激活与目标标签之间的信息量
- **信息熵（Entropy）**：衡量层表征的多样性
- **信息瓶颈（Information Bottleneck）**：即将支持

### 2. 优化几何 (Optimization Geometry)
- **梯度范数（Gradient Norm）**：测量梯度流的强度
- **Hessian迹（Hessian Trace）**：衡量损失曲面的曲率
- **损失景观锐度（Loss Landscape Sharpness）**：评估局部最优的平坦程度

### 3. 概率贝叶斯 (Probabilistic Bayesian)
- **Fisher信息（Fisher Information）**：测量参数对输出分布的敏感度
- **Laplace后验方差（Laplace Posterior Variance）**：即将支持
- **PAC-Bayes层界（PAC-Bayes Layer Bound）**：即将支持

### 4. 谱与核方法 (Spectral & Kernel Methods)
- **CKA（Centered Kernel Alignment）**：测量表征相似性
- **CKTA（Centered Kernel Target Alignment）**：目标对齐度
- **NTK分解（Neural Tangent Kernel）**：神经切线核分析
- **谱有效秩（Spectral Effective Rank）**：表征多样性度量

### 5. 表征结构 (Representation Structure)
- **表征Jacobian秩（Representation Jacobian Rank）**：衡量层的表达能力
- **表征多样性（Representation Diversity）**：特征冗余分析

### 6. 鲁棒性 (Robustness)
- **对抗层敏感性（Adversarial Layer Sensitivity）**：即将支持
- **DropLayer鲁棒性（DropLayer Robustness）**：通过消融测试重要性
- **层扰动影响（Layer Perturbation Impact）**：即将支持

### 7. 架构特定 (Architecture-Specific)
- **注意力流（Attention Flow）**：Transformer专用
- **Patch归因（Patch Attribution）**：ViT专用
- **Token混合影响（Token Mixing Influence）**：MLP-Mixer专用
- **路径积分影响（Path Integral Influence）**：通用方法

## 🏗️ 框架架构

```
uni_layer/
├── metrics/              # 层贡献度指标
│   ├── information_theory/   # 信息论指标
│   ├── optimization/         # 优化几何指标
│   ├── spectral/            # 谱与核方法指标
│   ├── representation/      # 表征结构指标
│   ├── robustness/          # 鲁棒性指标
│   ├── bayesian/            # 贝叶斯指标
│   └── architecture_specific/ # 架构特定指标
├── models/               # 模型加载器与注册表
├── benchmark/            # 基准测试评估流程
├── visualization/        # 可视化工具
└── utils/                # 工具函数
```

## 🚀 快速开始

### 安装

```bash
pip install uni-layer
```

或从源码安装：

```bash
git clone https://github.com/GeoffreyWang1117/Uni-Layer.git
cd Uni-Layer
pip install -e .
```

### 基础用法

```python
from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, MutualInformation

# 初始化分析器
analyzer = LayerAnalyzer(model, task_type='classification')

# 计算层贡献度
contributions = analyzer.compute_metrics(
    metrics=[GradientNorm(), CKA(), MutualInformation()],
    data_loader=train_loader
)

# 可视化结果
analyzer.visualize(contributions, save_path='layer_analysis.png')

# 用于下游任务
pruning_strategy = analyzer.get_pruning_strategy(contributions)
distillation_layers = analyzer.get_distillation_layers(contributions, top_k=6)
```

## 🔬 支持的模型类别

| 类别 | 模型 | 状态 |
|----------|--------|--------|
| **NLP** | BERT、RoBERTa、GPT、Llama3 | ✅ |
| **视觉** | ViT、Swin、ResNet、ConvNeXt | ✅ |
| **语音** | Whisper、Conformer | ✅ |
| **图** | GCN、GraphSAGE、GAT | ✅ |
| **多模态** | CLIP、BLIP | ✅ |
| **推荐** | DeepFM、DCNv2、AutoInt、DLRM | ✅ |
| **强化学习** | Atari CNN、Transformer-RL | ✅ |
| **扩散** | UNet | ✅ |
| **MLP-Mixer** | Mixer-B/16 | ✅ |
| **基线** | MLP、CNN、RNN | ✅ |

## 📖 详细文档

- [快速开始指南（中文）](docs/QUICKSTART_CN.md)
- [指标详细说明（中文）](docs/METRICS_CN.md)
- [API参考文档（中文）](docs/API_CN.md)
- [使用示例](examples/)

## 💡 应用场景

### 1. 知识蒸馏
自动选择最有价值的中间层进行知识传递：
```python
distill_layers = analyzer.get_distillation_layers(
    contributions,
    metric_name="gradient_norm",
    top_k=6
)
```

### 2. 模型剪枝
基于层贡献度生成差异化剪枝策略：
```python
pruning_strategy = analyzer.get_pruning_strategy(
    contributions,
    metric_name="gradient_norm",
    prune_ratio=0.3
)
```

### 3. 参数高效微调（PEFT）
识别最佳Adapter插入位置：
```python
adapter_layers = analyzer.get_peft_insertion_points(
    contributions,
    metric_name="gradient_norm",
    num_adapters=4
)
```

### 4. 模型解释
统一解释不同架构的层级功能：
```python
rankings = analyzer.rank_layers(contributions, "gradient_norm")
depth_analysis = analyzer.aggregate_by_depth(contributions, "gradient_norm")
```

## 🎓 引用

如果您在研究中使用Uni-Layer，请引用：

```bibtex
@article{unilayer2025,
  title={Uni-Layer: 跨深度学习架构的通用层贡献度分析框架},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 贡献

欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 添加新指标

1. 继承`LayerMetric`基类
2. 实现`compute()`方法
3. 添加单元测试
4. 更新文档

示例：
```python
from uni_layer.core.base_metric import LayerMetric

class MyMetric(LayerMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="my_metric",
            category="custom",
            requires_gradient=True,
            requires_data=True,
            **kwargs
        )

    def compute(self, model, layer, layer_name, layer_idx,
                data_loader, device, **kwargs):
        # 您的指标计算逻辑
        return {"my_metric": value}
```

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🔗 相关链接

- [完整文档](https://uni-layer.readthedocs.io)
- [论文](https://arxiv.org/abs/xxx)
- [示例代码](examples/)
- [问题反馈](https://github.com/GeoffreyWang1117/Uni-Layer/issues)

## 🙏 致谢

本项目建立在神经网络可解释性、模型压缩和表征学习的研究基础之上。感谢开源社区的贡献。

## 📮 联系方式

- GitHub Issues: [问题追踪](https://github.com/GeoffreyWang1117/Uni-Layer/issues)
- 讨论区: [GitHub Discussions](https://github.com/GeoffreyWang1117/Uni-Layer/discussions)

---

**注意**：本框架仍在积极开发中。我们欢迎反馈和建议！
