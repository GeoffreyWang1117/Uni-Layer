"""
Uni-Layer框架基础使用示例（中文详解版）

本脚本演示如何：
1. 初始化LayerAnalyzer分析器
2. 计算多种层贡献度指标
3. 可视化分析结果
4. 应用于下游任务（剪枝、蒸馏、PEFT）

作者：Uni-Layer团队
日期：2025-01
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 导入Uni-Layer核心组件
from uni_layer import LayerAnalyzer
from uni_layer.metrics import (
    GradientNorm,          # 梯度范数指标
    CKA,                   # 核对齐指标
    EffectiveRank,         # 有效秩指标
    MutualInformation,     # 互信息指标
    ActivationEntropy,     # 激活熵指标
    DropLayerRobustness,   # DropLayer鲁棒性指标
)
from uni_layer.visualization import (
    plot_layer_contributions,    # 绘制层贡献度柱状图
    plot_contribution_heatmap,   # 绘制多指标热力图
    plot_depth_analysis,         # 绘制深度分析图
)


# ==================== 第一部分：定义模型 ====================

class SimpleClassifier(nn.Module):
    """
    简单的多层感知机（MLP）分类器

    架构：
        Input(784) -> Linear(512) -> ReLU -> Dropout(0.2) ->
        Linear(256) -> ReLU -> Dropout(0.2) ->
        Linear(128) -> ReLU -> Dropout(0.2) ->
        Linear(10)

    参数：
        input_dim: 输入维度，默认784（如MNIST的28x28展平）
        hidden_dims: 隐藏层维度列表
        num_classes: 输出类别数
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], num_classes=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        # 将输入展平为一维向量
        x = x.view(x.size(0), -1)
        return self.model(x)


# ==================== 第二部分：主函数 ====================

def main():
    """主函数：演示Uni-Layer的完整使用流程"""

    print("=" * 60)
    print("Uni-Layer框架 - 基础使用示例（中文版）")
    print("=" * 60)

    # ------------------ 步骤1：创建模拟数据集 ------------------
    print("\n[1/5] 创建模拟数据集...")

    num_samples = 1000      # 样本数量
    input_dim = 784         # 输入维度（28x28图像展平）
    num_classes = 10        # 类别数（如0-9数字）

    # 生成随机数据（实际使用时替换为真实数据）
    X = torch.randn(num_samples, input_dim)  # 输入特征
    y = torch.randint(0, num_classes, (num_samples,))  # 标签

    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"✓ 数据集创建完成：{num_samples}个样本，{num_classes}个类别")

    # ------------------ 步骤2：初始化模型 ------------------
    print("\n[2/5] 初始化模型...")

    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        num_classes=num_classes
    )

    print(f"✓ 模型初始化完成")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------ 步骤3：初始化LayerAnalyzer ------------------
    print("\n[3/5] 初始化LayerAnalyzer分析器...")

    analyzer = LayerAnalyzer(
        model=model,
        task_type="classification",  # 任务类型：classification/regression
        device="cpu"  # 使用CPU（如有GPU可改为"cuda"）
    )

    # 分析器会自动检测模型中的所有层
    print(f"✓ 分析器初始化完成")
    print(f"  检测到层数: {len(analyzer.layers)}")
    print(f"  设备: {analyzer.device}")
    print(f"  任务类型: {analyzer.task_type}")

    # ------------------ 步骤4：计算层贡献度指标 ------------------
    print("\n[4/5] 计算层贡献度指标...")
    print("这可能需要几分钟，请稍候...\n")

    # 定义要计算的指标列表
    metrics = [
        # 1. 梯度范数：测量梯度流强度，快速指标
        GradientNorm(num_batches=5),

        # 2. CKA：测量表征与输出的相似性，中等速度
        CKA(num_batches=5),

        # 3. 有效秩：测量表征多样性，中等速度
        EffectiveRank(num_batches=5),

        # 4. 激活熵：测量激活分布的多样性，快速指标
        ActivationEntropy(num_batches=5),

        # 5. DropLayer鲁棒性：通过消融测试层重要性，较慢
        DropLayerRobustness(num_batches=5),
    ]

    # 计算所有指标
    contributions = analyzer.compute_metrics(
        metrics=metrics,
        data_loader=data_loader,
        verbose=True  # 显示进度条
    )

    # ------------------ 步骤5：分析结果 ------------------
    print("\n[5/5] 分析结果展示：")
    print("=" * 60)

    # 5.1 打印每层的详细指标
    print("\n【层级详细指标】\n")
    for layer_name, layer_metrics in contributions.items():
        print(f"{layer_name}:")
        print(f"  层类型: {layer_metrics.get('layer_type', '未知')}")
        print(f"  梯度范数: {layer_metrics.get('gradient_norm', 'N/A'):.4f}")
        print(f"  CKA得分: {layer_metrics.get('cka_score', 'N/A'):.4f}")
        print(f"  有效秩: {layer_metrics.get('effective_rank', 'N/A'):.2f}")
        print(f"  激活熵: {layer_metrics.get('activation_entropy', 'N/A'):.4f}")

        # DropLayer指标（如果有）
        drop_loss = layer_metrics.get('droplayer_loss_increase', None)
        if drop_loss is not None:
            print(f"  DropLayer损失增加: {drop_loss:.4f}")
        print()

    # 5.2 层重要性排名
    print("=" * 60)
    print("【层重要性排名（基于梯度范数）】")
    print("=" * 60)

    rankings = analyzer.rank_layers(contributions, "gradient_norm")
    print("\n前5个最重要的层：")
    for i, (layer_name, value) in enumerate(rankings[:5], 1):
        layer_type = contributions[layer_name].get('layer_type', '未知')
        print(f"{i}. {layer_name} (类型: {layer_type})")
        print(f"   梯度范数: {value:.4f}")

    # 5.3 知识蒸馏层推荐
    print("\n" + "=" * 60)
    print("【知识蒸馏推荐层】")
    print("=" * 60)

    distill_layers = analyzer.get_distillation_layers(
        contributions,
        metric_name="gradient_norm",  # 可选：cka_score, effective_rank等
        top_k=3  # 选择前3层
    )

    print("\n推荐用于知识蒸馏的层：")
    for i, layer_name in enumerate(distill_layers, 1):
        grad_norm = contributions[layer_name]['gradient_norm']
        cka = contributions[layer_name].get('cka_score', 'N/A')
        print(f"{i}. {layer_name}")
        print(f"   梯度范数: {grad_norm:.4f}, CKA得分: {cka}")

    # 5.4 剪枝策略生成
    print("\n" + "=" * 60)
    print("【智能剪枝策略（总体剪枝率30%）】")
    print("=" * 60)

    pruning_strategy = analyzer.get_pruning_strategy(
        contributions,
        metric_name="gradient_norm",
        prune_ratio=0.3  # 总体30%剪枝率
    )

    print("\n各层推荐剪枝比例（前5层）：")
    for layer_name, prune_ratio in list(pruning_strategy.items())[:5]:
        grad_norm = contributions[layer_name]['gradient_norm']
        print(f"{layer_name}:")
        print(f"  剪枝比例: {prune_ratio:.1%}")
        print(f"  梯度范数: {grad_norm:.4f}")

    print("\n💡 提示：梯度范数低的层分配更高的剪枝比例")

    # 5.5 PEFT适配器位置推荐
    print("\n" + "=" * 60)
    print("【PEFT适配器插入位置推荐】")
    print("=" * 60)

    adapter_positions = analyzer.get_peft_insertion_points(
        contributions,
        metric_name="gradient_norm",
        num_adapters=3  # 插入3个适配器
    )

    print("\n推荐的适配器插入位置：")
    for i, layer_name in enumerate(adapter_positions, 1):
        grad_norm = contributions[layer_name]['gradient_norm']
        print(f"{i}. {layer_name} (梯度范数: {grad_norm:.4f})")

    print("\n💡 提示：梯度范数高的层更适合插入适配器以提升微调效果")

    # 5.6 深度分析
    print("\n" + "=" * 60)
    print("【按深度聚合分析】")
    print("=" * 60)

    depth_analysis = analyzer.aggregate_by_depth(
        contributions,
        metric_name="gradient_norm",
        num_bins=3  # 分为3个深度区间：早期/中期/后期
    )

    print("\n各深度段的平均梯度范数：")
    for depth_bin, avg_value in depth_analysis.items():
        print(f"{depth_bin}: {avg_value:.4f}")

    print("\n💡 提示：观察'中层关键性'现象（middle layers通常最重要）")

    # 5.7 统计摘要
    print("\n" + "=" * 60)
    print("【梯度范数统计摘要】")
    print("=" * 60)

    stats = analyzer.get_summary_statistics(contributions, "gradient_norm")

    print(f"\n  均值:   {stats['mean']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  25%分位: {stats['q25']:.4f}")
    print(f"  75%分位: {stats['q75']:.4f}")

    # ------------------ 步骤6：可视化 ------------------
    print("\n" + "=" * 60)
    print("【生成可视化图表】")
    print("=" * 60)

    try:
        # 图表1：梯度范数柱状图
        print("\n正在生成：梯度范数柱状图...")
        plot_layer_contributions(
            contributions,
            metric_name="gradient_norm",
            save_path="gradient_norm_analysis_cn.png",
            title="层梯度范数分析"
        )
        print("✓ 已保存：gradient_norm_analysis_cn.png")

        # 图表2：多指标热力图
        print("正在生成：多指标热力图...")
        plot_contribution_heatmap(
            contributions,
            metrics=["gradient_norm", "cka_score", "effective_rank", "activation_entropy"],
            save_path="contribution_heatmap_cn.png"
        )
        print("✓ 已保存：contribution_heatmap_cn.png")

        # 图表3：深度分析图
        print("正在生成：深度分析图...")
        plot_depth_analysis(
            contributions,
            metric_name="gradient_norm",
            num_bins=3,
            save_path="depth_analysis_cn.png"
        )
        print("✓ 已保存：depth_analysis_cn.png")

        print("\n✓ 所有可视化图表生成完成！")

    except Exception as e:
        print(f"⚠ 可视化生成失败（这通常是因为缺少matplotlib）：{e}")
        print("   您可以稍后安装：pip install matplotlib seaborn")

    # ------------------ 总结 ------------------
    print("\n" + "=" * 60)
    print("【分析完成！】")
    print("=" * 60)

    print("\n📊 本次分析涵盖：")
    print(f"  • {len(contributions)} 个层")
    print(f"  • {len(metrics)} 种指标")
    print(f"  • {num_samples} 个训练样本")

    print("\n💡 下一步建议：")
    print("  1. 根据层重要性排名进行选择性剪枝")
    print("  2. 在推荐的层上进行知识蒸馏")
    print("  3. 在高梯度层插入适配器进行PEFT")
    print("  4. 对比不同训练阶段的层贡献度变化")

    print("\n📖 了解更多：")
    print("  • 指标详解：docs/METRICS_CN.md")
    print("  • 快速开始：docs/QUICKSTART_CN.md")
    print("  • API文档：docs/API_CN.md")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
