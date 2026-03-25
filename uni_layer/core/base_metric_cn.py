"""
层贡献度指标基类（详细中文注释版）

本模块定义了所有层贡献度指标的抽象基类LayerMetric。
所有具体的指标类都必须继承此基类并实现compute()方法。

作者：Uni-Layer团队
许可：MIT License
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class LayerMetric(ABC):
    """
    层贡献度指标的抽象基类

    这个基类定义了所有层贡献度指标的统一接口，确保：
    1. 所有指标有一致的调用方式
    2. 提供通用的辅助方法（如提取激活、梯度）
    3. 标准化的配置和元数据管理

    继承此类时，子类必须实现compute()方法来计算具体的指标。

    属性说明：
        name (str): 指标名称，用于结果字典的键
        category (str): 指标类别，如'optimization'（优化）、'spectral'（谱方法）等
        requires_gradient (bool): 是否需要计算梯度
            - True: 需要模型处于训练模式，会执行backward()
            - False: 只需要前向传播，模型处于eval模式
        requires_data (bool): 是否需要数据样本
            - True: 需要data_loader参数
            - False: 可以只基于模型结构计算（如参数统计）
        batch_size (int): 每批次处理的样本数
        config (dict): 其他配置参数

    使用示例：
        >>> from uni_layer.core.base_metric import LayerMetric
        >>>
        >>> class MyMetric(LayerMetric):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(
        ...             name="my_metric",
        ...             category="custom",
        ...             requires_gradient=False,
        ...             requires_data=True,
        ...             **kwargs
        ...         )
        ...
        ...     def compute(self, model, layer, layer_name, layer_idx,
        ...                 data_loader, device, **kwargs):
        ...         # 您的计算逻辑
        ...         return {"my_metric": value}
    """

    def __init__(
        self,
        name: str,
        category: str,
        requires_gradient: bool = False,
        requires_data: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        初始化层贡献度指标

        参数：
            name: 指标名称（如"gradient_norm"、"cka"等）
            category: 指标类别（如"optimization"、"spectral"等）
            requires_gradient: 是否需要梯度计算
            requires_data: 是否需要数据样本
            batch_size: 批次大小
            **kwargs: 其他子类特定的配置参数
        """
        self.name = name
        self.category = category
        self.requires_gradient = requires_gradient
        self.requires_data = requires_data
        self.batch_size = batch_size
        self.config = kwargs

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        **kwargs,
    ) -> Dict[str, float]:
        """
        计算层贡献度指标（抽象方法，必须由子类实现）

        参数：
            model (nn.Module): 完整的神经网络模型
                - 用于执行完整的前向/反向传播
                - 某些指标需要完整的计算图

            layer (nn.Module): 要分析的特定层
                - 这是从model中提取出的单个层模块
                - 可以注册hook来捕获激活/梯度

            layer_name (str): 层的名称
                - 如"encoder.layer.0"、"transformer.h.5"等
                - 用于标识和记录

            layer_idx (int): 层在模型中的索引
                - 从0开始计数
                - 用于判断层的深度位置

            data_loader (Optional[Any]): 数据加载器
                - 如果requires_data=True，则必须提供
                - 应返回(inputs, targets)元组或只有inputs

            device (str): 计算设备
                - "cuda"或"cpu"
                - 指标计算会在此设备上进行

            **kwargs: 其他参数
                - criterion: 损失函数（某些指标需要）
                - num_classes: 类别数（某些指标需要）

        返回：
            Dict[str, float]: 指标值字典
                - 键是指标名称（通常包含self.name）
                - 值是浮点数
                - 示例：{"gradient_norm": 0.5, "gradient_norm_std": 0.1}

        实现要点：
            1. 处理异常情况，避免程序崩溃
            2. 返回有意义的默认值（如0.0）而不是抛出异常
            3. 对于无法计算的情况，可以返回None
            4. 使用try-except捕获可能的错误

        示例实现：
            >>> def compute(self, model, layer, layer_name, layer_idx,
            ...             data_loader, device, **kwargs):
            ...     try:
            ...         # 1. 提取层激活
            ...         activations = self._get_layer_activations(
            ...             model, layer, data_loader, num_batches=10
            ...         )
            ...
            ...         # 2. 计算指标
            ...         value = some_computation(activations)
            ...
            ...         # 3. 返回结果
            ...         return {self.name: float(value)}
            ...
            ...     except Exception as e:
            ...         print(f"计算{self.name}时出错: {e}")
            ...         return {self.name: 0.0}
        """
        pass

    def _get_layer_activations(
        self, model: nn.Module, layer: nn.Module, data_loader: Any, num_batches: int = 10
    ) -> List[torch.Tensor]:
        """
        提取层激活的辅助方法

        工作原理：
        1. 使用PyTorch的hook机制捕获层输出
        2. 在前向传播时自动记录激活值
        3. 收集多个批次的激活用于统计

        参数：
            model: 神经网络模型
            layer: 要提取激活的层
            data_loader: 数据加载器
            num_batches: 处理的批次数

        返回：
            List[torch.Tensor]: 激活张量列表
                - 每个元素是一个批次的激活
                - 已经移到CPU并detach（节省GPU内存）

        技术细节：
            - 使用register_forward_hook注册钩子函数
            - 钩子函数在层的forward()执行后被调用
            - 使用detach()避免保留计算图
            - 使用cpu()将数据移到CPU以节省GPU内存

        使用示例：
            >>> activations = self._get_layer_activations(
            ...     model=model,
            ...     layer=layer,
            ...     data_loader=train_loader,
            ...     num_batches=10
            ... )
            >>> # 拼接所有批次
            >>> all_acts = torch.cat(activations, dim=0)
            >>> print(all_acts.shape)  # (总样本数, 特征维度)
        """
        activations = []

        # 定义hook函数：捕获层输出
        def hook_fn(module, input, output):
            """
            Hook函数在层的forward()后自动调用

            参数：
                module: 注册hook的层（即layer）
                input: 层的输入（tuple）
                output: 层的输出（Tensor或tuple）
            """
            # 保存输出激活（detach并移到CPU）
            activations.append(output.detach().cpu())

        # 注册hook
        handle = layer.register_forward_hook(hook_fn)

        try:
            # 设置为评估模式（关闭dropout等）
            model.eval()

            # 不计算梯度（节省内存和时间）
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break

                    # 解析批次数据
                    # 支持两种格式：
                    # 1. (inputs, targets) - 监督学习
                    # 2. inputs - 无监督/半监督
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    # 移到指定设备
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()

                    # 前向传播（hook自动捕获激活）
                    model(inputs)

        finally:
            # 移除hook（避免内存泄漏）
            handle.remove()

        return activations

    def _get_layer_gradients(
        self,
        model: nn.Module,
        layer: nn.Module,
        data_loader: Any,
        criterion: nn.Module,
        num_batches: int = 10,
    ) -> List[torch.Tensor]:
        """
        提取层梯度的辅助方法

        工作原理：
        1. 使用backward hook捕获反向传播时的梯度
        2. 执行前向传播 -> 计算损失 -> 反向传播
        3. Hook在反向传播时自动记录梯度

        参数：
            model: 神经网络模型
            layer: 要提取梯度的层
            data_loader: 数据加载器（必须提供标签）
            criterion: 损失函数
            num_batches: 处理的批次数

        返回：
            List[torch.Tensor]: 梯度张量列表
                - 每个元素是一个批次的梯度
                - 形状与层输出相同

        技术细节：
            - 使用register_full_backward_hook注册反向hook
            - grad_output是损失对层输出的梯度：∂L/∂output
            - grad_input是损失对层输入的梯度：∂L/∂input
            - 我们通常关注grad_output（输出梯度）

        注意事项：
            - 需要模型处于训练模式（model.train()）
            - 需要计算图（不能使用torch.no_grad()）
            - 内存占用较大，建议num_batches不要太大

        使用示例：
            >>> gradients = self._get_layer_gradients(
            ...     model=model,
            ...     layer=layer,
            ...     data_loader=train_loader,
            ...     criterion=nn.CrossEntropyLoss(),
            ...     num_batches=5
            ... )
            >>> # 计算梯度范数
            >>> grad_norms = [g.norm().item() for g in gradients]
            >>> print(f"平均梯度范数: {np.mean(grad_norms):.4f}")
        """
        gradients = []

        # 定义backward hook函数
        def hook_fn(module, grad_input, grad_output):
            """
            Backward hook在反向传播时自动调用

            参数：
                module: 注册hook的层
                grad_input: 损失对层输入的梯度（tuple）
                grad_output: 损失对层输出的梯度（tuple）
            """
            # 检查梯度是否存在
            if grad_output[0] is not None:
                # 保存输出梯度
                gradients.append(grad_output[0].detach().cpu())

        # 注册backward hook
        handle = layer.register_full_backward_hook(hook_fn)

        try:
            # 设置为训练模式（启用dropout等）
            model.train()

            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break

                # 解析批次
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    # 如果没有标签，无法计算梯度
                    inputs, targets = batch, None

                # 移到设备
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    if targets is not None:
                        targets = targets.cuda()

                # 清零梯度（重要！）
                model.zero_grad()

                # 前向传播
                outputs = model(inputs)

                # 计算损失
                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    # 如果没有标签，使用输出的均值作为伪损失
                    loss = outputs.mean()

                # 反向传播（hook自动捕获梯度）
                loss.backward()

        finally:
            # 移除hook
            handle.remove()

        return gradients

    def __repr__(self) -> str:
        """返回指标的字符串表示"""
        return (
            f"{self.__class__.__name__}(" f"name='{self.name}', " f"category='{self.category}'" f")"
        )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    演示如何使用LayerMetric基类创建自定义指标
    """

    # 示例：创建一个简单的激活均值指标
    class ActivationMean(LayerMetric):
        """计算层激活的均值（示例指标）"""

        def __init__(self, num_batches=10, **kwargs):
            super().__init__(
                name="activation_mean",
                category="statistics",
                requires_gradient=False,  # 不需要梯度
                requires_data=True,  # 需要数据
                **kwargs,
            )
            self.num_batches = num_batches

        def compute(self, model, layer, layer_name, layer_idx, data_loader, device, **kwargs):
            """计算激活均值"""

            # 使用基类提供的辅助方法提取激活
            activations = self._get_layer_activations(
                model=model, layer=layer, data_loader=data_loader, num_batches=self.num_batches
            )

            if not activations:
                return {self.name: 0.0}

            # 拼接所有批次
            all_acts = torch.cat(activations, dim=0)

            # 计算均值
            mean_value = all_acts.mean().item()
            std_value = all_acts.std().item()

            return {self.name: mean_value, f"{self.name}_std": std_value}

    # 测试
    print("LayerMetric基类使用示例")
    print("=" * 60)

    metric = ActivationMean(num_batches=5)
    print(f"创建指标: {metric}")
    print(f"  需要梯度: {metric.requires_gradient}")
    print(f"  需要数据: {metric.requires_data}")
    print(f"  类别: {metric.category}")
