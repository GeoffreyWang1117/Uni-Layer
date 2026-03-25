"""
Example: Using Uni-Layer for metrics-driven knowledge distillation.

Instead of manually choosing which intermediate layers to align during
distillation, Uni-Layer analyzes the teacher model and recommends:
1. Which layers carry the most task-relevant information
2. How to pair teacher layers with student layers
3. How much weight to give each layer's distillation loss

Works with any distillation framework or custom training loop.

Requirements:
    pip install uni-layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TeacherModel(nn.Module):
    def __init__(self, dim=512, depth=8, num_classes=10):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.head(self.layers(x))


class StudentModel(nn.Module):
    def __init__(self, dim=512, depth=4, num_classes=10):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.head(self.layers(x))


def main():
    teacher = TeacherModel(dim=256, depth=8)
    student = StudentModel(dim=256, depth=4)

    # Dummy data
    X = torch.randn(200, 256)
    y = torch.randint(0, 10, (200,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # --- Step 1: Analyze teacher with Uni-Layer ---
    from uni_layer import LayerAnalyzer
    from uni_layer.metrics import GradientNorm, CKA, EffectiveRank

    analyzer = LayerAnalyzer(teacher, task_type='classification')
    contributions = analyzer.compute_metrics(
        metrics=[GradientNorm(), EffectiveRank()],
        data_loader=loader,
    )

    # --- Step 2: Get distillation recommendations ---
    from uni_layer.integrations import DistillationBridge

    bridge = DistillationBridge(teacher, student, contributions)

    # Which teacher layers to distill?
    important_layers = bridge.recommend_distillation_layers(
        metric_name='gradient_norm', top_k=4
    )
    print("=== Most Important Teacher Layers ===")
    for name in important_layers:
        print(f"  {name}")

    # How to pair teacher-student layers?
    pairs = bridge.recommend_layer_pairs(metric_name='gradient_norm', top_k=4)
    print("\n=== Recommended Layer Pairs ===")
    for t_name, s_name in pairs:
        print(f"  Teacher: {t_name}  <->  Student: {s_name}")

    # Per-layer distillation weights
    weights = bridge.recommend_layer_weights(
        metric_name='gradient_norm',
        target_layers=important_layers,
    )
    print("\n=== Per-Layer Distillation Weights ===")
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")

    # Full summary
    print(f"\n{bridge.summary('gradient_norm', top_k=4)}")

    # --- Step 3: Use in distillation training loop ---
    print("\n=== Example distillation code ===")
    print("""
    # Register hooks to capture intermediate activations
    teacher_acts, student_acts = {}, {}

    for teacher_name, student_name in pairs:
        # ... register forward hooks ...

    # Training loop
    for inputs, labels in loader:
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)

        # Soft distillation loss
        T = 4.0
        loss_soft = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)

        # Metrics-weighted intermediate layer loss
        loss_layer = 0
        for teacher_name, student_name in pairs:
            w = weights.get(teacher_name, 1.0)
            loss_layer += w * F.mse_loss(
                student_acts[student_name],
                teacher_acts[teacher_name]
            )

        loss = 0.5 * loss_soft + 0.5 * F.cross_entropy(student_logits, labels) + 0.3 * loss_layer
    """)


if __name__ == "__main__":
    main()
