"""
Example: Knowledge distillation with automatic layer selection.

This script demonstrates how to use the KnowledgeDistiller to distill
a large teacher model into a smaller student model, with intermediate
layer distillation based on contribution analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA, EffectiveRank
from uni_layer.compression import KnowledgeDistiller, DistillationConfig


# Define teacher model (large)
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


# Define student model (small)
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main():
    print("=" * 70)
    print("Knowledge Distillation Example - Layer Contribution Based")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ========== Step 1: Create Models and Data ==========
    print("\n[1/6] Creating teacher and student models...")

    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"✓ Teacher model: {teacher_params:,} parameters")
    print(f"✓ Student model: {student_params:,} parameters")
    print(f"✓ Compression ratio: {teacher_params / student_params:.2f}x")

    # Create dataset
    X_train = torch.randn(2000, 784)
    y_train = torch.randint(0, 10, (2000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_test = torch.randn(500, 784)
    y_test = torch.randint(0, 10, (500,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # ========== Step 2: Train Teacher Model ==========
    print("\n[2/6] Training teacher model (simplified)...")

    # In practice, teacher is pre-trained. Here we do quick training
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    teacher.train()

    for epoch in range(5):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            teacher_optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            teacher_optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}/5, Loss: {total_loss / len(train_loader):.4f}")

    teacher_accuracy = evaluate_model(teacher, test_loader, device)
    print(f"✓ Teacher accuracy: {teacher_accuracy:.2%}")

    # ========== Step 3: Analyze Teacher Layers ==========
    print("\n[3/6] Analyzing teacher layer contributions...")

    analyzer = LayerAnalyzer(teacher, task_type="classification", device=device)

    contributions = analyzer.compute_metrics(
        metrics=[
            GradientNorm(num_batches=10),
            CKA(num_batches=10),
            EffectiveRank(num_batches=10),
        ],
        data_loader=train_loader,
        verbose=False
    )

    print("✓ Layer analysis complete")

    # Display key layers
    print("\nTop layers for distillation (based on CKA):")
    cka_scores = [(name, m.get('cka_score', 0)) for name, m in contributions.items()]
    cka_scores.sort(key=lambda x: x[1], reverse=True)

    for name, score in cka_scores[:3]:
        print(f"  {name:30s}: CKA={score:.4f}")

    # ========== Step 4: Setup Knowledge Distiller ==========
    print("\n[4/6] Setting up knowledge distiller...")

    distill_config = DistillationConfig(
        temperature=4.0,
        alpha=0.7,  # 70% soft targets, 30% hard targets
        layer_weight=0.5,  # 50% weight for intermediate layer distillation
        distance_metric="mse",
        top_k_layers=3  # Distill top 3 important layers
    )

    distiller = KnowledgeDistiller(
        teacher_model=teacher,
        student_model=student,
        contributions=contributions,
        config=distill_config
    )

    # Display distillation info
    info = distiller.get_distillation_info()
    print(f"✓ Temperature: {info['temperature']}")
    print(f"✓ Distillation layers: {info['num_distillation_layers']}")
    print(f"✓ Alpha (soft/hard ratio): {info['alpha']:.1f}")

    # ========== Step 5: Train Student with Distillation ==========
    print("\n[5/6] Training student with distillation...")

    student_optimizer = optim.Adam(student.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Distillation training step
            loss_components = distiller.train_step(inputs, labels, student_optimizer)
            total_loss += loss_components['total']

        if (epoch + 1) % 2 == 0:
            accuracy = evaluate_model(student, test_loader, device)
            print(f"  Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.4f}, "
                  f"Accuracy: {accuracy:.2%}")

    print("✓ Student training complete")

    # ========== Step 6: Compare Results ==========
    print("\n[6/6] Evaluating final models...")

    # Train student without distillation (baseline)
    print("\n  Training baseline student (no distillation)...")
    baseline_student = StudentModel().to(device)
    baseline_optimizer = optim.Adam(baseline_student.parameters(), lr=0.001)

    baseline_student.train()
    for epoch in range(10):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            baseline_optimizer.zero_grad()
            outputs = baseline_student(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            baseline_optimizer.step()

    baseline_accuracy = evaluate_model(baseline_student, test_loader, device)
    distilled_accuracy = evaluate_model(student, test_loader, device)

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("Distillation Summary")
    print("=" * 70)

    print(f"\nTeacher Model:")
    print(f"  Parameters: {teacher_params:,}")
    print(f"  Accuracy: {teacher_accuracy:.2%}")

    print(f"\nStudent Model (Baseline - No Distillation):")
    print(f"  Parameters: {student_params:,}")
    print(f"  Accuracy: {baseline_accuracy:.2%}")
    print(f"  Gap from teacher: {(teacher_accuracy - baseline_accuracy):.2%}")

    print(f"\nStudent Model (With Distillation):")
    print(f"  Parameters: {student_params:,}")
    print(f"  Accuracy: {distilled_accuracy:.2%}")
    print(f"  Gap from teacher: {(teacher_accuracy - distilled_accuracy):.2%}")

    print(f"\n📊 Improvement from distillation: {(distilled_accuracy - baseline_accuracy):.2%}")
    print(f"📊 Knowledge recovered: {((distilled_accuracy - baseline_accuracy) / (teacher_accuracy - baseline_accuracy)):.1%}")

    print("\n💡 Key Features:")
    print("  • Automatic layer selection based on CKA/GradNorm")
    print("  • Intermediate layer distillation for rich representations")
    print("  • Temperature-scaled soft targets")
    print("  • Configurable loss weights (soft/hard/layer)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
