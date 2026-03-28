"""
Integrations with popular deep learning optimization libraries.

Uni-Layer acts as the "analytical brain" that provides layer importance insights,
while specialized tools handle the actual optimization:

- Torch-Pruning: Structural pruning with DepGraph
- HuggingFace PEFT: LoRA/AdaLoRA/Adapter fine-tuning
- Knowledge Distillation: Layer-guided distillation workflows

Each integration converts Uni-Layer's contribution analysis into the format
expected by the target library, enabling metrics-driven optimization.
"""

from uni_layer.integrations.distillation import DistillationBridge
from uni_layer.integrations.export_hints import ExportHintsBridge
from uni_layer.integrations.huggingface_peft import HuggingFacePEFTBridge
from uni_layer.integrations.llm_frameworks import AxolotlConfigBridge, LLaMAFactoryConfigBridge
from uni_layer.integrations.torch_pruning import TorchPruningBridge

__all__ = [
    "TorchPruningBridge",
    "HuggingFacePEFTBridge",
    "DistillationBridge",
    "ExportHintsBridge",
    "AxolotlConfigBridge",
    "LLaMAFactoryConfigBridge",
]
