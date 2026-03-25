"""Utility functions for Uni-Layer framework"""

from uni_layer.utils.fast_math import minibatch_cka, randomized_svd
from uni_layer.utils.hook_utils import register_hooks, remove_hooks
from uni_layer.utils.layer_utils import get_model_layers, identify_layer_type
from uni_layer.utils.model_adapter import compute_loss, extract_logits, model_forward
from uni_layer.utils.report import ReportGenerator

__all__ = [
    "get_model_layers",
    "identify_layer_type",
    "register_hooks",
    "remove_hooks",
    "ReportGenerator",
    "randomized_svd",
    "minibatch_cka",
    "extract_logits",
    "compute_loss",
    "model_forward",
]
