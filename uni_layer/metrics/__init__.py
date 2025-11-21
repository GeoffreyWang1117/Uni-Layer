"""Layer contribution metrics"""

# Optimization metrics
from uni_layer.metrics.optimization.gradient_norm import GradientNorm
from uni_layer.metrics.optimization.hessian_trace import HessianTrace
from uni_layer.metrics.optimization.fisher_information import FisherInformation

# Spectral metrics
from uni_layer.metrics.spectral.cka import CKA
from uni_layer.metrics.spectral.effective_rank import EffectiveRank
from uni_layer.metrics.spectral.ntk import NTKTrace

# Information theory metrics
from uni_layer.metrics.information_theory.mutual_information import MutualInformation
from uni_layer.metrics.information_theory.entropy import ActivationEntropy

# Representation metrics
from uni_layer.metrics.representation.jacobian_rank import JacobianRank

# Robustness metrics
from uni_layer.metrics.robustness.droplayer import DropLayerRobustness

__all__ = [
    # Optimization
    "GradientNorm",
    "HessianTrace",
    "FisherInformation",
    # Spectral
    "CKA",
    "EffectiveRank",
    "NTKTrace",
    # Information Theory
    "MutualInformation",
    "ActivationEntropy",
    # Representation
    "JacobianRank",
    # Robustness
    "DropLayerRobustness",
]
