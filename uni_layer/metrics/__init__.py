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
from uni_layer.metrics.representation.block_influence import BlockInfluence

# Robustness metrics
from uni_layer.metrics.robustness.droplayer import DropLayerRobustness

# Bayesian metrics
from uni_layer.metrics.bayesian.laplace_posterior import LaplacePosterior

# Architecture-specific metrics
from uni_layer.metrics.architecture_specific.attention_flow import AttentionFlow

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
    "BlockInfluence",
    # Robustness
    "DropLayerRobustness",
    # Bayesian
    "LaplacePosterior",
    # Architecture-Specific
    "AttentionFlow",
]
