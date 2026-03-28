"""Layer contribution metrics"""

# Optimization metrics
# Architecture-specific metrics
from uni_layer.metrics.architecture_specific.attention_flow import AttentionFlow
from uni_layer.metrics.architecture_specific.diffusion_timestep import (
    DiffusionTimestepAnalysis,
    get_diffusion_blocks,
)
from uni_layer.metrics.architecture_specific.moe_router import MoERouterAnalysis

# Bayesian metrics
from uni_layer.metrics.bayesian.laplace_posterior import LaplacePosterior
from uni_layer.metrics.information_theory.entropy import ActivationEntropy

# Information theory metrics
from uni_layer.metrics.information_theory.mutual_information import MutualInformation
from uni_layer.metrics.optimization.fisher_information import FisherInformation
from uni_layer.metrics.optimization.gradient_norm import GradientNorm
from uni_layer.metrics.optimization.hessian_trace import HessianTrace
from uni_layer.metrics.representation.block_influence import BlockInfluence

# Representation metrics
from uni_layer.metrics.representation.jacobian_rank import JacobianRank

# Robustness metrics
from uni_layer.metrics.robustness.droplayer import DropLayerRobustness
from uni_layer.metrics.robustness.residual_droplayer import ResidualDropLayer

# Spectral metrics
from uni_layer.metrics.spectral.cka import CKA
from uni_layer.metrics.spectral.effective_rank import EffectiveRank
from uni_layer.metrics.spectral.ntk import NTKTrace

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
    "ResidualDropLayer",
    # Bayesian
    "LaplacePosterior",
    # Architecture-Specific
    "AttentionFlow",
    "DiffusionTimestepAnalysis",
    "MoERouterAnalysis",
    "get_diffusion_blocks",
]
