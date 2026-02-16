"""阶段 3 扰动与反事实模块。"""

from src.perturbation.diffusion_cf import CausalDiffusionCF, DiffusionConfig
from src.perturbation.latent_arithmetic import LatentArithmetic
from src.perturbation.spatial_propagation import propagate_perturbation
from src.perturbation.target_ranking import (
    PriorKnowledge,
    load_prior_knowledge,
    rank_counterfactual_interaction_targets,
)

__all__ = [
    "CausalDiffusionCF",
    "DiffusionConfig",
    "LatentArithmetic",
    "propagate_perturbation",
    "PriorKnowledge",
    "load_prior_knowledge",
    "rank_counterfactual_interaction_targets",
]

