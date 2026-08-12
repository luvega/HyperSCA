# HyperSCA evaluation package

from src.evaluation.cf_metrics import evaluate_counterfactual
from src.evaluation.benchmark_contract import (
    build_run_manifest,
    evaluate_promotion,
    load_benchmark_contract,
)
from src.evaluation.spatial_metrics import evaluate_spatial_propagation
from src.evaluation.task_c_benchmark import MeanDifferenceNetworkBaseline
from src.evaluation.cross_sample_metrics import evaluate_cross_sample

__all__ = [
    "build_run_manifest",
    "evaluate_counterfactual",
    "evaluate_promotion",
    "evaluate_spatial_propagation",
    "evaluate_cross_sample",
    "load_benchmark_contract",
    "MeanDifferenceNetworkBaseline",
]
