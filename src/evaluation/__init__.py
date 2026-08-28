"""Lazy public exports for evaluation code.

Keeping package initialization stdlib-only lets dependency-free contracts be
imported without loading numerical or model-oriented evaluation modules.
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_run_manifest",
    "evaluate_counterfactual",
    "evaluate_promotion",
    "evaluate_spatial_propagation",
    "evaluate_cross_sample",
    "load_benchmark_contract",
    "MeanDifferenceNetworkBaseline",
    "predict_task_s_baseline",
]

_EXPORTS = {
    "build_run_manifest": ("src.evaluation.benchmark_contract", "build_run_manifest"),
    "evaluate_counterfactual": ("src.evaluation.cf_metrics", "evaluate_counterfactual"),
    "evaluate_promotion": ("src.evaluation.benchmark_contract", "evaluate_promotion"),
    "evaluate_spatial_propagation": ("src.evaluation.spatial_metrics", "evaluate_spatial_propagation"),
    "evaluate_cross_sample": ("src.evaluation.cross_sample_metrics", "evaluate_cross_sample"),
    "load_benchmark_contract": ("src.evaluation.benchmark_contract", "load_benchmark_contract"),
    "MeanDifferenceNetworkBaseline": ("src.evaluation.task_c_benchmark", "MeanDifferenceNetworkBaseline"),
    "predict_task_s_baseline": ("src.evaluation.task_s_benchmark", "predict_task_s_baseline"),
}


def __getattr__(name: str) -> object:
    """Resolve only the fixed public export set when it is explicitly used."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attribute_name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
