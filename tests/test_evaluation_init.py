from __future__ import annotations

import subprocess
import sys


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


def test_package_and_outcome_blind_submodule_cold_import_no_numeric_dependencies() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.evaluation.spatial_perturbation_registry; import sys; "
            "assert not set(sys.modules) & {'numpy', 'pandas', 'scipy', 'torch', 'anndata'}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_existing_exports_remain_exactly_available_when_accessed() -> None:
    checks = "; ".join(
        f"assert package.{name} is getattr(importlib.import_module('{module}'), '{attribute}')"
        for name, (module, attribute) in _EXPORTS.items()
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; import src.evaluation as package; "
            f"assert tuple(package.__all__) == {tuple(_EXPORTS)!r}; {checks}; "
            "assert not hasattr(package, 'not_a_public_export')",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
