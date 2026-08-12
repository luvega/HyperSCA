"""Tests for packaging and continuous-integration contracts."""
from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        names.add(re.split(r"[<>=!~\s\[]", line, maxsplit=1)[0].lower())
    return names


def test_pyproject_defines_editable_src_package_and_test_path() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert config["project"]["name"] == "hypersca"
    assert config["project"]["version"] == "0.6.1.dev0"
    assert config["project"]["requires-python"] == ">=3.10"
    assert config["project"]["dynamic"] == ["dependencies"]
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["src*"]
    assert config["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_dependency_profiles_keep_compiled_gpu_extensions_out_of_core() -> None:
    core = _requirement_names(ROOT / "requirements-core.txt")
    gpu = _requirement_names(ROOT / "requirements-gpu.txt")

    assert {
        "scanpy",
        "torch",
        "torch-geometric",
        "geoopt",
        "dowhy",
        "tabulate",
    } <= core
    assert {
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    } <= gpu
    assert core.isdisjoint(gpu)


def test_ci_targets_python_310_core_cpu_contract() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "python-version: \"3.10\"" in workflow
    assert 'python -m pip install -e ".[dev]"' in workflow
    assert "python scripts/validate_env.py --profile core-cpu" in workflow
    assert "pytest tests -q" in workflow


def test_ci_runs_plain_language_check_before_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    check = "python scripts/check_plain_language.py"
    tests = "pytest tests -q"
    assert check in workflow
    assert workflow.index(check) < workflow.index(tests)
