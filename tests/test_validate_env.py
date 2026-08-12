"""Tests for environment validation behavior."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_validate_env_module():
    spec = importlib.util.spec_from_file_location(
        "validate_env", ROOT / "scripts" / "validate_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_optional_import_failure_is_warning_not_error() -> None:
    validate_env = _load_validate_env_module()

    def fake_import(name: str):
        if name == "scgen":
            raise ModuleNotFoundError("No module named 'scvi._compat'")

        class FakeModule:
            __version__ = "1.0.0"

        return FakeModule()

    result = validate_env.check_imports(
        required=[validate_env.ImportSpec("scanpy", "scanpy")],
        optional=[validate_env.ImportSpec("scgen", "scgen")],
        import_func=fake_import,
    )

    assert result.errors == []
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("scgen:")


def test_core_cpu_profile_excludes_full_only_imports_and_accelerator_checks() -> None:
    validate_env = _load_validate_env_module()

    plan = validate_env.build_validation_plan("core-cpu")

    assert plan.required_imports == tuple(validate_env.CORE_IMPORTS)
    assert plan.optional_imports == ()
    assert plan.check_gpu is False
    assert plan.pyg_extensions == ()


def test_gpu_profile_enables_gpu_and_pyg_extension_checks() -> None:
    validate_env = _load_validate_env_module()

    plan = validate_env.build_validation_plan("gpu")

    assert plan.required_imports == tuple(validate_env.CORE_IMPORTS)
    assert plan.optional_imports == ()
    assert plan.check_gpu is True
    assert plan.pyg_extensions == tuple(validate_env.PYG_EXTENSION_IMPORTS)


def test_full_profile_preserves_current_validation_scope() -> None:
    validate_env = _load_validate_env_module()

    plan = validate_env.build_validation_plan("full")

    assert validate_env.parse_args([]).profile == "full"
    assert plan.required_imports == tuple(validate_env.REQUIRED_IMPORTS)
    assert plan.optional_imports == tuple(validate_env.OPTIONAL_IMPORTS)
    assert plan.check_gpu is True
    assert plan.pyg_extensions == tuple(validate_env.PYG_EXTENSION_IMPORTS)
