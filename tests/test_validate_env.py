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
