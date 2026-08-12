"""Validate the HyperSCA runtime environment."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ImportSpec:
    """A package import check."""

    module: str
    label: str
    dist_name: str | None = None


@dataclass
class ImportCheckResult:
    """Import check summary."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationPlan:
    """Checks enabled for one environment validation profile."""

    required_imports: tuple[ImportSpec, ...]
    optional_imports: tuple[ImportSpec, ...]
    check_gpu: bool
    pyg_extensions: tuple[ImportSpec, ...]


ImportFunc = Callable[[str], object]


REQUIRED_IMPORTS = [
    ImportSpec("scanpy", "scanpy"),
    ImportSpec("squidpy", "squidpy"),
    ImportSpec("torch", "torch"),
    ImportSpec("torch_geometric", "torch_geometric", dist_name="torch-geometric"),
    ImportSpec("geoopt", "geoopt"),
    ImportSpec("dowhy", "dowhy"),
    ImportSpec("anndata", "anndata"),
    ImportSpec("scvi", "scvi-tools", dist_name="scvi-tools"),
    ImportSpec("econml", "econml"),
    ImportSpec("pgmpy", "pgmpy"),
    ImportSpec("pingouin", "pingouin"),
    ImportSpec("diffusers", "diffusers"),
]

CORE_IMPORTS = [
    ImportSpec("scanpy", "scanpy"),
    ImportSpec("torch", "torch"),
    ImportSpec("torch_geometric", "torch_geometric", dist_name="torch-geometric"),
    ImportSpec("geoopt", "geoopt"),
    ImportSpec("dowhy", "dowhy"),
    ImportSpec("anndata", "anndata"),
]

OPTIONAL_IMPORTS = [
    ImportSpec("scgen", "scgen"),
]

PYG_EXTENSION_IMPORTS = [
    ImportSpec("torch_scatter", "torch_scatter", dist_name="torch-scatter"),
    ImportSpec("torch_sparse", "torch_sparse", dist_name="torch-sparse"),
    ImportSpec("torch_cluster", "torch_cluster", dist_name="torch-cluster"),
    ImportSpec("torch_spline_conv", "torch_spline_conv", dist_name="torch-spline-conv"),
]


def build_validation_plan(profile: str) -> ValidationPlan:
    """Return the checks associated with a named validation profile."""
    if profile == "core-cpu":
        return ValidationPlan(tuple(CORE_IMPORTS), (), False, ())
    if profile == "gpu":
        return ValidationPlan(
            tuple(CORE_IMPORTS),
            (),
            True,
            tuple(PYG_EXTENSION_IMPORTS),
        )
    if profile == "full":
        return ValidationPlan(
            tuple(REQUIRED_IMPORTS),
            tuple(OPTIONAL_IMPORTS),
            True,
            tuple(PYG_EXTENSION_IMPORTS),
        )
    raise ValueError(f"Unknown validation profile: {profile}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("core-cpu", "gpu", "full"),
        default="full",
        help="validation scope (default: full, preserving legacy behavior)",
    )
    return parser.parse_args(argv)


def _version_text(module: object, spec: ImportSpec) -> str:
    dist_name = spec.dist_name or spec.module
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "")
    return f" {version}" if version else ""


def check_imports(
    required: list[ImportSpec] | tuple[ImportSpec, ...] = REQUIRED_IMPORTS,
    optional: list[ImportSpec] | tuple[ImportSpec, ...] = OPTIONAL_IMPORTS,
    *,
    import_func: ImportFunc = importlib.import_module,
) -> ImportCheckResult:
    """Check required and optional imports.

    Optional imports are reported as warnings so legacy baseline packages do
    not block the core HyperSCA pipeline.
    """
    result = ImportCheckResult()

    for spec in required:
        try:
            module = import_func(spec.module)
            print(f"  {spec.label}{_version_text(module, spec)} ... OK")
        except Exception as exc:
            result.errors.append(f"{spec.label}: {exc}")

    for spec in optional:
        try:
            module = import_func(spec.module)
            print(f"  {spec.label}{_version_text(module, spec)} ... OK (optional)")
        except Exception as exc:
            result.warnings.append(f"{spec.label}: {exc}")
            print(f"  {spec.label} ... OPTIONAL MISSING ({exc})")

    return result


def check_gpu() -> list[str]:
    errors: list[str] = []
    try:
        import torch

        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            tensor = torch.randn(3, 3).cuda()
            print(f"  GPU tensor test: {tensor.device} ... OK")
        else:
            errors.append("CUDA not available")
    except Exception as exc:
        errors.append(f"GPU test: {exc}")
    return errors


def check_data_readability(root: Path) -> None:
    data_dir = root / "data"
    if data_dir.exists():
        subdirs = [
            name for name in os.listdir(data_dir)
            if (data_dir / name).is_dir()
        ]
        print(f"  Found {len(subdirs)} data directories: {subdirs}")
    else:
        print(f"  Data directory not found at {data_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_validation_plan(args.profile)
    project_root = Path(__file__).resolve().parents[1]

    print("=" * 60)
    print(f"HyperSCA Environment Validation ({args.profile})")
    print("=" * 60)

    errors: list[str] = []
    warnings: list[str] = []

    print("\n[1] Core package imports...")
    import_result = check_imports(
        required=plan.required_imports,
        optional=plan.optional_imports,
    )
    errors.extend(import_result.errors)
    warnings.extend(import_result.warnings)

    if plan.check_gpu:
        print("\n[2] GPU check...")
        errors.extend(check_gpu())
    else:
        print("\n[2] GPU check... SKIPPED by profile")

    if plan.pyg_extensions:
        print("\n[3] PyG extensions...")
        pyg_result = check_imports(required=plan.pyg_extensions, optional=[])
        errors.extend(pyg_result.errors)
    else:
        print("\n[3] PyG extensions... SKIPPED by profile")

    print("\n[4] Data readability check...")
    check_data_readability(project_root)

    print("\n" + "=" * 60)
    if warnings:
        print(f"VALIDATION WARNINGS - {len(warnings)} optional issue(s):")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("All required validations PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
