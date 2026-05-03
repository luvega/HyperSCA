"""Artifact writing and manifest tracking for target discovery runs."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return _to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, set):
        try:
            values = sorted(obj)
        except TypeError:
            values = list(obj)
        return [_to_jsonable(value) for value in values]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def json_default(obj: Any) -> Any:
    converted = _to_jsonable(obj)
    if converted is obj:
        return str(obj)
    return converted


def _validate_relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe artifact path: {label}={value!r}")
    if label == "name" and not path.parts:
        raise ValueError(f"unsafe artifact path: {label}={value!r}")
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


class ArtifactWriter:
    """Write run artifacts into section directories and track a manifest."""

    def __init__(self, output_base: str | Path, run_id: str):
        self.output_base = Path(output_base)
        self.run_id = str(run_id)
        self.run_dir = self.output_base / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.time()
        self._artifacts: list[str] = []
        self._stages: list[dict[str, Any]] = []
        self._warnings: list[str] = []

    def section_dir(self, section: str) -> Path:
        section_path = _validate_relative_path(section, "section")
        path = self.run_dir / section_path
        run_dir_resolved = self.run_dir.resolve()
        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, run_dir_resolved):
            raise ValueError(f"unsafe artifact path: section={section!r}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path(self, section: str, name: str) -> Path:
        name_path = _validate_relative_path(name, "name")
        path = self.section_dir(section) / name_path
        run_dir_resolved = self.run_dir.resolve()
        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, run_dir_resolved):
            raise ValueError(f"unsafe artifact path: section={section!r}, name={name!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _record(self, path: Path) -> Path:
        rel = path.relative_to(self.run_dir).as_posix()
        self._artifacts.append(rel)
        return path

    def write_table(self, name: str, df: pd.DataFrame, section: str) -> Path:
        path = self._artifact_path(section, name)
        df.to_csv(path, index=False)
        return self._record(path)

    def write_json(self, name: str, payload: Mapping[str, Any], section: str) -> Path:
        path = self._artifact_path(section, name)
        path.write_text(
            json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return self._record(path)

    def write_array(self, name: str, arr: np.ndarray, section: str) -> Path:
        path = self._artifact_path(section, name)
        np.save(path, arr)
        return self._record(path)

    def write_markdown(self, name: str, text: str, section: str) -> Path:
        path = self._artifact_path(section, name)
        path.write_text(text, encoding="utf-8")
        return self._record(path)

    def write_figure(self, name: str, fig: Any, section: str, metadata: Mapping[str, Any] | None = None) -> Path:
        path = self._artifact_path(section, name)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        self._record(path)
        if metadata is not None:
            self.write_json(f"{name}.meta.json", dict(metadata), section=section)
        return path

    def record_stage(self, name: str, seconds: float, outputs: Mapping[str, Any]) -> None:
        self._stages.append(
            {
                "name": name,
                "seconds": float(seconds),
                "outputs": sorted(outputs.keys()),
            }
        )

    def warn(self, message: str) -> None:
        self._warnings.append(str(message))

    def finalize(self) -> Path:
        manifest = {
            "run_id": self.run_id,
            "elapsed_seconds": float(time.time() - self._started_at),
            "artifacts": sorted(set(self._artifacts)),
            "stages": self._stages,
            "warnings": self._warnings,
        }
        path = self.run_dir / "manifest.json"
        path.write_text(
            json.dumps(_to_jsonable(manifest), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return path
