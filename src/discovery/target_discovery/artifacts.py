"""Artifact writing and manifest tracking for target discovery runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


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
        path = self.run_dir / section
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _record(self, path: Path) -> Path:
        rel = path.relative_to(self.run_dir).as_posix()
        self._artifacts.append(rel)
        return path

    def write_table(self, name: str, df: pd.DataFrame, section: str) -> Path:
        path = self.section_dir(section) / name
        df.to_csv(path, index=False)
        return self._record(path)

    def write_json(self, name: str, payload: Mapping[str, Any], section: str) -> Path:
        path = self.section_dir(section) / name
        path.write_text(json.dumps(payload, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")
        return self._record(path)

    def write_array(self, name: str, arr: np.ndarray, section: str) -> Path:
        path = self.section_dir(section) / name
        np.save(path, arr)
        return self._record(path)

    def write_markdown(self, name: str, text: str, section: str) -> Path:
        path = self.section_dir(section) / name
        path.write_text(text, encoding="utf-8")
        return self._record(path)

    def write_figure(self, name: str, fig: Any, section: str, metadata: Mapping[str, Any] | None = None) -> Path:
        path = self.section_dir(section) / name
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
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
