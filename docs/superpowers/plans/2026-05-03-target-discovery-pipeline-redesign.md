# Target Discovery Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `scripts/run_target_discovery.py` into a reusable `src.discovery.target_discovery` pipeline package with structured stages, artifacts, tests, and migration notes.

**Architecture:** Keep the current scientific behavior but move it behind explicit config, stage, artifact, and pipeline boundaries. `scripts/run_target_discovery.py` becomes a thin CLI entrypoint; new modules own constants, external data loading, candidate construction, geometry, causal/perturbation wrappers, scoring, niche mapping, reporting, and figures.

**Tech Stack:** Python 3.10, dataclasses, pathlib, pandas, numpy, torch, sklearn, matplotlib, pytest. Use `E:\ProgramData\Anaconda3\envs\hypersca\python.exe` for verification.

---

## Scope And Safety Notes

- Work in `E:\HyperSCA`.
- Preserve unrelated dirty worktree changes. Current unrelated changes include design images, MSI inference files/tests, `src/evaluation/cross_sample_metrics.py`, `src/utils/plot_style.py`, and `src/visualization/hyperbolic.py`.
- Do not move or modify `data/`, `results/`, or `references/`.
- Do not rewrite Step2 or Step3 algorithms in this pass. Wrap and relocate the existing behavior.
- Prefer small commits after each task. Only stage files named in that task.
- If pytest fails with Windows temp directory permissions, rerun targeted tests outside sandbox or set a writable temp directory with user approval. Do not treat temp permission errors as business logic failures.

## File Structure

Create:

- `src/discovery/__init__.py` - package marker.
- `src/discovery/target_discovery/__init__.py` - public exports.
- `src/discovery/target_discovery/constants.py` - biological constants and scoring weights copied from the old script.
- `src/discovery/target_discovery/config.py` - `DiscoveryPaths`, `GeometryModeConfig`, `TargetDiscoveryConfig`.
- `src/discovery/target_discovery/artifacts.py` - JSON serialization, run manifest, artifact writer.
- `src/discovery/target_discovery/stage.py` - stage protocol and run context.
- `src/discovery/target_discovery/utils.py` - `_minmax`, adjacency normalization, kNN adjacency, small shared helpers.
- `src/discovery/target_discovery/loaders.py` - data mode detection and external table readers.
- `src/discovery/target_discovery/candidates.py` - candidate pool aggregation.
- `src/discovery/target_discovery/expression.py` - cluster expression assembly.
- `src/discovery/target_discovery/spatial.py` - spatial co-localization adjacency.
- `src/discovery/target_discovery/geometry.py` - hyperbolic/euclidean geometry and adjacency blending.
- `src/discovery/target_discovery/causal_stage.py` - Step2 causal stage wrapper.
- `src/discovery/target_discovery/perturbation_stage.py` - Step3 perturbation stage wrapper.
- `src/discovery/target_discovery/scoring.py` - evidence scoring, retained hubs/combos, mode comparison.
- `src/discovery/target_discovery/niche.py` - unified niche inventory, merge, build, and target mapping.
- `src/discovery/target_discovery/reporting.py` - markdown reports and migration notes.
- `src/discovery/target_discovery/figures.py` - figure pack generation.
- `src/discovery/target_discovery/pipeline.py` - `TargetDiscoveryPipeline` orchestration.
- `tests/discovery/__init__.py` - package marker for discovery tests.
- `tests/discovery/test_target_discovery_artifacts.py`
- `tests/discovery/test_target_discovery_candidates.py`
- `tests/discovery/test_target_discovery_geometry.py`
- `tests/discovery/test_target_discovery_scoring.py`
- `tests/discovery/test_target_discovery_pipeline.py`

Modify:

- `scripts/run_target_discovery.py` - replace monolithic implementation with thin CLI calling the new pipeline.

---

### Task 1: Scaffold Package, Constants, Config, And Shared Helpers

**Files:**
- Create: `src/discovery/__init__.py`
- Create: `src/discovery/target_discovery/__init__.py`
- Create: `src/discovery/target_discovery/constants.py`
- Create: `src/discovery/target_discovery/config.py`
- Create: `src/discovery/target_discovery/utils.py`
- Test: `tests/discovery/test_target_discovery_geometry.py`

- [ ] **Step 1: Create the failing helper tests**

Add `tests/discovery/__init__.py` as an empty file.

Add `tests/discovery/test_target_discovery_geometry.py`:

```python
from __future__ import annotations

import numpy as np

from src.discovery.target_discovery.config import TargetDiscoveryConfig
from src.discovery.target_discovery.utils import knn_adjacency, minmax, normalize_adjacency


def test_default_config_resolves_output_base():
    cfg = TargetDiscoveryConfig.default()
    assert cfg.paths.output_base.name == "target_discovery"
    assert cfg.geometry.geometry_k == 4
    assert "hyperbolic" in cfg.geometry.modes


def test_minmax_constant_returns_zeros():
    out = minmax(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(out, np.zeros(3))


def test_normalize_adjacency_clears_diagonal_and_scales():
    adj = np.array([[10.0, 2.0], [4.0, 10.0]])
    out = normalize_adjacency(adj)
    assert np.allclose(np.diag(out), [0.0, 0.0])
    assert float(out.max()) == 1.0


def test_knn_adjacency_is_symmetric():
    dist = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
    )
    out = knn_adjacency(dist, k=1)
    assert out.shape == (3, 3)
    assert np.allclose(out, out.T)
    assert np.allclose(np.diag(out), [0.0, 0.0, 0.0])
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_geometry.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.discovery'`.

- [ ] **Step 3: Create package markers**

Create `src/discovery/__init__.py`:

```python
"""Discovery workflows for HyperSCA."""
```

Create `src/discovery/target_discovery/__init__.py`:

```python
"""Target discovery pipeline package."""

from src.discovery.target_discovery.config import TargetDiscoveryConfig

__all__ = ["TargetDiscoveryConfig"]
```

- [ ] **Step 4: Move constants into `constants.py`**

Create `src/discovery/target_discovery/constants.py` by copying these values from `scripts/run_target_discovery.py`:

```python
"""Constants for the target discovery workflow."""

ANCHOR_GENES = ("MFAP2", "POSTN", "INHBA")
IFNG_FOCUS_GENES = ("CD74", "INHBA", "CXCL10", "IFNG", "COL1A1", "MFAP5", "FN1")

CELLTYPES = (
    "Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3",
    "Macrophage", "Macrophage_cycling",
    "Pericyte",
    "T_cell_CD4", "T_cell_CD8", "T_cell_CD8_cycling", "T_cell_regulatory",
    "NK",
    "cDC1", "cDC2", "DC_mature", "pDC",
    "Neutrophil", "Mast_cell",
    "Monocyte_classical",
    "Endothelial_venous", "Endothelial_arterial",
)

TYPE_MAPPING = {
    "Fibroblast_S1": "CAF", "Fibroblast_S2": "CAF", "Fibroblast_S3": "CAF",
    "Macrophage": "TAM", "Macrophage_cycling": "TAM",
    "Pericyte": "Stromal",
    "T_cell_CD4": "CD4T", "T_cell_CD8": "CD8T",
    "T_cell_CD8_cycling": "CD8T", "T_cell_regulatory": "Treg",
    "NK": "NK",
    "cDC1": "DC", "cDC2": "DC", "DC_mature": "DC", "pDC": "DC",
    "Neutrophil": "Neutrophil", "Mast_cell": "Mast",
    "Monocyte_classical": "Monocyte",
    "Endothelial_venous": "Endothelial", "Endothelial_arterial": "Endothelial",
}

ST_DECONV_MAP = {
    "Fibroblast_S1": ("Fibro_ADAMDEC1", "Fibro_CXCL8", "Fibro_CXCL14"),
    "Fibroblast_S2": ("Fibro_GPM6B", "Fibro_KCNN3", "Fibro_MYH11"),
    "Fibroblast_S3": ("Fibro_NOTCH3", "Fibro_PI16"),
    "Macrophage": ("Mac_M1", "Mac_M2", "Mac_SPP1"),
    "Macrophage_cycling": ("Mac_M1",),
    "Pericyte": ("Endo",),
    "T_cell_CD4": ("CD4_CXCL13", "CD4_Tcm", "CD4_Treg", "CD4_act"),
    "T_cell_CD8": ("CD8_Cyto", "CD8_HSP", "CD8_Teff", "CD8_Tem", "CD8_Tex"),
    "T_cell_CD8_cycling": ("CD8_Cyto",),
    "T_cell_regulatory": ("CD4_Treg",),
    "NK": ("NK_gdT",),
    "cDC1": ("cDC1",), "cDC2": ("cDC2",), "DC_mature": ("DC_LAMP3",), "pDC": ("pDC",),
    "Neutrophil": ("Monocyte_S100A8",),
    "Mast_cell": ("Mast",),
    "Monocyte_classical": ("Monocyte_S100A8",),
    "Endothelial_venous": ("Endo",), "Endothelial_arterial": ("Endo",),
}

ICB_TO_NEU_MAP = {
    "Fibro": ("Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3"),
    "Mph": ("Macrophage", "Macrophage_cycling"),
    "CD8": ("T_cell_CD8", "T_cell_CD8_cycling"),
    "T": ("T_cell_CD4", "T_cell_CD8", "T_cell_regulatory"),
    "Endo": ("Endothelial_venous", "Endothelial_arterial"),
    "Pericyte": ("Pericyte",),
    "Tumor": (),
    "Coloncyte": (), "Goblet": (), "Glia": (), "Tuft": (),
}

PRIOR_AXES = (
    ("CAF", "TAM", 0.3),
    ("CAF", "Treg", 0.3),
    ("TAM", "CD8T", 0.3),
    ("DC", "CD8T", 0.2),
    ("Neutrophil", "TAM", 0.2),
    ("CAF", "Endothelial", 0.2),
)

SCORE_WEIGHTS = {
    "causal": 0.25,
    "spatial": 0.25,
    "consistency": 0.25,
    "actionability": 0.10,
    "niche": 0.15,
}
```

- [ ] **Step 5: Implement config dataclasses**

Create `src/discovery/target_discovery/config.py`:

```python
"""Configuration for target discovery runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DiscoveryPaths:
    root: Path
    data_dir: Path
    neu_dir: Path
    ifng_dir: Path
    icb_dir: Path
    st_dir: Path
    output_base: Path
    icb_h5ad_path: Path
    reference_manifest_path: Path

    @classmethod
    def default(cls, root: Path | None = None, output_base: Path | None = None) -> "DiscoveryPaths":
        root = project_root() if root is None else Path(root)
        data_dir = root / "data"
        return cls(
            root=root,
            data_dir=data_dir,
            neu_dir=Path(
                r"G:\scCRC_Neu\downstream_analyses_de_analysis"
                r"\0downstream_analyses_de_analysis\de_analysis"
                r"\de_analysis_tumor_mss_msi\deseq2_dgea"
            ),
            ifng_dir=Path(r"F:\scCRC_IFNG"),
            icb_dir=Path(r"G:\scCRC_ICB\output"),
            st_dir=Path(r"G:\ST_CRC_MSS"),
            output_base=Path(output_base) if output_base else root / "results" / "discovery" / "target_discovery",
            icb_h5ad_path=data_dir / "scRNA" / "scCRC_ICB" / "expression.h5ad",
            reference_manifest_path=data_dir / "ref" / "manifest" / "reference_manifest.json",
        )


@dataclass(frozen=True)
class GeometryModeConfig:
    modes: tuple[str, ...] = ("hyperbolic", "euclidean")
    geometry_k: int = 4
    geometry_blend: float = 0.30


@dataclass(frozen=True)
class TargetDiscoveryConfig:
    paths: DiscoveryPaths = field(default_factory=DiscoveryPaths.default)
    geometry: GeometryModeConfig = field(default_factory=GeometryModeConfig)
    max_perturb: int = 50
    platform: str = "all"
    focused_genes: tuple[str, ...] = ()
    hierarchy_levels: int = 3
    run_id: str | None = None
    random_seed: int = 42
    device: str = "cuda"
    skip_figures: bool = False

    @classmethod
    def default(cls) -> "TargetDiscoveryConfig":
        return cls()

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return datetime.now().strftime("%Y%m%d_%H%M%S")
```

- [ ] **Step 6: Implement shared helpers**

Create `src/discovery/target_discovery/utils.py`:

```python
"""Shared helper functions for target discovery."""
from __future__ import annotations

import numpy as np


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    out = np.array(adj, dtype=float, copy=True)
    if out.size == 0:
        return out
    np.fill_diagonal(out, 0.0)
    mx = float(out.max())
    return out / mx if mx > 0 else out


def knn_adjacency(dist: np.ndarray, k: int) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    k = max(1, min(int(k), n - 1))
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    finite = d[np.isfinite(d)]
    scale = max(float(np.median(finite)) if finite.size else 1.0, 1e-6)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.argsort(d[i])[:k]:
            weight = float(np.exp(-float(dist[i, j]) / scale))
            adj[i, j] = max(adj[i, j], weight)
            adj[j, i] = max(adj[j, i], weight)
    return normalize_adjacency(adj)


def minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    denom = mx - mn
    if denom <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / denom
```

- [ ] **Step 7: Run tests to verify Task 1 passes**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_geometry.py -q
```

Expected: `4 passed`.

- [ ] **Step 8: Commit Task 1**

Run:

```powershell
git add src\discovery tests\discovery\__init__.py tests\discovery\test_target_discovery_geometry.py
git commit -m "feat: scaffold target discovery pipeline package"
```

---

### Task 2: Artifact Writer And Manifest

**Files:**
- Create: `src/discovery/target_discovery/artifacts.py`
- Test: `tests/discovery/test_target_discovery_artifacts.py`

- [ ] **Step 1: Write artifact tests**

Create `tests/discovery/test_target_discovery_artifacts.py`:

```python
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.discovery.target_discovery.artifacts import ArtifactWriter, json_default


def test_json_default_serializes_common_science_objects():
    assert json_default(np.int64(3)) == 3
    assert json_default(np.float32(1.5)) == 1.5
    assert json_default(np.array([1, 2])) == [1, 2]
    assert json_default({"A", "B"}) in (["A", "B"], ["B", "A"])
    df_payload = json_default(pd.DataFrame({"x": [1]}))
    assert df_payload == [{"x": 1}]


def test_artifact_writer_creates_sections_and_manifest(tmp_path):
    writer = ArtifactWriter(tmp_path, run_id="unit")
    table_path = writer.write_table("example.csv", pd.DataFrame({"gene": ["A"]}), section="candidates")
    json_path = writer.write_json("metrics.json", {"value": np.float32(2.0)}, section="scoring")
    array_path = writer.write_array("adj.npy", np.eye(2), section="spatial")
    md_path = writer.write_markdown("report.md", "# Report\n", section="reports")
    writer.finalize()

    assert table_path.exists()
    assert json_path.exists()
    assert array_path.exists()
    assert md_path.exists()
    manifest = json.loads((tmp_path / "unit" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit"
    assert "candidates/example.csv" in manifest["artifacts"]
```

- [ ] **Step 2: Run artifact tests to verify they fail**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_artifacts.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing `artifacts`.

- [ ] **Step 3: Implement `artifacts.py`**

Create `src/discovery/target_discovery/artifacts.py`:

```python
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
```

- [ ] **Step 4: Run artifact tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_artifacts.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src\discovery\target_discovery\artifacts.py tests\discovery\test_target_discovery_artifacts.py
git commit -m "feat: add target discovery artifact writer"
```

---

### Task 3: Stage Protocol, Run Context, Pipeline Shell, And Smoke Test

**Files:**
- Create: `src/discovery/target_discovery/stage.py`
- Create: `src/discovery/target_discovery/pipeline.py`
- Modify: `src/discovery/target_discovery/__init__.py`
- Test: `tests/discovery/test_target_discovery_pipeline.py`

- [ ] **Step 1: Write pipeline smoke test with fake stages**

Create `tests/discovery/test_target_discovery_pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.discovery.target_discovery.artifacts import ArtifactWriter
from src.discovery.target_discovery.config import DiscoveryPaths, TargetDiscoveryConfig
from src.discovery.target_discovery.pipeline import TargetDiscoveryPipeline
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


@dataclass
class FakeStage:
    name: str
    key: str
    value: Any

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        context.writer.write_json(f"{self.name}.json", {"ok": True}, section="fake")
        return {self.key: self.value}


def _config(tmp_path):
    paths = DiscoveryPaths.default(root=tmp_path, output_base=tmp_path / "runs")
    return TargetDiscoveryConfig(paths=paths, run_id="smoke", device="cpu")


def test_pipeline_runs_fake_stages_and_writes_manifest(tmp_path):
    cfg = _config(tmp_path)
    pipeline = TargetDiscoveryPipeline(
        config=cfg,
        stages=[
            FakeStage("one", "a", 1),
            FakeStage("two", "b", 2),
        ],
        icb_data_mode_detector=lambda config: "deg_only",
    )
    outputs = pipeline.run()

    assert outputs["a"] == 1
    assert outputs["b"] == 2
    assert (tmp_path / "runs" / "smoke" / "manifest.json").exists()
    assert (tmp_path / "runs" / "smoke" / "fake" / "one.json").exists()
```

- [ ] **Step 2: Run pipeline smoke test to verify it fails**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_pipeline.py -q
```

Expected: FAIL with missing pipeline/stage modules.

- [ ] **Step 3: Implement stage protocol and context**

Create `src/discovery/target_discovery/stage.py`:

```python
"""Stage protocol and run context for target discovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from src.discovery.target_discovery.artifacts import ArtifactWriter
from src.discovery.target_discovery.config import TargetDiscoveryConfig


@dataclass
class TargetDiscoveryRunContext:
    config: TargetDiscoveryConfig
    writer: ArtifactWriter
    started_at: float
    icb_data_mode: str


class DiscoveryStage(Protocol):
    name: str

    def run(
        self,
        context: TargetDiscoveryRunContext,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...
```

- [ ] **Step 4: Implement pipeline shell**

Create `src/discovery/target_discovery/pipeline.py`:

```python
"""Pipeline orchestration for target discovery."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from src.discovery.target_discovery.artifacts import ArtifactWriter
from src.discovery.target_discovery.config import TargetDiscoveryConfig
from src.discovery.target_discovery.stage import DiscoveryStage, TargetDiscoveryRunContext


def default_icb_data_mode_detector(config: TargetDiscoveryConfig) -> str:
    if config.paths.reference_manifest_path.exists():
        return "reference"
    if config.paths.icb_h5ad_path.exists():
        return "h5ad"
    return "deg_only"


class TargetDiscoveryPipeline:
    def __init__(
        self,
        config: TargetDiscoveryConfig,
        stages: Iterable[DiscoveryStage] | None = None,
        icb_data_mode_detector: Callable[[TargetDiscoveryConfig], str] = default_icb_data_mode_detector,
    ):
        self.config = config
        self.stages = list(stages or [])
        self.icb_data_mode_detector = icb_data_mode_detector

    def run(self) -> dict[str, Any]:
        run_id = self.config.resolved_run_id()
        writer = ArtifactWriter(self.config.paths.output_base, run_id=run_id)
        context = TargetDiscoveryRunContext(
            config=self.config,
            writer=writer,
            started_at=time.time(),
            icb_data_mode=self.icb_data_mode_detector(self.config),
        )
        outputs: dict[str, Any] = {}
        try:
            for stage in self.stages:
                t0 = time.time()
                produced = dict(stage.run(context, outputs))
                outputs.update(produced)
                writer.record_stage(stage.name, time.time() - t0, produced)
            outputs["run_dir"] = writer.run_dir
            outputs["manifest_path"] = writer.finalize()
            return outputs
        except Exception:
            writer.finalize()
            raise
```

- [ ] **Step 5: Export public classes**

Update `src/discovery/target_discovery/__init__.py`:

```python
"""Target discovery pipeline package."""

from src.discovery.target_discovery.config import TargetDiscoveryConfig
from src.discovery.target_discovery.pipeline import TargetDiscoveryPipeline

__all__ = ["TargetDiscoveryConfig", "TargetDiscoveryPipeline"]
```

- [ ] **Step 6: Run pipeline test**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_pipeline.py -q
```

Expected: `1 passed`.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src\discovery\target_discovery\stage.py src\discovery\target_discovery\pipeline.py src\discovery\target_discovery\__init__.py tests\discovery\test_target_discovery_pipeline.py
git commit -m "feat: add target discovery pipeline orchestration"
```

---

### Task 4: Candidate, Expression, And Spatial Lightweight Stages

**Files:**
- Create: `src/discovery/target_discovery/candidates.py`
- Create: `src/discovery/target_discovery/expression.py`
- Create: `src/discovery/target_discovery/spatial.py`
- Test: `tests/discovery/test_target_discovery_candidates.py`

- [ ] **Step 1: Write candidate/expression/spatial tests**

Create `tests/discovery/test_target_discovery_candidates.py`:

```python
from __future__ import annotations

import pandas as pd

from src.discovery.target_discovery.candidates import aggregate_candidate_pool
from src.discovery.target_discovery.expression import assemble_cluster_expression
from src.discovery.target_discovery.spatial import build_spatial_adjacency_from_tables


def test_aggregate_candidate_pool_scores_multisource_genes():
    neu = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "celltype_neu": ["Fibroblast_S1", "Macrophage"],
            "lfc_neu": [1.0, -0.7],
            "padj_neu": [0.01, 0.02],
        }
    )
    icb = pd.DataFrame(
        {
            "gene": ["A"],
            "celltype_icb": ["Fibro"],
            "lfc_icb": [0.8],
            "padj_icb": [0.03],
            "source_file": ["unit.csv"],
        }
    )
    ifng = pd.DataFrame(
        {
            "gene": ["C"],
            "celltype_ifng": ["IFNG_focus"],
            "lfc_ifng": [0.5],
            "mmr_group": ["MSS"],
        }
    )
    out = aggregate_candidate_pool(neu, icb, ifng)
    assert out.iloc[0]["gene"] == "A"
    assert int(out[out["gene"] == "A"].iloc[0]["cross_queue_count"]) == 2
    assert "init_score" in out.columns


def test_assemble_cluster_expression_log1p_means():
    tables = {
        "Fibroblast_S1": pd.DataFrame({"s1": [0.0, 3.0], "s2": [2.0, 5.0]}, index=["A", "B"]),
        "Macrophage": pd.DataFrame({"s1": [1.0, 1.0], "s2": [1.0, 3.0]}, index=["A", "B"]),
    }
    expr, labels = assemble_cluster_expression(tables)
    assert labels == ["Fibroblast_S1", "Macrophage"]
    assert expr.shape == (2, 2)
    assert float(expr.loc["Fibroblast_S1", "A"]) > 0


def test_build_spatial_adjacency_from_tables_returns_square_matrix():
    table = pd.DataFrame(
        {
            "Fibro_ADAMDEC1": [1.0, 0.0, 1.0],
            "Mac_M1": [0.0, 1.0, 0.5],
        }
    )
    out = build_spatial_adjacency_from_tables([table], ["Fibroblast_S1", "Macrophage"])
    assert out.shape == (2, 2)
    assert out[0, 0] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_candidates.py -q
```

Expected: FAIL with missing modules/functions.

- [ ] **Step 3: Implement candidate aggregation**

Create `src/discovery/target_discovery/candidates.py`:

```python
"""Candidate target discovery from multi-source DEG tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.discovery.target_discovery.constants import ANCHOR_GENES, IFNG_FOCUS_GENES
from src.discovery.target_discovery.utils import minmax


def aggregate_candidate_pool(
    neu_df: pd.DataFrame,
    icb_df: pd.DataFrame,
    ifng_df: pd.DataFrame,
) -> pd.DataFrame:
    all_genes: set[str] = set()
    if not neu_df.empty:
        all_genes |= set(neu_df["gene"].dropna().astype(str).unique())
    if not icb_df.empty:
        all_genes |= set(icb_df["gene"].dropna().astype(str).unique())
    if not ifng_df.empty:
        all_genes |= set(ifng_df["gene"].dropna().astype(str).unique())
    all_genes -= {"", "nan", "None"}

    rows: list[dict] = []
    for gene in sorted(all_genes):
        n_sub = neu_df[neu_df["gene"].astype(str) == gene] if not neu_df.empty else pd.DataFrame()
        i_sub = icb_df[icb_df["gene"].astype(str) == gene] if not icb_df.empty else pd.DataFrame()
        f_sub = ifng_df[ifng_df["gene"].astype(str) == gene] if not ifng_df.empty else pd.DataFrame()

        lfcs: list[float] = []
        for frame, col in [(n_sub, "lfc_neu"), (i_sub, "lfc_icb"), (f_sub, "lfc_ifng")]:
            if not frame.empty and col in frame:
                lfcs.extend(pd.to_numeric(frame[col], errors="coerce").dropna().astype(float).tolist())
        padjs: list[float] = []
        for frame, col in [(n_sub, "padj_neu"), (i_sub, "padj_icb")]:
            if not frame.empty and col in frame:
                padjs.extend(pd.to_numeric(frame[col], errors="coerce").dropna().astype(float).tolist())

        signs = np.sign(lfcs) if lfcs else np.array([])
        majority = np.sign(np.sum(signs)) if signs.size else 0
        direction_consistency = float(np.mean(signs == majority)) if majority != 0 else (0.5 if signs.size else 0.0)

        n_ct_neu = int(n_sub["celltype_neu"].nunique()) if "celltype_neu" in n_sub else 0
        n_ct_icb = int(i_sub["celltype_icb"].nunique()) if "celltype_icb" in i_sub else 0
        n_ct_ifng = int(f_sub["celltype_ifng"].nunique()) if "celltype_ifng" in f_sub else 0
        min_padj = float(np.min(padjs)) if padjs else 1.0

        rows.append(
            {
                "gene": gene,
                "n_celltypes_neu": n_ct_neu,
                "n_celltypes_icb": n_ct_icb,
                "n_celltypes_ifng": n_ct_ifng,
                "cross_queue_count": int(n_ct_neu > 0) + int(n_ct_icb > 0) + int(n_ct_ifng > 0),
                "mean_lfc": float(np.mean(lfcs)) if lfcs else 0.0,
                "mean_abs_lfc": float(np.mean(np.abs(lfcs))) if lfcs else 0.0,
                "direction_consistency": direction_consistency,
                "min_padj": min_padj,
                "neg_log10_padj": -float(np.log10(max(min_padj, 1e-300))),
                "is_anchor": gene in ANCHOR_GENES,
                "is_ifng_target": gene in IFNG_FOCUS_GENES,
                "celltypes_neu": ";".join(sorted(n_sub["celltype_neu"].astype(str).unique())) if "celltype_neu" in n_sub else "",
                "celltypes_icb": ";".join(sorted(i_sub["celltype_icb"].astype(str).unique())) if "celltype_icb" in i_sub else "",
                "celltypes_ifng": ";".join(sorted(f_sub["celltype_ifng"].astype(str).unique())) if "celltype_ifng" in f_sub else "",
            }
        )

    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool
    pool["init_score"] = (
        pool["cross_queue_count"] * 2.0
        + minmax(pool["mean_abs_lfc"].values) * 1.5
        + minmax(pool["neg_log10_padj"].values) * 1.5
        + pool["direction_consistency"] * 1.0
        + minmax(pool["n_celltypes_neu"].values) * 0.5
        + minmax(pool["n_celltypes_ifng"].values) * 0.5
    )
    return pool.sort_values("init_score", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Implement expression assembly**

Create `src/discovery/target_discovery/expression.py`:

```python
"""Cluster-level expression assembly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_normalized_count_tables(neu_dir: Path, celltypes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for celltype in celltypes:
        path = Path(neu_dir) / f"{celltype}-NormalizedCounts.tsv"
        if path.exists():
            tables[celltype] = pd.read_csv(path, sep="\t", index_col=0)
    return tables


def assemble_cluster_expression(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    if not tables:
        raise RuntimeError("No NormalizedCounts loaded")
    series = {celltype: table.mean(axis=1) for celltype, table in tables.items()}
    expr = pd.DataFrame(series).T.fillna(0.0)
    expr = np.log1p(expr)
    labels = list(expr.index)
    return expr, labels
```

- [ ] **Step 5: Implement spatial adjacency**

Create `src/discovery/target_discovery/spatial.py`:

```python
"""Spatial co-localization context for target discovery."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.discovery.target_discovery.constants import ST_DECONV_MAP
from src.discovery.target_discovery.utils import normalize_adjacency


def read_st_metadata_tables(st_dir: Path) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for path in sorted(Path(st_dir).glob("STmetadata_*.csv")):
        try:
            tables.append(pd.read_csv(path, low_memory=False))
        except Exception:
            continue
    return tables


def build_spatial_adjacency_from_tables(tables: list[pd.DataFrame], node_labels: list[str]) -> np.ndarray:
    all_corr: list[np.ndarray] = []
    n_nodes = len(node_labels)
    for table in tables:
        scores = np.zeros((len(table), n_nodes), dtype=float)
        for i, celltype in enumerate(node_labels):
            cols = [col for col in ST_DECONV_MAP.get(celltype, ()) if col in table.columns]
            if cols:
                scores[:, i] = table[cols].mean(axis=1).values
        corr = np.nan_to_num(np.corrcoef(scores.T), nan=0.0)
        all_corr.append(corr)
    if not all_corr:
        return np.eye(n_nodes)
    adj = np.mean(all_corr, axis=0)
    adj = np.where(adj > 0.05, adj, 0.0)
    return normalize_adjacency(adj)
```

- [ ] **Step 6: Run tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_candidates.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Commit Task 4**

Run:

```powershell
git add src\discovery\target_discovery\candidates.py src\discovery\target_discovery\expression.py src\discovery\target_discovery\spatial.py tests\discovery\test_target_discovery_candidates.py
git commit -m "feat: add lightweight target discovery data stages"
```

---

### Task 5: Geometry Module And Stage

**Files:**
- Create: `src/discovery/target_discovery/geometry.py`
- Modify: `tests/discovery/test_target_discovery_geometry.py`

- [ ] **Step 1: Extend geometry tests**

Append to `tests/discovery/test_target_discovery_geometry.py`:

```python
import pandas as pd

from src.discovery.target_discovery.geometry import blend_adjacencies, compute_geometry


def test_blend_adjacencies_normalizes_result():
    spatial = np.array([[0.0, 1.0], [1.0, 0.0]])
    geom = np.array([[0.0, 0.5], [0.5, 0.0]])
    out = blend_adjacencies(spatial, geom, blend=0.5)
    assert np.allclose(out, out.T)
    assert float(out.max()) == 1.0


def test_compute_geometry_euclidean_small_expression():
    expr = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [3.0, 2.0, 1.0],
            "C": [0.5, 0.7, 0.9],
        },
        index=["n1", "n2", "n3"],
    )
    out = compute_geometry(expr, ["n1", "n2", "n3"], mode="euclidean", k=1)
    assert out["embedding"].shape[0] == 3
    assert out["dist_matrix"].shape == (3, 3)
    assert out["adjacency"].shape == (3, 3)
    assert out["metrics"]["mode"] == "euclidean"
```

- [ ] **Step 2: Run geometry tests to verify new tests fail**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_geometry.py -q
```

Expected: FAIL with missing `geometry` module.

- [ ] **Step 3: Implement geometry module**

Create `src/discovery/target_discovery/geometry.py`:

```python
"""Geometry comparison helpers for target discovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.discovery.target_discovery.constants import TYPE_MAPPING
from src.discovery.target_discovery.utils import knn_adjacency, normalize_adjacency


def compute_geometry(
    cluster_expr: pd.DataFrame,
    node_labels: list[str],
    mode: str,
    k: int = 4,
) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = cluster_expr.values.astype(np.float32)
    xz = StandardScaler().fit_transform(x)
    n_comp = min(8, xz.shape[0], xz.shape[1])
    z = PCA(n_components=n_comp).fit_transform(xz)
    z2 = z[:, :2]
    n_nodes = x.shape[0]

    if mode == "hyperbolic":
        from src.models.hyperbolic.lorentz import lorentz_to_poincare, polar_project
        from src.models.hyperbolic.poincare import poincare_distance

        zt = torch.tensor(z2, dtype=torch.float32)
        zt = zt / (zt.std() + 1e-6) * 0.5
        emb = lorentz_to_poincare(polar_project(zt)).detach().cpu().numpy()
        dist = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = poincare_distance(
                    torch.tensor(emb[i : i + 1], dtype=torch.float32),
                    torch.tensor(emb[j : j + 1], dtype=torch.float32),
                    c=1.0,
                ).item()
                dist[i, j] = dist[j, i] = d
    else:
        from scipy.spatial.distance import cdist

        emb = z2
        dist = cdist(emb, emb)

    adj = knn_adjacency(dist, k)
    type_map = {label: TYPE_MAPPING.get(label, label) for label in node_labels}
    within: list[float] = []
    between: list[float] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if type_map[node_labels[i]] == type_map[node_labels[j]]:
                within.append(float(dist[i, j]))
            else:
                between.append(float(dist[i, j]))

    metrics = {
        "mode": mode,
        "radius_mean": float(np.linalg.norm(emb, axis=1).mean()),
        "within_dist": float(np.mean(within)) if within else 0.0,
        "between_dist": float(np.mean(between)) if between else 0.0,
        "separation": float(np.mean(between) / max(np.mean(within), 1e-8)) if within else 0.0,
        "n_edges": int((adj > 0).sum()),
    }
    return {"mode": mode, "embedding": emb, "dist_matrix": dist, "adjacency": adj, "metrics": metrics}


def blend_adjacencies(spatial_adj: np.ndarray, geometry_adj: np.ndarray, blend: float) -> np.ndarray:
    blend = float(np.clip(blend, 0.0, 1.0))
    return normalize_adjacency((1.0 - blend) * spatial_adj + blend * geometry_adj)
```

- [ ] **Step 4: Run geometry tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_geometry.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add src\discovery\target_discovery\geometry.py tests\discovery\test_target_discovery_geometry.py
git commit -m "feat: add target discovery geometry helpers"
```

---

### Task 6: Scoring, Retention, And Mode Comparison

**Files:**
- Create: `src/discovery/target_discovery/scoring.py`
- Test: `tests/discovery/test_target_discovery_scoring.py`

- [ ] **Step 1: Write scoring tests**

Create `tests/discovery/test_target_discovery_scoring.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from src.discovery.target_discovery.scoring import compare_modes, retain_hubs_and_combos, score_candidates


def _candidate_pool():
    return pd.DataFrame(
        {
            "gene": ["A", "MFAP2"],
            "cross_queue_count": [2, 1],
            "direction_consistency": [1.0, 0.5],
            "mean_abs_lfc": [1.0, 0.2],
            "celltypes_neu": ["Fibroblast_S1", "Macrophage"],
            "is_anchor": [False, True],
            "is_ifng_target": [False, False],
            "mean_lfc": [1.0, 0.2],
            "min_padj": [0.01, 0.05],
        }
    )


def _step2():
    return {
        "node_labels": ["Fibroblast_S1", "Macrophage"],
        "betweenness": {"Fibroblast_S1": 0.7, "Macrophage": 0.2},
        "flow_edges": [{"source": "A", "target": "B"}],
        "metrics": {"graph_sparsity": 0.5, "hsic_independence": 0.8, "known_axis_recall": 0.4, "mean_bootstrap_freq": 0.3},
    }


def test_score_candidates_returns_ranked_frame():
    cluster_expr = pd.DataFrame({"A": [1.0, 2.0]}, index=["Fibroblast_S1", "Macrophage"])
    ranked = score_candidates(_candidate_pool(), _step2(), _step2(), {"A": {"n_ranked": 3}}, {}, cluster_expr)
    assert list(ranked["rank"]) == [1, 2]
    assert "final_score" in ranked.columns
    assert ranked.iloc[0]["final_score"] >= ranked.iloc[1]["final_score"]


def test_retain_hubs_and_combos_keeps_anchor_and_combo_rows():
    ranking = _candidate_pool()
    ranking["rank"] = [1, 2]
    step3 = {"A": {"ranked_targets": pd.DataFrame([{"ligand": "A", "receptor": "B", "target_priority_score": 0.9}])}}
    hubs, combos = retain_hubs_and_combos(ranking, step3)
    assert "MFAP2" in set(hubs["gene"])
    assert len(combos) == 1


def test_compare_modes_returns_geometry_and_step2_summary():
    geom = {"metrics": {"separation": 2.0}}
    comp = compare_modes(geom, geom, _step2(), _step2(), {}, {}, pd.DataFrame({"gene": ["A"], "final_score": [1.0]}))
    assert comp["geometry"]["hyp_separation"] == 2.0
    assert "hyp_graph_sparsity" in comp["step2"]
```

- [ ] **Step 2: Run scoring tests to verify they fail**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_scoring.py -q
```

Expected: FAIL with missing `scoring` module.

- [ ] **Step 3: Implement scoring module**

Create `src/discovery/target_discovery/scoring.py` by extracting the pure logic from old `score_and_rank()`, `retain_hubs_and_combos()`, and `compare_modes()`.

Required function signatures:

```python
def score_candidates(
    candidate_pool: pd.DataFrame,
    step2_hyp: dict,
    step2_euc: dict,
    step3_hyp: dict,
    step3_euc: dict,
    cluster_expr: pd.DataFrame,
) -> pd.DataFrame:
    ...


def retain_hubs_and_combos(
    ranking: pd.DataFrame,
    step3_results_hyp: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...


def compare_modes(
    geom_hyp: dict,
    geom_euc: dict,
    s2_hyp: dict,
    s2_euc: dict,
    s3_hyp: dict,
    s3_euc: dict,
    ranking: pd.DataFrame,
) -> dict:
    ...
```

Implementation requirements:

- Use `minmax()` from `utils.py`.
- Use `SCORE_WEIGHTS` and `ANCHOR_GENES` from `constants.py`.
- Do not write files in this module.
- Return dataframes/dicts only.

- [ ] **Step 4: Run scoring tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_scoring.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add src\discovery\target_discovery\scoring.py tests\discovery\test_target_discovery_scoring.py
git commit -m "feat: add target discovery scoring helpers"
```

---

### Task 7: Loaders And Full Lightweight Stages

**Files:**
- Create: `src/discovery/target_discovery/loaders.py`
- Create: `src/discovery/target_discovery/lightweight_stages.py`
- Modify: `src/discovery/target_discovery/pipeline.py`
- Test: `tests/discovery/test_target_discovery_pipeline.py`

- [ ] **Step 1: Add fake-free stage smoke test**

Append to `tests/discovery/test_target_discovery_pipeline.py`:

```python
import pandas as pd

from src.discovery.target_discovery.lightweight_stages import (
    CandidateDiscoveryStage,
    ExpressionAssemblyStage,
    SpatialContextStage,
)


def test_lightweight_stages_can_run_on_synthetic_files(tmp_path):
    paths = DiscoveryPaths(
        root=tmp_path,
        data_dir=tmp_path / "data",
        neu_dir=tmp_path / "neu",
        ifng_dir=tmp_path / "ifng",
        icb_dir=tmp_path / "icb",
        st_dir=tmp_path / "st",
        output_base=tmp_path / "runs",
        icb_h5ad_path=tmp_path / "data" / "scRNA" / "scCRC_ICB" / "expression.h5ad",
        reference_manifest_path=tmp_path / "data" / "ref" / "manifest" / "reference_manifest.json",
    )
    paths.neu_dir.mkdir(parents=True, exist_ok=True)
    paths.icb_dir.mkdir(parents=True, exist_ok=True)
    (paths.ifng_dir / "results" / "tables").mkdir(parents=True, exist_ok=True)
    paths.st_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"symbol": ["A"], "padj": [0.01], "log2FoldChange": [1.0]}).to_csv(
        paths.neu_dir / "Fibroblast_S1-DESeq2_result.tsv", sep="\t", index=False
    )
    pd.DataFrame({"s1": [1.0], "s2": [3.0]}, index=["A"]).to_csv(
        paths.neu_dir / "Fibroblast_S1-NormalizedCounts.tsv", sep="\t"
    )
    pd.DataFrame({"Fibro_ADAMDEC1": [1.0, 0.5]}).to_csv(paths.st_dir / "STmetadata_unit.csv", index=False)

    cfg = TargetDiscoveryConfig(paths=paths, run_id="light", device="cpu")
    pipeline = TargetDiscoveryPipeline(
        cfg,
        stages=[CandidateDiscoveryStage(), ExpressionAssemblyStage(), SpatialContextStage()],
        icb_data_mode_detector=lambda config: "deg_only",
    )
    out = pipeline.run()
    assert not out["candidate_pool"].empty
    assert out["cluster_expression"].shape[0] == 1
    assert out["spatial_adjacency"].shape == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_pipeline.py -q
```

Expected: FAIL with missing `lightweight_stages`.

- [ ] **Step 3: Implement loaders**

Create `src/discovery/target_discovery/loaders.py`:

```python
"""External data loading helpers for target discovery."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_neu_deg_tables(neu_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted(Path(neu_dir).glob("*-DESeq2_result.tsv")):
        celltype = path.stem.replace("-DESeq2_result", "")
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            continue
        if "padj" not in df.columns or "log2FoldChange" not in df.columns:
            continue
        sig = df[(df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 0.5)].copy()
        for _, row in sig.iterrows():
            records.append(
                {
                    "gene": str(row.get("symbol", "")),
                    "celltype_neu": celltype,
                    "lfc_neu": float(row["log2FoldChange"]),
                    "padj_neu": float(row["padj"]),
                }
            )
    return pd.DataFrame(records, columns=["gene", "celltype_neu", "lfc_neu", "padj_neu"])


def read_icb_deg_tables(icb_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for name in ["DEGs_MSS_response_Mid_lfc0.5.csv", "DEGs_MSS_Mid.csv", "DEGs_MSS_response_Major_lfc0.5.csv", "DEGs_MSS_Major.csv"]:
        path = Path(icb_dir) / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        gene_col = "gene" if "gene" in df.columns else df.columns[0]
        lfc_col = "avg_log2FC" if "avg_log2FC" in df.columns else None
        padj_col = "p_val_adj" if "p_val_adj" in df.columns else None
        ct_col = "celltype" if "celltype" in df.columns else None
        for _, row in df.iterrows():
            if lfc_col and padj_col:
                try:
                    padj = float(row[padj_col])
                    lfc = float(row[lfc_col])
                except (TypeError, ValueError):
                    continue
                if padj > 0.05 or abs(lfc) < 0.3:
                    continue
            records.append(
                {
                    "gene": str(row[gene_col]),
                    "celltype_icb": str(row[ct_col]) if ct_col else name,
                    "lfc_icb": float(row[lfc_col]) if lfc_col else float("nan"),
                    "padj_icb": float(row[padj_col]) if padj_col else float("nan"),
                    "source_file": name,
                }
            )
    return pd.DataFrame(records, columns=["gene", "celltype_icb", "lfc_icb", "padj_icb", "source_file"])


def read_ifng_tables(ifng_dir: Path, focus_genes: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict] = []
    path = Path(ifng_dir) / "results" / "tables" / "targets_shared_specific_by_mmr.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            gene_col = "gene" if "gene" in df.columns else df.columns[0]
            for _, row in df.iterrows():
                gene = str(row[gene_col])
                if gene and gene != "nan":
                    records.append(
                        {
                            "gene": gene,
                            "celltype_ifng": str(row.get("celltype", "unknown")),
                            "lfc_ifng": float(row.get("log2FoldChange", row.get("avg_log2FC", 0))),
                            "mmr_group": str(row.get("mmr_group", "")),
                        }
                    )
        except Exception:
            records = []
    for gene in focus_genes:
        if not any(row["gene"] == gene for row in records):
            records.append({"gene": gene, "celltype_ifng": "IFNG_focus", "lfc_ifng": float("nan"), "mmr_group": ""})
    return pd.DataFrame(records, columns=["gene", "celltype_ifng", "lfc_ifng", "mmr_group"])
```

- [ ] **Step 4: Implement lightweight stages**

Create `src/discovery/target_discovery/lightweight_stages.py`:

```python
"""Lightweight target discovery stages with synthetic-testable behavior."""
from __future__ import annotations

from typing import Any, Mapping

from src.discovery.target_discovery.candidates import aggregate_candidate_pool
from src.discovery.target_discovery.constants import CELLTYPES, IFNG_FOCUS_GENES
from src.discovery.target_discovery.expression import assemble_cluster_expression, read_normalized_count_tables
from src.discovery.target_discovery.loaders import read_icb_deg_tables, read_ifng_tables, read_neu_deg_tables
from src.discovery.target_discovery.spatial import build_spatial_adjacency_from_tables, read_st_metadata_tables
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


class CandidateDiscoveryStage:
    name = "candidate_discovery"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        paths = context.config.paths
        pool = aggregate_candidate_pool(
            read_neu_deg_tables(paths.neu_dir),
            read_icb_deg_tables(paths.icb_dir),
            read_ifng_tables(paths.ifng_dir, IFNG_FOCUS_GENES),
        )
        context.writer.write_table("candidate_pool.csv", pool, section="candidates")
        return {"candidate_pool": pool}


class ExpressionAssemblyStage:
    name = "expression_assembly"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        tables = read_normalized_count_tables(context.config.paths.neu_dir, CELLTYPES)
        expr, labels = assemble_cluster_expression(tables)
        context.writer.write_table("cluster_expression.csv", expr.reset_index().rename(columns={"index": "celltype"}), section="expression")
        context.writer.write_json("node_labels.json", {"node_labels": labels}, section="expression")
        return {"cluster_expression": expr, "node_labels": labels}


class SpatialContextStage:
    name = "spatial_context"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        labels = list(inputs["node_labels"])
        tables = read_st_metadata_tables(context.config.paths.st_dir)
        adj = build_spatial_adjacency_from_tables(tables, labels)
        context.writer.write_array("spatial_adjacency.npy", adj, section="spatial")
        return {"spatial_adjacency": adj}
```

- [ ] **Step 5: Run pipeline tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery\test_target_discovery_pipeline.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 6: Commit Task 7**

Run:

```powershell
git add src\discovery\target_discovery\loaders.py src\discovery\target_discovery\lightweight_stages.py tests\discovery\test_target_discovery_pipeline.py
git commit -m "feat: add target discovery lightweight stages"
```

---

### Task 8: Heavy Stage Wrappers For Geometry, Causal, Perturbation, Niche, Reporting, And Figures

**Files:**
- Create: `src/discovery/target_discovery/heavy_stages.py`
- Create: `src/discovery/target_discovery/causal_stage.py`
- Create: `src/discovery/target_discovery/perturbation_stage.py`
- Create: `src/discovery/target_discovery/niche.py`
- Create: `src/discovery/target_discovery/reporting.py`
- Create: `src/discovery/target_discovery/figures.py`
- Modify: `src/discovery/target_discovery/pipeline.py`

- [ ] **Step 1: Create heavy stage shell file**

Create `src/discovery/target_discovery/heavy_stages.py`:

```python
"""Stage wrappers for geometry, causal, perturbation, scoring, niche, and reporting."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.discovery.target_discovery.geometry import blend_adjacencies, compute_geometry
from src.discovery.target_discovery.scoring import compare_modes, retain_hubs_and_combos, score_candidates
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


class GeometryComparisonStage:
    name = "geometry_comparison"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        expr = inputs["cluster_expression"]
        labels = inputs["node_labels"]
        spatial_adj = inputs["spatial_adjacency"]
        geom_results: dict[str, dict] = {}
        blended: dict[str, Any] = {}
        for mode in context.config.geometry.modes:
            geom = compute_geometry(expr, labels, mode=mode, k=context.config.geometry.geometry_k)
            geom_results[mode] = geom
            blended_adj = blend_adjacencies(spatial_adj, geom["adjacency"], context.config.geometry.geometry_blend)
            blended[mode] = blended_adj
            context.writer.write_table(
                "embedding.csv",
                pd.DataFrame(geom["embedding"], index=labels, columns=["d1", "d2"]).reset_index().rename(columns={"index": "node"}),
                section=f"geometry/{mode}",
            )
            context.writer.write_array("distance.npy", geom["dist_matrix"], section=f"geometry/{mode}")
            context.writer.write_array("adjacency.npy", geom["adjacency"], section=f"geometry/{mode}")
            context.writer.write_array("blended.npy", blended_adj, section=f"geometry/{mode}")
            context.writer.write_json("metrics.json", geom["metrics"], section=f"geometry/{mode}")
        return {"geometry_results": geom_results, "blended_adjacencies": blended}


class EvidenceScoringStage:
    name = "evidence_scoring"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        causal = inputs["causal_results"]
        perturb = inputs["perturbation_results"]
        ranking = score_candidates(
            inputs["candidate_pool"],
            causal["hyperbolic"],
            causal["euclidean"],
            perturb["hyperbolic"],
            perturb["euclidean"],
            inputs["cluster_expression"],
        )
        hubs, combos = retain_hubs_and_combos(ranking, perturb["hyperbolic"])
        comparison = compare_modes(
            inputs["geometry_results"]["hyperbolic"],
            inputs["geometry_results"]["euclidean"],
            causal["hyperbolic"],
            causal["euclidean"],
            perturb["hyperbolic"],
            perturb["euclidean"],
            ranking,
        )
        context.writer.write_table("target_ranking.csv", ranking, section="scoring")
        evidence_cols = [col for col in ["gene", "rank", "final_score", "s_causal", "s_spatial", "s_consistency", "s_actionability", "s_niche"] if col in ranking]
        context.writer.write_table("evidence_matrix.csv", ranking[evidence_cols], section="scoring")
        context.writer.write_table("hub_targets_retained.csv", hubs, section="scoring")
        context.writer.write_table("spatiotemporal_regulatory_combos.csv", combos, section="scoring")
        context.writer.write_json("mode_comparison.json", comparison, section="scoring")
        return {
            "target_ranking": ranking,
            "retained_hubs": hubs,
            "retained_combos": combos,
            "mode_comparison": comparison,
        }
```

- [ ] **Step 2: Move `run_step2()` into `causal_stage.py`**

Create `src/discovery/target_discovery/causal_stage.py`.

Implementation instructions:

- Copy the body of old `run_step2()` from `scripts/run_target_discovery.py`.
- Rename it to `run_causal_discovery()`.
- Add parameters `device: str`, `section: str`, and `writer: ArtifactWriter`.
- Replace writes to `out_dir / "step2"` with writer calls under `causal/<mode>`.
- Import `PRIOR_AXES` and `TYPE_MAPPING` from `constants.py`.
- Import `json_default` from `artifacts.py`.
- Preserve training, bootstrap, prior injection, DoWhy validation, signaling flow, metrics, and betweenness behavior.

Also add:

```python
class CausalDiscoveryStage:
    name = "causal_discovery"

    def run(self, context, inputs):
        results = {}
        for mode, adjacency in inputs["blended_adjacencies"].items():
            results[mode] = run_causal_discovery(
                cluster_expr=inputs["cluster_expression"],
                cluster_adj=adjacency,
                node_labels=inputs["node_labels"],
                writer=context.writer,
                section=f"causal/{mode}",
                device=context.config.device,
            )
        return {"causal_results": results}
```

- [ ] **Step 3: Move `run_step3_batch()` into `perturbation_stage.py`**

Create `src/discovery/target_discovery/perturbation_stage.py`.

Implementation instructions:

- Copy old `run_step3_batch()` into `run_perturbation_screen()`.
- Replace direct file writes with writer calls under `perturbation/<mode>`.
- Use `json_default` from `artifacts.py`.
- Preserve propagation, target ranking, counterfactual quality, and spatial quality behavior.
- Add a pure helper:

```python
def select_perturbation_targets(candidate_pool, cluster_expr, max_perturb):
    gene_upper = {c.upper(): c for c in cluster_expr.columns}
    available = [g for g in candidate_pool["gene"] if str(g).upper() in gene_upper]
    targets = []
    for anchor in ANCHOR_GENES:
        if anchor in available and anchor not in targets:
            targets.append(anchor)
    for gene in available:
        if gene not in targets and len(targets) < max_perturb:
            targets.append(gene)
    return targets
```

Add:

```python
class PerturbationScreenStage:
    name = "perturbation_screen"

    def run(self, context, inputs):
        targets = select_perturbation_targets(
            inputs["candidate_pool"],
            inputs["cluster_expression"],
            context.config.max_perturb,
        )
        results = {}
        for mode, causal_result in inputs["causal_results"].items():
            results[mode] = run_perturbation_screen(
                step2_results=causal_result,
                target_genes=targets,
                writer=context.writer,
                section=f"perturbation/{mode}",
            )
        return {"perturbation_targets": targets, "perturbation_results": results}
```

- [ ] **Step 4: Move niche functions into `niche.py`**

Create `src/discovery/target_discovery/niche.py`.

Implementation instructions:

- Copy these functions from old script:
  - `_broad_type_from_deconv_col`
  - `_normalize_rows`
  - `_read_st_deconv_table`
  - `_read_cosmx_deconv_like`
  - `_read_visiumhd_deconv_like`
  - `_merge_multimodal_deconv_tables`
  - `_dist_matrix`
  - `_hex_from_rgba`
  - `_assign_niche_colors`
  - `collect_available_data_inventory`
  - `build_unified_niche_definition`
  - `map_targets_to_unified_niches`
- Replace global paths with `context.config.paths`.
- Replace direct `NICHE_DIR` writes with `ArtifactWriter` writes under `niche/`.
- Preserve platform filtering.

Add:

```python
class UnifiedNicheStage:
    name = "unified_niche"

    def run(self, context, inputs):
        inventory = collect_available_data_inventory(context.config.paths, context.writer)
        niche_pack = build_unified_niche_definition(
            paths=context.config.paths,
            writer=context.writer,
            n_clusters=None,
            k_min=8,
            k_max=18,
            fallback_node_labels=inputs["node_labels"],
            platform=context.config.platform,
        )
        niche_map = map_targets_to_unified_niches(
            writer=context.writer,
            ranking=inputs["target_ranking"],
            cluster_expr=inputs["cluster_expression"],
            node_labels=inputs["node_labels"],
            niche_pack=niche_pack,
            combos=inputs["retained_combos"],
        )
        return {
            "available_data_inventory": inventory,
            "niche_pack": niche_pack,
            "target_niche": niche_map.get("target_niche"),
            "combo_niche": niche_map.get("combo_niche"),
        }
```

- [ ] **Step 5: Move reporting into `reporting.py`**

Create `src/discovery/target_discovery/reporting.py`.

Implementation requirements:

- Move old `generate_report()` body into `build_target_discovery_report()`, returning markdown text instead of writing directly.
- Add `build_migration_notes(run_dir: Path) -> str`.
- Migration notes must map old root `results/integration/discovery/` to new root `results/discovery/target_discovery/<run_id>/`.
- Include old-to-new mappings:
  - `candidate_pool.csv` -> `candidates/candidate_pool.csv`
  - `target_ranking.csv` -> `scoring/target_ranking.csv`
  - `evidence_matrix.csv` -> `scoring/evidence_matrix.csv`
  - `hub_targets_retained.csv` -> `scoring/hub_targets_retained.csv`
  - `spatiotemporal_regulatory_combos.csv` -> `scoring/spatiotemporal_regulatory_combos.csv`
  - `mode_comparison.json` -> `scoring/mode_comparison.json`
  - `target_discovery_report.md` -> `reports/target_discovery_report.md`
  - `comparison_report.md` -> `reports/migration_notes.md` plus `scoring/mode_comparison.json`

- [ ] **Step 6: Move figures into `figures.py`**

Create `src/discovery/target_discovery/figures.py`.

Implementation instructions:

- Move old `generate_figures()` into `generate_figure_pack()`.
- Replace direct `FIG_DIR` writes with `ArtifactWriter.write_figure(...)` under section `figures`.
- Import `TYPE_MAPPING` and `ANCHOR_GENES` from constants.
- Keep `matplotlib.use("Agg")`.
- Preserve current figure names.

- [ ] **Step 7: Add report/figure stage**

In `heavy_stages.py`, add:

```python
class ReportAndFigureStage:
    name = "report_and_figure"

    def run(self, context, inputs):
        from src.discovery.target_discovery.figures import generate_figure_pack
        from src.discovery.target_discovery.reporting import build_migration_notes, build_target_discovery_report

        if not context.config.skip_figures:
            generate_figure_pack(context.writer, inputs)
        report = build_target_discovery_report(context, inputs)
        migration = build_migration_notes(context.writer.run_dir)
        report_path = context.writer.write_markdown("target_discovery_report.md", report, section="reports")
        migration_path = context.writer.write_markdown("migration_notes.md", migration, section="reports")
        return {"target_discovery_report": report_path, "migration_notes": migration_path}
```

- [ ] **Step 8: Add default stage factory**

In `pipeline.py`, add:

```python
def default_target_discovery_stages():
    from src.discovery.target_discovery.heavy_stages import EvidenceScoringStage, GeometryComparisonStage, ReportAndFigureStage
    from src.discovery.target_discovery.lightweight_stages import CandidateDiscoveryStage, ExpressionAssemblyStage, SpatialContextStage
    from src.discovery.target_discovery.causal_stage import CausalDiscoveryStage
    from src.discovery.target_discovery.niche import UnifiedNicheStage
    from src.discovery.target_discovery.perturbation_stage import PerturbationScreenStage

    return [
        CandidateDiscoveryStage(),
        ExpressionAssemblyStage(),
        SpatialContextStage(),
        GeometryComparisonStage(),
        CausalDiscoveryStage(),
        PerturbationScreenStage(),
        EvidenceScoringStage(),
        UnifiedNicheStage(),
        ReportAndFigureStage(),
    ]
```

Change `TargetDiscoveryPipeline.__init__` so `stages=None` uses `default_target_discovery_stages()`.

- [ ] **Step 9: Run focused import/compile checks**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m compileall -q src\discovery scripts\run_target_discovery.py
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -c "from src.discovery.target_discovery.pipeline import TargetDiscoveryPipeline; from src.discovery.target_discovery.pipeline import default_target_discovery_stages; print(len(default_target_discovery_stages()))"
```

Expected:

- compileall exits 0.
- second command prints `9`.

- [ ] **Step 10: Commit Task 8**

Run:

```powershell
git add src\discovery\target_discovery\heavy_stages.py src\discovery\target_discovery\causal_stage.py src\discovery\target_discovery\perturbation_stage.py src\discovery\target_discovery\niche.py src\discovery\target_discovery\reporting.py src\discovery\target_discovery\figures.py src\discovery\target_discovery\pipeline.py
git commit -m "feat: wrap target discovery heavy stages"
```

---

### Task 9: Thin CLI Entrypoint

**Files:**
- Modify: `scripts/run_target_discovery.py`

- [ ] **Step 1: Replace script with thin CLI**

Replace `scripts/run_target_discovery.py` with:

```python
"""CLI entrypoint for the HyperSCA target discovery pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.discovery.target_discovery.config import DiscoveryPaths, GeometryModeConfig, TargetDiscoveryConfig
from src.discovery.target_discovery.pipeline import TargetDiscoveryPipeline


def _parse_genes(value: str) -> tuple[str, ...]:
    return tuple(g.strip() for g in value.split(",") if g.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperSCA Target Discovery")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "discovery" / "target_discovery")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-perturb", type=int, default=50)
    parser.add_argument("--geometry-k", type=int, default=4)
    parser.add_argument("--geometry-blend", type=float, default=0.30)
    parser.add_argument("--platform", choices=["cosmx", "visium", "visiumhd", "all"], default="all")
    parser.add_argument("--genes", type=str, default="")
    parser.add_argument("--hierarchy-levels", type=int, default=3)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = DiscoveryPaths.default(root=ROOT, output_base=args.output_dir)
    config = TargetDiscoveryConfig(
        paths=paths,
        geometry=GeometryModeConfig(geometry_k=args.geometry_k, geometry_blend=args.geometry_blend),
        max_perturb=args.max_perturb,
        platform=args.platform,
        focused_genes=_parse_genes(args.genes),
        hierarchy_levels=args.hierarchy_levels,
        run_id=args.run_id,
        device=args.device,
        skip_figures=args.skip_figures,
    )
    outputs = TargetDiscoveryPipeline(config).run()
    print(f"Run directory: {outputs['run_dir']}")
    print(f"Manifest: {outputs['manifest_path']}")
    if "target_discovery_report" in outputs:
        print(f"Report: {outputs['target_discovery_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify CLI help**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe scripts\run_target_discovery.py --help
```

Expected: command exits 0 and prints the new options, including `--output-dir`, `--run-id`, and `--skip-figures`.

- [ ] **Step 3: Commit Task 9**

Run:

```powershell
git add scripts\run_target_discovery.py
git commit -m "refactor: make target discovery script a thin cli"
```

---

### Task 10: Verification, Documentation, And Cleanup

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-05-03-target-discovery-pipeline-redesign.md`
- Verify generated runtime file: `reports/migration_notes.md` through smoke run or writer test.

- [ ] **Step 1: Run discovery tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\discovery -q
```

Expected: all discovery tests pass.

- [ ] **Step 2: Run existing targeted regression tests**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m pytest tests\test_step2_pipeline.py tests\test_step3_pipeline.py tests\test_perturbation.py -q
```

Expected: pass, unless blocked by environment/temp permissions. If blocked, record the exact permission error.

- [ ] **Step 3: Run compile check**

Run:

```powershell
E:\ProgramData\Anaconda3\envs\hypersca\python.exe -m compileall -q src scripts tests
```

Expected: exits 0.

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected: only unrelated pre-existing changes remain unstaged, plus any intentional files from this plan if not committed.

- [ ] **Step 5: Commit verification doc adjustment if any**

If implementation required changing the spec or adding hand-written docs, run:

```powershell
git add docs\superpowers\specs\2026-05-03-target-discovery-pipeline-redesign.md
git commit -m "docs: update target discovery migration notes"
```

Skip this commit if no docs changed.
