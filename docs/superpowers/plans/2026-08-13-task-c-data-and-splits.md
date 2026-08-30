# Task C Data and Split Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 CausalBench 官方缓存建立可追溯的 K562/RPE1 统一输入，并生成不会泄漏封存干预结果的固定数据划分。

**Architecture:** 数据下载仍由固定版本的 CausalBench 完成；HyperSCA 只负责核验、记录来源、选择共同基因和物化被允许的数据范围。公开划分记录只暴露学习与参数调节范围，封存关系和细胞只写入独立的评估目录；后续方法运行只能接收显式文件路径。

**Tech Stack:** Python 3.10、NumPy、pandas、JSON、pytest、conda、CausalBench commit `1a2143cffdc85f835b41ce8d52034be1bf903e71`

---

## 执行位置与依赖

在隔离工作区执行：

```bash
cd /home/a/.config/superpowers/worktrees/HyperSCA/real-data-readiness-design
```

本计划是四份计划中的第 1 份。完成后再执行：

1. `2026-08-13-hypersca-c-model.md`
2. `2026-08-13-task-c-comparison-runtime.md`
3. `2026-08-13-task-c-rehearsal-and-aggregation.md`

## 文件结构

- Create: `envs/task_c/causalbench.yml` — 固定官方数据导出环境。
- Create: `src/evaluation/task_c_data.py` — 数据核验、来源记录、共同划分和物化。
- Create: `scripts/export_causalbench_data.py` — 在官方环境中生成两个 NPZ 缓存、汇总生物关系和 ChIP 有向关系。
- Create: `scripts/prepare_task_c_data.py` — 核验缓存并生成五个固定划分。
- Create: `tests/test_task_c_data.py` — 数据与泄漏防护单元测试。
- Create: `tests/test_task_c_data_cli.py` — 小型端到端命令测试。
- Create: `docs/research/task_c_data_readiness_v1.md` — 面向生物医学读者的数据说明。
- Modify: `.gitignore` — 明确忽略任务 C 大型本地缓存。

### Task 1: 固定官方数据来源和导出环境

**Files:**
- Create: `envs/task_c/causalbench.yml`
- Create: `scripts/export_causalbench_data.py`
- Test: `tests/test_task_c_data_cli.py`

- [ ] **Step 1: 写导出命令的失败测试**

在 `tests/test_task_c_data_cli.py` 写入：

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_export_command_reports_pinned_source_without_downloading(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_causalbench_data.py"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--describe-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["repository"] == "https://github.com/causalbench/causalbench.git"
    assert payload["commit"] == "1a2143cffdc85f835b41ce8d52034be1bf903e71"
    assert payload["datasets"] == ["dataset_k562.npz", "dataset_rpe1.npz"]
    assert payload["references"] == [
        "reference_k562_pooled.csv",
        "reference_k562_chipseq.csv",
        "reference_rpe1_pooled.csv",
        "reference_rpe1_chipseq.csv",
    ]
```

- [ ] **Step 2: 运行测试并确认失败原因正确**

Run:

```bash
pytest tests/test_task_c_data_cli.py::test_export_command_reports_pinned_source_without_downloading -q -p no:cacheprovider
```

Expected: FAIL，因为 `scripts/export_causalbench_data.py` 尚不存在。

- [ ] **Step 3: 写固定环境文件**

创建 `envs/task_c/causalbench.yml`：

```yaml
name: hypersca-task-c-causalbench
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71
```

- [ ] **Step 4: 实现只负责官方导出的薄命令**

创建 `scripts/export_causalbench_data.py`：

```python
"""使用固定版本 CausalBench 生成 K562 和 RPE1 官方缓存。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPOSITORY = "https://github.com/causalbench/causalbench.git"
COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载并整理任务 C 的 K562/RPE1 官方单细胞干预数据。"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    description = {
        "schema_version": "1.0",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "datasets": ["dataset_k562.npz", "dataset_rpe1.npz"],
        "references": [
            "reference_k562_pooled.csv",
            "reference_k562_chipseq.csv",
            "reference_rpe1_pooled.csv",
            "reference_rpe1_chipseq.csv",
        ],
        "data_dir": str(args.data_dir),
        "filter": bool(args.filter),
    }
    if args.describe_only:
        print(json.dumps(description, ensure_ascii=False, sort_keys=True))
        return 0

    try:
        from causalscbench.data_access.create_dataset import CreateDataset
        from causalscbench.data_access.create_evaluation_datasets import (
            CreateEvaluationDatasets,
        )
    except ImportError as exc:
        raise SystemExit(
            "无法导入固定版本 CausalBench；请先创建 envs/task_c/causalbench.yml 中的环境。"
        ) from exc

    args.data_dir.mkdir(parents=True, exist_ok=True)
    k562_path, rpe1_path = CreateDataset(
        str(args.data_dir), bool(args.filter)
    ).load()
    description["paths"] = {
        "k562": str(Path(k562_path).resolve()),
        "rpe1": str(Path(rpe1_path).resolve()),
    }
    reference_paths = {}
    for context_id, dataset_name in (
        ("k562", "weissmann_k562"),
        ("rpe1", "weissmann_rpe1"),
    ):
        corum, ligand_receptor, string_network, string_physical, chipseq = (
            CreateEvaluationDatasets(str(args.data_dir), dataset_name).load()
        )
        pooled = set().union(
            corum, ligand_receptor, string_network, string_physical, chipseq
        )
        pooled_directed = pooled | {(target, source) for source, target in pooled}
        for reference_id, edges in (("pooled", pooled_directed), ("chipseq", chipseq)):
            path = args.data_dir / f"reference_{context_id}_{reference_id}.csv"
            pd.DataFrame(sorted(edges), columns=["source", "target"]).to_csv(
                path, index=False
            )
            reference_paths[f"{context_id}_{reference_id}"] = str(path.resolve())
    description["reference_paths"] = reference_paths
    description["reference_scope"] = {
        "pooled": "CausalBench pooled biological evidence expanded in both directions",
        "chipseq": "CausalBench bundled directed ChIP evidence; the RPE1 branch uses the bundled HepG2 file in this pinned commit",
    }
    description["reference_sources"] = {
        "corum": "https://mips.helmholtz-muenchen.de/corum/",
        "ligand_receptor": "https://github.com/LewisLabUCSD/Ligand-Receptor-Pairs/tree/ba44c3c4b4a3e501667309dd9ce7208501aeb961",
        "string_db": "https://string-db.org/cgi/download.pl",
        "chip_atlas": "https://dbarchive.biosciencedbc.jp/en/chip-atlas/lic.html",
    }
    description["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.data_dir / "export_manifest.json").write_text(
        json.dumps(description, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(description, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试并提交**

Run:

```bash
pytest tests/test_task_c_data_cli.py::test_export_command_reports_pinned_source_without_downloading -q -p no:cacheprovider
```

Expected: PASS。

Commit:

```bash
git add envs/task_c/causalbench.yml scripts/export_causalbench_data.py tests/test_task_c_data_cli.py
git commit -m "feat: pin CausalBench data exporter"
```

### Task 2: 建立严格的数据核验与来源记录

**Files:**
- Create: `src/evaluation/task_c_data.py`
- Test: `tests/test_task_c_data.py`

- [ ] **Step 1: 写数据核验和来源指纹测试**

创建 `tests/test_task_c_data.py`，先加入：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.task_c_data import (
    TaskCDataError,
    build_task_c_reference_provenance,
    build_task_c_provenance,
    load_task_c_dataset,
)


def write_dataset(path: Path, genes: list[str], labels: list[str]) -> None:
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    np.savez(
        path,
        expression_matrix=expression,
        interventions=np.asarray(labels),
        var_names=np.asarray(genes),
    )


def test_load_dataset_validates_shape_labels_and_gene_names(tmp_path: Path) -> None:
    path = tmp_path / "k562.npz"
    write_dataset(path, ["A", "B", "C"], ["non-targeting", "A", "B"])
    dataset = load_task_c_dataset(path, context_id="k562")
    assert dataset.expression.shape == (3, 3)
    assert dataset.gene_names == ("A", "B", "C")
    assert dataset.interventions.tolist() == ["non-targeting", "A", "B"]


def test_duplicate_gene_names_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    write_dataset(path, ["A", "A"], ["non-targeting", "A"])
    with pytest.raises(TaskCDataError, match="gene names must be unique"):
        load_task_c_dataset(path, context_id="k562")


def test_provenance_binds_file_and_official_commit(tmp_path: Path) -> None:
    path = tmp_path / "k562.npz"
    write_dataset(path, ["A", "B"], ["non-targeting", "A"])
    dataset = load_task_c_dataset(path, context_id="k562")
    record = build_task_c_provenance(dataset)
    assert record["context_id"] == "k562"
    assert record["causalbench_commit"] == (
        "1a2143cffdc85f835b41ce8d52034be1bf903e71"
    )
    assert record["input_sha256"].startswith("sha256:")
    json.dumps(record, allow_nan=False)


def test_reference_provenance_distinguishes_pooled_and_directed_evidence(
    tmp_path: Path,
) -> None:
    pooled = tmp_path / "reference_k562_pooled.csv"
    chipseq = tmp_path / "reference_k562_chipseq.csv"
    pooled.write_text("source,target\nA,B\nB,A\n", encoding="utf-8")
    chipseq.write_text("source,target\nA,B\n", encoding="utf-8")
    record = build_task_c_reference_provenance(
        context_id="k562", pooled_path=pooled, chipseq_path=chipseq
    )
    assert record["primary_reference_id"] == "causalbench_pooled_biological_v1"
    assert record["directed_reference_id"] == "causalbench_chipseq_v1"
    assert record["pooled_sha256"].startswith("sha256:")
    assert record["chipseq_sha256"].startswith("sha256:")
```

- [ ] **Step 2: 运行测试并确认缺少模块**

Run:

```bash
pytest tests/test_task_c_data.py -q -p no:cacheprovider
```

Expected: collection FAIL，提示 `src.evaluation.task_c_data` 不存在。

- [ ] **Step 3: 实现数据对象、核验和来源记录**

创建 `src/evaluation/task_c_data.py`，先写入：

```python
"""任务 C 官方数据核验、固定划分和封存数据物化。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CAUSALBENCH_REPOSITORY = "https://github.com/causalbench/causalbench.git"
CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
CONTROL_LABEL = "non-targeting"


class TaskCDataError(ValueError):
    """任务 C 数据或划分不满足固定规则。"""


@dataclass(frozen=True)
class TaskCDataset:
    expression: np.ndarray
    interventions: np.ndarray
    gene_names: tuple[str, ...]
    context_id: str
    source_path: Path
    source_sha256: str


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def load_task_c_dataset(path: str | Path, *, context_id: str) -> TaskCDataset:
    source = Path(path)
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("context_id must be k562 or rpe1")
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {"expression_matrix", "interventions", "var_names"}
            missing = required - set(archive.files)
            if missing:
                raise TaskCDataError(f"dataset is missing arrays: {sorted(missing)}")
            expression = np.asarray(archive["expression_matrix"])
            interventions = np.asarray(archive["interventions"], dtype=str)
            gene_names = tuple(str(value) for value in archive["var_names"].tolist())
    except (OSError, ValueError) as exc:
        raise TaskCDataError(f"could not load {source}: {exc}") from exc
    if expression.ndim != 2 or expression.shape[0] != len(interventions):
        raise TaskCDataError("expression rows must match intervention labels")
    if expression.shape[1] != len(gene_names):
        raise TaskCDataError("expression columns must match gene names")
    if len(gene_names) != len(set(gene_names)):
        raise TaskCDataError("gene names must be unique")
    if any(not gene for gene in gene_names):
        raise TaskCDataError("gene names must be non-empty")
    if not np.isfinite(expression).all():
        raise TaskCDataError("expression values must be finite")
    if CONTROL_LABEL not in set(interventions.tolist()):
        raise TaskCDataError("non-targeting control cells are required")
    return TaskCDataset(
        expression=expression,
        interventions=interventions,
        gene_names=gene_names,
        context_id=context_id,
        source_path=source.resolve(),
        source_sha256=sha256_path(source),
    )


def build_task_c_provenance(dataset: TaskCDataset) -> dict[str, Any]:
    labels, counts = np.unique(dataset.interventions, return_counts=True)
    return {
        "schema_version": "1.0",
        "context_id": dataset.context_id,
        "causalbench_repository": CAUSALBENCH_REPOSITORY,
        "causalbench_commit": CAUSALBENCH_COMMIT,
        "dataset_source_url": (
            "https://plus.figshare.com/ndownloader/files/35773219"
            if dataset.context_id == "k562"
            else "https://plus.figshare.com/ndownloader/files/35775606"
        ),
        "source_path": str(dataset.source_path),
        "input_sha256": dataset.source_sha256,
        "n_cells": int(dataset.expression.shape[0]),
        "n_genes": int(dataset.expression.shape[1]),
        "control_label": CONTROL_LABEL,
        "intervention_counts": {
            str(label): int(count) for label, count in zip(labels, counts)
        },
        "licenses": {
            "causalbench_code": "Apache-2.0",
            "replogle_perturb_seq": "CC-BY-4.0",
        },
    }


def build_task_c_reference_provenance(
    *,
    context_id: str,
    pooled_path: str | Path,
    chipseq_path: str | Path,
) -> dict[str, Any]:
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("reference context_id must be k562 or rpe1")
    return {
        "schema_version": "1.0",
        "context_id": context_id,
        "causalbench_repository": CAUSALBENCH_REPOSITORY,
        "causalbench_commit": CAUSALBENCH_COMMIT,
        "primary_reference_id": "causalbench_pooled_biological_v1",
        "primary_reference_scope": "pooled biological evidence expanded in both directions",
        "directed_reference_id": "causalbench_chipseq_v1",
        "directed_reference_scope": (
            "K562 ChIP file"
            if context_id == "k562"
            else "HepG2 ChIP file bundled by the pinned CausalBench RPE1 branch"
        ),
        "pooled_sha256": sha256_path(pooled_path),
        "chipseq_sha256": sha256_path(chipseq_path),
        "licenses": {
            "causalbench_code": "Apache-2.0",
            "chip_atlas_adapted_data": "CC-BY-SA-4.0",
            "corum": "CC-BY-NC",
            "string_db": "CC-BY-4.0",
            "ligand_receptor_resource_as_declared_by_causalbench": "GPL-3.0",
        },
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: 运行测试并提交**

Run:

```bash
pytest tests/test_task_c_data.py -q -p no:cacheprovider
```

Expected: 4 tests PASS。

Commit:

```bash
git add src/evaluation/task_c_data.py tests/test_task_c_data.py
git commit -m "feat: validate Task C benchmark data"
```

### Task 3: 生成两个细胞系共用的固定来源基因划分

**Files:**
- Modify: `src/evaluation/task_c_data.py`
- Modify: `tests/test_task_c_data.py`

- [ ] **Step 1: 写共同划分、控制细胞划分和泄漏测试**

在 `tests/test_task_c_data.py` 追加：

```python
from src.evaluation.task_c_data import (
    build_shared_task_c_split,
    validate_task_c_split,
)


def dataset_for_split(path: Path, context: str) -> object:
    genes = ["A", "B", "C", "D", "E", "Z"]
    labels = ["non-targeting"] * 10
    for gene in genes[:5]:
        labels.extend([gene] * 5)
    write_dataset(path, genes, labels)
    return load_task_c_dataset(path, context_id=context)


def test_shared_split_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    k562 = dataset_for_split(tmp_path / "k562.npz", "k562")
    rpe1 = dataset_for_split(tmp_path / "rpe1.npz", "rpe1")
    first = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    second = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    assert first == second
    assert len(first.train_sources) == 3
    assert len(first.tune_sources) == 1
    assert len(first.holdout_sources) == 1
    validate_task_c_split(first, k562, rpe1)


def test_split_rejects_source_overlap(tmp_path: Path) -> None:
    from dataclasses import replace

    k562 = dataset_for_split(tmp_path / "k562.npz", "k562")
    rpe1 = dataset_for_split(tmp_path / "rpe1.npz", "rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    corrupted = replace(
        split,
        tune_sources=(split.train_sources[0],),
    )
    with pytest.raises(TaskCDataError, match="source partitions overlap"):
        validate_task_c_split(corrupted, k562, rpe1)


def test_holdout_sources_are_identical_across_contexts(tmp_path: Path) -> None:
    k562 = dataset_for_split(tmp_path / "k562.npz", "k562")
    rpe1 = dataset_for_split(tmp_path / "rpe1.npz", "rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=23, min_cells=5)
    for dataset in (k562, rpe1):
        observed = set(dataset.interventions.tolist())
        assert set(split.holdout_sources) <= observed
```

- [ ] **Step 2: 运行新增测试并确认导入失败**

Run:

```bash
pytest tests/test_task_c_data.py -q -p no:cacheprovider
```

Expected: collection FAIL，因为划分函数尚不存在。

- [ ] **Step 3: 实现不可变划分对象与验证**

在 `src/evaluation/task_c_data.py` 追加：

```python
@dataclass(frozen=True)
class TaskCSplit:
    schema_version: str
    split_id: str
    seed: int
    train_sources: tuple[str, ...]
    tune_sources: tuple[str, ...]
    holdout_sources: tuple[str, ...]
    control_indices: Mapping[str, Mapping[str, tuple[int, ...]]]
    min_cells_per_intervention: int


def _eligible_sources(dataset: TaskCDataset, min_cells: int) -> set[str]:
    labels, counts = np.unique(dataset.interventions, return_counts=True)
    measured = set(dataset.gene_names)
    return {
        str(label)
        for label, count in zip(labels, counts)
        if label != CONTROL_LABEL and label in measured and int(count) >= min_cells
    }


def _control_partitions(dataset: TaskCDataset, seed: int) -> dict[str, tuple[int, ...]]:
    indices = np.flatnonzero(dataset.interventions == CONTROL_LABEL)
    if len(indices) < 5:
        raise TaskCDataError("at least five control cells are required")
    shuffled = np.random.default_rng(seed).permutation(indices)
    n_train = max(1, int(np.floor(0.6 * len(shuffled))))
    n_tune = max(1, int(np.floor(0.2 * len(shuffled))))
    return {
        "train": tuple(int(value) for value in np.sort(shuffled[:n_train])),
        "tune": tuple(
            int(value) for value in np.sort(shuffled[n_train : n_train + n_tune])
        ),
        "holdout": tuple(
            int(value) for value in np.sort(shuffled[n_train + n_tune :])
        ),
    }


def build_shared_task_c_split(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    *,
    seed: int,
    min_cells: int = 5,
) -> TaskCSplit:
    if seed not in {11, 23, 47, 71, 97}:
        raise TaskCDataError("seed must be one of 11, 23, 47, 71, 97")
    common = sorted(_eligible_sources(k562, min_cells) & _eligible_sources(rpe1, min_cells))
    if len(common) < 5:
        raise TaskCDataError("at least five shared intervention sources are required")
    shuffled = np.random.default_rng(seed).permutation(np.asarray(common, dtype=str))
    n_train = max(1, int(np.floor(0.6 * len(common))))
    n_tune = max(1, int(np.floor(0.2 * len(common))))
    split = TaskCSplit(
        schema_version="1.0",
        split_id=f"C-context-intervention-holdout-v1-seed-{seed}",
        seed=seed,
        train_sources=tuple(sorted(shuffled[:n_train].tolist())),
        tune_sources=tuple(sorted(shuffled[n_train : n_train + n_tune].tolist())),
        holdout_sources=tuple(sorted(shuffled[n_train + n_tune :].tolist())),
        control_indices={
            "k562": _control_partitions(k562, seed),
            "rpe1": _control_partitions(rpe1, seed),
        },
        min_cells_per_intervention=min_cells,
    )
    validate_task_c_split(split, k562, rpe1)
    return split


def validate_task_c_split(
    split: TaskCSplit,
    k562: TaskCDataset,
    rpe1: TaskCDataset,
) -> None:
    source_sets = [
        set(split.train_sources),
        set(split.tune_sources),
        set(split.holdout_sources),
    ]
    if any(source_sets[i] & source_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise TaskCDataError("source partitions overlap")
    shared = _eligible_sources(k562, split.min_cells_per_intervention) & _eligible_sources(
        rpe1, split.min_cells_per_intervention
    )
    if set.union(*source_sets) != shared:
        raise TaskCDataError("source partitions do not cover the shared eligible sources")
    for dataset in (k562, rpe1):
        parts = split.control_indices[dataset.context_id]
        control_sets = [set(parts[name]) for name in ("train", "tune", "holdout")]
        if any(control_sets[i] & control_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise TaskCDataError("control partitions overlap")
        expected = set(np.flatnonzero(dataset.interventions == CONTROL_LABEL).tolist())
        if set.union(*control_sets) != expected:
            raise TaskCDataError("control partitions do not cover all control cells")
```

- [ ] **Step 4: 运行测试并提交**

Run:

```bash
pytest tests/test_task_c_data.py -q -p no:cacheprovider
```

Expected: 7 tests PASS。

Commit:

```bash
git add src/evaluation/task_c_data.py tests/test_task_c_data.py
git commit -m "feat: freeze shared Task C intervention splits"
```

### Task 4: 物化允许读取的数据文件和私有封存记录

**Files:**
- Modify: `src/evaluation/task_c_data.py`
- Create: `scripts/prepare_task_c_data.py`
- Modify: `tests/test_task_c_data.py`
- Modify: `tests/test_task_c_data_cli.py`

- [ ] **Step 1: 写物化文件和公开记录不泄漏的测试**

在 `tests/test_task_c_data.py` 追加：

```python
from src.evaluation.task_c_data import materialize_task_c_split


def test_materialized_training_files_exclude_holdout_sources(tmp_path: Path) -> None:
    k562 = dataset_for_split(tmp_path / "k562.npz", "k562")
    rpe1 = dataset_for_split(tmp_path / "rpe1.npz", "rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    train = load_task_c_dataset(result["within"]["k562"]["train"], context_id="k562")
    assert not (set(train.interventions.tolist()) & set(split.holdout_sources))
    public = json.loads(Path(result["public_manifest"]).read_text(encoding="utf-8"))
    assert "holdout_sources" not in public
    private = json.loads(Path(result["private_manifest"]).read_text(encoding="utf-8"))
    assert private["holdout_sources"] == list(split.holdout_sources)


def test_cross_context_adaptation_contains_only_target_controls(tmp_path: Path) -> None:
    k562 = dataset_for_split(tmp_path / "k562.npz", "k562")
    rpe1 = dataset_for_split(tmp_path / "rpe1.npz", "rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    adapt = load_task_c_dataset(
        result["cross"]["k562_to_rpe1"]["target_adapt_refit"], context_id="rpe1"
    )
    assert set(adapt.interventions.tolist()) == {"non-targeting"}
    source_train = load_task_c_dataset(
        result["cross"]["k562_to_rpe1"]["source_train"], context_id="k562"
    )
    source_tune = load_task_c_dataset(
        result["cross"]["k562_to_rpe1"]["source_tune"], context_id="k562"
    )
    assert not (set(source_train.interventions) & set(split.tune_sources))
    assert not (set(source_tune.interventions) & set(split.train_sources))
```

在 `tests/test_task_c_data_cli.py` 追加一个使用本文件 `write_dataset` 等价小夹具的命令测试；不要从另一个测试模块导入辅助函数：

```python
import numpy as np


def _write_cli_dataset(path: Path) -> None:
    genes = np.asarray(["A", "B", "C", "D", "E", "Z"])
    labels = ["non-targeting"] * 10
    for gene in genes[:5]:
        labels.extend([str(gene)] * 5)
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    np.savez(
        path,
        expression_matrix=expression,
        interventions=np.asarray(labels),
        var_names=genes,
    )


def test_prepare_cli_writes_five_reproducible_splits(tmp_path: Path) -> None:
    k562 = tmp_path / "k562.npz"
    rpe1 = tmp_path / "rpe1.npz"
    _write_cli_dataset(k562)
    _write_cli_dataset(rpe1)
    references = {}
    for context in ("k562", "rpe1"):
        for reference_id in ("pooled", "chipseq"):
            path = tmp_path / f"reference_{context}_{reference_id}.csv"
            path.write_text("source,target\nA,B\n", encoding="utf-8")
            references[f"{context}_{reference_id}"] = path
    output = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_task_c_data.py"),
            "--k562-npz",
            str(k562),
            "--rpe1-npz",
            str(rpe1),
            "--k562-pooled-reference",
            str(references["k562_pooled"]),
            "--k562-chipseq-reference",
            str(references["k562_chipseq"]),
            "--rpe1-pooled-reference",
            str(references["rpe1_pooled"]),
            "--rpe1-chipseq-reference",
            str(references["rpe1_chipseq"]),
            "--output-dir",
            str(output),
            "--min-cells-per-intervention",
            "5",
        ],
        cwd=ROOT,
        check=True,
    )
    for seed in (11, 23, 47, 71, 97):
        assert (output / "splits" / f"seed_{seed}" / "public_manifest.json").exists()
```

- [ ] **Step 2: 运行测试并确认物化函数和命令缺失**

Run:

```bash
pytest tests/test_task_c_data.py tests/test_task_c_data_cli.py -q -p no:cacheprovider
```

Expected: FAIL，缺少 `materialize_task_c_split` 和 `prepare_task_c_data.py`。

- [ ] **Step 3: 实现分区 NPZ 写入和指纹记录**

在 `src/evaluation/task_c_data.py` 追加以下职责完整的辅助函数；保持字典键与测试一致：

```python
def _write_dataset_subset(
    dataset: TaskCDataset,
    indices: np.ndarray,
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        expression_matrix=dataset.expression[indices],
        interventions=dataset.interventions[indices],
        var_names=np.asarray(dataset.gene_names),
    )
    return str(path.resolve())


def _indices_for_sources(
    dataset: TaskCDataset,
    sources: Sequence[str],
    control_indices: Sequence[int],
) -> np.ndarray:
    source_mask = np.isin(dataset.interventions, np.asarray(tuple(sources), dtype=str))
    selected = np.flatnonzero(source_mask).tolist() + [int(value) for value in control_indices]
    return np.asarray(sorted(set(selected)), dtype=int)


def _private_split_payload(split: TaskCSplit) -> dict[str, Any]:
    return {
        "schema_version": split.schema_version,
        "split_id": split.split_id,
        "seed": split.seed,
        "train_sources": list(split.train_sources),
        "tune_sources": list(split.tune_sources),
        "holdout_sources": list(split.holdout_sources),
        "control_indices": {
            context: {name: list(values) for name, values in parts.items()}
            for context, parts in split.control_indices.items()
        },
    }


def materialize_task_c_split(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    output_dir: str | Path,
) -> dict[str, Any]:
    validate_task_c_split(split, k562, rpe1)
    root = Path(output_dir)
    datasets = {"k562": k562, "rpe1": rpe1}
    within: dict[str, dict[str, str]] = {}
    for context, dataset in datasets.items():
        parts = split.control_indices[context]
        within[context] = {
            "train": _write_dataset_subset(
                dataset,
                _indices_for_sources(dataset, split.train_sources, parts["train"]),
                root / "within" / context / "train.npz",
            ),
            "tune": _write_dataset_subset(
                dataset,
                _indices_for_sources(dataset, split.tune_sources, parts["tune"]),
                root / "within" / context / "tune.npz",
            ),
            "refit": _write_dataset_subset(
                dataset,
                _indices_for_sources(
                    dataset,
                    split.train_sources + split.tune_sources,
                    parts["train"] + parts["tune"],
                ),
                root / "within" / context / "refit.npz",
            ),
            "holdout": _write_dataset_subset(
                dataset,
                _indices_for_sources(dataset, split.holdout_sources, parts["holdout"]),
                root / "private" / "within" / context / "holdout.npz",
            ),
        }

    cross: dict[str, dict[str, str]] = {}
    for source_name, target_name in (("k562", "rpe1"), ("rpe1", "k562")):
        source = datasets[source_name]
        target = datasets[target_name]
        source_controls = split.control_indices[source_name]
        target_controls = split.control_indices[target_name]
        direction = f"{source_name}_to_{target_name}"
        all_target_sources = split.train_sources + split.tune_sources + split.holdout_sources
        cross[direction] = {
            "source_train": _write_dataset_subset(
                source,
                _indices_for_sources(source, split.train_sources, source_controls["train"]),
                root / "cross" / direction / "source_train.npz",
            ),
            "source_tune": _write_dataset_subset(
                source,
                _indices_for_sources(source, split.tune_sources, source_controls["tune"]),
                root / "cross" / direction / "source_tune.npz",
            ),
            "source_refit": _write_dataset_subset(
                source,
                _indices_for_sources(
                    source,
                    split.train_sources + split.tune_sources,
                    source_controls["train"] + source_controls["tune"],
                ),
                root / "cross" / direction / "source_refit.npz",
            ),
            "target_adapt_train": _write_dataset_subset(
                target,
                np.asarray(target_controls["train"], dtype=int),
                root / "cross" / direction / "target_adapt_train.npz",
            ),
            "target_adapt_tune": _write_dataset_subset(
                target,
                np.asarray(target_controls["tune"], dtype=int),
                root / "cross" / direction / "target_adapt_tune.npz",
            ),
            "target_adapt_refit": _write_dataset_subset(
                target,
                np.asarray(
                    target_controls["train"] + target_controls["tune"], dtype=int
                ),
                root / "cross" / direction / "target_adapt_refit.npz",
            ),
            "target_holdout": _write_dataset_subset(
                target,
                _indices_for_sources(target, all_target_sources, target_controls["holdout"]),
                root / "private" / "cross" / direction / "target_holdout.npz",
            ),
        }

    private_path = root / "private" / "private_manifest.json"
    public_path = root / "public_manifest.json"
    write_json(private_path, _private_split_payload(split))
    write_json(
        public_path,
        {
            "schema_version": split.schema_version,
            "split_id": split.split_id,
            "seed": split.seed,
            "train_sources": list(split.train_sources),
            "tune_sources": list(split.tune_sources),
            "holdout_source_count": len(split.holdout_sources),
            "input_sha256": {
                "k562": k562.source_sha256,
                "rpe1": rpe1.source_sha256,
            },
        },
    )
    return {
        "within": within,
        "cross": cross,
        "public_manifest": str(public_path.resolve()),
        "private_manifest": str(private_path.resolve()),
    }
```

- [ ] **Step 4: 实现五随机种子准备命令**

创建 `scripts/prepare_task_c_data.py`。命令只解析参数、调用上述函数并输出摘要；完整主体使用以下代码：

```python
"""核验 K562/RPE1，并生成任务 C 的五个固定数据划分。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_data import (
    TaskCDataError,
    build_shared_task_c_split,
    build_task_c_reference_provenance,
    build_task_c_provenance,
    load_task_c_dataset,
    materialize_task_c_split,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="核验任务 C 官方数据，并隔离模型建立与最终检验所用细胞。"
    )
    parser.add_argument("--k562-npz", type=Path, required=True)
    parser.add_argument("--rpe1-npz", type=Path, required=True)
    parser.add_argument("--k562-pooled-reference", type=Path, required=True)
    parser.add_argument("--k562-chipseq-reference", type=Path, required=True)
    parser.add_argument("--rpe1-pooled-reference", type=Path, required=True)
    parser.add_argument("--rpe1-chipseq-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cells-per-intervention", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        k562 = load_task_c_dataset(args.k562_npz, context_id="k562")
        rpe1 = load_task_c_dataset(args.rpe1_npz, context_id="rpe1")
        provenance_dir = args.output_dir / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        write_json(provenance_dir / "k562.json", build_task_c_provenance(k562))
        write_json(provenance_dir / "rpe1.json", build_task_c_provenance(rpe1))
        for context_id in ("k562", "rpe1"):
            write_json(
                provenance_dir / f"{context_id}_references.json",
                build_task_c_reference_provenance(
                    context_id=context_id,
                    pooled_path=getattr(args, f"{context_id}_pooled_reference"),
                    chipseq_path=getattr(args, f"{context_id}_chipseq_reference"),
                ),
            )
        summaries = []
        for seed in (11, 23, 47, 71, 97):
            split = build_shared_task_c_split(
                k562,
                rpe1,
                seed=seed,
                min_cells=args.min_cells_per_intervention,
            )
            result = materialize_task_c_split(
                k562,
                rpe1,
                split,
                args.output_dir / "splits" / f"seed_{seed}",
            )
            summaries.append(
                {
                    "seed": seed,
                    "split_id": split.split_id,
                    "public_manifest": result["public_manifest"],
                }
            )
    except TaskCDataError as exc:
        parser.error(f"无法准备任务 C 数据：{exc}")
    print(json.dumps({"status": "prepared", "splits": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行数据测试并提交**

Run:

```bash
pytest tests/test_task_c_data.py tests/test_task_c_data_cli.py -q -p no:cacheprovider
```

Expected: all tests PASS。

Commit:

```bash
git add src/evaluation/task_c_data.py scripts/prepare_task_c_data.py tests/test_task_c_data.py tests/test_task_c_data_cli.py
git commit -m "feat: materialize leak-resistant Task C splits"
```

### Task 5: 文档、忽略规则与真实数据准备命令

**Files:**
- Modify: `.gitignore`
- Create: `docs/research/task_c_data_readiness_v1.md`

- [ ] **Step 1: 加入大型缓存忽略规则**

在 `.gitignore` 项目数据部分加入：

```gitignore
# 任务 C 官方缓存与封存划分
/data/task_c/
/results/benchmarks/task_c/
```

- [ ] **Step 2: 写面向研究人员的数据说明**

创建 `docs/research/task_c_data_readiness_v1.md`，必须包含以下完整章节和命令：

```markdown
# 任务 C 数据准备与封存规则

任务 C 使用 K562 和 RPE1 单细胞基因干预数据，检验方法能否从部分已见干预推广到未见干预。数据来源于固定版本 CausalBench；HyperSCA 不重新定义原始实验标签。

## 数据和许可

- CausalBench 代码：Apache-2.0。
- Replogle Perturb-seq 数据：CausalBench 记录为 CC-BY-4.0。
- 固定代码提交：`1a2143cffdc85f835b41ce8d52034be1bf903e71`。

## 数据隔离

同一个被干预基因在 K562 和 RPE1 使用相同的学习、参数调节和最终检验归属。模型运行只接收公开记录列出的学习文件；最终检验文件位于 `private/`，只由评分步骤读取。

## 本机准备

```bash
conda env create -f envs/task_c/causalbench.yml
TASK_C_DATA_ROOT=/home/a/Data/HyperSCA_external/task_c
mkdir -p "$TASK_C_DATA_ROOT/official"
conda run -n hypersca-task-c-causalbench python scripts/export_causalbench_data.py \
  --data-dir "$TASK_C_DATA_ROOT/official"
python scripts/prepare_task_c_data.py \
  --k562-npz "$TASK_C_DATA_ROOT/official/dataset_k562.npz" \
  --rpe1-npz "$TASK_C_DATA_ROOT/official/dataset_rpe1.npz" \
  --k562-pooled-reference "$TASK_C_DATA_ROOT/official/reference_k562_pooled.csv" \
  --k562-chipseq-reference "$TASK_C_DATA_ROOT/official/reference_k562_chipseq.csv" \
  --rpe1-pooled-reference "$TASK_C_DATA_ROOT/official/reference_rpe1_pooled.csv" \
  --rpe1-chipseq-reference "$TASK_C_DATA_ROOT/official/reference_rpe1_chipseq.csv" \
  --output-dir "$TASK_C_DATA_ROOT/prepared"
```

导出命令同时保存 CausalBench 的汇总生物关系和 ChIP 有向关系。汇总关系用于主 AP，ChIP 关系只用于方向性补充；固定提交的 RPE1 分支实际使用随包提供的 HepG2 ChIP 文件，报告必须明确这一局限。

这些文件只用于研究评测。封存检验通过前，不得据此声称因果关系已被实验确认。
```

- [ ] **Step 3: 运行文档与回归检查**

Run:

```bash
python scripts/check_plain_language.py
pytest tests/test_task_c_data.py tests/test_task_c_data_cli.py tests/test_task_c_benchmark.py tests/test_benchmark_contract.py -q -p no:cacheprovider
git diff --check
```

Expected: 通俗术语检查通过；测试无失败；`git diff --check` 无输出。

- [ ] **Step 4: 提交并记录真实下载前状态**

Commit:

```bash
git add .gitignore docs/research/task_c_data_readiness_v1.md
git commit -m "docs: explain Task C data isolation"
```

Run:

```bash
git status --short
```

Expected: 无未提交文件。真实数据下载留到第 4 份计划统一执行，以便同时记录资源消耗和外部方法状态。
