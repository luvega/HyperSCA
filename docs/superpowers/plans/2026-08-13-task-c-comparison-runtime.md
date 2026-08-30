# Task C Comparison Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用隔离环境运行任务 C 的公开比较方法，把所有输出转换为相同的完整有向关系表，并准确记录完成、超时、资源不足、无效输出和官方资产不可用状态。

**Architecture:** 一个机器可读的方法登记表固定来源、提交版本、训练信息范围和运行命令。外部方法只在各自环境中读取被允许的 NPZ 文件并写原始关系；HyperSCA 主环境负责补齐共同评分范围、写运行记录和捕获失败，不复制或无记录修改外部方法代码。

**Tech Stack:** Python 3.10、conda、subprocess、NumPy、pandas、JSON、pytest、CausalBench、LightGBM

---

## 执行位置与前置条件

```bash
cd /home/a/.config/superpowers/worktrees/HyperSCA/real-data-readiness-design
```

先完成数据划分和 HyperSCA-C 两份计划。本计划不打开 `private/` 下的最终检验文件。

## 文件结构

- Create: `configs/task_c_methods_v1.json` — 方法、来源、环境、输入权限和状态规则。
- Create: `envs/task_c/psgrn.yml` — PSGRN/Guanlab 独立环境。
- Create: `src/evaluation/task_c_method_registry.py` — 登记表读取与严格核验。
- Create: `src/evaluation/task_c_predictions.py` — 完整关系范围和分数标准化。
- Create: `src/evaluation/task_c_runtime.py` — 隔离子进程、超时和资源记录。
- Create: `src/evaluation/task_c_tuning.py` — 只用参数调节干预响应选择登记配置。
- Create: `scripts/task_c_workers/causalbench_worker.py` — 官方内置方法运行入口。
- Create: `scripts/task_c_workers/psgrn_worker.py` — 固定 PSGRN 提交运行入口。
- Create: `scripts/bootstrap_task_c_methods.py` — 获取官方代码并创建环境记录。
- Create: `scripts/run_task_c_method.py` — 运行一个方法并写标准结果。
- Create: `scripts/select_task_c_configuration.py` — 从不超过 20 次的调节结果中选择配置。
- Create: `tests/test_task_c_method_registry.py`
- Create: `tests/test_task_c_predictions.py`
- Create: `tests/test_task_c_runtime.py`
- Create: `tests/test_task_c_external_workers.py`
- Create: `docs/research/task_c_method_compatibility_v1.md`

### Task 1: 固定全面比较方法登记表

**Files:**
- Create: `configs/task_c_methods_v1.json`
- Create: `src/evaluation/task_c_method_registry.py`
- Create: `tests/test_task_c_method_registry.py`

- [ ] **Step 1: 写登记表覆盖和资产状态测试**

创建 `tests/test_task_c_method_registry.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistryError,
    load_task_c_method_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registry_covers_confirmed_comparison_methods() -> None:
    registry = load_task_c_method_registry(ROOT / "configs/task_c_methods_v1.json")
    expected = {
        "hypersca_c",
        "mean_difference",
        "random1000",
        "grnboost",
        "pc",
        "ges",
        "gies",
        "gsp",
        "igsp",
        "notears_linear",
        "dcdi_g",
        "dcdi_dsf",
        "dcdfg_linear",
        "dcdfg_mlp",
        "sortnregress",
        "guanlab_psgrn",
        "betterboost",
        "sparse_rc",
        "catran",
    }
    assert expected == set(registry.methods)


def test_publication_only_methods_cannot_claim_runnable_code() -> None:
    registry = load_task_c_method_registry(ROOT / "configs/task_c_methods_v1.json")
    for method_id in ("betterboost", "sparse_rc", "catran"):
        method = registry.methods[method_id]
        assert method.source_kind == "publication_only"
        assert method.command is None


def test_every_method_has_explicit_output_semantics() -> None:
    registry = load_task_c_method_registry(ROOT / "configs/task_c_methods_v1.json")
    assert all(
        method.output_semantics in {"official_return_order", "no_output"}
        for method in registry.methods.values()
    )
    assert registry.methods["guanlab_psgrn"].output_semantics == "official_return_order"
    assert registry.methods["catran"].output_semantics == "no_output"


def test_registry_rejects_interventional_access_for_observational_method(tmp_path: Path) -> None:
    payload = json.loads(
        (ROOT / "configs/task_c_methods_v1.json").read_text(encoding="utf-8")
    )
    payload["methods"]["pc"]["training_information"] = "partial_interventional"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskCMethodRegistryError, match="pc must remain observational"):
        load_task_c_method_registry(path)
```

- [ ] **Step 2: 运行测试并确认登记模块缺失**

Run:

```bash
pytest tests/test_task_c_method_registry.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 写方法登记表**

创建 `configs/task_c_methods_v1.json`。`methods` 必须使用下列记录；同类 CausalBench 记录均固定到同一个官方提交：

```json
{
  "schema_version": "1.0",
  "causalbench": {
    "repository": "https://github.com/causalbench/causalbench.git",
    "commit": "1a2143cffdc85f835b41ce8d52034be1bf903e71",
    "environment": "hypersca-task-c-causalbench"
  },
  "output_semantics": {
    "official_return_order": ["hypersca_c", "mean_difference", "random1000", "grnboost", "pc", "ges", "gies", "gsp", "igsp", "notears_linear", "dcdi_g", "dcdi_dsf", "dcdfg_linear", "dcdfg_mlp", "sortnregress", "guanlab_psgrn"],
    "no_output": ["betterboost", "sparse_rc", "catran"]
  },
  "methods": {
    "hypersca_c": {"role": "candidate", "source_kind": "local", "training_information": "partial_interventional", "command": "local_hypersca_c", "required_for_core_rehearsal": true},
    "mean_difference": {"role": "simple_baseline", "source_kind": "local", "training_information": "partial_interventional", "command": "local_mean_difference", "required_for_core_rehearsal": true},
    "random1000": {"role": "null_control", "source_kind": "causalbench", "training_information": "observational", "command": "random1000", "required_for_core_rehearsal": true},
    "grnboost": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "observational", "command": "grnboost", "required_for_core_rehearsal": true},
    "pc": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "observational", "command": "pc", "required_for_core_rehearsal": true},
    "ges": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "observational", "command": "ges", "required_for_core_rehearsal": false},
    "gies": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "partial_interventional", "command": "gies", "required_for_core_rehearsal": true},
    "gsp": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "observational", "command": "gsp", "required_for_core_rehearsal": false},
    "igsp": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "partial_interventional", "command": "igsp", "required_for_core_rehearsal": false},
    "notears_linear": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "observational", "command": "notears-lin-sparse", "required_for_core_rehearsal": true},
    "dcdi_g": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "partial_interventional", "command": "DCDI-G", "required_for_core_rehearsal": false},
    "dcdi_dsf": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "partial_interventional", "command": "DCDI-DSF", "required_for_core_rehearsal": false},
    "dcdfg_linear": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "partial_interventional", "command": "DCDFG-LIN", "required_for_core_rehearsal": false},
    "dcdfg_mlp": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "partial_interventional", "command": "DCDFG-MLP", "required_for_core_rehearsal": false},
    "sortnregress": {"role": "external_comparator", "source_kind": "causalbench", "training_information": "observational", "command": "sortnregress", "required_for_core_rehearsal": false},
    "guanlab_psgrn": {"role": "external_comparator", "source_kind": "git", "training_information": "partial_interventional", "command": "psgrn", "required_for_core_rehearsal": false, "repository": "https://github.com/GuanLab/PSGRN.git", "commit": "74aa640f7c472b23a69811f6795bb17678efd344", "environment": "hypersca-task-c-psgrn"},
    "betterboost": {"role": "external_comparator", "source_kind": "publication_only", "training_information": "partial_interventional", "command": null, "required_for_core_rehearsal": false, "publication": "https://openreview.net/forum?id=gpDOOAOmMe"},
    "sparse_rc": {"role": "external_comparator", "source_kind": "publication_only", "training_information": "partial_interventional", "command": null, "required_for_core_rehearsal": false, "publication": "https://openreview.net/forum?id=TOaPl9tXlmD"},
    "catran": {"role": "external_comparator", "source_kind": "publication_only", "training_information": "partial_interventional", "command": null, "required_for_core_rehearsal": false, "publication": "https://openreview.net/forum?id=Wf0QRYUkhwV"}
  }
}
```

- [ ] **Step 4: 实现严格登记类型**

创建 `src/evaluation/task_c_method_registry.py`：

```python
"""任务 C 比较方法、来源和输入权限登记。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class TaskCMethodRegistryError(ValueError):
    """方法登记表缺少来源、权限或运行信息。"""


@dataclass(frozen=True)
class TaskCMethodSpec:
    method_id: str
    role: str
    source_kind: str
    training_information: str
    command: str | None
    required_for_core_rehearsal: bool
    output_semantics: str
    repository: str | None = None
    commit: str | None = None
    environment: str | None = None
    publication: str | None = None


@dataclass(frozen=True)
class TaskCMethodRegistry:
    schema_version: str
    methods: Mapping[str, TaskCMethodSpec]
    causalbench: Mapping[str, str]


def load_task_c_method_registry(path: str | Path) -> TaskCMethodRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise TaskCMethodRegistryError("schema_version must be 1.0")
    semantic_groups = payload.get("output_semantics", {})
    method_semantics = {
        method_id: semantic
        for semantic, method_ids in semantic_groups.items()
        for method_id in method_ids
    }
    methods = {}
    for method_id, raw in payload.get("methods", {}).items():
        if method_id not in method_semantics:
            raise TaskCMethodRegistryError(f"missing output semantics for {method_id}")
        spec = TaskCMethodSpec(
            method_id=method_id,
            output_semantics=method_semantics[method_id],
            **raw,
        )
        if spec.role not in {
            "candidate",
            "simple_baseline",
            "external_comparator",
            "null_control",
        }:
            raise TaskCMethodRegistryError(f"unsupported role for {method_id}")
        if spec.source_kind not in {"local", "causalbench", "git", "publication_only"}:
            raise TaskCMethodRegistryError(f"unsupported source kind for {method_id}")
        if spec.training_information not in {"observational", "partial_interventional"}:
            raise TaskCMethodRegistryError(f"unsupported training information for {method_id}")
        if spec.source_kind == "publication_only" and spec.command is not None:
            raise TaskCMethodRegistryError(
                f"publication-only method {method_id} cannot declare a command"
            )
        if spec.source_kind == "git" and (not spec.repository or not spec.commit):
            raise TaskCMethodRegistryError(f"git method {method_id} needs repository and commit")
        if spec.output_semantics not in {"official_return_order", "no_output"}:
            raise TaskCMethodRegistryError(f"unsupported output semantics for {method_id}")
        if spec.source_kind == "publication_only" and spec.output_semantics != "no_output":
            raise TaskCMethodRegistryError(
                f"publication-only method {method_id} must declare no_output"
            )
        methods[method_id] = spec
    for observational_id in ("random1000", "grnboost", "pc", "ges", "gsp", "notears_linear", "sortnregress"):
        if methods[observational_id].training_information != "observational":
            raise TaskCMethodRegistryError(f"{observational_id} must remain observational")
    if not methods:
        raise TaskCMethodRegistryError("method registry must not be empty")
    return TaskCMethodRegistry(
        schema_version="1.0",
        methods=methods,
        causalbench=payload["causalbench"],
    )
```

- [ ] **Step 5: 运行测试并提交**

```bash
pytest tests/test_task_c_method_registry.py -q -p no:cacheprovider
git add configs/task_c_methods_v1.json src/evaluation/task_c_method_registry.py tests/test_task_c_method_registry.py
git commit -m "feat: register Task C comparison methods"
```

Expected: tests PASS；提交成功。

### Task 2: 把所有方法映射到相同完整关系范围

**Files:**
- Create: `src/evaluation/task_c_predictions.py`
- Create: `tests/test_task_c_predictions.py`

- [ ] **Step 1: 写缺失关系补零、重复合并和并列顺序测试**

创建 `tests/test_task_c_predictions.py`：

```python
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.task_c_predictions import (
    TaskCPredictionError,
    normalize_task_c_predictions,
)


def test_sparse_output_is_completed_to_all_directed_nonself_edges() -> None:
    raw = pd.DataFrame({"source": ["A"], "target": ["B"], "score": [0.8]})
    completed = normalize_task_c_predictions(raw, ["A", "B", "C"])
    assert len(completed) == 6
    assert completed.loc[
        (completed["source"] == "A") & (completed["target"] == "B"), "score"
    ].item() == pytest.approx(0.8)
    assert int((completed["score"] == 0.0).sum()) == 5


def test_duplicate_edges_keep_the_highest_finite_score() -> None:
    raw = pd.DataFrame(
        {"source": ["A", "A"], "target": ["B", "B"], "score": [0.2, 0.7]}
    )
    completed = normalize_task_c_predictions(raw, ["A", "B"])
    assert completed.loc[
        (completed["source"] == "A") & (completed["target"] == "B"), "score"
    ].item() == pytest.approx(0.7)


def test_unknown_gene_or_negative_score_fails_closed() -> None:
    with pytest.raises(TaskCPredictionError):
        normalize_task_c_predictions(
            pd.DataFrame({"source": ["A"], "target": ["Z"], "score": [1.0]}),
            ["A", "B"],
        )
    with pytest.raises(TaskCPredictionError):
        normalize_task_c_predictions(
            pd.DataFrame({"source": ["A"], "target": ["B"], "score": [-1.0]}),
            ["A", "B"],
        )
```

- [ ] **Step 2: 运行测试并确认模块缺失**

```bash
pytest tests/test_task_c_predictions.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 实现完整关系标准化**

创建 `src/evaluation/task_c_predictions.py`：

```python
"""把稀疏或前若干条外部关系映射到共同的完整评分范围。"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np
import pandas as pd


class TaskCPredictionError(ValueError):
    """外部方法结果不能安全进入统一评分。"""


def normalize_task_c_predictions(
    raw: pd.DataFrame,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    required = {"source", "target", "score"}
    missing = required - set(raw.columns)
    if missing:
        raise TaskCPredictionError(f"prediction table is missing {sorted(missing)}")
    genes = tuple(str(gene) for gene in gene_names)
    gene_set = set(genes)
    selected = raw[["source", "target", "score"]].copy()
    selected["source"] = selected["source"].astype(str)
    selected["target"] = selected["target"].astype(str)
    selected["score"] = pd.to_numeric(selected["score"], errors="coerce")
    if not np.isfinite(selected["score"]).all() or (selected["score"] < 0).any():
        raise TaskCPredictionError("scores must be finite and non-negative")
    if not set(selected["source"]) <= gene_set or not set(selected["target"]) <= gene_set:
        raise TaskCPredictionError("prediction table contains genes outside the fixed set")
    selected = selected[selected["source"] != selected["target"]]
    selected = selected.groupby(["source", "target"], as_index=False)["score"].max()
    universe = pd.DataFrame(permutations(genes, 2), columns=["source", "target"])
    completed = universe.merge(selected, how="left", on=["source", "target"])
    completed["score"] = completed["score"].fillna(0.0)
    completed["returned_by_method"] = completed["score"] > 0.0
    return completed.sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
```

- [ ] **Step 4: 运行测试并提交**

```bash
pytest tests/test_task_c_predictions.py -q -p no:cacheprovider
git add src/evaluation/task_c_predictions.py tests/test_task_c_predictions.py
git commit -m "feat: normalize Task C edge universes"
```

Expected: 3 tests PASS。

### Task 3: 实现官方 CausalBench 与 PSGRN 外部工作进程

**Files:**
- Create: `envs/task_c/psgrn.yml`
- Create: `scripts/task_c_workers/causalbench_worker.py`
- Create: `scripts/task_c_workers/psgrn_worker.py`
- Create: `tests/test_task_c_external_workers.py`

- [ ] **Step 1: 写工作进程的静态接口测试**

创建 `tests/test_task_c_external_workers.py`：

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_workers_expose_plain_language_help() -> None:
    for relative in (
        "scripts/task_c_workers/causalbench_worker.py",
        "scripts/task_c_workers/psgrn_worker.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "单细胞" in completed.stdout
        assert "--input-npz" in completed.stdout
        assert "--output-csv" in completed.stdout
```

- [ ] **Step 2: 运行测试并确认工作进程缺失**

```bash
pytest tests/test_task_c_external_workers.py -q -p no:cacheprovider
```

Expected: FAIL。

- [ ] **Step 3: 写 PSGRN 环境**

创建 `envs/task_c/psgrn.yml`：

```yaml
name: hypersca-task-c-psgrn
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - numpy==1.24.4
      - pandas==2.0.3
      - scikit-learn==1.3.2
      - lightgbm==4.5.0
      - tqdm==4.66.5
      - git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71
```

- [ ] **Step 4: 实现 CausalBench 工作进程**

创建 `scripts/task_c_workers/causalbench_worker.py`。参数解析和 `--help` 在任何外部导入前完成；模型类导入与 `MODEL_BUILDERS` 都放在 `main()` 中。在 `main()` 中读取三数组 NPZ；观察方法只保留 `non-targeting` 细胞并传入 `TrainingRegime.Observational`，干预方法传入 `TrainingRegime.PartialIntervational`。使用官方类映射：

```python
MODEL_BUILDERS = {
    "random1000": lambda: RandomWithSize(1000),
    "grnboost": GRNBoost,
    "pc": lambda: PC(missing_value=False),
    "ges": GES,
    "gies": GIES,
    "gsp": GreedySparsestPermutation,
    "igsp": InterventionalGreedySparsestPermutation,
    "notears-lin-sparse": lambda: NotearsLin(lambda1=0.001),
    "DCDI-G": lambda: DCDI("DCDI-G"),
    "DCDI-DSF": lambda: DCDI("DCDI-DSF"),
    "DCDFG-LIN": lambda: DCDFG("linear"),
    "DCDFG-MLP": lambda: DCDFG("mlplr"),
    "sortnregress": Sortnregress,
}
```

输出规则由登记表的 `output_semantics` 决定。所有可运行连接当前均登记为 `official_return_order`：保留官方返回顺序，第一条得 `N` 分、最后一条得 `1` 分；这只把已有顺序转成单调分数，不重新选择关系。若实际返回对象是 `set` 或顺序在固定提交中不可证明，工作进程以 `failed_invalid_output` 停止，不自行猜测顺序。输出 CSV 固定为 `source,target,score`。

参数解析必须包含：

```python
parser.add_argument("--input-npz", type=Path, required=True)
parser.add_argument("--output-csv", type=Path, required=True)
parser.add_argument("--model-name", choices=sorted(MODEL_BUILDERS), required=True)
parser.add_argument(
    "--training-information",
    choices=["observational", "partial_interventional"],
    required=True,
)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument(
    "--output-semantics",
    choices=["official_return_order"],
    required=True,
)
```

不要调用 CausalBench 自带 `DatasetSplitter`，因为它会重新划分并破坏 HyperSCA 的封存规则。

- [ ] **Step 5: 实现 PSGRN 工作进程**

创建 `scripts/task_c_workers/psgrn_worker.py`。参数除 `--model-name` 外增加 `--psgrn-source`。通过 `importlib.util.spec_from_file_location` 从固定检出目录加载 `src/main.py` 的 `Custom` 类，直接传入 HyperSCA 已允许的表达矩阵、标签和基因名。把返回的前 1,000 条关系按返回顺序转为递减分数并写 `source,target,score`。

在加载前必须执行：

```python
expected = "74aa640f7c472b23a69811f6795bb17678efd344"
completed = subprocess.run(
    ["git", "-C", str(args.psgrn_source), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
)
if completed.stdout.strip() != expected:
    raise SystemExit("PSGRN source revision does not match the registered commit")
```

- [ ] **Step 6: 运行接口测试并提交**

```bash
pytest tests/test_task_c_external_workers.py -q -p no:cacheprovider
git add envs/task_c/psgrn.yml scripts/task_c_workers tests/test_task_c_external_workers.py
git commit -m "feat: add isolated Task C method workers"
```

Expected: help tests PASS。真实外部导入留到环境预演，不能在主测试环境中伪造。

### Task 4: 获取官方资产并记录不可运行方法

**Files:**
- Create: `scripts/bootstrap_task_c_methods.py`
- Create: `src/evaluation/task_c_runtime.py`
- Create: `tests/test_task_c_runtime.py`

- [ ] **Step 1: 写资产状态和子进程失败分类测试**

创建 `tests/test_task_c_runtime.py`：

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.task_c_method_registry import load_task_c_method_registry
from src.evaluation.task_c_runtime import (
    classify_publication_only_method,
    run_isolated_method,
)


ROOT = Path(__file__).resolve().parents[1]


def test_publication_only_method_gets_explicit_unavailable_status() -> None:
    registry = load_task_c_method_registry(ROOT / "configs/task_c_methods_v1.json")
    status = classify_publication_only_method(registry.methods["betterboost"])
    assert status["status"] == "official_assets_unavailable"
    assert status["publication"].startswith("https://openreview.net/")


def test_timeout_is_not_mislabeled_as_code_incompatibility(tmp_path: Path) -> None:
    result = run_isolated_method(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        output_dir=tmp_path,
        timeout_seconds=1,
    )
    assert result["status"] == "failed_timeout"
    assert (tmp_path / "method_status.json").exists()


def test_nonzero_exit_is_recorded_with_stderr(tmp_path: Path) -> None:
    result = run_isolated_method(
        [sys.executable, "-c", "import sys; print('bad input', file=sys.stderr); sys.exit(3)"],
        output_dir=tmp_path,
        timeout_seconds=10,
    )
    assert result["status"] == "official_code_incompatible"
    assert "bad input" in result["stderr_tail"]
```

- [ ] **Step 2: 运行测试并确认运行模块缺失**

```bash
pytest tests/test_task_c_runtime.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 实现失败分类与资源记录**

创建 `src/evaluation/task_c_runtime.py`，实现：

```python
"""隔离运行任务 C 外部方法并保留失败和资源证据。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Sequence

from src.evaluation.task_c_method_registry import TaskCMethodSpec


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def classify_publication_only_method(spec: TaskCMethodSpec) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "method_id": spec.method_id,
        "status": "official_assets_unavailable",
        "publication": spec.publication,
        "reason": "The registered primary source provides a method report but no runnable official code asset.",
    }


def run_isolated_method(
    command: Sequence[str],
    *,
    output_dir: str | Path,
    timeout_seconds: int,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    time_report = destination / "resource_usage.txt"
    timed_command = ["/usr/bin/time", "-v", "-o", str(time_report), *command]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            timed_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stderr_lower = completed.stderr.lower()
        if completed.returncode == 0:
            status = "completed_raw_inference"
        elif completed.returncode == 137 or "out of memory" in stderr_lower:
            status = "failed_resource_limit"
        else:
            status = "official_code_incompatible"
        return_code = completed.returncode
        stdout_tail = completed.stdout[-4000:]
        stderr_tail = completed.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        status = "failed_timeout"
        return_code = None
        stdout_tail = (exc.stdout or "")[-4000:]
        stderr_tail = (exc.stderr or "")[-4000:]
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "command": list(command),
    }
    _write_json(destination / "method_status.json", payload)
    resource_values: dict[str, object] = {
        "schema_version": "1.0",
        "elapsed_seconds": payload["elapsed_seconds"],
        "maximum_resident_kib": None,
    }
    if time_report.exists():
        for line in time_report.read_text(encoding="utf-8").splitlines():
            if "Maximum resident set size" in line:
                resource_values["maximum_resident_kib"] = int(line.rsplit(":", 1)[1])
                break
    _write_json(destination / "resource_usage.json", resource_values)
    return payload
```

- [ ] **Step 4: 实现官方资产引导命令**

创建 `scripts/bootstrap_task_c_methods.py`：

- `--cache-root` 必填；
- `--registry` 默认 `configs/task_c_methods_v1.json`；
- `git` 资产只克隆到参数 `--cache-root` 下的 `sources/{method_id}` 并检出登记提交；
- 使用 `conda env create -f envs/task_c/causalbench.yml` 和 `envs/task_c/psgrn.yml`；已存在环境时运行 `conda env update --prune`；
- 对 `publication_only` 方法写入缓存根目录的 `status/{method_id}/method_status.json`，状态为 `official_assets_unavailable`；
- 对每个环境运行 `conda list --json`，保存到缓存根目录的 `environment_manifests/`；
- 不复制外部源文件到项目目录。

所有 `subprocess.run` 使用参数列表和 `check=True`；不得使用 `shell=True`。

- [ ] **Step 5: 运行测试并提交**

```bash
pytest tests/test_task_c_runtime.py tests/test_task_c_method_registry.py -q -p no:cacheprovider
git add src/evaluation/task_c_runtime.py scripts/bootstrap_task_c_methods.py tests/test_task_c_runtime.py
git commit -m "feat: track Task C method availability"
```

Expected: tests PASS。

### Task 5: 实现单方法统一运行命令

**Files:**
- Create: `scripts/run_task_c_method.py`
- Modify: `tests/test_task_c_runtime.py`
- Create: `docs/research/task_c_method_compatibility_v1.md`

- [ ] **Step 1: 写本地均值变化与稀疏外部结果标准化测试**

在 `tests/test_task_c_runtime.py` 追加一个小 NPZ 测试；用 NumPy 写入 `expression_matrix`、`interventions` 和 `var_names`，再运行：

```python
input_path = tmp_path / "allowed_train.npz"
np.savez(
    input_path,
    expression_matrix=np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 1.0], [0.0, 1.2]],
        dtype=np.float32,
    ),
    interventions=np.asarray(["non-targeting", "non-targeting", "A", "A"]),
    var_names=np.asarray(["A", "B"]),
)
output = tmp_path / "run"
subprocess.run(
    [
        sys.executable,
        "scripts/run_task_c_method.py",
        "--method-id",
        "mean_difference",
        "--input-npz",
        str(input_path),
        "--output-dir",
        str(output),
        "--seed",
        "11",
        "--registry",
        "configs/task_c_methods_v1.json",
        "--asset-root",
        str(tmp_path / "method_assets"),
    ],
    cwd=ROOT,
    check=True,
)
```

然后断言：

```python
predictions = pd.read_csv(output / "predictions.csv")
assert len(predictions) == gene_count * (gene_count - 1)
status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
assert status["status"] == "completed_standardized_output"
```

- [ ] **Step 2: 实现方法分派和结果核验**

创建 `scripts/run_task_c_method.py`，执行以下固定分派：

- `hypersca_c`：调用当前环境的 `scripts/run_hypersca_c.py`；
- `mean_difference`：直接调用 `score_mean_difference_network` 并写原始分数；
- `causalbench`：使用 `conda run -n hypersca-task-c-causalbench python scripts/task_c_workers/causalbench_worker.py`；
- `guanlab_psgrn`：使用 `conda run -n hypersca-task-c-psgrn python scripts/task_c_workers/psgrn_worker.py`；
- `publication_only`：写 `official_assets_unavailable` 后返回，不产生伪结果。

外部进程成功后：

1. 读取输入 NPZ 的固定基因顺序；
2. 用 `normalize_task_c_predictions` 生成完整关系表；
3. 写 `raw_predictions.csv` 和 `predictions.csv`；
4. 核对行数等于 `G × (G - 1)`、分数有限且无重复；
5. 通过后把状态从 `completed_raw_inference` 改为 `completed_standardized_output`；真实数据控制器完成封存评分和强制文件核验后，才可改为 `passed_real_rehearsal`；
6. 输入文件 SHA-256、登记表 SHA-256、方法提交和命令写入 `environment_manifest.json`。

若输出 CSV 缺列、含未知基因、负分或无效数值，状态写 `failed_invalid_output`。

- [ ] **Step 3: 运行统一命令测试**

```bash
pytest tests/test_task_c_runtime.py tests/test_task_c_predictions.py tests/test_task_c_external_workers.py -q -p no:cacheprovider
```

Expected: tests PASS。

- [ ] **Step 4: 写公开方法可用性说明**

创建 `docs/research/task_c_method_compatibility_v1.md`，记录：

- CausalBench 内置方法和固定提交；
- Varsortability 在固定 CausalBench 提交中是排序偏倚诊断概念，实际可运行的网络估计器是 `Sortnregress`；登记表只把 `sortnregress` 作为方法运行，不把同一实现重复计数为两个独立比较方法；
- PSGRN 固定仓库与提交；
- BetterBoost、SparseRC、CATRAN 当前登记的主要来源只有公开方法报告，预演必须记录 `official_assets_unavailable`，不能以自写版本代替；
- 外部方法失败不阻止其他方法运行，但不会被从全面比较清单删除；
- 观察方法只能读取未干预细胞，部分干预方法只能读取公开学习文件；
- 所有输出补齐到共同有向关系范围，未返回关系为零分。

- [ ] **Step 5: 完整检查并提交**

```bash
python scripts/check_plain_language.py
pytest tests/test_task_c_method_registry.py tests/test_task_c_predictions.py tests/test_task_c_runtime.py tests/test_task_c_external_workers.py tests/test_task_c_benchmark.py -q -p no:cacheprovider
git diff --check
git add scripts/run_task_c_method.py tests/test_task_c_runtime.py docs/research/task_c_method_compatibility_v1.md
git commit -m "feat: run Task C methods through one contract"
git status --short
```

Expected: 检查和测试通过，工作区干净。

### Task 6: 建立不读取最终参考关系的参数选择接口

**Files:**
- Create: `configs/task_c_tuning_v1.json`
- Create: `src/evaluation/task_c_tuning.py`
- Create: `scripts/select_task_c_configuration.py`
- Create: `tests/test_task_c_tuning.py`

- [ ] **Step 1: 写调节响应标签、20 次上限和确定性选择测试**

创建 `tests/test_task_c_tuning.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_tuning import (
    TaskCTuningError,
    build_tuning_response_edges,
    select_task_c_configuration,
)


def test_tuning_edges_come_from_allowed_response_cells() -> None:
    controls = np.column_stack([np.linspace(0.0, 0.1, 10), np.zeros(10)])
    perturbed = np.column_stack([np.linspace(0.0, 0.1, 10), np.full(10, 5.0)])
    expression = np.vstack([controls, perturbed])
    labels = np.asarray(["non-targeting"] * 10 + ["A"] * 10)
    edges = build_tuning_response_edges(
        expression,
        labels,
        ["A", "B"],
        eligible_sources={"A"},
        q_value_threshold=0.1,
    )
    assert ("A", "B") in edges


def test_configuration_selection_maximizes_ap_and_breaks_ties_by_trial_index() -> None:
    universe = pd.DataFrame(
        {"source": ["A", "A"], "target": ["B", "C"]}
    )
    trials = []
    for trial_index, scores in ((0, [0.9, 0.1]), (1, [0.1, 0.9])):
        trial = universe.assign(score=scores)
        trials.append((trial_index, {"lambda": trial_index + 1}, trial))
    selected = select_task_c_configuration(
        trials,
        tuning_edges={("A", "B")},
        maximum_trials=20,
    )
    assert selected["selected_trial_index"] == 0
    assert selected["average_precision"] == pytest.approx(1.0)


def test_more_than_twenty_trials_fails_closed() -> None:
    trial = pd.DataFrame({"source": ["A"], "target": ["B"], "score": [1.0]})
    with pytest.raises(TaskCTuningError, match="twenty"):
        select_task_c_configuration(
            [(index, {"index": index}, trial) for index in range(21)],
            tuning_edges={("A", "B")},
            maximum_trials=20,
        )
```

- [ ] **Step 2: 写冻结调节配置**

创建 `configs/task_c_tuning_v1.json`：

```json
{
  "schema_version": "1.0",
  "maximum_trials_per_method": 20,
  "selection_metric": "average_precision_against_tuning_response_edges",
  "q_value_threshold": 0.1,
  "tie_break": ["average_precision_descending", "trial_index_ascending"],
  "external_biological_references_allowed": false,
  "final_holdout_allowed": false
}
```

- [ ] **Step 3: 实现只从调节响应构建关系标签**

创建 `src/evaluation/task_c_tuning.py`，核心实现为：

```python
"""任务 C 参数调节：只使用公开调节干预响应，不读取最终参考关系。"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score


class TaskCTuningError(ValueError):
    """调节数据、候选次数或预测范围违反固定规则。"""


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def build_tuning_response_edges(
    expression: np.ndarray,
    interventions: Sequence[str],
    gene_names: Sequence[str],
    *,
    eligible_sources: Iterable[str],
    q_value_threshold: float,
) -> set[tuple[str, str]]:
    values = np.asarray(expression, dtype=float)
    labels = np.asarray([str(label) for label in interventions])
    genes = tuple(str(gene) for gene in gene_names)
    controls = labels == "non-targeting"
    tests = []
    for source in sorted(set(eligible_sources)):
        perturbed = labels == source
        if int(perturbed.sum()) < 2 or int(controls.sum()) < 2:
            continue
        for target_index, target in enumerate(genes):
            if source == target:
                continue
            p_value = mannwhitneyu(
                values[perturbed, target_index],
                values[controls, target_index],
                alternative="two-sided",
            ).pvalue
            tests.append((source, target, float(p_value)))
    if not tests:
        raise TaskCTuningError("tuning responses contain no testable directed relations")
    q_values = _benjamini_hochberg(np.asarray([test[2] for test in tests]))
    edges = {
        (source, target)
        for (source, target, _), q_value in zip(tests, q_values)
        if q_value <= q_value_threshold
    }
    if not edges:
        raise TaskCTuningError("tuning responses contain no positive relation at the fixed q-value")
    return edges
```

- [ ] **Step 4: 实现确定性配置选择**

在同一文件追加：

```python
def select_task_c_configuration(
    trials: Sequence[tuple[int, Mapping[str, object], pd.DataFrame]],
    *,
    tuning_edges: Iterable[tuple[str, str]],
    maximum_trials: int,
) -> dict[str, object]:
    if maximum_trials != 20 or len(trials) > maximum_trials:
        raise TaskCTuningError("configuration selection allows at most twenty trials")
    positives = set(tuning_edges)
    evaluated = []
    for trial_index, parameters, predictions in trials:
        if predictions.duplicated(["source", "target"]).any():
            raise TaskCTuningError("trial predictions contain duplicate relations")
        labels = np.asarray(
            [
                (source, target) in positives
                for source, target in predictions[["source", "target"]].itertuples(
                    index=False, name=None
                )
            ],
            dtype=int,
        )
        if labels.sum() == 0 or labels.sum() == len(labels):
            raise TaskCTuningError("tuning relation universe needs positives and negatives")
        metric = float(average_precision_score(labels, predictions["score"]))
        evaluated.append((metric, int(trial_index), dict(parameters)))
    if not evaluated:
        raise TaskCTuningError("at least one completed tuning trial is required")
    metric, trial_index, parameters = sorted(
        evaluated, key=lambda item: (-item[0], item[1])
    )[0]
    return {
        "schema_version": "1.0",
        "selected_trial_index": trial_index,
        "selected_parameters": parameters,
        "average_precision": metric,
        "completed_trial_count": len(evaluated),
        "external_biological_references_used": False,
        "final_holdout_used": False,
    }
```

- [ ] **Step 5: 实现薄选择命令并提交**

创建 `scripts/select_task_c_configuration.py`，接收 `--tune-npz`、重复的 `--trial-dir`、`--output-json` 和 `--config`。每个试验目录必须包含 `predictions.csv` 和 `trial_parameters.json`；命令核对输入指纹、调用上述函数并写选择结果，不接受参考关系或私有目录参数。

```bash
pytest tests/test_task_c_tuning.py tests/test_task_c_runtime.py tests/test_task_c_predictions.py -q -p no:cacheprovider
git add configs/task_c_tuning_v1.json src/evaluation/task_c_tuning.py scripts/select_task_c_configuration.py tests/test_task_c_tuning.py
git commit -m "feat: select Task C settings without holdout leakage"
```

Expected: tests PASS。若调节集没有固定阈值下的阳性关系，该条件记录参数选择失败，不读取最终参考关系补救。
