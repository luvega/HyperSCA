# Task C Rehearsal and Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真实 K562/RPE1 数据上完成 64 基因连接检查和 256 基因全方法预演，保留零效应、失败和资源证据，并只判断能否启动正式五随机种子比较。

**Architecture:** 预演控制器只编排前三份计划提供的数据、模型和隔离方法接口。独立评分进程读取封存文件和参考关系；方法进程看不到这些内容。预演按运行目录追加结果，汇总器核验五个强制文件、比较相同评分范围、生成配对区间和资源估计，但把结论固定为 `workflow_validation_only`。

**Tech Stack:** Python 3.10、NumPy、pandas、scikit-learn、psutil、pytest、conda、CausalBench

---

## 执行位置与前置条件

```bash
cd /home/a/.config/superpowers/worktrees/HyperSCA/real-data-readiness-design
```

依次完成数据划分、HyperSCA-C 和比较方法运行三份计划。预演可安装软件、下载公开数据并写入调用者指定的外部目录；不得提交大型缓存、外部代码副本或生成结果。

## 文件结构

- Create: `configs/task_c_rehearsal_v1.json` — 两级预演、零效应和准入规则。
- Create: `src/evaluation/task_c_rehearsal.py` — 固定基因/细胞子集、运行身份和强制文件核验。
- Create: `src/evaluation/task_c_null_controls.py` — 标签打乱和假干预数据。
- Create: `src/evaluation/task_c_aggregation.py` — 评分、配对分组区间、失败保留和资源估计。
- Create: `scripts/task_c_workers/causalbench_evaluation_worker.py` — 在官方环境中计算补充干预分布指标。
- Create: `scripts/run_task_c_rehearsal.py` — 运行连接检查或全方法预演。
- Create: `scripts/summarize_task_c_rehearsal.py` — 只读汇总并生成全量作业清单草案。
- Create: `tests/test_task_c_rehearsal.py`
- Create: `tests/test_task_c_null_controls.py`
- Create: `tests/test_task_c_aggregation.py`
- Create: `tests/test_task_c_rehearsal_cli.py`
- Create: `docs/research/task_c_rehearsal_v1.md` — 真实预演结果、阻断和资源报告。
- Modify: `docs/research/task_c_mean_difference_baseline_v1.md` — 声明共同参考关系和完整评分范围。
- Modify: `docs/technical_roadmap.md` — 记录任务 C 能否进入正式实测。

### Task 1: 固定预演范围和确定性子集

**Files:**
- Create: `configs/task_c_rehearsal_v1.json`
- Create: `src/evaluation/task_c_rehearsal.py`
- Create: `tests/test_task_c_rehearsal.py`

- [ ] **Step 1: 写配置和子集选择失败测试**

创建 `tests/test_task_c_rehearsal.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.task_c_rehearsal import (
    center_and_merge_allowed_contexts,
    choose_rehearsal_cells,
    choose_rehearsal_genes,
    load_task_c_rehearsal_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_rehearsal_profiles_are_frozen_and_never_promotion_eligible() -> None:
    config = load_task_c_rehearsal_config(
        ROOT / "configs/task_c_rehearsal_v1.json"
    )
    assert config.profiles["connection"].maximum_genes == 64
    assert config.profiles["connection"].maximum_cells_per_context == 2000
    assert config.profiles["comprehensive"].maximum_genes == 256
    assert config.profiles["comprehensive"].maximum_cells_per_context == 20000
    assert config.seed == 11
    assert config.promotion_eligible is False


def test_gene_selection_uses_only_allowed_control_variance() -> None:
    k562 = np.asarray([[0.0, 0.0, 0.0], [1.0, 5.0, 2.0], [2.0, 10.0, 4.0]])
    rpe1 = np.asarray([[0.0, 0.0, 0.0], [2.0, 4.0, 1.0], [4.0, 8.0, 2.0]])
    selected = choose_rehearsal_genes(
        {"k562": k562, "rpe1": rpe1},
        gene_names=["A", "B", "C"],
        maximum_genes=2,
    )
    assert selected == ("B", "A")


def test_cell_selection_is_stratified_and_reproducible() -> None:
    labels = np.asarray(["non-targeting"] * 6 + ["A"] * 6 + ["B"] * 6)
    first = choose_rehearsal_cells(labels, maximum_cells=9, seed=11)
    second = choose_rehearsal_cells(labels, maximum_cells=9, seed=11)
    assert first.tolist() == second.tolist()
    selected_labels = labels[first]
    assert {label: int((selected_labels == label).sum()) for label in set(labels)} == {
        "A": 3,
        "B": 3,
        "non-targeting": 3,
    }


def test_cross_context_merge_centers_each_context_from_its_controls() -> None:
    merged, labels, environments = center_and_merge_allowed_contexts(
        {
            "k562": (
                np.asarray([[1.0, 2.0], [3.0, 4.0], [9.0, 8.0]]),
                np.asarray(["non-targeting", "non-targeting", "A"]),
            ),
            "rpe1": (
                np.asarray([[10.0, 20.0], [14.0, 24.0]]),
                np.asarray(["non-targeting", "non-targeting"]),
            ),
        }
    )
    for environment in ("k562", "rpe1"):
        controls = (environments == environment) & (labels == "non-targeting")
        assert merged[controls].mean(axis=0).tolist() == pytest.approx([0.0, 0.0])
```

- [ ] **Step 2: 确认配置和模块尚不存在**

```bash
pytest tests/test_task_c_rehearsal.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 写冻结预演配置**

创建 `configs/task_c_rehearsal_v1.json`：

```json
{
  "schema_version": "1.0",
  "seed": 11,
  "promotion_eligible": false,
  "feature_selection": "mean_control_variance_across_contexts",
  "profiles": {
    "connection": {"maximum_genes": 64, "maximum_cells_per_context": 2000, "timeout_seconds_per_method": 1800},
    "comprehensive": {"maximum_genes": 256, "maximum_cells_per_context": 20000, "timeout_seconds_per_method": 14400}
  },
  "null_controls": {"repeats": 20, "minimum_empirical_advantage": 0.0, "maximum_empirical_p_value": 0.05},
  "required_core_methods": ["hypersca_c", "mean_difference", "random1000", "grnboost", "pc", "notears_linear"],
  "required_interventional_method_count": 1,
  "required_artifacts": ["run_manifest.json", "input_summary.json", "metrics.json", "predictions.csv", "promotion_decision.json"],
  "full_run_seeds": [11, 23, 47, 71, 97]
}
```

- [ ] **Step 4: 实现配置核验和稳定子集**

创建 `src/evaluation/task_c_rehearsal.py`，先写入：

```python
"""任务 C 真实数据预演的固定范围、抽样和运行核验。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class TaskCRehearsalError(ValueError):
    """预演配置、输入或结果不满足已确认规则。"""


@dataclass(frozen=True)
class RehearsalProfile:
    maximum_genes: int
    maximum_cells_per_context: int
    timeout_seconds_per_method: int


@dataclass(frozen=True)
class TaskCRehearsalConfig:
    schema_version: str
    seed: int
    promotion_eligible: bool
    feature_selection: str
    profiles: Mapping[str, RehearsalProfile]
    null_controls: Mapping[str, float | int]
    required_core_methods: tuple[str, ...]
    required_interventional_method_count: int
    required_artifacts: tuple[str, ...]
    full_run_seeds: tuple[int, ...]


def load_task_c_rehearsal_config(path: str | Path) -> TaskCRehearsalConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise TaskCRehearsalError("schema_version must be 1.0")
    profiles = {
        name: RehearsalProfile(**values)
        for name, values in payload["profiles"].items()
    }
    if set(profiles) != {"connection", "comprehensive"}:
        raise TaskCRehearsalError("profiles must be connection and comprehensive")
    if payload.get("promotion_eligible") is not False:
        raise TaskCRehearsalError("real-data rehearsal must not be promotion eligible")
    return TaskCRehearsalConfig(
        schema_version="1.0",
        seed=int(payload["seed"]),
        promotion_eligible=False,
        feature_selection=str(payload["feature_selection"]),
        profiles=profiles,
        null_controls=payload["null_controls"],
        required_core_methods=tuple(payload["required_core_methods"]),
        required_interventional_method_count=int(
            payload["required_interventional_method_count"]
        ),
        required_artifacts=tuple(payload["required_artifacts"]),
        full_run_seeds=tuple(int(seed) for seed in payload["full_run_seeds"]),
    )


def choose_rehearsal_genes(
    allowed_control_expression: Mapping[str, np.ndarray],
    gene_names: Sequence[str],
    maximum_genes: int,
) -> tuple[str, ...]:
    genes = tuple(str(gene) for gene in gene_names)
    if maximum_genes < 2 or not allowed_control_expression:
        raise TaskCRehearsalError("gene selection needs contexts and at least two genes")
    variances = []
    for context_id, expression in sorted(allowed_control_expression.items()):
        values = np.asarray(expression, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(genes):
            raise TaskCRehearsalError(f"control expression shape mismatch for {context_id}")
        variances.append(np.var(values, axis=0))
    mean_variance = np.mean(np.stack(variances), axis=0)
    order = sorted(range(len(genes)), key=lambda index: (-mean_variance[index], genes[index]))
    return tuple(genes[index] for index in order[:maximum_genes])


def choose_rehearsal_cells(
    interventions: Sequence[str],
    maximum_cells: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray([str(label) for label in interventions])
    if maximum_cells >= len(labels):
        return np.arange(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    groups = sorted(set(labels.tolist()))
    base = maximum_cells // len(groups)
    remainder = maximum_cells % len(groups)
    selected: list[int] = []
    for group_index, label in enumerate(groups):
        candidates = np.flatnonzero(labels == label)
        quota = min(len(candidates), base + int(group_index < remainder))
        selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
    if len(selected) < maximum_cells:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected))
        extra = rng.choice(remaining, size=maximum_cells - len(selected), replace=False)
        selected.extend(extra.tolist())
    return np.asarray(sorted(selected), dtype=int)


def center_and_merge_allowed_contexts(
    contexts: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    control_label: str = "non-targeting",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = []
    labels = []
    environments = []
    expected_genes = None
    for context_id, (expression, interventions) in sorted(contexts.items()):
        values = np.asarray(expression, dtype=np.float32)
        context_labels = np.asarray(interventions, dtype=str)
        if values.ndim != 2 or values.shape[0] != len(context_labels):
            raise TaskCRehearsalError(f"expression and labels mismatch for {context_id}")
        if expected_genes is None:
            expected_genes = values.shape[1]
        elif values.shape[1] != expected_genes:
            raise TaskCRehearsalError("cross-context files must use the same genes")
        controls = context_labels == control_label
        if int(controls.sum()) < 2:
            raise TaskCRehearsalError(f"at least two controls are required for {context_id}")
        center = values[controls].mean(axis=0)
        scale = values[controls].std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        centered.append((values - center) / scale)
        labels.append(context_labels)
        environments.append(np.asarray([context_id] * len(context_labels)))
    return (
        np.concatenate(centered, axis=0),
        np.concatenate(labels),
        np.concatenate(environments),
    )
```

- [ ] **Step 5: 运行测试并提交**

```bash
pytest tests/test_task_c_rehearsal.py -q -p no:cacheprovider
git add configs/task_c_rehearsal_v1.json src/evaluation/task_c_rehearsal.py tests/test_task_c_rehearsal.py
git commit -m "feat: freeze Task C rehearsal profiles"
```

Expected: 4 tests PASS。

### Task 2: 实现两类零效应数据并预先固定判定

**Files:**
- Create: `src/evaluation/task_c_null_controls.py`
- Create: `tests/test_task_c_null_controls.py`

- [ ] **Step 1: 写组大小保持和假干预测试**

创建 `tests/test_task_c_null_controls.py`：

```python
from __future__ import annotations

import numpy as np

from src.evaluation.task_c_null_controls import (
    build_control_resampling_null,
    empirical_null_check,
    permute_intervention_labels,
)


def test_permutation_keeps_every_intervention_group_size() -> None:
    labels = np.asarray(["non-targeting"] * 4 + ["A"] * 3 + ["B"] * 2)
    permuted = permute_intervention_labels(labels, seed=11)
    assert {label: int((permuted == label).sum()) for label in set(labels)} == {
        label: int((labels == label).sum()) for label in set(labels)
    }
    assert not np.array_equal(labels, permuted)


def test_control_resampling_returns_expression_with_fake_group_sizes() -> None:
    expression = np.arange(60, dtype=float).reshape(20, 3)
    labels = np.asarray(["non-targeting"] * 12 + ["A"] * 5 + ["B"] * 3)
    null_expression, null_labels = build_control_resampling_null(
        expression, labels, seed=11
    )
    assert null_expression.shape == expression.shape
    assert int((null_labels == "A").sum()) == 5
    assert int((null_labels == "B").sum()) == 3
    assert set(null_labels) == {"non-targeting", "A", "B"}


def test_empirical_null_check_requires_real_metric_to_exceed_all_twenty_nulls() -> None:
    passed = empirical_null_check(0.5, np.linspace(0.1, 0.4, 20), maximum_p_value=0.05)
    failed = empirical_null_check(0.3, np.linspace(0.1, 0.4, 20), maximum_p_value=0.05)
    assert passed["passed"] is True
    assert passed["empirical_p_value"] == 1.0 / 21.0
    assert failed["passed"] is False
```

- [ ] **Step 2: 运行测试并确认模块缺失**

```bash
pytest tests/test_task_c_null_controls.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 实现零效应生成和判定**

创建 `src/evaluation/task_c_null_controls.py`：

```python
"""任务 C 的标签打乱、假干预和预先固定零效应判定。"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class TaskCNullControlError(ValueError):
    """零效应输入或重复结果无效。"""


def permute_intervention_labels(labels: Sequence[str], seed: int) -> np.ndarray:
    values = np.asarray([str(label) for label in labels])
    if len(values) < 2:
        raise TaskCNullControlError("at least two labels are required")
    return np.random.default_rng(seed).permutation(values)


def build_control_resampling_null(
    expression: np.ndarray,
    labels: Sequence[str],
    seed: int,
    control_label: str = "non-targeting",
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(expression)
    interventions = np.asarray([str(label) for label in labels])
    control_rows = np.flatnonzero(interventions == control_label)
    if not len(control_rows):
        raise TaskCNullControlError("control resampling needs control cells")
    rng = np.random.default_rng(seed)
    sampled_rows = rng.choice(control_rows, size=len(interventions), replace=True)
    return values[sampled_rows].copy(), interventions.copy()


def empirical_null_check(
    real_metric: float,
    null_metrics: Sequence[float],
    maximum_p_value: float,
) -> dict[str, float | bool | int]:
    null = np.asarray(null_metrics, dtype=float)
    if len(null) != 20 or not np.isfinite(null).all() or not np.isfinite(real_metric):
        raise TaskCNullControlError("exactly twenty finite null metrics are required")
    p_value = float((1 + np.count_nonzero(null >= real_metric)) / (len(null) + 1))
    return {
        "real_metric": float(real_metric),
        "null_median": float(np.median(null)),
        "empirical_p_value": p_value,
        "repeat_count": int(len(null)),
        "passed": bool(real_metric > float(np.max(null)) and p_value <= maximum_p_value),
    }
```

- [ ] **Step 4: 运行测试并提交**

```bash
pytest tests/test_task_c_null_controls.py -q -p no:cacheprovider
git add src/evaluation/task_c_null_controls.py tests/test_task_c_null_controls.py
git commit -m "feat: add Task C null controls"
```

Expected: 3 tests PASS。

### Task 3: 实现统一评分、配对区间和失败保留

**Files:**
- Create: `src/evaluation/task_c_aggregation.py`
- Create: `tests/test_task_c_aggregation.py`

- [ ] **Step 1: 写参考网络、配对区间和失败分母测试**

创建 `tests/test_task_c_aggregation.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_aggregation import (
    aggregate_task_c_runs,
    evaluate_declared_references,
    paired_cluster_interval,
)


def test_primary_reference_and_directed_reference_are_reported_separately() -> None:
    scores = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C", "C"],
            "target": ["B", "C", "A", "C", "A", "B"],
            "score": [0.9, 0.1, 0.2, 0.8, 0.0, 0.0],
            "effect": [1.0, -0.1, -0.2, 0.7, 0.0, 0.0],
        }
    )
    metrics = evaluate_declared_references(
        scores,
        pooled_reference={("A", "B"), ("B", "A")},
        directed_chip_reference={("A", "B")},
        eligible_sources={"A"},
        directed_reference_context_match=True,
        precision_values=(2, 5),
    )
    assert metrics["primary_reference_id"] == "causalbench_pooled_biological_v1"
    assert metrics["average_precision"] > 0.0
    assert metrics["directed_chip_edge_count"] == 1
    assert metrics["edge_direction_accuracy"] == 1.0


def test_paired_cluster_interval_resamples_seed_condition_pairs() -> None:
    table = pd.DataFrame(
        {
            "seed": [11, 11, 23, 23, 47, 47],
            "condition": ["k562", "rpe1"] * 3,
            "candidate": [0.5, 0.6, 0.55, 0.65, 0.52, 0.62],
            "baseline": [0.4, 0.5, 0.45, 0.55, 0.42, 0.52],
        }
    )
    interval = paired_cluster_interval(table, repeats=1000, seed=11)
    assert interval["estimate"] == pytest.approx(0.1)
    assert interval["ci_lower"] > 0.0
    assert interval["cluster_count"] == 6


def test_failed_runs_remain_in_method_summary(tmp_path: Path) -> None:
    passed = tmp_path / "passed"
    failed = tmp_path / "failed"
    passed.mkdir()
    failed.mkdir()
    (passed / "method_status.json").write_text(
        json.dumps({"method_id": "pc", "status": "passed_real_rehearsal"}),
        encoding="utf-8",
    )
    (passed / "metrics.json").write_text(
        json.dumps({"average_precision": 0.2}), encoding="utf-8"
    )
    (failed / "method_status.json").write_text(
        json.dumps({"method_id": "gies", "status": "failed_timeout"}),
        encoding="utf-8",
    )
    summary = aggregate_task_c_runs([passed, failed])
    assert summary["attempted_run_count"] == 2
    assert summary["completed_run_count"] == 1
    assert summary["status_counts"]["failed_timeout"] == 1
```

- [ ] **Step 2: 运行测试并确认模块缺失**

```bash
pytest tests/test_task_c_aggregation.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 实现参考关系评分和方向检查**

创建 `src/evaluation/task_c_aggregation.py`，先加入：

```python
"""任务 C 真实预演的统一评分、不确定性和失败汇总。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.evaluation.task_c_benchmark import evaluate_task_c_scores


class TaskCAggregationError(ValueError):
    """运行结果不完整或不能进行公平汇总。"""


def evaluate_declared_references(
    scores: pd.DataFrame,
    *,
    pooled_reference: Iterable[tuple[str, str]],
    directed_chip_reference: Iterable[tuple[str, str]],
    eligible_sources: Iterable[str],
    directed_reference_context_match: bool,
    precision_values: Sequence[int] = (1000, 5000),
) -> dict[str, object]:
    pooled = set(pooled_reference)
    directed = set(directed_chip_reference)
    allowed_sources = {str(source) for source in eligible_sources}
    scored = scores[scores["source"].isin(allowed_sources)].copy()
    if scored.empty:
        raise TaskCAggregationError("holdout scoring has no eligible source relations")
    metrics = evaluate_task_c_scores(
        scored, pooled, precision_at_k=int(precision_values[0])
    )
    ordered = scored.sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    for k in precision_values:
        effective_k = min(int(k), len(ordered))
        positives = [
            (source, target) in pooled
            for source, target in ordered.head(effective_k)[["source", "target"]].itertuples(index=False, name=None)
        ]
        metrics[f"precision_at_{k}"] = float(np.mean(positives))
    score_map = {
        (source, target): float(score)
        for source, target, score in scores[["source", "target", "score"]].itertuples(
            index=False, name=None
        )
    }
    directed_in_universe = [
        (source, target)
        for source, target in directed
        if source in allowed_sources
        and (source, target) in score_map
        and (target, source) in score_map
    ]
    direction_accuracy = (
        float(
            np.mean(
                [
                    score_map[(source, target)] > score_map[(target, source)]
                    for source, target in directed_in_universe
                ]
            )
        )
        if directed_in_universe and directed_reference_context_match
        else None
    )
    metrics.update(
        {
            "primary_reference_id": "causalbench_pooled_biological_v1",
            "primary_reference_scope": "directed expansion of pooled biological evidence; incomplete reference, not causal ground truth",
            "directed_reference_id": "causalbench_chipseq_v1",
            "directed_chip_edge_count": int(len(directed_in_universe)),
            "directed_reference_context_match": directed_reference_context_match,
            "edge_direction_accuracy": direction_accuracy,
            "eligible_source_count": int(len(allowed_sources)),
            "scored_edge_count": int(len(scored)),
        }
    )
    return metrics
```

这里的主参考关系严格复现 CausalBench 汇总参考网络的双向展开；ChIP 关系仅作为有向补充，不把双向蛋白互作误称为方向证据。固定提交的 RPE1 分支绑定 HepG2 ChIP 文件，因此 RPE1 条件必须传 `directed_reference_context_match=False`，只报告参考关系数量，不计算方向正确率。每个方法仍提交全部 `G × (G - 1)` 条关系，但 AP 只在该条件私有清单声明、且确有封存干预细胞的来源基因上计算。完整输出防止方法选择容易关系，封存来源过滤防止无检验证据的来源混入指标。

- [ ] **Step 4: 实现配对分组区间和失败汇总**

在同一文件追加：

```python
def paired_cluster_interval(
    table: pd.DataFrame,
    *,
    candidate_column: str = "candidate",
    baseline_column: str = "baseline",
    repeats: int = 10000,
    seed: int = 11,
) -> dict[str, float | int]:
    required = {"seed", "condition", candidate_column, baseline_column}
    if required - set(table.columns):
        raise TaskCAggregationError("paired table is missing seed, condition, or metrics")
    paired = table.dropna(subset=[candidate_column, baseline_column]).copy()
    differences = (
        paired[candidate_column].to_numpy(dtype=float)
        - paired[baseline_column].to_numpy(dtype=float)
    )
    if not len(differences):
        raise TaskCAggregationError("paired interval needs completed matched runs")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(differences), size=len(differences))
        estimates[repeat] = float(np.mean(differences[indices]))
    return {
        "estimate": float(np.mean(differences)),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
        "cluster_count": int(len(differences)),
        "bootstrap_repeats": int(repeats),
    }


def aggregate_task_c_runs(run_directories: Sequence[str | Path]) -> dict[str, object]:
    statuses = []
    completed = 0
    for directory in run_directories:
        run_dir = Path(directory)
        status_path = run_dir / "method_status.json"
        if not status_path.exists():
            raise TaskCAggregationError(f"missing method_status.json in {run_dir}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        statuses.append(status)
        if status.get("status") == "passed_real_rehearsal":
            if not (run_dir / "metrics.json").exists():
                raise TaskCAggregationError(f"completed run lacks metrics.json in {run_dir}")
            completed += 1
    counts = Counter(str(status["status"]) for status in statuses)
    return {
        "attempted_run_count": len(statuses),
        "completed_run_count": completed,
        "failed_or_unavailable_count": len(statuses) - completed,
        "status_counts": dict(sorted(counts.items())),
        "runs": statuses,
    }
```

- [ ] **Step 5: 运行测试并提交**

```bash
pytest tests/test_task_c_aggregation.py -q -p no:cacheprovider
git add src/evaluation/task_c_aggregation.py tests/test_task_c_aggregation.py
git commit -m "feat: aggregate Task C rehearsal evidence"
```

Expected: 3 tests PASS。

### Task 4: 隔离封存评分和 CausalBench 补充指标

**Files:**
- Create: `scripts/task_c_workers/causalbench_evaluation_worker.py`
- Modify: `src/evaluation/task_c_rehearsal.py`
- Modify: `tests/test_task_c_rehearsal.py`

- [ ] **Step 1: 写封存边界和帮助信息测试**

在 `tests/test_task_c_rehearsal.py` 追加：

```python
import subprocess
import sys

from src.evaluation.task_c_rehearsal import validate_private_scoring_command


def test_method_command_cannot_receive_private_paths(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    try:
        validate_private_scoring_command(
            ["python", "worker.py", "--input", str(private_root / "test.npz")],
            private_root=private_root,
        )
    except ValueError as exc:
        assert "private scoring path" in str(exc)
    else:
        raise AssertionError("private path must be rejected")


def test_official_evaluation_worker_exposes_help_without_external_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/task_c_workers/causalbench_evaluation_worker.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "封存" in completed.stdout
    assert "--prediction-csv" in completed.stdout
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest tests/test_task_c_rehearsal.py -q -p no:cacheprovider
```

Expected: FAIL，缺少路径核验和评分进程。

- [ ] **Step 3: 加入封存路径核验**

在 `src/evaluation/task_c_rehearsal.py` 追加：

```python
def validate_private_scoring_command(
    command: Sequence[str],
    *,
    private_root: str | Path,
) -> None:
    private = Path(private_root).resolve()
    for argument in command:
        candidate = Path(str(argument)).expanduser()
        if not candidate.is_absolute():
            continue
        resolved = candidate.resolve()
        if resolved == private or private in resolved.parents:
            raise TaskCRehearsalError("method command contains a private scoring path")
```

- [ ] **Step 4: 实现官方补充评分工作进程**

创建 `scripts/task_c_workers/causalbench_evaluation_worker.py`。参数解析必须在导入 CausalBench 之前完成，使主环境可以显示帮助；实际评分时读取 `--prediction-csv`、`--heldout-npz`、`--output-json`、`--seed`，再从固定版本导入 `causalscbench.evaluation.statistical_evaluation.Evaluator`。生物参考关系由 HyperSCA 主评分器单独处理，不传给官方干预分布评分器。

核心入口写为：

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions = pd.read_csv(args.prediction_csv)
    expression, interventions, genes = load_three_array_npz(args.heldout_npz)
    eligible_sources = set(interventions) - {"non-targeting", "excluded"}
    ordered_edges = list(
        predictions[predictions["source"].isin(eligible_sources)].sort_values(
            ["score", "source", "target"],
            ascending=[False, True, True],
            kind="mergesort",
        )[["source", "target"]].itertuples(index=False, name=None)
    )
    from causalscbench.evaluation.statistical_evaluation import Evaluator

    evaluator = Evaluator(
        expression,
        interventions,
        genes,
    )
    raw_metrics = evaluator.evaluate_network(
        ordered_edges,
        max_path_length=1,
        check_false_omission_rate=False,
        omission_estimation_size=0,
        seed=args.seed,
    )
    payload = {
        "schema_version": "1.0",
        "status": "supplementary_official_metrics",
        "seed": args.seed,
        "eligible_source_count": len(eligible_sources),
        "metrics": make_json_safe(raw_metrics),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0
```

该调用签名来自固定提交 `1a2143c` 的 `statistical_evaluation.py`。若返回结构不能转换为有限 JSON，状态必须是 `failed_invalid_output`，不得改变指标定义。

- [ ] **Step 5: 运行测试并提交**

```bash
pytest tests/test_task_c_rehearsal.py -q -p no:cacheprovider
git add src/evaluation/task_c_rehearsal.py scripts/task_c_workers/causalbench_evaluation_worker.py tests/test_task_c_rehearsal.py
git commit -m "feat: isolate Task C holdout scoring"
```

Expected: tests PASS；此时只验证边界，不伪造官方包运行。

### Task 5: 实现预演控制器和强制结果文件

**Files:**
- Modify: `src/evaluation/task_c_rehearsal.py`
- Create: `scripts/run_task_c_rehearsal.py`
- Create: `tests/test_task_c_rehearsal_cli.py`

- [ ] **Step 1: 写小型模拟闭环命令测试**

创建 `tests/test_task_c_rehearsal_cli.py`，在测试中生成两个三数组 NPZ、公开划分清单和两份参考 CSV，然后运行：

```python
completed = subprocess.run(
    [
        sys.executable,
        "scripts/run_task_c_rehearsal.py",
        "--profile",
        "connection",
        "--prepared-root",
        str(prepared_root),
        "--method-assets-root",
        str(tmp_path / "method_assets"),
        "--output-root",
        str(tmp_path / "results"),
        "--methods",
        "hypersca_c,mean_difference,random1000",
        "--synthetic-smoke",
    ],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
summary = json.loads(completed.stdout)
assert summary["claim_level"] == "workflow_validation_only"
assert summary["promotion_eligible"] is False
assert summary["profile"] == "connection"
assert summary["attempted_methods"] == [
    "hypersca_c",
    "mean_difference",
    "random1000",
]
```

测试还应遍历每个成功运行目录，断言五个强制文件和 `method_status.json`、`resource_usage.json`、`environment_manifest.json` 均存在；任意成功运行若缺文件必须使命令非零退出。

- [ ] **Step 2: 运行测试并确认命令缺失**

```bash
pytest tests/test_task_c_rehearsal_cli.py -q -p no:cacheprovider
```

Expected: FAIL。

- [ ] **Step 3: 实现运行身份、不可覆盖和强制文件核验**

在 `src/evaluation/task_c_rehearsal.py` 追加：

```python
def build_rehearsal_run_id(
    *, profile: str, condition: str, method_id: str, seed: int
) -> str:
    safe = [profile, condition, method_id, f"seed-{seed}"]
    if any(not value or "/" in value or ".." in value for value in safe):
        raise TaskCRehearsalError("run identity contains unsafe text")
    return "__".join(safe)


def validate_required_run_artifacts(
    run_dir: str | Path, required_artifacts: Sequence[str]
) -> None:
    destination = Path(run_dir)
    missing = [name for name in required_artifacts if not (destination / name).is_file()]
    extras = [
        "method_status.json",
        "resource_usage.json",
        "environment_manifest.json",
    ]
    missing.extend(name for name in extras if not (destination / name).is_file())
    if missing:
        raise TaskCRehearsalError(f"run is missing required artifacts: {sorted(set(missing))}")
```

- [ ] **Step 4: 实现薄控制命令**

创建 `scripts/run_task_c_rehearsal.py`，参数固定为：

```python
parser.add_argument("--profile", choices=["connection", "comprehensive"], required=True)
parser.add_argument("--prepared-root", type=Path, required=True)
parser.add_argument("--prepared-identity-sha256")
parser.add_argument("--method-assets-root", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--methods", required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume-token")
parser.add_argument("--synthetic-smoke", action="store_true")
```

正式预演的 `--prepared-identity-sha256` 必须来自准备命令标准输出中 seed 11
对应的 `materialization_identity_sha256`，并由调用者保存在准备目录之外；从当前
`public_manifest.json` 重新计算的本地散列不是外部身份锚。`--resume-token` 只能与
`--resume` 一起使用，且必须来自初次运行标准输出并独立保存，不能从控制器目录内
重新取值。

主流程必须按下列固定顺序执行：

1. 若请求 `--resume`，先要求输出根目录已经存在；缺失时立即拒绝，不创建目录、
   不核对输入，也不启动任何方法；
2. 核对数据、划分、参考关系和方法登记指纹；
3. 只从公开学习/调节文件选择基因和细胞；
4. 依次建立 `within_k562`、`within_rpe1`、`k562_to_rpe1`、`rpe1_to_k562` 四种条件；
5. 细胞系内部最终拟合使用公开 `refit.npz`；跨环境时 HyperSCA-C 接收来源环境 `source_refit.npz` 和目标环境 `target_adapt_refit.npz` 两个上下文，其他方法接收由 `center_and_merge_allowed_contexts` 生成的临时合并 NPZ，并在其中保留 `environment_labels` 供支持环境标签的方法读取；
6. 连接检查为 HyperSCA-C 和 Mean Difference 各建立两个调节试验，调用 `select_task_c_configuration.py` 核验“调节文件选择、公开 refit 重拟合、封存评分”边界；全方法预演只测登记默认配置，不能把单试验预演描述为正式参数公平比较；
7. 向每个方法只传上述允许的公开文件，并把合并方式和中心化参数写入 `input_summary.json`；
8. 将原始输出转换为完整有向关系表；
9. 在独立评分步骤读取封存文件和两类参考关系；
10. 对 HyperSCA-C 和 Mean Difference 各运行 20 次两类零效应检查；
11. 写强制文件、失败状态和资源记录；
12. 将所有预演 `promotion_decision.json` 固定为：

```json
{
  "schema_version": "1.0",
  "status": "workflow_validation_only",
  "claim_level": "workflow_validation_only",
  "promotion_eligible": false,
  "reason": "Single-seed reduced-data rehearsal validates execution and resource readiness only."
}
```

任何已有运行目录若身份或输入指纹不同则拒绝覆盖；`--resume` 只复用身份、实际
文件、重建摘要和外部 `--resume-token` 全部一致的完整运行。没有独立保存初次标准
输出中的 token 时不得恢复，必须重新运行。

- [ ] **Step 5: 运行模拟闭环并提交**

模拟闭环测试也必须把初次命令的完整标准输出排他写到输出根目录之外，再从这份
外部记录读取 `resume_token`；不能直接从控制器记录取令牌，也不能用内部令牌替代
调用者独立保存的副本。

```bash
pytest tests/test_task_c_rehearsal_cli.py -q -p no:cacheprovider
git add src/evaluation/task_c_rehearsal.py scripts/run_task_c_rehearsal.py tests/test_task_c_rehearsal_cli.py
git commit -m "feat: orchestrate Task C real-data rehearsal"
```

Expected: test PASS，模拟结果仍为 `workflow_validation_only`。

### Task 6: 实现只读汇总和全量作业资源草案

**Files:**
- Modify: `src/evaluation/task_c_aggregation.py`
- Modify: `src/evaluation/task_c_rehearsal.py`
- Create: `scripts/summarize_task_c_rehearsal.py`
- Modify: `tests/test_task_c_aggregation.py`

- [ ] **Step 1: 写准入规则和作业数测试**

在 `tests/test_task_c_aggregation.py` 追加：

```python
from src.evaluation.task_c_aggregation import build_full_run_draft, evaluate_rehearsal_readiness


def test_rehearsal_readiness_requires_core_and_interventional_method() -> None:
    registered = {
        "hypersca_c", "mean_difference", "random1000", "grnboost", "pc", "ges",
        "gies", "gsp", "igsp", "notears_linear", "dcdi_g", "dcdi_dsf",
        "dcdfg_linear", "dcdfg_mlp", "sortnregress", "guanlab_psgrn",
        "betterboost", "sparse_rc", "catran",
    }
    statuses = {method: "failed_timeout" for method in registered}
    for method in ("betterboost", "sparse_rc", "catran"):
        statuses[method] = "official_assets_unavailable"
    statuses.update(
        {
            method: "passed_real_rehearsal"
            for method in (
            "hypersca_c",
            "mean_difference",
            "random1000",
            "grnboost",
            "pc",
            "notears_linear",
            "gies",
        )
        }
    )
    decision = evaluate_rehearsal_readiness(
        statuses,
        data_checks_passed=True,
        five_splits_reproduced=True,
        null_controls_passed=True,
        tuning_boundary_passed=True,
        project_tests_passed=True,
    )
    assert decision["ready_for_full_run"] is True
    statuses["notears_linear"] = "failed_resource_limit"
    assert evaluate_rehearsal_readiness(
        statuses,
        data_checks_passed=True,
        five_splits_reproduced=True,
        null_controls_passed=True,
        tuning_boundary_passed=True,
        project_tests_passed=True,
    )["ready_for_full_run"] is False


def test_full_run_draft_has_five_seeds_and_never_starts_jobs() -> None:
    draft = build_full_run_draft(
        runnable_methods=["hypersca_c", "mean_difference", "gies"],
        conditions=["within_k562", "within_rpe1", "k562_to_rpe1", "rpe1_to_k562"],
        seeds=[11, 23, 47, 71, 97],
        median_runtime_seconds={"hypersca_c": 100.0, "mean_difference": 2.0, "gies": 20.0},
        maximum_tuning_trials=20,
    )
    assert draft["job_count"] == 1260
    assert draft["tuning_job_count"] == 1200
    assert draft["final_fit_job_count"] == 60
    assert draft["authorization_status"] == "not_authorized_to_start"
```

- [ ] **Step 2: 实现准入判断和资源估计**

在 `src/evaluation/task_c_aggregation.py` 追加：

```python
CORE_METHODS = {
    "hypersca_c",
    "mean_difference",
    "random1000",
    "grnboost",
    "pc",
    "notears_linear",
}
REGISTERED_METHODS = {
    "hypersca_c", "mean_difference", "random1000", "grnboost", "pc", "ges",
    "gies", "gsp", "igsp", "notears_linear", "dcdi_g", "dcdi_dsf",
    "dcdfg_linear", "dcdfg_mlp", "sortnregress", "guanlab_psgrn",
    "betterboost", "sparse_rc", "catran",
}
INTERVENTIONAL_METHODS = {
    "gies",
    "igsp",
    "dcdi_g",
    "dcdi_dsf",
    "dcdfg_linear",
    "dcdfg_mlp",
    "guanlab_psgrn",
}


def evaluate_rehearsal_readiness(
    method_statuses: dict[str, str],
    *,
    data_checks_passed: bool,
    five_splits_reproduced: bool,
    null_controls_passed: bool,
    tuning_boundary_passed: bool,
    project_tests_passed: bool,
) -> dict[str, object]:
    passed = {
        method_id
        for method_id, status in method_statuses.items()
        if status == "passed_real_rehearsal"
    }
    checks = {
        "data_checks": data_checks_passed,
        "five_splits_reproduced": five_splits_reproduced,
        "core_methods": CORE_METHODS <= passed,
        "interventional_method": bool(INTERVENTIONAL_METHODS & passed),
        "all_methods_classified": (
            set(method_statuses) == REGISTERED_METHODS
            and all(status != "not_attempted" for status in method_statuses.values())
        ),
        "null_controls": null_controls_passed,
        "tuning_boundary": tuning_boundary_passed,
        "project_tests": project_tests_passed,
    }
    return {
        "ready_for_full_run": all(checks.values()),
        "checks": checks,
        "claim_level": "workflow_validation_only",
        "authorization_status": "not_authorized_to_start",
    }


def build_full_run_draft(
    *,
    runnable_methods: Sequence[str],
    conditions: Sequence[str],
    seeds: Sequence[int],
    median_runtime_seconds: dict[str, float],
    maximum_tuning_trials: int,
) -> dict[str, object]:
    tuning_jobs = [
        {
            "phase": "tune",
            "trial_index": trial_index,
            "method_id": method,
            "condition": condition,
            "seed": int(seed),
        }
        for method in sorted(runnable_methods)
        for condition in conditions
        for seed in seeds
        for trial_index in range(maximum_tuning_trials)
    ]
    final_jobs = [
        {
            "phase": "refit_and_private_evaluation",
            "method_id": method,
            "condition": condition,
            "seed": int(seed),
        }
        for method in sorted(runnable_methods)
        for condition in conditions
        for seed in seeds
    ]
    jobs = tuning_jobs + final_jobs
    estimated_seconds = sum(
        float(median_runtime_seconds[method]) * (maximum_tuning_trials + 1)
        for method in runnable_methods
        for _condition in conditions
        for _seed in seeds
    )
    return {
        "job_count": len(jobs),
        "tuning_job_count": len(tuning_jobs),
        "final_fit_job_count": len(final_jobs),
        "maximum_tuning_trials_per_method": maximum_tuning_trials,
        "estimated_serial_runtime_seconds": estimated_seconds,
        "jobs": jobs,
        "authorization_status": "not_authorized_to_start",
    }
```

- [ ] **Step 3: 实现只读汇总命令**

创建 `scripts/summarize_task_c_rehearsal.py`，接收 `--rehearsal-root`、`--output-dir`
和必填的 `--resume-token`。令牌必须来自初次预演标准输出在结果目录之外独立保存的
记录；汇总命令不得从 `controller_manifest.json` 读取令牌作为预期值。它必须：

- 读取而不修改每个方法运行目录；
- 对完整运行计算指标，对失败运行只汇总状态；
- 写 `rehearsal_summary.json`、`method_compatibility.csv`、`resource_estimate.json`、`full_run_jobs_draft.json`；
- 不调用 `evaluate_promotion()`，不启动草案中的任何命令；
- 标准输出只给出 `ready_for_full_run`、阻断项和四个汇总文件路径。
- 以外部令牌重新核对全部证据，并限制汇总期间保留的运行 JSON 总字节数；
- 把 20 次调节试验明确标作保守的 `worst_case_upper_bound`，不把预演说成正式资源实测。

- [ ] **Step 4: 运行测试并提交**

```bash
pytest tests/test_task_c_aggregation.py -q -p no:cacheprovider
git add src/evaluation/task_c_aggregation.py src/evaluation/task_c_rehearsal.py scripts/summarize_task_c_rehearsal.py tests/test_task_c_aggregation.py
git commit -m "feat: summarize Task C rehearsal readiness"
```

Expected: tests PASS。

### Task 7: 建立本机环境并完成 64 基因真实连接检查

**Files:**
- Generate outside Git: `/home/a/Data/HyperSCA_external/task_c/`
- Generate ignored results: `results/benchmarks/task_c/connection/`

- [ ] **Step 1: 建立主环境和隔离环境**

```bash
conda create -n hypersca python=3.10 -y
conda run -n hypersca python -m pip install -r requirements.txt
conda env create -f envs/task_c/causalbench.yml
conda env create -f envs/task_c/psgrn.yml
conda run -n hypersca python scripts/validate_env.py
```

若同名环境已存在，用 `conda env update -n hypersca-task-c-causalbench -f envs/task_c/causalbench.yml --prune` 和对应 PSGRN 命令；不得删除用户已有环境。

- [ ] **Step 2: 获取官方数据、参考关系和方法资产**

```bash
set -euo pipefail
export TASK_C_DATA_ROOT=/home/a/Data/HyperSCA_external/task_c
mkdir -p "$TASK_C_DATA_ROOT/raw" "$TASK_C_DATA_ROOT/prepared" "$TASK_C_DATA_ROOT/method_assets"
conda run -n hypersca-task-c-causalbench python scripts/export_causalbench_data.py --data-dir "$TASK_C_DATA_ROOT/raw"
export TASK_C_PREPARED_IDENTITY_RECORD="$TASK_C_DATA_ROOT/prepared_identity_summary.json"
TASK_C_PREPARED_IDENTITY_TMP="$(mktemp "$TASK_C_DATA_ROOT/.prepared_identity_summary.XXXXXX")"
if conda run --no-capture-output -n hypersca python scripts/prepare_task_c_data.py \
    --k562-npz "$TASK_C_DATA_ROOT/raw/dataset_k562.npz" \
    --rpe1-npz "$TASK_C_DATA_ROOT/raw/dataset_rpe1.npz" \
    --k562-pooled-reference "$TASK_C_DATA_ROOT/raw/reference_k562_pooled.csv" \
    --k562-chipseq-reference "$TASK_C_DATA_ROOT/raw/reference_k562_chipseq.csv" \
    --rpe1-pooled-reference "$TASK_C_DATA_ROOT/raw/reference_rpe1_pooled.csv" \
    --rpe1-chipseq-reference "$TASK_C_DATA_ROOT/raw/reference_rpe1_chipseq.csv" \
    --output-dir "$TASK_C_DATA_ROOT/prepared" \
    > "$TASK_C_PREPARED_IDENTITY_TMP"; then
  chmod 0400 "$TASK_C_PREPARED_IDENTITY_TMP"
  if ! ln -T -- "$TASK_C_PREPARED_IDENTITY_TMP" "$TASK_C_PREPARED_IDENTITY_RECORD"; then
    rm -f "$TASK_C_PREPARED_IDENTITY_TMP"
    echo "prepared identity record already exists; refusing to replace it" >&2
    exit 1
  fi
  rm -f "$TASK_C_PREPARED_IDENTITY_TMP"
else
  rm -f "$TASK_C_PREPARED_IDENTITY_TMP"
  exit 1
fi
conda run -n hypersca python scripts/bootstrap_task_c_methods.py --cache-root "$TASK_C_DATA_ROOT/method_assets"
```

`prepared_identity_summary.json` 位于 `prepared/` 之外，并且在准备命令成功后才用同目录
硬链接排他发布；目标已存在时 `ln` 会失败，不会替换旧记录。随后删除临时文件名，
最终记录只保留一个文件名。它是调用者独立保留的身份记录，不是数字签名。核对
`raw/export_manifest.json` 的下载时间、来源和官方提交，以及 `prepared/provenance/` 中
两个表达缓存、汇总生物关系和 ChIP 有向关系的许可与 SHA-256。

- [ ] **Step 3: 重现五份划分并运行连接检查**

```bash
set -euo pipefail
for TASK_C_SEED in 11 23 47 71 97; do
  test -f "$TASK_C_DATA_ROOT/prepared/splits/seed_$TASK_C_SEED/public_manifest.json"
done
TASK_C_PREPARED_IDENTITY_SHA256="$(
  conda run --no-capture-output -n hypersca python - "$TASK_C_PREPARED_IDENTITY_RECORD" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
matches = [entry for entry in summary.get("splits", []) if entry.get("seed") == 11]
if len(matches) != 1:
    raise SystemExit("prepared identity record must contain exactly one seed 11 split")
value = matches[0].get("materialization_identity_sha256")
if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
    raise SystemExit("seed 11 prepared identity fingerprint is malformed")
print(value)
PY
)"
export TASK_C_PREPARED_IDENTITY_SHA256
export TASK_C_CONNECTION_STDOUT_RECORD="$TASK_C_DATA_ROOT/connection_rehearsal_initial_stdout.json"
TASK_C_CONNECTION_STDOUT_TMP="$(mktemp "$TASK_C_DATA_ROOT/.connection_rehearsal_stdout.XXXXXX")"
if conda run --no-capture-output -n hypersca python scripts/run_task_c_rehearsal.py \
    --profile connection \
    --prepared-root "$TASK_C_DATA_ROOT/prepared/splits/seed_11" \
    --prepared-identity-sha256 "$TASK_C_PREPARED_IDENTITY_SHA256" \
    --method-assets-root "$TASK_C_DATA_ROOT/method_assets" \
    --output-root results/benchmarks/task_c/connection \
    --methods hypersca_c,mean_difference,random1000,grnboost,pc,notears_linear,gies \
    > "$TASK_C_CONNECTION_STDOUT_TMP"; then
  chmod 0400 "$TASK_C_CONNECTION_STDOUT_TMP"
  if ! ln -T -- "$TASK_C_CONNECTION_STDOUT_TMP" "$TASK_C_CONNECTION_STDOUT_RECORD"; then
    rm -f "$TASK_C_CONNECTION_STDOUT_TMP"
    echo "connection stdout record already exists; refusing to replace it" >&2
    exit 1
  fi
  rm -f "$TASK_C_CONNECTION_STDOUT_TMP"
else
  rm -f "$TASK_C_CONNECTION_STDOUT_TMP"
  exit 1
fi
```

Expected: 64 个共同基因、每环境不超过 2,000 个细胞、种子 11；所有预演决定保持 `workflow_validation_only`。
初次标准输出只在命令成功后排他发布到结果目录之外；目标已存在时停止，不能覆盖。

- [ ] **Step 4: 检查连接结果而不解读性能优劣**

```bash
export TASK_C_DATA_ROOT=/home/a/Data/HyperSCA_external/task_c
export TASK_C_CONNECTION_STDOUT_RECORD="$TASK_C_DATA_ROOT/connection_rehearsal_initial_stdout.json"
TASK_C_CONNECTION_RESUME_TOKEN="$(
  conda run --no-capture-output -n hypersca python - "$TASK_C_CONNECTION_STDOUT_RECORD" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
value = record.get("resume_token")
if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
    raise SystemExit("independently saved connection resume token is malformed")
print(value)
PY
)"
conda run -n hypersca python scripts/summarize_task_c_rehearsal.py \
  --rehearsal-root results/benchmarks/task_c/connection \
  --output-dir results/benchmarks/task_c/connection_summary \
  --resume-token "$TASK_C_CONNECTION_RESUME_TOKEN"
```

只有数据、路径隔离、结果完整性或核心方法连接失败时才修复代码；不得根据 AP 高低调整已冻结基因、划分、评分或零效应阈值。

### Task 8: 完成 256 基因全方法真实预演并写报告

**Files:**
- Generate ignored results: `results/benchmarks/task_c/comprehensive/`
- Create: `docs/research/task_c_rehearsal_v1.md`
- Modify: `docs/research/task_c_mean_difference_baseline_v1.md`
- Modify: `docs/technical_roadmap.md`

- [ ] **Step 1: 尝试全部登记方法**

```bash
set -euo pipefail
export TASK_C_DATA_ROOT=/home/a/Data/HyperSCA_external/task_c
export TASK_C_PREPARED_IDENTITY_RECORD="$TASK_C_DATA_ROOT/prepared_identity_summary.json"
TASK_C_PREPARED_IDENTITY_SHA256="$(
  conda run --no-capture-output -n hypersca python - "$TASK_C_PREPARED_IDENTITY_RECORD" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
matches = [entry for entry in summary.get("splits", []) if entry.get("seed") == 11]
if len(matches) != 1:
    raise SystemExit("prepared identity record must contain exactly one seed 11 split")
value = matches[0].get("materialization_identity_sha256")
if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
    raise SystemExit("seed 11 prepared identity fingerprint is malformed")
print(value)
PY
)"
export TASK_C_PREPARED_IDENTITY_SHA256
export TASK_C_COMPREHENSIVE_STDOUT_RECORD="$TASK_C_DATA_ROOT/comprehensive_rehearsal_initial_stdout.json"
TASK_C_COMPREHENSIVE_STDOUT_TMP="$(mktemp "$TASK_C_DATA_ROOT/.comprehensive_rehearsal_stdout.XXXXXX")"
if conda run --no-capture-output -n hypersca python scripts/run_task_c_rehearsal.py \
    --profile comprehensive \
    --prepared-root "$TASK_C_DATA_ROOT/prepared/splits/seed_11" \
    --prepared-identity-sha256 "$TASK_C_PREPARED_IDENTITY_SHA256" \
    --method-assets-root "$TASK_C_DATA_ROOT/method_assets" \
    --output-root results/benchmarks/task_c/comprehensive \
    --methods hypersca_c,mean_difference,random1000,grnboost,pc,ges,gies,gsp,igsp,notears_linear,dcdi_g,dcdi_dsf,dcdfg_linear,dcdfg_mlp,sortnregress,guanlab_psgrn,betterboost,sparse_rc,catran \
    > "$TASK_C_COMPREHENSIVE_STDOUT_TMP"; then
  chmod 0400 "$TASK_C_COMPREHENSIVE_STDOUT_TMP"
  if ! ln -T -- "$TASK_C_COMPREHENSIVE_STDOUT_TMP" "$TASK_C_COMPREHENSIVE_STDOUT_RECORD"; then
    rm -f "$TASK_C_COMPREHENSIVE_STDOUT_TMP"
    echo "comprehensive stdout record already exists; refusing to replace it" >&2
    exit 1
  fi
  rm -f "$TASK_C_COMPREHENSIVE_STDOUT_TMP"
else
  rm -f "$TASK_C_COMPREHENSIVE_STDOUT_TMP"
  exit 1
fi
```

Expected: 每个方法得到六种允许状态之一；BetterBoost、SparseRC 和 CATRAN 若仍无官方可执行资产，记录 `official_assets_unavailable`，不编写替代实现。
初次标准输出必须独立留在结果目录之外，后续汇总只接受这里保存的令牌。

- [ ] **Step 2: 运行全部预先登记消融**

```bash
conda run -n hypersca python scripts/run_hypersca_c_ablations.py \
  --context "k562=$TASK_C_DATA_ROOT/prepared/splits/seed_11/within/k562/refit.npz" \
  --context "rpe1=$TASK_C_DATA_ROOT/prepared/splits/seed_11/within/rpe1/refit.npz" \
  --config configs/hypersca_c_v1.json \
  --ablation-registry configs/hypersca_c_ablations_v1.json \
  --output-root results/benchmarks/task_c/comprehensive/ablations \
  --seed 11 \
  --device cuda
```

Expected: 主版本、共享关系、分开环境、不使用干预、无稳定性修正和两种循环限制均得到完成或准确失败状态。若尚未登记不与评分参考重叠的知识先验，`prior_on_secondary` 保留 `no_nonoverlapping_preregistered_prior` 阻断；绝不能把 CausalBench 评分关系用作模型先验。

- [ ] **Step 3: 生成资源和阻断汇总**

```bash
export TASK_C_DATA_ROOT=/home/a/Data/HyperSCA_external/task_c
export TASK_C_COMPREHENSIVE_STDOUT_RECORD="$TASK_C_DATA_ROOT/comprehensive_rehearsal_initial_stdout.json"
TASK_C_COMPREHENSIVE_RESUME_TOKEN="$(
  conda run --no-capture-output -n hypersca python - "$TASK_C_COMPREHENSIVE_STDOUT_RECORD" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
value = record.get("resume_token")
if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
    raise SystemExit("independently saved comprehensive resume token is malformed")
print(value)
PY
)"
conda run -n hypersca python scripts/summarize_task_c_rehearsal.py \
  --rehearsal-root results/benchmarks/task_c/comprehensive \
  --output-dir results/benchmarks/task_c/comprehensive_summary \
  --resume-token "$TASK_C_COMPREHENSIVE_RESUME_TOKEN"
```

核对 19 个方法和八项消融均进入兼容性记录，并确认作业草案为 `not_authorized_to_start`。

- [ ] **Step 4: 写面向生物医学读者的预演报告**

创建 `docs/research/task_c_rehearsal_v1.md`，只从生成的清单摘录：

- 数据来源、许可、指纹和共同基因/细胞数；
- 64 与 256 两级预演是否完成；
- 每个方法的准确状态、时间、内存、显存和磁盘；
- 两类零效应是否端到端完成；
- 核心准入条件逐项结果；
- 正式五随机种子作业数和资源区间；
- 明确结论“流程验证，不是性能结论”；
- 若未就绪，列出可复查阻断，不淡化失败。

同步修改均值变化文档，说明主 AP 使用 CausalBench 汇总生物关系的双向展开；方向正确率只在 ChIP 文件与目标环境匹配的 K562 条件计算，RPE1/HepG2 不匹配条件报告为不可解释。修改路线图只写 `ready_for_full_run` 状态，不写算法领先。

- [ ] **Step 5: 提交预演报告，不提交结果缓存**

```bash
git status --short
git add docs/research/task_c_rehearsal_v1.md docs/research/task_c_mean_difference_baseline_v1.md docs/technical_roadmap.md
git commit -m "docs: report Task C real-data rehearsal"
```

Expected: `results/`、外部数据和方法资产不进入提交。

### Task 9: 全项目回归和最终边界核验

**Files:**
- Modify only if a regression is caused by this implementation.

- [ ] **Step 1: 运行任务 C 和通俗术语专项测试**

```bash
conda run -n hypersca pytest \
  tests/test_task_c_benchmark.py \
  tests/test_benchmark_contract.py \
  tests/test_task_c_data.py \
  tests/test_task_c_data_cli.py \
  tests/test_hypersca_c.py \
  tests/test_hypersca_c_stability.py \
  tests/test_hypersca_c_cli.py \
  tests/test_task_c_method_registry.py \
  tests/test_task_c_predictions.py \
  tests/test_task_c_runtime.py \
  tests/test_task_c_tuning.py \
  tests/test_task_c_external_workers.py \
  tests/test_task_c_rehearsal.py \
  tests/test_task_c_null_controls.py \
  tests/test_task_c_aggregation.py \
  tests/test_task_c_rehearsal_cli.py \
  tests/test_plain_language_cli.py \
  -q -p no:cacheprovider
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行完整回归**

```bash
conda run -n hypersca pytest tests -q -p no:cacheprovider
```

Expected: 全部 PASS；空间分析、靶点发现和现有因果流程无退化。

- [ ] **Step 3: 核验没有误启动或误晋级**

```bash
rg -n '"status": "promoted"|benchmark_supported_candidate' \
  results/benchmarks/task_c/connection \
  results/benchmarks/task_c/comprehensive
```

Expected: 无匹配。若有匹配，预演失败并停止，不发布报告。

- [ ] **Step 4: 核验版本记录范围并提交剩余修正**

```bash
git status --short
git diff --check
git diff --stat origin/main...HEAD
```

Expected: 只包含源码、测试、共享配置和文档；不包含数据、外部方法副本、权重或生成结果。若有为回归测试所需的修正：

```bash
git add src scripts tests configs envs docs .gitignore
git commit -m "fix: close Task C rehearsal regressions"
```

## 停止点

完成本计划后只交付以下结论之一：

- **任务 C 已具备真实全量比较条件。**
- **核心比较已具备实测条件，全面比较仍有明确缺口。**
- **任务 C 尚未具备全量比较条件，并附可复查阻断。**

不得自动执行 `full_run_jobs_draft.json`。正式五随机种子全量作业需要用户另行确认。
