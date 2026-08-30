# HyperSCA-C Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个以细胞为观测、以基因为变量、能够联合学习共享关系和细胞环境差异的线性干预结构模型，并输出稳定性和暂不判断信息。

**Architecture:** HyperSCA-C 使用 `A_context = A_shared + Delta_context` 表示每个细胞环境的有向基因关系。被直接干预基因的自身解释误差会被遮挡，其他基因仍参与学习；重复分层抽样产生关系强度中位数、选择比例和方向一致性，三者按已确认公式形成排名分数。

**Tech Stack:** Python 3.10、PyTorch、NumPy、pandas、pytest、JSON

---

## 执行位置与前置条件

```bash
cd /home/a/.config/superpowers/worktrees/HyperSCA/real-data-readiness-design
```

先完成 `docs/superpowers/plans/2026-08-13-task-c-data-and-splits.md`。本计划不读取最终检验文件，也不计算正式评测指标。

## 文件结构

- Create: `configs/hypersca_c_v1.json` — 固定第一版优化、稳定性和暂不判断参数。
- Create: `configs/hypersca_c_ablations_v1.json` — 固定候选贡献消融，不根据封存结果删减。
- Create: `src/causal/hypersca_c.py` — 单次共享结构拟合与输入标准化。
- Create: `src/causal/hypersca_c_stability.py` — 重复抽样、关系分数和完整结果表。
- Create: `src/causal/hypersca_c_ablation.py` — 把固定消融映射为模型配置和允许的数据范围。
- Create: `scripts/run_hypersca_c.py` — 对被允许的训练文件运行候选方法。
- Create: `scripts/run_hypersca_c_ablations.py` — 运行全部固定消融并保留失败。
- Create: `tests/test_hypersca_c.py` — 数学目标和共享/特有关系测试。
- Create: `tests/test_hypersca_c_stability.py` — 稳定性、分数和暂不判断测试。
- Create: `tests/test_hypersca_c_cli.py` — 命令端到端测试。
- Create: `docs/research/hypersca_c_method_v1.md` — 方法假设、候选创新和限制。

### Task 1: 固定配置并实现干预遮挡损失

**Files:**
- Create: `configs/hypersca_c_v1.json`
- Create: `src/causal/hypersca_c.py`
- Create: `tests/test_hypersca_c.py`

- [ ] **Step 1: 写遮挡、自我关系和配置测试**

创建 `tests/test_hypersca_c.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.causal.hypersca_c import (
    HyperSCACConfig,
    HyperSCACError,
    build_intervention_mask,
    masked_sem_loss,
    zero_diagonal,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_frozen_and_has_no_prior_discount() -> None:
    payload = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    config = HyperSCACConfig.from_mapping(payload)
    assert config.schema_version == "1.0"
    assert config.prior_discount == 0.0
    assert config.enable_context_adjustments is True
    assert config.bootstrap_success_fraction == 0.8


def test_intervention_mask_excludes_only_the_direct_target() -> None:
    mask = build_intervention_mask(
        ["non-targeting", "B", "excluded"],
        ["A", "B", "C"],
    )
    assert mask.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]


def test_masked_loss_ignores_perturbed_gene_self_error() -> None:
    observed = torch.tensor([[1.0, 7.0]])
    prediction = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert float(masked_sem_loss(prediction, observed, mask)) == pytest.approx(0.0)


def test_zero_diagonal_removes_self_edges() -> None:
    matrix = torch.ones((3, 3))
    result = zero_diagonal(matrix)
    assert torch.diagonal(result).tolist() == [0.0, 0.0, 0.0]
    assert float(result.sum()) == pytest.approx(6.0)
```

- [ ] **Step 2: 运行测试并确认模块或配置缺失**

Run:

```bash
pytest tests/test_hypersca_c.py -q -p no:cacheprovider
```

Expected: collection FAIL，缺少模块或配置。

- [ ] **Step 3: 写冻结配置**

创建 `configs/hypersca_c_v1.json`：

```json
{
  "schema_version": "1.0",
  "learning_rate": 0.01,
  "maximum_epochs": 200,
  "early_stopping_patience": 10,
  "shared_l1": 0.001,
  "context_l1": 0.002,
  "acyclicity_weight": 0.01,
  "enable_context_adjustments": true,
  "prior_discount": 0.0,
  "selection_threshold": 0.0001,
  "bootstrap_repeats": 20,
  "bootstrap_success_fraction": 0.8,
  "minimum_source_variance": 1e-08,
  "control_label": "non-targeting",
  "excluded_label": "excluded"
}
```

- [ ] **Step 4: 实现配置、遮挡和基础损失**

创建 `src/causal/hypersca_c.py`，先写入：

```python
"""HyperSCA-C：跨细胞环境的线性干预基因关系模型。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional


class HyperSCACError(ValueError):
    """HyperSCA-C 输入或优化结果不满足固定规则。"""


@dataclass(frozen=True)
class HyperSCACConfig:
    schema_version: str
    learning_rate: float
    maximum_epochs: int
    early_stopping_patience: int
    shared_l1: float
    context_l1: float
    acyclicity_weight: float
    enable_context_adjustments: bool
    prior_discount: float
    selection_threshold: float
    bootstrap_repeats: int
    bootstrap_success_fraction: float
    minimum_source_variance: float
    control_label: str
    excluded_label: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "HyperSCACConfig":
        config = cls(**payload)
        if config.schema_version != "1.0":
            raise HyperSCACError("schema_version must be 1.0")
        if not 0.0 <= config.prior_discount < 1.0:
            raise HyperSCACError("prior_discount must be in [0, 1)")
        if not 0.0 < config.bootstrap_success_fraction <= 1.0:
            raise HyperSCACError("bootstrap_success_fraction must be in (0, 1]")
        if config.maximum_epochs < 1 or config.bootstrap_repeats < 1:
            raise HyperSCACError("epoch and bootstrap counts must be positive")
        return config


def build_intervention_mask(
    interventions: Sequence[str],
    gene_names: Sequence[str],
    *,
    excluded_label: str = "excluded",
) -> np.ndarray:
    gene_index = {str(gene): index for index, gene in enumerate(gene_names)}
    mask = np.ones((len(interventions), len(gene_names)), dtype=np.float32)
    for row, label_value in enumerate(interventions):
        label = str(label_value)
        if label == excluded_label:
            mask[row, :] = 0.0
        elif label in gene_index:
            mask[row, gene_index[label]] = 0.0
    if float(mask.sum()) == 0.0:
        raise HyperSCACError("intervention mask contains no usable expression values")
    return mask


def zero_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return matrix * (1.0 - identity)


def masked_sem_loss(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != observed.shape or mask.shape != observed.shape:
        raise HyperSCACError("prediction, observed, and mask shapes must match")
    denominator = mask.sum().clamp_min(1.0)
    return functional.smooth_l1_loss(
        prediction * mask,
        observed * mask,
        reduction="sum",
    ) / denominator
```

- [ ] **Step 5: 运行测试并提交**

Run:

```bash
pytest tests/test_hypersca_c.py -q -p no:cacheprovider
```

Expected: 4 tests PASS。

Commit:

```bash
git add configs/hypersca_c_v1.json src/causal/hypersca_c.py tests/test_hypersca_c.py
git commit -m "feat: define HyperSCA-C intervention loss"
```

### Task 2: 实现共享关系与环境特有调整的单次拟合

**Files:**
- Modify: `src/causal/hypersca_c.py`
- Modify: `tests/test_hypersca_c.py`

- [ ] **Step 1: 写控制细胞标准化和双环境恢复测试**

在 `tests/test_hypersca_c.py` 追加：

```python
from src.causal.hypersca_c import (
    HyperSCACContext,
    fit_hypersca_c_once,
    standardize_context,
)


def small_config() -> HyperSCACConfig:
    payload = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    payload.update(
        {
            "maximum_epochs": 120,
            "early_stopping_patience": 20,
            "acyclicity_weight": 0.001,
        }
    )
    return HyperSCACConfig.from_mapping(payload)


def test_standardization_uses_control_cells_only() -> None:
    expression = np.asarray([[0.0, 0.0], [2.0, 2.0], [100.0, 100.0]])
    scaled, center, scale = standardize_context(
        expression,
        ["non-targeting", "non-targeting", "A"],
        control_label="non-targeting",
    )
    assert center.tolist() == pytest.approx([1.0, 1.0])
    assert scaled[:2].mean(axis=0).tolist() == pytest.approx([0.0, 0.0])
    assert np.all(scale > 0.0)


def test_joint_fit_returns_shared_and_context_specific_matrices() -> None:
    rng = np.random.default_rng(11)
    genes = ("A", "B", "C")
    contexts = []
    for name, shift in (("k562", 0.0), ("rpe1", 0.25)):
        a = rng.normal(size=80)
        b = 1.5 * a + shift + rng.normal(scale=0.05, size=80)
        c = rng.normal(size=80)
        expression = np.column_stack([a, b, c]).astype(np.float32)
        contexts.append(
            HyperSCACContext(
                context_id=name,
                expression=expression,
                interventions=np.asarray(["non-targeting"] * 80),
                gene_names=genes,
            )
        )
    result = fit_hypersca_c_once(contexts, small_config(), seed=11, device="cpu")
    assert result.shared.shape == (3, 3)
    assert set(result.context_adjustments) == {"k562", "rpe1"}
    assert np.diag(result.shared).tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert np.isfinite(result.loss_history).all()
```

- [ ] **Step 2: 运行测试并确认新接口缺失**

Run:

```bash
pytest tests/test_hypersca_c.py -q -p no:cacheprovider
```

Expected: collection FAIL，缺少拟合接口。

- [ ] **Step 3: 实现上下文对象、标准化和无环软限制**

在 `src/causal/hypersca_c.py` 追加：

```python
@dataclass(frozen=True)
class HyperSCACContext:
    context_id: str
    expression: np.ndarray
    interventions: np.ndarray
    gene_names: tuple[str, ...]


@dataclass(frozen=True)
class HyperSCACFit:
    shared: np.ndarray
    context_adjustments: Mapping[str, np.ndarray]
    context_adjacencies: Mapping[str, np.ndarray]
    loss_history: np.ndarray
    converged: bool
    epochs_run: int


def standardize_context(
    expression: np.ndarray,
    interventions: Sequence[str],
    *,
    control_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(expression, dtype=np.float32)
    labels = np.asarray(interventions, dtype=str)
    controls = labels == control_label
    if int(controls.sum()) < 2:
        raise HyperSCACError("at least two control cells are required")
    center = values[controls].mean(axis=0)
    scale = values[controls].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (values - center) / scale, center, scale


def acyclicity_penalty(adjacency: torch.Tensor) -> torch.Tensor:
    dimension = adjacency.shape[0]
    return torch.trace(torch.matrix_exp(adjacency * adjacency)) - dimension


def _validate_contexts(contexts: Sequence[HyperSCACContext]) -> tuple[str, ...]:
    if not contexts:
        raise HyperSCACError("at least one context is required")
    genes = contexts[0].gene_names
    names = set()
    for context in contexts:
        if context.context_id in names:
            raise HyperSCACError("context identifiers must be unique")
        names.add(context.context_id)
        if context.gene_names != genes:
            raise HyperSCACError("all contexts must use the same ordered genes")
        if context.expression.shape != (
            len(context.interventions),
            len(context.gene_names),
        ):
            raise HyperSCACError("context expression shape is inconsistent")
        if not np.isfinite(context.expression).all():
            raise HyperSCACError("context expression values must be finite")
    return genes
```

- [ ] **Step 4: 实现单次联合优化**

继续在 `src/causal/hypersca_c.py` 追加：

```python
def fit_hypersca_c_once(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None = None,
) -> HyperSCACFit:
    genes = _validate_contexts(contexts)
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device)
    dimension = len(genes)
    shared_raw = torch.zeros(
        (dimension, dimension), device=target_device, requires_grad=True
    )
    delta_raw = {
        context.context_id: torch.zeros(
            (dimension, dimension),
            device=target_device,
            requires_grad=config.enable_context_adjustments,
        )
        for context in contexts
    }
    parameters = [shared_raw] + (
        list(delta_raw.values()) if config.enable_context_adjustments else []
    )
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    prepared = {}
    for context in contexts:
        scaled, _, _ = standardize_context(
            context.expression,
            context.interventions,
            control_label=config.control_label,
        )
        prepared[context.context_id] = (
            torch.as_tensor(scaled, dtype=torch.float32, device=target_device),
            torch.as_tensor(
                build_intervention_mask(
                    context.interventions,
                    genes,
                    excluded_label=config.excluded_label,
                ),
                device=target_device,
            ),
        )
    if prior_mask is None:
        prior_weights = torch.ones((dimension, dimension), device=target_device)
    else:
        if prior_mask.shape != (dimension, dimension):
            raise HyperSCACError("prior mask shape must match the gene network")
        prior_weights = 1.0 - config.prior_discount * torch.as_tensor(
            prior_mask, dtype=torch.float32, device=target_device
        )

    history: list[float] = []
    best = float("inf")
    stale_epochs = 0
    best_state: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
    for _ in range(config.maximum_epochs):
        optimizer.zero_grad()
        shared = zero_diagonal(shared_raw)
        total = config.shared_l1 * (shared.abs() * prior_weights).mean()
        for context in contexts:
            delta = zero_diagonal(delta_raw[context.context_id])
            adjacency = shared + delta
            values, mask = prepared[context.context_id]
            prediction = values @ adjacency
            total = total + masked_sem_loss(prediction, values, mask)
            total = total + config.context_l1 * delta.abs().mean()
            total = total + config.acyclicity_weight * acyclicity_penalty(adjacency)
        total.backward()
        optimizer.step()
        numeric = float(total.detach().cpu())
        if not np.isfinite(numeric):
            raise HyperSCACError("optimization produced a non-finite loss")
        history.append(numeric)
        if numeric < best - 1e-7:
            best = numeric
            stale_epochs = 0
            best_state = (
                zero_diagonal(shared_raw).detach().cpu().numpy().copy(),
                {
                    name: zero_diagonal(value).detach().cpu().numpy().copy()
                    for name, value in delta_raw.items()
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise HyperSCACError("optimization did not produce a usable state")
    shared_value, delta_values = best_state
    context_values = {
        name: shared_value + delta for name, delta in delta_values.items()
    }
    return HyperSCACFit(
        shared=shared_value,
        context_adjustments=delta_values,
        context_adjacencies=context_values,
        loss_history=np.asarray(history, dtype=float),
        converged=stale_epochs >= config.early_stopping_patience,
        epochs_run=len(history),
    )
```

- [ ] **Step 5: 运行测试并提交**

Run:

```bash
pytest tests/test_hypersca_c.py -q -p no:cacheprovider
```

Expected: all tests PASS。

Commit:

```bash
git add src/causal/hypersca_c.py tests/test_hypersca_c.py
git commit -m "feat: fit shared HyperSCA-C structures"
```

### Task 3: 实现分层重复抽样、固定分数和暂不判断

**Files:**
- Create: `src/causal/hypersca_c_stability.py`
- Create: `tests/test_hypersca_c_stability.py`

- [ ] **Step 1: 写分数公式、完整关系范围和失败重复测试**

创建 `tests/test_hypersca_c_stability.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext
from src.causal.hypersca_c_stability import (
    build_stability_table,
    stratified_bootstrap_context,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stability_score_uses_frozen_product_formula() -> None:
    genes = ("A", "B")
    matrices = [
        {"k562": np.asarray([[0.0, 2.0], [0.0, 0.0]])},
        {"k562": np.asarray([[0.0, 4.0], [0.0, 0.0]])},
        {"k562": np.asarray([[0.0, -3.0], [0.0, 0.0]])},
    ]
    table, summary = build_stability_table(
        matrices,
        genes,
        selection_threshold=0.1,
        requested_repeats=3,
        minimum_success_fraction=0.8,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    ab = table[(table["source"] == "A") & (table["target"] == "B")].iloc[0]
    assert ab["median_effect"] == pytest.approx(2.0)
    assert ab["selection_frequency"] == pytest.approx(1.0)
    assert ab["direction_agreement"] == pytest.approx(2.0 / 3.0)
    assert ab["context_consistency"] == pytest.approx(1.0)
    assert ab["score"] == pytest.approx(4.0 / 3.0)
    assert len(table) == 2
    assert summary["successful_repeats"] == 3


def test_insufficient_success_marks_every_source_as_abstained() -> None:
    table, summary = build_stability_table(
        [{"k562": np.zeros((2, 2))}],
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    assert table["abstained"].all()
    assert summary["coverage"] == 0.0


def test_bootstrap_preserves_intervention_group_sizes() -> None:
    context = HyperSCACContext(
        context_id="k562",
        expression=np.arange(18, dtype=np.float32).reshape(6, 3),
        interventions=np.asarray(["non-targeting", "non-targeting", "A", "A", "B", "B"]),
        gene_names=("A", "B", "C"),
    )
    sampled = stratified_bootstrap_context(context, np.random.default_rng(11))
    labels, counts = np.unique(sampled.interventions, return_counts=True)
    assert dict(zip(labels, counts)) == {"A": 2, "B": 2, "non-targeting": 2}
```

- [ ] **Step 2: 运行测试并确认模块缺失**

Run:

```bash
pytest tests/test_hypersca_c_stability.py -q -p no:cacheprovider
```

Expected: collection FAIL，缺少稳定性模块。

- [ ] **Step 3: 实现分层抽样和固定分数表**

创建 `src/causal/hypersca_c_stability.py`：

```python
"""HyperSCA-C 重复稳定性、完整关系表和暂不判断规则。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.causal.hypersca_c import (
    HyperSCACConfig,
    HyperSCACContext,
    HyperSCACError,
    fit_hypersca_c_once,
)


@dataclass(frozen=True)
class HyperSCAStabilityResult:
    predictions: pd.DataFrame
    summary: Mapping[str, object]
    failures: tuple[str, ...]


def stratified_bootstrap_context(
    context: HyperSCACContext,
    rng: np.random.Generator,
) -> HyperSCACContext:
    sampled_indices: list[int] = []
    for label in sorted(set(context.interventions.tolist())):
        indices = np.flatnonzero(context.interventions == label)
        sampled_indices.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
    selected = np.asarray(sampled_indices, dtype=int)
    return HyperSCACContext(
        context_id=context.context_id,
        expression=context.expression[selected],
        interventions=context.interventions[selected],
        gene_names=context.gene_names,
    )


def build_stability_table(
    context_matrices: Sequence[Mapping[str, np.ndarray]],
    gene_names: Sequence[str],
    *,
    selection_threshold: float,
    requested_repeats: int,
    minimum_success_fraction: float,
    source_variance: Mapping[str, float],
    minimum_source_variance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    genes = tuple(str(gene) for gene in gene_names)
    successful = len(context_matrices)
    repeat_success = successful / requested_repeats
    rows: list[dict[str, object]] = []
    contexts = sorted({name for result in context_matrices for name in result})
    for source_index, source in enumerate(genes):
        source_abstained = (
            repeat_success < minimum_success_fraction
            or float(source_variance.get(source, 0.0)) < minimum_source_variance
        )
        reason = ""
        if repeat_success < minimum_success_fraction:
            reason = "insufficient_successful_bootstraps"
        elif float(source_variance.get(source, 0.0)) < minimum_source_variance:
            reason = "source_has_no_control_variation"
        for target_index, target in enumerate(genes):
            if source == target:
                continue
            values_by_context = {
                context: np.asarray(
                    [
                        result[context][source_index, target_index]
                        for result in context_matrices
                        if context in result
                    ],
                    dtype=float,
                )
                for context in contexts
            }
            values = np.asarray(
                [
                    value
                    for context in contexts
                    for value in values_by_context[context]
                ],
                dtype=float,
            )
            if len(values) == 0:
                median_effect = 0.0
                selection_frequency = 0.0
                direction_agreement = 0.0
            else:
                median_effect = float(np.median(values))
                selected = np.abs(values) >= selection_threshold
                selection_frequency = float(selected.mean())
                if selected.any():
                    selected_signs = np.sign(values[selected])
                    positive = float((selected_signs > 0).mean())
                    negative = float((selected_signs < 0).mean())
                    direction_agreement = max(positive, negative)
                else:
                    direction_agreement = 0.0
            context_effects = {
                context: (
                    float(np.median(context_values)) if len(context_values) else 0.0
                )
                for context, context_values in values_by_context.items()
            }
            selected_context_signs = np.asarray(
                [
                    np.sign(effect)
                    for effect in context_effects.values()
                    if abs(effect) >= selection_threshold
                ],
                dtype=float,
            )
            if len(selected_context_signs):
                context_consistency = float(
                    max(
                        (selected_context_signs > 0).mean(),
                        (selected_context_signs < 0).mean(),
                    )
                )
            else:
                context_consistency = 0.0
            score = abs(median_effect) * selection_frequency * direction_agreement
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "effect": median_effect,
                    "median_effect": median_effect,
                    "direction": int(np.sign(median_effect)),
                    "selection_frequency": selection_frequency,
                    "direction_agreement": direction_agreement,
                    "context_consistency": context_consistency,
                    **{
                        f"effect_{context}": effect
                        for context, effect in context_effects.items()
                    },
                    "score": float(score),
                    "abstained": source_abstained,
                    "abstention_reason": reason,
                }
            )
    predictions = pd.DataFrame(rows).sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    source_status = predictions.groupby("source", sort=True)["abstained"].first()
    coverage = float((~source_status).mean()) if len(source_status) else 0.0
    return predictions.reset_index(drop=True), {
        "requested_repeats": requested_repeats,
        "successful_repeats": successful,
        "repeat_success_fraction": repeat_success,
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
        "score_formula": (
            "abs_median_effect_times_selection_frequency_times_direction_agreement"
        ),
    }


def fit_stable_hypersca_c(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None = None,
) -> HyperSCAStabilityResult:
    rng = np.random.default_rng(seed)
    matrices: list[Mapping[str, np.ndarray]] = []
    failures: list[str] = []
    for repeat in range(config.bootstrap_repeats):
        sampled = [stratified_bootstrap_context(context, rng) for context in contexts]
        try:
            fit = fit_hypersca_c_once(
                sampled,
                config,
                seed=seed + repeat,
                device=device,
                prior_mask=prior_mask,
            )
        except HyperSCACError as exc:
            failures.append(f"repeat_{repeat}:{exc}")
            continue
        matrices.append(fit.context_adjacencies)
    source_variance = {
        gene: float(
            np.mean(
                [
                    context.expression[
                        context.interventions == config.control_label, index
                    ].var()
                    for context in contexts
                ]
            )
        )
        for index, gene in enumerate(contexts[0].gene_names)
    }
    predictions, summary = build_stability_table(
        matrices,
        contexts[0].gene_names,
        selection_threshold=config.selection_threshold,
        requested_repeats=config.bootstrap_repeats,
        minimum_success_fraction=config.bootstrap_success_fraction,
        source_variance=source_variance,
        minimum_source_variance=config.minimum_source_variance,
    )
    return HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=tuple(failures),
    )
```

- [ ] **Step 4: 运行测试并提交**

Run:

```bash
pytest tests/test_hypersca_c_stability.py -q -p no:cacheprovider
```

Expected: 3 tests PASS。

Commit:

```bash
git add src/causal/hypersca_c_stability.py tests/test_hypersca_c_stability.py
git commit -m "feat: score HyperSCA-C edge stability"
```

### Task 4: 增加训练命令和可恢复的原始结果

**Files:**
- Create: `scripts/run_hypersca_c.py`
- Create: `tests/test_hypersca_c_cli.py`

- [ ] **Step 1: 写双上下文命令测试**

创建 `tests/test_hypersca_c_cli.py`：

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_context(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=30)
    b = 1.2 * a + rng.normal(scale=0.1, size=30)
    c = rng.normal(size=30)
    np.savez(
        path,
        expression_matrix=np.column_stack([a, b, c]).astype(np.float32),
        interventions=np.asarray(["non-targeting"] * 10 + ["A"] * 10 + ["B"] * 10),
        var_names=np.asarray(["A", "B", "C"]),
    )


def test_hypersca_c_cli_writes_complete_raw_results(tmp_path: Path) -> None:
    k562 = tmp_path / "k562.npz"
    rpe1 = tmp_path / "rpe1.npz"
    write_context(k562, 11)
    write_context(rpe1, 23)
    config = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    config.update({"maximum_epochs": 5, "bootstrap_repeats": 2})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_hypersca_c.py"),
            "--context",
            f"k562={k562}",
            "--context",
            f"rpe1={rpe1}",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--seed",
            "11",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )
    predictions = pd.read_csv(output / "raw_predictions.csv")
    assert len(predictions) == 6
    assert (output / "fit_summary.json").exists()
    status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed_raw_inference"
```

- [ ] **Step 2: 运行测试并确认命令缺失**

Run:

```bash
pytest tests/test_hypersca_c_cli.py -q -p no:cacheprovider
```

Expected: FAIL，因为命令不存在。

- [ ] **Step 3: 实现薄命令**

创建 `scripts/run_hypersca_c.py`，使用 `load_task_c_dataset` 读取每个 `name=path` 参数，确保所有文件基因顺序相同，然后调用稳定性拟合。主体必须包含：

```python
"""在允许查看的任务 C 文件上运行 HyperSCA-C。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext, HyperSCACError
from src.causal.hypersca_c_stability import fit_stable_hypersca_c
from src.evaluation.task_c_data import load_task_c_dataset, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从允许查看的单细胞干预数据学习 HyperSCA-C 候选基因关系。"
    )
    parser.add_argument("--context", action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def _parse_context(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise HyperSCACError("--context must use name=path")
    name, path = value.split("=", 1)
    if not name or not path:
        raise HyperSCACError("--context must use non-empty name=path")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = HyperSCACConfig.from_mapping(
            json.loads(args.config.read_text(encoding="utf-8"))
        )
        contexts = []
        expected_genes = None
        for raw in args.context:
            name, path = _parse_context(raw)
            dataset = load_task_c_dataset(path, context_id=name)
            if expected_genes is None:
                expected_genes = dataset.gene_names
            elif dataset.gene_names != expected_genes:
                raise HyperSCACError("all context files must use the same ordered genes")
            contexts.append(
                HyperSCACContext(
                    context_id=name,
                    expression=dataset.expression,
                    interventions=dataset.interventions,
                    gene_names=dataset.gene_names,
                )
            )
        if args.device == "cuda" and not torch.cuda.is_available():
            raise HyperSCACError("CUDA was requested but is not available")
        result = fit_stable_hypersca_c(
            contexts,
            config,
            seed=args.seed,
            device=args.device,
        )
    except (HyperSCACError, OSError, json.JSONDecodeError) as exc:
        parser.error(f"无法运行 HyperSCA-C：{exc}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output_dir / "raw_predictions.csv", index=False)
    write_json(
        args.output_dir / "fit_summary.json",
        {**result.summary, "failures": list(result.failures)},
    )
    write_json(
        args.output_dir / "method_status.json",
        {
            "schema_version": "1.0",
            "method_id": "hypersca_c",
            "status": "completed_raw_inference",
            "seed": args.seed,
            "contexts": [context.context_id for context in contexts],
        },
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行命令测试并提交**

Run:

```bash
pytest tests/test_hypersca_c_cli.py tests/test_hypersca_c.py tests/test_hypersca_c_stability.py -q -p no:cacheprovider
```

Expected: all tests PASS。

Commit:

```bash
git add scripts/run_hypersca_c.py tests/test_hypersca_c_cli.py
git commit -m "feat: add HyperSCA-C training command"
```

### Task 5: 固定并运行候选贡献消融

**Files:**
- Create: `configs/hypersca_c_ablations_v1.json`
- Create: `src/causal/hypersca_c_ablation.py`
- Create: `scripts/run_hypersca_c_ablations.py`
- Create: `tests/test_hypersca_c_ablation.py`

- [ ] **Step 1: 写消融覆盖、数据权限和主配置不变测试**

创建 `tests/test_hypersca_c_ablation.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext
from src.causal.hypersca_c_ablation import (
    apply_hypersca_c_ablation,
    load_hypersca_c_ablations,
)


ROOT = Path(__file__).resolve().parents[1]


def context() -> HyperSCACContext:
    return HyperSCACContext(
        context_id="k562",
        expression=np.arange(18, dtype=np.float32).reshape(6, 3),
        interventions=np.asarray(
            ["non-targeting", "non-targeting", "A", "A", "B", "B"]
        ),
        gene_names=("A", "B", "C"),
    )


def test_ablation_registry_covers_every_confirmed_candidate_component() -> None:
    registry = load_hypersca_c_ablations(
        ROOT / "configs/hypersca_c_ablations_v1.json"
    )
    assert set(registry) == {
        "primary",
        "shared_only",
        "separate_contexts",
        "observational_only",
        "no_stability_weighting",
        "acyclicity_off",
        "acyclicity_strong",
        "prior_on_secondary",
    }


def test_observational_ablation_removes_perturbed_cells_instead_of_relabeling() -> None:
    payload = json.loads((ROOT / "configs/hypersca_c_v1.json").read_text())
    config = HyperSCACConfig.from_mapping(payload)
    transformed, transformed_config = apply_hypersca_c_ablation(
        [context()], config, "observational_only"
    )
    assert transformed[0].expression.shape[0] == 2
    assert set(transformed[0].interventions) == {"non-targeting"}
    assert transformed_config == config


def test_shared_only_disables_context_adjustments_without_changing_primary() -> None:
    payload = json.loads((ROOT / "configs/hypersca_c_v1.json").read_text())
    config = HyperSCACConfig.from_mapping(payload)
    _, shared_only = apply_hypersca_c_ablation([context()], config, "shared_only")
    assert shared_only.enable_context_adjustments is False
    assert config.enable_context_adjustments is True
```

- [ ] **Step 2: 运行测试并确认消融模块缺失**

```bash
pytest tests/test_hypersca_c_ablation.py -q -p no:cacheprovider
```

Expected: collection FAIL。

- [ ] **Step 3: 写不可按结果删减的消融登记表**

创建 `configs/hypersca_c_ablations_v1.json`：

```json
{
  "schema_version": "1.0",
  "ablations": {
    "primary": {"mode": "joint", "configuration_changes": {}},
    "shared_only": {"mode": "joint", "configuration_changes": {"enable_context_adjustments": false}},
    "separate_contexts": {"mode": "one_context_per_fit", "configuration_changes": {}},
    "observational_only": {"mode": "control_cells_only", "configuration_changes": {}},
    "no_stability_weighting": {"mode": "joint", "configuration_changes": {"bootstrap_repeats": 1}},
    "acyclicity_off": {"mode": "joint", "configuration_changes": {"acyclicity_weight": 0.0}},
    "acyclicity_strong": {"mode": "joint", "configuration_changes": {"acyclicity_weight": 0.1}},
    "prior_on_secondary": {"mode": "joint_with_external_prior", "configuration_changes": {"prior_discount": 0.5}}
  }
}
```

- [ ] **Step 4: 实现消融映射，严格保护主配置**

创建 `src/causal/hypersca_c_ablation.py`：

```python
"""HyperSCA-C 预先登记消融的数据范围和配置映射。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext, HyperSCACError


def load_hypersca_c_ablations(path: str | Path) -> Mapping[str, Mapping[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or not payload.get("ablations"):
        raise HyperSCACError("ablation registry must use schema 1.0 and be non-empty")
    return payload["ablations"]


def apply_hypersca_c_ablation(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    ablation_id: str,
) -> tuple[list[HyperSCACContext], HyperSCACConfig]:
    registry = load_hypersca_c_ablations(
        Path(__file__).resolve().parents[2] / "configs/hypersca_c_ablations_v1.json"
    )
    if ablation_id not in registry:
        raise HyperSCACError(f"unknown ablation: {ablation_id}")
    spec = registry[ablation_id]
    transformed = list(contexts)
    if spec["mode"] == "control_cells_only":
        transformed = []
        for context in contexts:
            selected = context.interventions == config.control_label
            transformed.append(
                HyperSCACContext(
                    context_id=context.context_id,
                    expression=context.expression[selected],
                    interventions=context.interventions[selected],
                    gene_names=context.gene_names,
                )
            )
    values = asdict(config)
    values.update(spec["configuration_changes"])
    return transformed, HyperSCACConfig.from_mapping(values)
```

`separate_contexts` 由运行命令对每个环境分别调用拟合；`prior_on_secondary` 若没有已登记且不与评估关系重用的外部先验，必须写阻断状态，不能用 CausalBench 参考关系代替。

- [ ] **Step 5: 实现消融运行命令**

创建 `scripts/run_hypersca_c_ablations.py`，沿用 `run_hypersca_c.py` 的 `--context`、`--config`、`--seed` 和 `--device`，增加：

```python
parser.add_argument("--ablation-registry", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--prior-edges", type=Path)
```

命令按登记顺序运行全部八项；每项写独立 `raw_predictions.csv`、`fit_summary.json` 和 `method_status.json`。`prior_on_secondary` 只有在提供先验 CSV、先验来源清单和 SHA-256，且其指纹不等于两类评分参考关系时才可运行，否则状态写 `official_assets_unavailable`，理由写 `no_nonoverlapping_preregistered_prior`。单项失败不得删除其他消融结果。

- [ ] **Step 6: 运行测试并提交**

```bash
pytest tests/test_hypersca_c_ablation.py tests/test_hypersca_c.py tests/test_hypersca_c_stability.py tests/test_hypersca_c_cli.py -q -p no:cacheprovider
git add configs/hypersca_c_ablations_v1.json src/causal/hypersca_c.py src/causal/hypersca_c_ablation.py scripts/run_hypersca_c_ablations.py tests/test_hypersca_c.py tests/test_hypersca_c_ablation.py
git commit -m "feat: preregister HyperSCA-C ablations"
```

Expected: tests PASS；主配置仍是无外部知识版本。

### Task 6: 写方法边界并完成模型回归检查

**Files:**
- Create: `docs/research/hypersca_c_method_v1.md`

- [ ] **Step 1: 写方法说明**

创建 `docs/research/hypersca_c_method_v1.md`，内容至少完整覆盖：

```markdown
# HyperSCA-C 第一版方法说明

HyperSCA-C 学习“干预一个基因后，哪些其他基因可能随之改变”的有向候选关系。它与 HyperSCA 原有的细胞群通讯图是两个不同研究对象，不能互相替代。

## 第一版模型

每个细胞环境的关系由共享部分和环境特有调整组成。被直接干预基因的自身表达不用于拟合其输入关系，其他基因仍用于估计可能的下游作用。第一版只采用线性关系，以便检查每项假设和失败原因。

## 固定排名分数

`score = |重复拟合关系强度中位数| × 选择比例 × 方向一致比例`。

该分数不使用最终检验结果。来源基因没有足够控制组变化，或成功重复比例低于 80% 时，模型明确标记暂不判断。

## 候选贡献与限制

带干预标记的结构学习已有 DCDI、NOTEARS 和 SparseRC。HyperSCA-C 的候选贡献是共享/环境特有关系的联合学习、只用目标环境未干预细胞进行适配，以及正式报告稳定性和暂不判断。只有消融比较支持这些组成部分时，才讨论算法贡献。

线性关系和无环软限制都是简化假设。无环限制不表示真实基因网络没有反馈。模型输出是待验证关系，不是已证实机制或治疗靶点。
```

- [ ] **Step 2: 运行术语、模型和既有因果测试**

Run:

```bash
python scripts/check_plain_language.py
pytest tests/test_hypersca_c.py tests/test_hypersca_c_stability.py tests/test_hypersca_c_cli.py tests/test_causal_stability_audit.py tests/test_task_c_benchmark.py -q -p no:cacheprovider
git diff --check
```

Expected: 检查和测试均通过，差异格式无错误。

- [ ] **Step 3: 提交**

```bash
git add docs/research/hypersca_c_method_v1.md
git commit -m "docs: define HyperSCA-C evidence boundary"
git status --short
```

Expected: 工作区干净。
