"""HyperSCA-C：跨细胞环境的线性干预基因关系模型。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as functional


class HyperSCACError(ValueError):
    """HyperSCA-C 输入或计算结果不满足固定规则。"""


@dataclass(frozen=True)
class HyperSCACConfig:
    """第一版候选方法在正式比较前固定的设置。"""

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
        """读取并严格检查一组已固定的候选方法设置。"""

        if not isinstance(payload, Mapping):
            raise HyperSCACError("configuration must be a mapping of frozen fields")

        expected = {field.name for field in fields(cls)}
        try:
            supplied = set(payload)
        except Exception as exc:
            raise HyperSCACError(
                "configuration must be a mapping of frozen fields"
            ) from exc
        missing = expected - supplied
        extra = supplied - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing fields: {sorted(missing)}")
            if extra:
                details.append(f"unexpected fields: {sorted(extra, key=str)}")
            raise HyperSCACError(
                "configuration fields do not match: " + "; ".join(details)
            )

        schema_version = _required_text(payload["schema_version"], "schema_version")
        if schema_version != "1.0":
            raise HyperSCACError("schema_version must be 1.0")

        learning_rate = _finite_number(payload["learning_rate"], "learning_rate")
        if learning_rate <= 0.0:
            raise HyperSCACError("learning_rate must be greater than 0")

        maximum_epochs = _positive_integer(payload["maximum_epochs"], "maximum_epochs")
        early_stopping_patience = _positive_integer(
            payload["early_stopping_patience"], "early_stopping_patience"
        )
        bootstrap_repeats = _positive_integer(
            payload["bootstrap_repeats"], "bootstrap_repeats"
        )

        nonnegative_names = (
            "shared_l1",
            "context_l1",
            "acyclicity_weight",
            "selection_threshold",
        )
        numeric: dict[str, float] = {"learning_rate": learning_rate}
        for name in nonnegative_names:
            value = _finite_number(payload[name], name)
            if value < 0.0:
                raise HyperSCACError(f"{name} must be non-negative")
            numeric[name] = value

        prior_discount = _finite_number(payload["prior_discount"], "prior_discount")
        if not 0.0 <= prior_discount < 1.0:
            raise HyperSCACError("prior_discount must be in [0, 1)")

        bootstrap_success_fraction = _finite_number(
            payload["bootstrap_success_fraction"], "bootstrap_success_fraction"
        )
        if not 0.0 < bootstrap_success_fraction <= 1.0:
            raise HyperSCACError("bootstrap_success_fraction must be in (0, 1]")

        minimum_source_variance = _finite_number(
            payload["minimum_source_variance"], "minimum_source_variance"
        )
        if minimum_source_variance <= 0.0:
            raise HyperSCACError("minimum_source_variance must be greater than 0")

        enabled = payload["enable_context_adjustments"]
        if type(enabled) is not bool:
            raise HyperSCACError("enable_context_adjustments must be a bool")

        control_label = _required_text(payload["control_label"], "control_label")
        excluded_label = _required_text(payload["excluded_label"], "excluded_label")
        if control_label == excluded_label:
            raise HyperSCACError("control_label and excluded_label must be different")

        return cls(
            schema_version=schema_version,
            learning_rate=learning_rate,
            maximum_epochs=maximum_epochs,
            early_stopping_patience=early_stopping_patience,
            shared_l1=numeric["shared_l1"],
            context_l1=numeric["context_l1"],
            acyclicity_weight=numeric["acyclicity_weight"],
            enable_context_adjustments=enabled,
            prior_discount=prior_discount,
            selection_threshold=numeric["selection_threshold"],
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_success_fraction=bootstrap_success_fraction,
            minimum_source_variance=minimum_source_variance,
            control_label=control_label,
            excluded_label=excluded_label,
        )


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HyperSCACError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HyperSCACError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise HyperSCACError(f"{name} must be a finite number")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HyperSCACError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 1:
        raise HyperSCACError(f"{name} must be a positive integer")
    return normalized


def _text_vector(
    values: Sequence[str],
    name: str,
    *,
    forbid_whitespace: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise HyperSCACError(
            f"{name} must be a one-dimensional sequence of text labels"
        )
    try:
        array = np.asarray(values, dtype=object)
    except Exception as exc:
        raise HyperSCACError(
            f"{name} must be a one-dimensional sequence of text labels"
        ) from exc
    if array.ndim != 1:
        raise HyperSCACError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise HyperSCACError(f"{name} must contain at least one label")

    result: list[str] = []
    for value in array.tolist():
        if not isinstance(value, str):
            raise HyperSCACError(f"{name} must contain only text labels")
        if not value or value != value.strip():
            raise HyperSCACError(
                f"{name} labels must be non-empty text without surrounding whitespace"
            )
        if forbid_whitespace and any(character.isspace() for character in value):
            raise HyperSCACError(f"{name} labels must not contain whitespace")
        result.append(value)
    return tuple(result)


def build_intervention_mask(
    interventions: Sequence[str],
    gene_names: Sequence[str],
    *,
    excluded_label: str = "excluded",
) -> np.ndarray:
    """标出可用于学习的表达值，并遮挡被直接干预的基因。"""

    labels = _text_vector(interventions, "interventions", forbid_whitespace=True)
    genes = _text_vector(gene_names, "gene_names")
    _required_text(excluded_label, "excluded_label")
    if len(set(genes)) != len(genes):
        raise HyperSCACError("gene_names must be unique")

    gene_index = {gene: index for index, gene in enumerate(genes)}
    mask = np.ones((len(labels), len(genes)), dtype=np.float32)
    for row, label in enumerate(labels):
        if label == excluded_label:
            mask[row, :] = 0.0
        elif label in gene_index:
            mask[row, gene_index[label]] = 0.0
    if not bool(mask.any()):
        raise HyperSCACError("intervention mask contains no usable expression values")
    return mask


def zero_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """移除基因对自身的关系，同时保留张量的计算梯度。"""

    if not isinstance(matrix, torch.Tensor):
        raise HyperSCACError("matrix must be a torch tensor")
    if matrix.ndim != 2:
        raise HyperSCACError("matrix must be two-dimensional")
    if matrix.shape[0] != matrix.shape[1]:
        raise HyperSCACError("matrix must be square")
    if not matrix.is_floating_point():
        raise HyperSCACError("matrix must use a floating-point dtype")
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return matrix * (1.0 - identity)


def masked_sem_loss(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """只用未被干预遮挡的表达值计算平滑绝对误差。"""

    tensors = (prediction, observed, mask)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise HyperSCACError("prediction, observed, and mask must be torch tensors")
    if prediction.shape != observed.shape or mask.shape != observed.shape:
        raise HyperSCACError("prediction, observed, and mask shapes must match")
    if prediction.device != observed.device or mask.device != observed.device:
        raise HyperSCACError("prediction, observed, and mask must use the same device")
    if prediction.numel() == 0:
        raise HyperSCACError("prediction, observed, and mask must not be empty")
    if any(not value.is_floating_point() for value in tensors):
        raise HyperSCACError(
            "prediction, observed, and mask must use floating-point dtypes"
        )
    if prediction.dtype != observed.dtype or mask.dtype != observed.dtype:
        raise HyperSCACError("prediction, observed, and mask must use the same dtype")
    if prediction.device.type == "meta":
        raise HyperSCACError(
            "prediction, observed, and mask must use a materialized device"
        )
    try:
        all_finite = all(bool(torch.isfinite(value).all().item()) for value in tensors)
    except (RuntimeError, TypeError) as exc:
        raise HyperSCACError("could not validate tensor values") from exc
    if not all_finite:
        raise HyperSCACError(
            "prediction, observed, and mask must contain only finite values"
        )
    if not bool(torch.logical_or(mask == 0.0, mask == 1.0).all().item()):
        raise HyperSCACError("mask values must be 0 or 1")
    denominator = mask.sum()
    if float(denominator.detach().cpu()) == 0.0:
        raise HyperSCACError("mask must contain at least one usable value")

    losses = functional.smooth_l1_loss(prediction, observed, reduction="none")
    return (losses * mask).sum() / denominator
