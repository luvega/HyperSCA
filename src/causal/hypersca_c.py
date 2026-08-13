"""HyperSCA-C：跨细胞环境的线性干预基因关系模型。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from types import MappingProxyType

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

    def __post_init__(self) -> None:
        """阻止直接构造绕过第一版候选方法的固定规则。"""

        schema_version = _required_text(self.schema_version, "schema_version")
        if schema_version != "1.0":
            raise HyperSCACError("schema_version must be 1.0")

        learning_rate = _finite_number(self.learning_rate, "learning_rate")
        if learning_rate <= 0.0:
            raise HyperSCACError("learning_rate must be greater than 0")

        maximum_epochs = _positive_integer(self.maximum_epochs, "maximum_epochs")
        early_stopping_patience = _positive_integer(
            self.early_stopping_patience, "early_stopping_patience"
        )
        bootstrap_repeats = _positive_integer(
            self.bootstrap_repeats, "bootstrap_repeats"
        )

        nonnegative_names = (
            "shared_l1",
            "context_l1",
            "acyclicity_weight",
            "selection_threshold",
        )
        normalized_nonnegative: dict[str, float] = {}
        for name in nonnegative_names:
            value = _finite_number(getattr(self, name), name)
            if value < 0.0:
                raise HyperSCACError(f"{name} must be non-negative")
            normalized_nonnegative[name] = value

        prior_discount = _finite_number(self.prior_discount, "prior_discount")
        if not 0.0 <= prior_discount < 1.0:
            raise HyperSCACError("prior_discount must be in [0, 1)")

        bootstrap_success_fraction = _finite_number(
            self.bootstrap_success_fraction, "bootstrap_success_fraction"
        )
        if not 0.0 < bootstrap_success_fraction <= 1.0:
            raise HyperSCACError("bootstrap_success_fraction must be in (0, 1]")

        minimum_source_variance = _finite_number(
            self.minimum_source_variance, "minimum_source_variance"
        )
        if minimum_source_variance <= 0.0:
            raise HyperSCACError("minimum_source_variance must be greater than 0")

        if type(self.enable_context_adjustments) is not bool:
            raise HyperSCACError("enable_context_adjustments must be a bool")

        control_label = _required_text(self.control_label, "control_label")
        excluded_label = _required_text(self.excluded_label, "excluded_label")
        if control_label == excluded_label:
            raise HyperSCACError("control_label and excluded_label must be different")

        normalized_values: dict[str, object] = {
            "schema_version": schema_version,
            "learning_rate": learning_rate,
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": early_stopping_patience,
            **normalized_nonnegative,
            "prior_discount": prior_discount,
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_success_fraction": bootstrap_success_fraction,
            "minimum_source_variance": minimum_source_variance,
            "control_label": control_label,
            "excluded_label": excluded_label,
        }
        for name, value in normalized_values.items():
            object.__setattr__(self, name, value)

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

        try:
            return cls(**dict(payload))  # type: ignore[arg-type]
        except HyperSCACError:
            raise
        except Exception as exc:
            raise HyperSCACError("configuration values could not be validated") from exc


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HyperSCACError(f"{name} must be non-empty text without whitespace")
    if any(character.isspace() for character in value):
        raise HyperSCACError(f"{name} must not contain whitespace")
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
        result.append(_required_text(value, f"{name} label"))
    return tuple(result)


def build_intervention_mask(
    interventions: Sequence[str],
    gene_names: Sequence[str],
    *,
    excluded_label: str = "excluded",
) -> np.ndarray:
    """标出可用于学习的表达值，并遮挡被直接干预的基因。"""

    labels = _text_vector(interventions, "interventions")
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
    if any(value.layout != torch.strided for value in tensors):
        raise HyperSCACError(
            "prediction, observed, and mask must be dense tensors"
        )
    if prediction.device.type == "meta":
        raise HyperSCACError(
            "prediction, observed, and mask must use a materialized device"
        )
    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if any(value.dtype not in supported_dtypes for value in tensors):
        raise HyperSCACError(
            "prediction, observed, and mask must use a supported floating-point dtype"
        )
    if prediction.dtype != observed.dtype or mask.dtype != observed.dtype:
        raise HyperSCACError("prediction, observed, and mask must use the same dtype")
    if mask.requires_grad:
        raise HyperSCACError(
            "mask is a fixed experimental rule and must not require a gradient"
        )

    try:
        all_finite = all(bool(torch.isfinite(value).all().item()) for value in tensors)
    except Exception as exc:
        raise HyperSCACError("could not validate tensor values") from exc
    if not all_finite:
        raise HyperSCACError(
            "prediction, observed, and mask must contain only finite values"
        )
    try:
        mask_is_binary = bool(
            torch.logical_or(mask == 0.0, mask == 1.0).all().item()
        )
    except Exception as exc:
        raise HyperSCACError("could not validate mask values") from exc
    if not mask_is_binary:
        raise HyperSCACError("mask values must be 0 or 1")

    usable = mask == 1.0
    try:
        selected_prediction = prediction[usable]
        selected_observed = observed[usable]
    except Exception as exc:
        raise HyperSCACError("could not select usable expression values") from exc
    usable_count = selected_prediction.numel()
    if usable_count == 0:
        raise HyperSCACError("mask must contain at least one usable value")

    if prediction.dtype in {torch.float16, torch.bfloat16}:
        selected_prediction = selected_prediction.to(torch.float32)
        selected_observed = selected_observed.to(torch.float32)
    try:
        loss = functional.smooth_l1_loss(
            selected_prediction,
            selected_observed,
            reduction="sum",
        ) / usable_count
        loss_is_finite = bool(torch.isfinite(loss).item())
    except Exception as exc:
        raise HyperSCACError("could not calculate the masked expression loss") from exc
    if not loss_is_finite:
        raise HyperSCACError("masked expression loss must be finite")
    return loss


def _numeric_matrix(values: object, name: str) -> np.ndarray:
    """检查并复制一个有限的实数二维矩阵。"""

    try:
        array = np.asarray(values)
    except Exception as exc:
        raise HyperSCACError(f"{name} must be a two-dimensional numeric array") from exc
    if array.ndim != 2:
        raise HyperSCACError(f"{name} must be two-dimensional")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise HyperSCACError(f"{name} must contain at least one row and one column")
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise HyperSCACError(f"{name} must contain real numeric values, not bool")
    try:
        finite = bool(np.isfinite(array).all())
    except Exception as exc:
        raise HyperSCACError(f"{name} values could not be validated") from exc
    if not finite:
        raise HyperSCACError(f"{name} values must be finite")
    try:
        normalized = np.array(array, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HyperSCACError(f"{name} could not be converted to float32") from exc
    if not bool(np.isfinite(normalized).all()):
        raise HyperSCACError(f"{name} values must remain finite as float32")
    return normalized


@dataclass(frozen=True)
class HyperSCACContext:
    """一个细胞环境中按细胞排列的表达和干预标签。"""

    context_id: str
    expression: np.ndarray
    interventions: np.ndarray
    gene_names: tuple[str, ...]

    def __post_init__(self) -> None:
        context_id = _required_text(self.context_id, "context_id")
        genes = _text_vector(self.gene_names, "gene_names")
        if len(set(genes)) != len(genes):
            raise HyperSCACError("gene_names must be unique")
        labels = _text_vector(self.interventions, "interventions")
        expression = _numeric_matrix(self.expression, "expression")
        expected_shape = (len(labels), len(genes))
        if expression.shape != expected_shape:
            raise HyperSCACError(
                "expression shape must match intervention rows and gene columns"
            )

        label_array = np.array(labels, dtype=str, copy=True)
        expression.setflags(write=False)
        label_array.setflags(write=False)
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "interventions", label_array)
        object.__setattr__(self, "gene_names", genes)


def _validated_fit_matrix(
    values: object,
    name: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    result = _numeric_matrix(values, name)
    raw = np.asarray(values)
    if expected_shape is None:
        if result.shape[0] != result.shape[1]:
            raise HyperSCACError(f"{name} matrix must be square")
        if result.shape[0] < 2:
            raise HyperSCACError(f"{name} matrix must contain at least two genes")
    elif result.shape != expected_shape:
        raise HyperSCACError(f"{name} matrix shape must match the shared matrix")
    # Check before float32 conversion so a tiny self-edge cannot silently underflow.
    if bool(np.any(np.diag(raw) != 0.0)):
        raise HyperSCACError(f"{name} matrix diagonal must be exactly zero")
    result.setflags(write=False)
    return result


def _validated_loss_history(values: object) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise HyperSCACError("loss_history must be a one-dimensional numeric array")
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise HyperSCACError(
            "loss_history must be a one-dimensional numeric array"
        ) from exc
    if array.ndim != 1:
        raise HyperSCACError("loss_history must be one-dimensional")
    if array.size < 1:
        raise HyperSCACError("loss_history must contain at least one value")
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise HyperSCACError("loss_history must contain real numeric values")
    result = np.array(array, dtype=np.float64, copy=True)
    if not bool(np.isfinite(result).all()):
        raise HyperSCACError("loss_history values must be finite")
    result.setflags(write=False)
    return result


def _fit_matrix_mapping(
    values: object,
    name: str,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise HyperSCACError(f"{name} must be a mapping by context identifier")
    try:
        items = tuple(values.items())
    except Exception as exc:
        raise HyperSCACError(f"{name} could not be read") from exc
    if not items:
        raise HyperSCACError(f"{name} must contain at least one context")
    normalized: dict[str, np.ndarray] = {}
    singular = "adjustment" if name == "context_adjustments" else "adjacency"
    for raw_identifier, matrix in items:
        identifier = _required_text(raw_identifier, "context identifier")
        normalized[identifier] = _validated_fit_matrix(
            matrix,
            f"context {singular}",
            expected_shape=expected_shape,
        )
    return normalized


@dataclass(frozen=True)
class HyperSCACFit:
    """一次联合拟合保留下来的最佳关系矩阵和损失轨迹。"""

    shared: np.ndarray
    context_adjustments: Mapping[str, np.ndarray]
    context_adjacencies: Mapping[str, np.ndarray]
    loss_history: np.ndarray
    converged: bool
    epochs_run: int
    seed: int
    config: HyperSCACConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, HyperSCACConfig):
            raise HyperSCACError("config must be a validated HyperSCACConfig")
        shared = _validated_fit_matrix(self.shared, "shared")
        adjustments = _fit_matrix_mapping(
            self.context_adjustments,
            "context_adjustments",
            expected_shape=shared.shape,
        )
        adjacencies_unordered = _fit_matrix_mapping(
            self.context_adjacencies,
            "context_adjacencies",
            expected_shape=shared.shape,
        )
        if set(adjustments) != set(adjacencies_unordered):
            raise HyperSCACError(
                "adjustments and adjacencies must contain the same context identifiers"
            )
        adjacencies: dict[str, np.ndarray] = {}
        for identifier, adjustment in adjustments.items():
            adjacency = adjacencies_unordered[identifier]
            expected = shared + adjustment
            if not np.array_equal(adjacency, expected):
                raise HyperSCACError(
                    "context adjacency must equal the shared matrix plus its adjustment"
                )
            adjacencies[identifier] = adjacency

        history = _validated_loss_history(self.loss_history)
        if type(self.converged) is not bool:
            raise HyperSCACError("converged must be a bool")
        epochs_run = _positive_integer(self.epochs_run, "epochs_run")
        if epochs_run != len(history):
            raise HyperSCACError("epochs_run must equal the loss_history length")
        if epochs_run > self.config.maximum_epochs:
            raise HyperSCACError("epochs_run must not exceed config.maximum_epochs")
        seed = _validated_seed(self.seed)
        object.__setattr__(self, "shared", shared)
        object.__setattr__(
            self, "context_adjustments", MappingProxyType(adjustments)
        )
        object.__setattr__(
            self, "context_adjacencies", MappingProxyType(adjacencies)
        )
        object.__setattr__(self, "loss_history", history)
        object.__setattr__(self, "epochs_run", epochs_run)
        object.__setattr__(self, "seed", seed)


def standardize_context(
    expression: np.ndarray,
    interventions: Sequence[str],
    *,
    control_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只用对照细胞估计每个基因的中心和尺度。"""

    values = _numeric_matrix(expression, "expression")
    labels = _text_vector(interventions, "interventions")
    normalized_control_label = _required_text(control_label, "control_label")
    if len(labels) != values.shape[0]:
        raise HyperSCACError("intervention rows must match expression rows")
    controls = np.fromiter(
        (label == normalized_control_label for label in labels),
        dtype=bool,
        count=len(labels),
    )
    if int(controls.sum()) < 2:
        raise HyperSCACError("at least two control cells are required")

    # float64 accumulation avoids avoidable cancellation before the public float32 output.
    control_values = values[controls].astype(np.float64, copy=False)
    center64 = control_values.mean(axis=0)
    scale64 = control_values.std(axis=0, ddof=0)
    scale64 = np.where(scale64 > 1e-6, scale64, 1.0)
    scaled64 = (values.astype(np.float64, copy=False) - center64) / scale64
    scaled = np.array(scaled64, dtype=np.float32, order="C", copy=True)
    center = np.array(center64, dtype=np.float32, copy=True)
    scale = np.array(scale64, dtype=np.float32, copy=True)
    if not all(bool(np.isfinite(value).all()) for value in (scaled, center, scale)):
        raise HyperSCACError("standardized expression values must be finite")
    return scaled, center, scale


def acyclicity_penalty(adjacency: torch.Tensor) -> torch.Tensor:
    """计算 NOTEARS 无环软限制 ``tr(exp(A⊙A))-d``。"""

    if not isinstance(adjacency, torch.Tensor):
        raise HyperSCACError("adjacency must be a torch tensor")
    if adjacency.layout != torch.strided:
        raise HyperSCACError("adjacency must be a dense tensor")
    if adjacency.device.type == "meta":
        raise HyperSCACError("adjacency must use a materialized device")
    if adjacency.ndim != 2:
        raise HyperSCACError("adjacency must be two-dimensional")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise HyperSCACError("adjacency must be square")
    if adjacency.shape[0] < 1:
        raise HyperSCACError("adjacency must contain at least one gene")
    if adjacency.dtype not in {torch.float32, torch.float64}:
        raise HyperSCACError("adjacency must use a supported floating-point dtype")
    try:
        if not bool(torch.isfinite(adjacency).all().item()):
            raise HyperSCACError("adjacency values must be finite")
        without_self_edges = zero_diagonal(adjacency)
        dimension = adjacency.shape[0]
        penalty = (
            torch.trace(torch.matrix_exp(without_self_edges * without_self_edges))
            - dimension
        )
        if not bool(torch.isfinite(penalty).item()):
            raise HyperSCACError("acyclicity penalty must be finite")
        return penalty
    except HyperSCACError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HyperSCACError("could not calculate the acyclicity penalty") from exc


def _validated_contexts(
    contexts: Sequence[HyperSCACContext],
) -> tuple[tuple[HyperSCACContext, ...], tuple[str, ...]]:
    if isinstance(contexts, (str, bytes)):
        raise HyperSCACError("contexts must be a sequence of context objects")
    try:
        normalized = tuple(contexts)
    except (TypeError, RuntimeError) as exc:
        raise HyperSCACError("contexts must be a sequence of context objects") from exc
    if not normalized:
        raise HyperSCACError("at least one context is required")
    if any(not isinstance(context, HyperSCACContext) for context in normalized):
        raise HyperSCACError("contexts must contain only HyperSCACContext objects")

    genes = normalized[0].gene_names
    if len(genes) < 2:
        raise HyperSCACError("at least two genes are required for structure learning")
    identifiers: set[str] = set()
    for context in normalized:
        if context.context_id in identifiers:
            raise HyperSCACError("context identifiers must be unique")
        identifiers.add(context.context_id)
        if context.gene_names != genes:
            raise HyperSCACError("all contexts must use the same ordered genes")
    return normalized, genes


def _validated_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise HyperSCACError("seed must be a non-negative integer supported by torch")
    normalized = int(seed)
    if not 0 <= normalized <= 2**64 - 1:
        raise HyperSCACError("seed must be a non-negative integer supported by torch")
    return normalized


def _validated_device(device: object) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise HyperSCACError("device must identify an available torch device")
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HyperSCACError("device must identify an available torch device") from exc
    if target.type == "meta":
        raise HyperSCACError("device must be materialized")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise HyperSCACError("requested CUDA device is not available")
        if target.index is not None and target.index >= torch.cuda.device_count():
            raise HyperSCACError("requested CUDA device is not available")
    elif target.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise HyperSCACError("requested MPS device is not available")
    elif target.type != "cpu":
        raise HyperSCACError("device type is not supported by HyperSCA-C")
    try:
        torch.empty(0, device=target)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HyperSCACError("requested torch device is not available") from exc
    return target


def _prior_weights(
    prior_mask: np.ndarray | None,
    *,
    dimension: int,
    prior_discount: float,
    device: torch.device,
) -> torch.Tensor:
    if prior_mask is None:
        return torch.ones((dimension, dimension), dtype=torch.float32, device=device)
    if not isinstance(prior_mask, np.ndarray):
        raise HyperSCACError("prior mask must be a dense numeric array")
    if prior_mask.ndim != 2 or prior_mask.shape != (dimension, dimension):
        raise HyperSCACError("prior mask shape must match the gene network")
    if (
        not (
            np.issubdtype(prior_mask.dtype, np.bool_)
            or np.issubdtype(prior_mask.dtype, np.integer)
            or np.issubdtype(prior_mask.dtype, np.floating)
        )
        or np.issubdtype(prior_mask.dtype, np.complexfloating)
    ):
        raise HyperSCACError("prior mask must contain numeric or bool values")
    try:
        if not bool(np.isfinite(prior_mask).all()):
            raise HyperSCACError("prior mask values must be finite")
        if not bool(np.logical_or(prior_mask == 0, prior_mask == 1).all()):
            raise HyperSCACError("prior mask values must be 0 or 1")
        normalized = np.array(prior_mask, dtype=np.float32, order="C", copy=True)
        np.fill_diagonal(normalized, 0.0)
        mask_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        return 1.0 - prior_discount * mask_tensor
    except HyperSCACError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HyperSCACError("prior mask could not be prepared") from exc


def _all_finite_tensor(value: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(value).all().item())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HyperSCACError("could not validate optimization values") from exc


def _hypersca_c_objective(
    shared_raw: torch.Tensor,
    delta_raw: Mapping[str, torch.Tensor],
    contexts: Sequence[HyperSCACContext],
    prepared: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    config: HyperSCACConfig,
    prior_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """用一份公式同时计算训练目标和更新后留档目标。"""

    shared = zero_diagonal(shared_raw)
    total = config.shared_l1 * (shared.abs() * prior_weights).mean()
    current_deltas: dict[str, torch.Tensor] = {}
    for context in contexts:
        delta = zero_diagonal(delta_raw[context.context_id])
        current_deltas[context.context_id] = delta
        adjacency = shared + delta
        values, mask = prepared[context.context_id]
        # Convention: adjacency[source, target], so prediction = X @ adjacency.
        prediction = values @ adjacency
        total = total + masked_sem_loss(prediction, values, mask)
        total = total + config.context_l1 * delta.abs().mean()
        total = total + config.acyclicity_weight * acyclicity_penalty(adjacency)
    return total, shared, current_deltas


def fit_hypersca_c_once(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None = None,
) -> HyperSCACFit:
    """联合学习共享关系与各细胞环境的轻量调整。"""

    try:
        return _fit_hypersca_c_once(
            contexts,
            config,
            seed=seed,
            device=device,
            prior_mask=prior_mask,
        )
    except HyperSCACError:
        raise
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise HyperSCACError("could not fit HyperSCA-C on the supplied contexts") from exc


def _fit_hypersca_c_once(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None,
) -> HyperSCACFit:
    if not isinstance(config, HyperSCACConfig):
        raise HyperSCACError("config must be a validated HyperSCACConfig")
    normalized_contexts, genes = _validated_contexts(contexts)
    normalized_seed = _validated_seed(seed)
    target_device = _validated_device(device)
    dimension = len(genes)
    prior_weights = _prior_weights(
        prior_mask,
        dimension=dimension,
        prior_discount=config.prior_discount,
        device=target_device,
    )

    prepared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for context in normalized_contexts:
        scaled, _, _ = standardize_context(
            context.expression,
            context.interventions,
            control_label=config.control_label,
        )
        mask = build_intervention_mask(
            context.interventions,
            genes,
            excluded_label=config.excluded_label,
        )
        prepared[context.context_id] = (
            torch.as_tensor(scaled, dtype=torch.float32, device=target_device),
            torch.as_tensor(mask, dtype=torch.float32, device=target_device),
        )

    shared_raw = torch.zeros(
        (dimension, dimension),
        dtype=torch.float32,
        device=target_device,
        requires_grad=True,
    )
    if config.enable_context_adjustments:
        delta_raw = {
            context.context_id: torch.zeros(
                (dimension, dimension),
                dtype=torch.float32,
                device=target_device,
                requires_grad=True,
            )
            for context in normalized_contexts
        }
    else:
        delta_raw = {
            context.context_id: torch.zeros(
                (dimension, dimension), dtype=torch.float32, device=target_device
            )
            for context in normalized_contexts
        }
    parameters = [shared_raw]
    if config.enable_context_adjustments:
        parameters.extend(delta_raw.values())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    history: list[float] = []
    best_loss = float("inf")
    early_stopping_best = float("inf")
    stale_epochs = 0
    converged = False
    best_state: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
    for _ in range(config.maximum_epochs):
        optimizer.zero_grad(set_to_none=True)
        training_total, _, _ = _hypersca_c_objective(
            shared_raw,
            delta_raw,
            normalized_contexts,
            prepared,
            config,
            prior_weights,
        )
        if not _all_finite_tensor(training_total):
            raise HyperSCACError("optimization loss must be finite at every step")
        try:
            training_total.backward()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HyperSCACError("optimization gradient could not be calculated") from exc
        for parameter in parameters:
            if parameter.grad is None or not _all_finite_tensor(parameter.grad):
                raise HyperSCACError("optimization gradient must be finite at every step")
        try:
            optimizer.step()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HyperSCACError("optimizer could not update the model") from exc
        if any(not _all_finite_tensor(parameter) for parameter in parameters):
            raise HyperSCACError("model values must remain finite after every step")

        with torch.no_grad():
            updated_total, updated_shared, updated_deltas = _hypersca_c_objective(
                shared_raw,
                delta_raw,
                normalized_contexts,
                prepared,
                config,
                prior_weights,
            )
        if not _all_finite_tensor(updated_total):
            raise HyperSCACError("updated optimization loss must be finite")
        numeric_loss = float(updated_total.detach().cpu().item())
        history.append(numeric_loss)

        # History and the saved state now describe the same post-update parameters.
        if numeric_loss < best_loss:
            best_loss = numeric_loss
            best_state = (
                updated_shared.detach().cpu().numpy().copy(),
                {
                    name: delta.detach().cpu().numpy().copy()
                    for name, delta in updated_deltas.items()
                },
            )

        if numeric_loss < early_stopping_best - 1e-7:
            early_stopping_best = numeric_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                converged = True
                break

    if best_state is None or not history:
        raise HyperSCACError("optimization did not produce a usable state")
    shared_value, delta_values = best_state
    if not config.enable_context_adjustments:
        delta_values = {
            context.context_id: np.zeros((dimension, dimension), dtype=np.float32)
            for context in normalized_contexts
        }
    context_values = {
        context.context_id: shared_value + delta_values[context.context_id]
        for context in normalized_contexts
    }
    return HyperSCACFit(
        shared=shared_value,
        context_adjustments=delta_values,
        context_adjacencies=context_values,
        loss_history=np.asarray(history, dtype=np.float64),
        converged=converged,
        epochs_run=len(history),
        seed=normalized_seed,
        config=config,
    )
