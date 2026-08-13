"""HyperSCA-C 重复稳定性、完整关系表和暂不判断规则。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.causal import hypersca_c as _core
from src.causal.hypersca_c import (
    HyperSCACConfig,
    HyperSCACContext,
    HyperSCACError,
    fit_hypersca_c_once,
)


_SCORE_FORMULA = (
    "abs_median_effect_times_selection_frequency_times_direction_agreement"
)
_REQUIRED_PREDICTION_COLUMNS = (
    "source",
    "target",
    "effect",
    "median_effect",
    "direction",
    "selection_frequency",
    "direction_agreement",
    "context_consistency",
    "score",
    "abstained",
    "abstention_reason",
)
_SUMMARY_FIELDS = (
    "requested_repeats",
    "successful_repeats",
    "repeat_success_fraction",
    "coverage",
    "abstention_rate",
    "score_formula",
)


class _FrozenJSONDict(dict[str, object]):
    """A JSON-serializable dictionary whose public mutations are disabled."""

    @staticmethod
    def _read_only(*args: object, **kwargs: object) -> None:
        raise TypeError("frozen JSON mappings are read-only")

    __setitem__ = _read_only
    __delitem__ = _read_only
    clear = _read_only
    pop = _read_only
    popitem = _read_only
    setdefault = _read_only
    update = _read_only
    __ior__ = _read_only


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
        raise HyperSCACError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise HyperSCACError(f"{name} must be a positive integer")
    return normalized


def _validated_genes(gene_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(gene_names, (str, bytes)):
        raise HyperSCACError("gene_names must be a sequence of text labels")
    try:
        raw_genes = tuple(gene_names)
    except (TypeError, RuntimeError) as exc:
        raise HyperSCACError("gene_names must be a sequence of text labels") from exc
    genes = tuple(_required_text(gene, "gene name") for gene in raw_genes)
    if len(genes) < 2:
        raise HyperSCACError("gene_names must contain at least two genes")
    if len(set(genes)) != len(genes):
        raise HyperSCACError("gene_names must be unique")
    return genes


def _validated_source_variance(
    source_variance: Mapping[str, float],
    genes: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(source_variance, Mapping):
        raise HyperSCACError("source_variance must be a mapping for exactly all genes")
    try:
        items = tuple(source_variance.items())
    except Exception as exc:
        raise HyperSCACError("source_variance could not be read") from exc
    normalized: dict[str, float] = {}
    for raw_gene, raw_value in items:
        gene = _required_text(raw_gene, "source_variance gene")
        if gene in normalized:
            raise HyperSCACError("source_variance must not contain duplicate genes")
        value = _finite_number(raw_value, f"source_variance for {gene}")
        if value < 0.0:
            raise HyperSCACError("source_variance values must be non-negative")
        normalized[gene] = value
    if set(normalized) != set(genes) or len(normalized) != len(genes):
        raise HyperSCACError("source_variance must cover exactly all gene_names")
    return {gene: normalized[gene] for gene in genes}


def _validated_matrix(
    values: object,
    *,
    dimension: int,
    repeat: int,
    context: str,
) -> np.ndarray:
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise HyperSCACError(
            f"repeat {repeat} context {context} matrix could not be read"
        ) from exc
    if array.shape != (dimension, dimension):
        raise HyperSCACError(
            f"repeat {repeat} context {context} matrix shape must be "
            f"{dimension} by {dimension}"
        )
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise HyperSCACError(
            f"repeat {repeat} context {context} matrix must contain real numbers"
        )
    try:
        if not bool(np.isfinite(array).all()):
            raise HyperSCACError(
                f"repeat {repeat} context {context} matrix must be finite"
            )
        if bool(np.any(np.diag(array) != 0.0)):
            raise HyperSCACError(
                f"repeat {repeat} context {context} matrix diagonal must be zero"
            )
        return np.array(array, dtype=np.float64, order="C", copy=True)
    except HyperSCACError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise HyperSCACError(
            f"repeat {repeat} context {context} matrix could not be validated"
        ) from exc


def _validated_repeat_matrices(
    context_matrices: Sequence[Mapping[str, np.ndarray]],
    *,
    dimension: int,
    requested_repeats: int,
) -> tuple[tuple[dict[str, np.ndarray], ...], tuple[str, ...]]:
    if isinstance(context_matrices, (str, bytes)):
        raise HyperSCACError("context_matrices must be a sequence of mappings")
    try:
        repeats = tuple(context_matrices)
    except (TypeError, RuntimeError) as exc:
        raise HyperSCACError(
            "context_matrices must be a sequence of mappings"
        ) from exc
    if len(repeats) > requested_repeats:
        raise HyperSCACError(
            "successful repeats must not exceed requested_repeats"
        )
    expected_contexts: set[str] | None = None
    normalized_repeats: list[dict[str, np.ndarray]] = []
    for repeat_index, result in enumerate(repeats):
        if not isinstance(result, Mapping):
            raise HyperSCACError("each successful repeat must be a context mapping")
        try:
            items = tuple(result.items())
        except Exception as exc:
            raise HyperSCACError(
                f"successful repeat {repeat_index} could not be read"
            ) from exc
        if not items:
            raise HyperSCACError(
                "each successful repeat must contain at least one context"
            )
        normalized: dict[str, np.ndarray] = {}
        for raw_context, matrix in items:
            context = _required_text(raw_context, "context identifier")
            if context in normalized:
                raise HyperSCACError(
                    "a successful repeat must not contain duplicate contexts"
                )
            normalized[context] = _validated_matrix(
                matrix,
                dimension=dimension,
                repeat=repeat_index,
                context=context,
            )
        current_contexts = set(normalized)
        if expected_contexts is None:
            expected_contexts = current_contexts
        elif current_contexts != expected_contexts:
            raise HyperSCACError(
                "all successful repeats must contain exactly the same contexts"
            )
        normalized_repeats.append(normalized)
    context_names = tuple(sorted(expected_contexts)) if expected_contexts else ()
    ordered_repeats = tuple(
        {context: result[context] for context in context_names}
        for result in normalized_repeats
    )
    return ordered_repeats, context_names


def _freeze_json(value: object, path: str) -> object:
    """Validate JSON-compatible content and return a recursively read-only copy."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise HyperSCACError(f"{path} must contain only finite JSON values")
        return normalized
    if isinstance(value, Mapping):
        try:
            items = tuple(value.items())
        except Exception as exc:
            raise HyperSCACError(f"{path} must contain readable JSON values") from exc
        copied: dict[str, object] = {}
        for key, nested in items:
            if not isinstance(key, str):
                raise HyperSCACError(f"{path} JSON mapping keys must be text")
            copied[key] = _freeze_json(nested, f"{path}.{key}")
        return _FrozenJSONDict(copied)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    raise HyperSCACError(f"{path} must contain only JSON-compatible values")


def _validated_failures(failures: object) -> tuple[str, ...]:
    if isinstance(failures, (str, bytes)):
        raise HyperSCACError("failures must be a sequence of text messages")
    try:
        normalized = tuple(failures)  # type: ignore[arg-type]
    except (TypeError, RuntimeError) as exc:
        raise HyperSCACError("failures must be a sequence of text messages") from exc
    if any(not isinstance(message, str) or not message.strip() for message in normalized):
        raise HyperSCACError("failures must contain only non-empty text messages")
    return normalized


def _validated_predictions(predictions: object) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise HyperSCACError("predictions must be a pandas DataFrame")
    if not predictions.columns.is_unique:
        raise HyperSCACError("prediction columns must be unique")
    missing = set(_REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise HyperSCACError(
            f"predictions are missing required columns: {sorted(missing)}"
        )
    if predictions.empty:
        raise HyperSCACError("predictions must contain directed relationships")
    result = predictions.copy(deep=True).reset_index(drop=True)

    for column in ("source", "target"):
        for value in result[column].tolist():
            _required_text(value, column)
    if bool(result.duplicated(["source", "target"]).any()):
        raise HyperSCACError("source-target relationships must be unique")
    if bool((result["source"] == result["target"]).any()):
        raise HyperSCACError("predictions must not contain a self edge")

    metric_columns = [
        "effect",
        "median_effect",
        "selection_frequency",
        "direction_agreement",
        "context_consistency",
        "score",
        *[
            str(column)
            for column in result.columns
            if str(column).startswith("effect_")
            and column not in {"effect", "median_effect"}
        ],
    ]
    for column in metric_columns:
        try:
            values = result[column].to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HyperSCACError(f"prediction metric {column} must be numeric") from exc
        if not bool(np.isfinite(values).all()):
            raise HyperSCACError(f"prediction metric {column} must be finite")
    if not np.array_equal(
        result["effect"].to_numpy(dtype=float),
        result["median_effect"].to_numpy(dtype=float),
    ):
        raise HyperSCACError("effect must equal median_effect")
    for column in (
        "selection_frequency",
        "direction_agreement",
        "context_consistency",
    ):
        values = result[column].to_numpy(dtype=float)
        if bool(np.any((values < 0.0) | (values > 1.0))):
            raise HyperSCACError(f"prediction metric {column} must be in [0, 1]")
    if bool((result["score"].to_numpy(dtype=float) < 0.0).any()):
        raise HyperSCACError("prediction score must be non-negative")

    try:
        directions = result["direction"].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HyperSCACError("prediction direction must be -1, 0, or 1") from exc
    if not bool(np.isfinite(directions).all()) or not bool(
        np.isin(directions, (-1.0, 0.0, 1.0)).all()
    ):
        raise HyperSCACError("prediction direction must be -1, 0, or 1")
    expected_directions = np.sign(
        result["median_effect"].to_numpy(dtype=float)
    )
    if not np.array_equal(directions, expected_directions):
        raise HyperSCACError("prediction direction must equal sign(median_effect)")
    expected_scores = (
        np.abs(result["median_effect"].to_numpy(dtype=float))
        * result["selection_frequency"].to_numpy(dtype=float)
        * result["direction_agreement"].to_numpy(dtype=float)
    )
    if not bool(
        np.isclose(
            result["score"].to_numpy(dtype=float),
            expected_scores,
            rtol=1e-12,
            atol=1e-12,
        ).all()
    ):
        raise HyperSCACError("prediction score must match the frozen score formula")

    if not pd.api.types.is_bool_dtype(result["abstained"].dtype) or bool(
        result["abstained"].isna().any()
    ):
        raise HyperSCACError("prediction abstained values must be bool")
    for reason in result["abstention_reason"].tolist():
        if not isinstance(reason, str):
            raise HyperSCACError("abstention_reason values must be text")
    for _, source_rows in result.groupby("source", sort=False):
        if source_rows["abstained"].nunique(dropna=False) != 1:
            raise HyperSCACError("abstention must be consistent for each source")
        if source_rows["abstention_reason"].nunique(dropna=False) != 1:
            raise HyperSCACError(
                "abstention_reason must be consistent for each source"
            )
        abstained = bool(source_rows["abstained"].iloc[0])
        reason = source_rows["abstention_reason"].iloc[0]
        if abstained == (reason == ""):
            raise HyperSCACError(
                "abstention_reason must be present only for abstained sources"
            )
    return result


def _validated_summary(
    summary: object,
    predictions: pd.DataFrame,
    failures: tuple[str, ...],
) -> Mapping[str, object]:
    if not isinstance(summary, Mapping):
        raise HyperSCACError("summary must be a mapping of JSON-safe values")
    try:
        payload = dict(summary)
    except Exception as exc:
        raise HyperSCACError("summary could not be read") from exc
    missing = set(_SUMMARY_FIELDS) - set(payload)
    if missing:
        raise HyperSCACError(f"summary is missing required fields: {sorted(missing)}")

    requested = _positive_integer(payload["requested_repeats"], "requested_repeats")
    successful_raw = payload["successful_repeats"]
    if isinstance(successful_raw, bool) or not isinstance(successful_raw, Integral):
        raise HyperSCACError("successful_repeats must be a non-negative integer")
    successful = int(successful_raw)
    if not 0 <= successful <= requested:
        raise HyperSCACError(
            "successful_repeats must be between zero and requested_repeats"
        )
    success_fraction = _finite_number(
        payload["repeat_success_fraction"], "repeat_success_fraction"
    )
    expected_success_fraction = successful / requested
    if not math.isclose(
        success_fraction, expected_success_fraction, rel_tol=0.0, abs_tol=1e-12
    ):
        raise HyperSCACError(
            "repeat_success_fraction must equal successful divided by requested"
        )
    if len(failures) != requested - successful:
        raise HyperSCACError(
            "failure count must equal requested_repeats minus successful_repeats"
        )

    coverage = _finite_number(payload["coverage"], "coverage")
    abstention_rate = _finite_number(payload["abstention_rate"], "abstention_rate")
    if not 0.0 <= coverage <= 1.0 or not 0.0 <= abstention_rate <= 1.0:
        raise HyperSCACError("coverage and abstention_rate must be in [0, 1]")
    source_status = predictions.groupby("source", sort=True)["abstained"].first()
    expected_coverage = float((~source_status).mean())
    if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
        raise HyperSCACError("coverage must match source-level abstention")
    if not math.isclose(
        abstention_rate, 1.0 - coverage, rel_tol=0.0, abs_tol=1e-12
    ):
        raise HyperSCACError("abstention_rate must equal one minus coverage")
    if payload["score_formula"] != _SCORE_FORMULA:
        raise HyperSCACError("score_formula does not match the frozen formula")

    payload.update(
        {
            "requested_repeats": requested,
            "successful_repeats": successful,
            "repeat_success_fraction": success_fraction,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
        }
    )
    frozen = _freeze_json(payload, "summary")
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by input check
        raise HyperSCACError("summary must be a JSON mapping")
    return frozen


@dataclass(frozen=True)
class HyperSCAStabilityResult:
    """重复拟合后的完整关系表、覆盖范围和可预期失败。"""

    predictions: pd.DataFrame
    summary: Mapping[str, object]
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        predictions = _validated_predictions(self.predictions)
        failures = _validated_failures(self.failures)
        summary = _validated_summary(self.summary, predictions, failures)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "failures", failures)


def stratified_bootstrap_context(
    context: HyperSCACContext,
    rng: np.random.Generator,
) -> HyperSCACContext:
    """在每个干预标签内部有放回抽样，并保持各组原始大小。"""

    if not isinstance(context, HyperSCACContext):
        raise HyperSCACError("context must be a HyperSCACContext")
    if not isinstance(rng, np.random.Generator):
        raise HyperSCACError("rng must be a numpy random Generator")
    labels_in_order = tuple(dict.fromkeys(context.interventions.tolist()))
    sampled_indices: list[int] = []
    for label in labels_in_order:
        indices = np.flatnonzero(context.interventions == label)
        chosen = rng.choice(indices, size=len(indices), replace=True)
        sampled_indices.extend(np.asarray(chosen, dtype=np.intp).tolist())
    selected = np.asarray(sampled_indices, dtype=np.intp)
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
    """用固定公式形成全部有向关系，并标出无法可靠判断的来源基因。

    每个细胞环境和每次成功重复具有一个等权观测。暂不判断的关系仍保留原始
    ``score`` 以便审计，但稳定排序始终把它们放在可判断关系之后。
    """

    genes = _validated_genes(gene_names)
    threshold = _finite_number(selection_threshold, "selection_threshold")
    if threshold <= 0.0:
        raise HyperSCACError("selection_threshold must be greater than zero")
    requested = _positive_integer(requested_repeats, "requested_repeats")
    minimum_fraction = _finite_number(
        minimum_success_fraction, "minimum_success_fraction"
    )
    if not 0.0 < minimum_fraction <= 1.0:
        raise HyperSCACError("minimum_success_fraction must be in (0, 1]")
    minimum_variance = _finite_number(
        minimum_source_variance, "minimum_source_variance"
    )
    if minimum_variance <= 0.0:
        raise HyperSCACError("minimum_source_variance must be greater than zero")
    variances = _validated_source_variance(source_variance, genes)
    repeats, contexts = _validated_repeat_matrices(
        context_matrices,
        dimension=len(genes),
        requested_repeats=requested,
    )

    successful = len(repeats)
    success_fraction = successful / requested
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(genes):
        if success_fraction < minimum_fraction:
            source_abstained = True
            abstention_reason = "insufficient_successful_bootstraps"
        elif variances[source] < minimum_variance:
            source_abstained = True
            abstention_reason = "source_has_no_control_variation"
        else:
            source_abstained = False
            abstention_reason = ""
        for target_index, target in enumerate(genes):
            if source == target:
                continue
            values_by_context = {
                context: np.asarray(
                    [
                        result[context][source_index, target_index]
                        for result in repeats
                    ],
                    dtype=np.float64,
                )
                for context in contexts
            }
            if contexts:
                values = np.concatenate(
                    [values_by_context[context] for context in contexts]
                )
            else:
                values = np.asarray([], dtype=np.float64)
            if values.size:
                median_effect = float(np.median(values))
                selected = np.abs(values) >= threshold
                selection_frequency = float(selected.mean())
                if bool(selected.any()):
                    selected_signs = np.sign(values[selected])
                    direction_agreement = float(
                        max(
                            (selected_signs > 0).mean(),
                            (selected_signs < 0).mean(),
                        )
                    )
                else:
                    direction_agreement = 0.0
            else:
                median_effect = 0.0
                selection_frequency = 0.0
                direction_agreement = 0.0
            context_effects = {
                context: float(np.median(context_values))
                for context, context_values in values_by_context.items()
            }
            selected_context_signs = np.asarray(
                [
                    np.sign(effect)
                    for effect in context_effects.values()
                    if abs(effect) >= threshold
                ],
                dtype=np.float64,
            )
            if selected_context_signs.size:
                context_consistency = float(
                    max(
                        (selected_context_signs > 0).mean(),
                        (selected_context_signs < 0).mean(),
                    )
                )
            else:
                context_consistency = 0.0
            score = (
                abs(median_effect)
                * selection_frequency
                * direction_agreement
            )
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
                        f"effect_{context}": context_effects[context]
                        for context in contexts
                    },
                    "score": float(score),
                    "abstained": source_abstained,
                    "abstention_reason": abstention_reason,
                }
            )
    columns = [
        "source",
        "target",
        "effect",
        "median_effect",
        "direction",
        "selection_frequency",
        "direction_agreement",
        "context_consistency",
        *[f"effect_{context}" for context in contexts],
        "score",
        "abstained",
        "abstention_reason",
    ]
    predictions = pd.DataFrame(rows, columns=columns).sort_values(
        ["abstained", "score", "source", "target"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    coverage = float(
        np.mean(
            [
                not (
                    success_fraction < minimum_fraction
                    or variances[source] < minimum_variance
                )
                for source in genes
            ]
        )
    )
    summary: dict[str, object] = {
        "requested_repeats": requested,
        "successful_repeats": successful,
        "repeat_success_fraction": success_fraction,
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
        "score_formula": _SCORE_FORMULA,
    }
    return predictions.reset_index(drop=True), summary


def _compact_failure(repeat: int, error: HyperSCACError) -> str:
    message = " ".join(str(error).split()) or error.__class__.__name__
    if len(message) > 200:
        message = message[:197] + "..."
    return f"repeat_{repeat}:{message}"


def fit_stable_hypersca_c(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None = None,
) -> HyperSCAStabilityResult:
    """重复拟合 HyperSCA-C，并把可预期失败转为覆盖范围说明。"""

    if not isinstance(config, HyperSCACConfig):
        raise HyperSCACError("config must be a validated HyperSCACConfig")
    normalized_contexts, genes = _core._validated_contexts(contexts)
    if config.selection_threshold <= 0.0:
        raise HyperSCACError("selection_threshold must be greater than zero")
    normalized_seed = _core._validated_seed(seed)
    if normalized_seed + config.bootstrap_repeats - 1 > 2**64 - 1:
        raise HyperSCACError(
            "seed plus bootstrap repeats must remain within the supported seed range"
        )
    target_device = _core._validated_device(device)
    _core._prior_weights(
        prior_mask,
        dimension=len(genes),
        prior_discount=config.prior_discount,
        device=target_device,
    )

    context_control_variances: list[np.ndarray] = []
    for context in normalized_contexts:
        control_mask = context.interventions == config.control_label
        if int(control_mask.sum()) < 2:
            raise HyperSCACError(
                f"context {context.context_id} requires at least two control cells"
            )
        control_values = context.expression[control_mask].astype(
            np.float64, copy=False
        )
        context_control_variances.append(control_values.var(axis=0, ddof=0))
    source_variance = {
        gene: float(
            np.mean([variance[index] for variance in context_control_variances])
        )
        for index, gene in enumerate(genes)
    }

    matrices: list[Mapping[str, np.ndarray]] = []
    failures: list[str] = []
    for repeat in range(config.bootstrap_repeats):
        repeat_seed = normalized_seed + repeat
        repeat_rng = np.random.default_rng(repeat_seed)
        sampled = [
            stratified_bootstrap_context(context, repeat_rng)
            for context in normalized_contexts
        ]
        try:
            fit = fit_hypersca_c_once(
                sampled,
                config,
                seed=repeat_seed,
                device=device,
                prior_mask=prior_mask,
            )
        except HyperSCACError as exc:
            failures.append(_compact_failure(repeat, exc))
            continue
        matrices.append(fit.context_adjacencies)

    predictions, summary = build_stability_table(
        matrices,
        genes,
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
