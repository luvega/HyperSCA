"""Task S own-only and fixed distance-decay spatial baselines.

Both baselines consume the same non-spatial own-effect prediction.  This keeps
upstream model quality fixed while testing whether a spatial method contributes
information about effects in non-perturbed neighboring units.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluation.benchmark_contract import (
    build_run_manifest,
    contract_digest,
    validate_benchmark_contract,
)


SUPPORTED_BASELINES = {"own_only", "fixed_distance_decay"}
SUPPORTED_DATA_STATUSES = {"external_benchmark", "synthetic_smoke"}
REQUIRED_COLUMNS = {
    "unit_id",
    "sample_id",
    "spatial_block",
    "perturbation_id",
    "feature_id",
    "distance",
    "is_perturbed",
    "own_effect_prediction",
    "observed_effect",
}


class TaskSBenchmarkError(ValueError):
    """Raised when Task S inputs or evidence violate the adapter contract."""


@dataclass(frozen=True)
class TaskSBaselineResult:
    """Per-unit predictions and coverage summary for one simple baseline."""

    predictions: pd.DataFrame
    summary: dict[str, Any]


def _coerce_boolean(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise TaskSBenchmarkError("is_perturbed must contain only boolean values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    coerced = values.map(
        lambda value: mapping.get(value.lower(), None)
        if isinstance(value, str)
        else mapping.get(value, None)
    )
    if coerced.isna().any():
        raise TaskSBenchmarkError("is_perturbed must contain only boolean values")
    return coerced.astype(bool)


def _validate_and_normalize_holdout(holdout: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(holdout, pd.DataFrame) or holdout.empty:
        raise TaskSBenchmarkError("holdout must be a non-empty pandas DataFrame")
    missing = REQUIRED_COLUMNS - set(holdout.columns)
    if missing:
        raise TaskSBenchmarkError(
            f"holdout is missing required columns: {sorted(missing)}"
        )
    frame = holdout.copy()
    for column in (
        "unit_id",
        "sample_id",
        "spatial_block",
        "perturbation_id",
        "feature_id",
    ):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise TaskSBenchmarkError(f"{column} must contain non-empty values")
        frame[column] = frame[column].astype(str)
    identity_columns = ["unit_id", "perturbation_id", "feature_id"]
    if frame.duplicated(identity_columns).any():
        raise TaskSBenchmarkError(
            "holdout must contain unique unit/perturbation/feature rows"
        )
    frame["is_perturbed"] = _coerce_boolean(frame["is_perturbed"])
    frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
    frame["observed_effect"] = pd.to_numeric(
        frame["observed_effect"],
        errors="coerce",
    )
    frame["own_effect_prediction"] = pd.to_numeric(
        frame["own_effect_prediction"],
        errors="coerce",
    )
    if not np.isfinite(frame["distance"]).all():
        raise TaskSBenchmarkError("distance must contain finite values")
    if (frame["distance"] < 0).any():
        raise TaskSBenchmarkError("distance must be non-negative")
    if not np.isfinite(frame["observed_effect"]).all():
        raise TaskSBenchmarkError("observed_effect must contain finite values")
    own_values = frame["own_effect_prediction"].to_numpy(dtype=float)
    if np.isinf(own_values).any():
        raise TaskSBenchmarkError(
            "own_effect_prediction may be missing but must not be infinite"
        )
    source_distance = frame.loc[frame["is_perturbed"], "distance"].to_numpy()
    if len(source_distance) == 0 or not np.allclose(source_distance, 0.0):
        raise TaskSBenchmarkError("perturbed units must have distance zero")
    if (~frame["is_perturbed"]).sum() == 0:
        raise TaskSBenchmarkError("holdout must include non-perturbed neighbor units")
    return frame


def predict_task_s_baseline(
    holdout: pd.DataFrame,
    *,
    baseline_id: str,
    length_scale: float | None = None,
) -> TaskSBaselineResult:
    """Predict own and neighbor effects with a pre-registered simple baseline."""
    if baseline_id not in SUPPORTED_BASELINES:
        raise TaskSBenchmarkError(
            f"baseline_id must be one of {sorted(SUPPORTED_BASELINES)}"
        )
    if baseline_id == "fixed_distance_decay":
        if (
            not isinstance(length_scale, (int, float))
            or isinstance(length_scale, bool)
            or not math.isfinite(float(length_scale))
            or float(length_scale) <= 0.0
        ):
            raise TaskSBenchmarkError(
                "fixed_distance_decay requires a finite positive length_scale"
            )
        numeric_length_scale = float(length_scale)
    else:
        if length_scale is not None:
            raise TaskSBenchmarkError("own_only does not accept a length_scale")
        numeric_length_scale = None

    frame = _validate_and_normalize_holdout(holdout)
    own_effect = frame["own_effect_prediction"].to_numpy(dtype=float)
    covered = np.isfinite(own_effect)
    perturbed = frame["is_perturbed"].to_numpy(dtype=bool)
    if baseline_id == "own_only":
        predicted = np.where(perturbed, own_effect, 0.0)
        formula = "own_effect_prediction for own units; 0 for neighbors"
    else:
        distance = frame["distance"].to_numpy(dtype=float)
        predicted = own_effect * np.exp(-distance / numeric_length_scale)
        formula = (
            "own_effect_prediction * exp(-distance / "
            f"{numeric_length_scale})"
        )
    predicted = np.where(covered, predicted, np.nan)
    frame["predicted_effect"] = predicted
    frame["endpoint"] = np.where(perturbed, "own", "neighbor")
    frame["abstained"] = ~covered
    coverage = float(covered.mean())
    summary = {
        "baseline_id": baseline_id,
        "formula": formula,
        "length_scale": numeric_length_scale,
        "n_rows": int(len(frame)),
        "n_own_rows": int(perturbed.sum()),
        "n_neighbor_rows": int((~perturbed).sum()),
        "n_samples": int(frame["sample_id"].nunique()),
        "n_spatial_blocks": int(frame["spatial_block"].nunique()),
        "n_perturbations": int(frame["perturbation_id"].nunique()),
        "n_features": int(frame["feature_id"].nunique()),
        "coverage": coverage,
        "abstention_rate": float(1.0 - coverage),
    }
    return TaskSBaselineResult(predictions=frame, summary=summary)


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) == 0:
        return None
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def _pearson(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 2 or np.std(observed) < 1e-12 or np.std(predicted) < 1e-12:
        return None
    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return correlation if math.isfinite(correlation) else None


def _distance_calibration_error(neighbor: pd.DataFrame) -> float | None:
    if neighbor.empty:
        return None
    unique_distances = int(neighbor["distance"].nunique())
    n_bins = min(4, unique_distances)
    if n_bins < 1:
        return None
    if n_bins == 1:
        bins = pd.Series("all", index=neighbor.index)
    else:
        bins = pd.qcut(neighbor["distance"], q=n_bins, duplicates="drop")
    group_columns = [
        "sample_id",
        "spatial_block",
        "perturbation_id",
        "feature_id",
        "_distance_bin",
    ]
    grouped = neighbor.assign(_distance_bin=bins).groupby(
        group_columns,
        observed=True,
    )
    errors = (
        grouped["predicted_effect"].mean()
        - grouped["observed_effect"].mean()
    ).abs()
    return float(np.mean(errors))


def evaluate_task_s_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate own and neighbor endpoints separately on covered holdout rows."""
    required = {
        "predicted_effect",
        "observed_effect",
        "is_perturbed",
        "abstained",
        "distance",
        "sample_id",
        "spatial_block",
        "perturbation_id",
        "feature_id",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise TaskSBenchmarkError(
            f"predictions are missing required columns: {sorted(missing)}"
        )
    frame = predictions.copy()
    covered = ~frame["abstained"].astype(bool)
    own = frame[covered & frame["is_perturbed"].astype(bool)]
    neighbor = frame[covered & ~frame["is_perturbed"].astype(bool)]
    own_observed = own["observed_effect"].to_numpy(dtype=float)
    own_predicted = own["predicted_effect"].to_numpy(dtype=float)
    neighbor_observed = neighbor["observed_effect"].to_numpy(dtype=float)
    neighbor_predicted = neighbor["predicted_effect"].to_numpy(dtype=float)
    primary = _rmse(neighbor_observed, neighbor_predicted)
    return {
        "status": "evaluated" if primary is not None else "insufficient_neighbor_rows",
        "primary_metric": "neighbor_effect_rmse",
        "neighbor_effect_rmse": primary,
        "own_effect_rmse": _rmse(own_observed, own_predicted),
        "neighbor_effect_pcc": _pearson(neighbor_observed, neighbor_predicted),
        "distance_decay_calibration_error": _distance_calibration_error(neighbor),
        "coverage": float(covered.mean()),
        "abstention_rate": float((~covered).mean()),
        "n_evaluated_own_rows": int(len(own)),
        "n_evaluated_neighbor_rows": int(len(neighbor)),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TaskSBenchmarkError(f"{name} must be a non-empty string")


def run_task_s_baseline(
    *,
    holdout: pd.DataFrame,
    contract: Mapping[str, Any],
    baseline_id: str,
    dataset_id: str,
    dataset_source: str,
    data_status: str,
    input_digest: str,
    own_effect_source_id: str,
    own_effect_source_digest: str,
    train_only_attested: bool,
    nonadjacent_blocks_attested: bool,
    code_revision: str,
    random_seed: int,
    output_dir: str | Path,
    length_scale: float | None = None,
    length_scale_source_id: str | None = None,
    length_scale_source_digest: str | None = None,
) -> dict[str, Any]:
    """Run one Task S baseline and write a contract-bound artifact bundle."""
    validate_benchmark_contract(contract)
    if baseline_id not in contract["tasks"]["S"]["required_simple_baselines"]:
        raise TaskSBenchmarkError(
            f"baseline_id is not registered for Task S: {baseline_id}"
        )
    if data_status not in SUPPORTED_DATA_STATUSES:
        raise TaskSBenchmarkError(
            f"data_status must be one of {sorted(SUPPORTED_DATA_STATUSES)}"
        )
    for name, value in {
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "own_effect_source_id": own_effect_source_id,
    }.items():
        _require_nonempty(value, name)
    if data_status == "external_benchmark" and not (
        train_only_attested and nonadjacent_blocks_attested
    ):
        raise TaskSBenchmarkError(
            "external benchmark requires train-only and nonadjacent-block attestations"
        )
    if baseline_id == "fixed_distance_decay":
        if length_scale_source_id is None or length_scale_source_digest is None:
            raise TaskSBenchmarkError(
                "fixed_distance_decay requires length-scale source provenance"
            )
        _require_nonempty(length_scale_source_id, "length_scale_source_id")
    elif any(
        value is not None
        for value in (
            length_scale,
            length_scale_source_id,
            length_scale_source_digest,
        )
    ):
        raise TaskSBenchmarkError(
            "own_only must not receive length-scale parameters or provenance"
        )

    result = predict_task_s_baseline(
        holdout,
        baseline_id=baseline_id,
        length_scale=length_scale,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(destination / "predictions.csv", index=False)

    input_artifacts = {
        "spatial_holdout": input_digest,
        "own_effect_source": own_effect_source_digest,
    }
    if length_scale_source_digest is not None:
        input_artifacts["length_scale_source"] = length_scale_source_digest
    manifest = build_run_manifest(
        contract,
        task_id="S",
        dataset_id=dataset_id,
        method_id=baseline_id,
        method_role="simple_baseline",
        code_revision=code_revision,
        random_seed=random_seed,
        input_artifacts=input_artifacts,
    )
    manifest.update(
        {
            "dataset_source": dataset_source,
            "data_status": data_status,
            "own_effect_source_id": own_effect_source_id,
            "parameters": {
                "length_scale": result.summary["length_scale"],
                "length_scale_source_id": length_scale_source_id,
                "formula": result.summary["formula"],
            },
        }
    )
    input_summary = {
        **result.summary,
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": "S",
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "data_status": data_status,
        "input_digest": input_digest,
        "own_effect_source_id": own_effect_source_id,
        "leakage_checks": {
            "own_effect_train_only_attested": bool(train_only_attested),
            "nonadjacent_spatial_blocks_attested": bool(
                nonadjacent_blocks_attested
            ),
            "holdout_outcomes_used_to_set_length_scale": False,
        },
    }
    metrics = {
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": "S",
        "method_id": baseline_id,
        **evaluate_task_s_predictions(result.predictions),
        "external_holdout_evaluated": (
            data_status == "external_benchmark"
            and train_only_attested
            and nonadjacent_blocks_attested
        ),
        "null_controls_passed": False,
        "null_control_status": "not_run",
        "interpretation": (
            "Own and neighbor endpoints are separate; this simple baseline does "
            "not establish a spatial causal mechanism."
        ),
    }
    promotion_decision = {
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": "S",
        "method_id": baseline_id,
        "status": "not_applicable_simple_baseline",
        "claim_level": "baseline_only",
        "synthetic_smoke": data_status == "synthetic_smoke",
        "promotion_eligible": False,
        "reason": (
            "This artifact defines a required simple spatial baseline and cannot "
            "promote itself or establish a HyperSCA superiority claim."
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    _write_json(destination / "run_manifest.json", manifest)
    _write_json(destination / "input_summary.json", input_summary)
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "promotion_decision.json", promotion_decision)
    return {
        "manifest": manifest,
        "input_summary": input_summary,
        "metrics": metrics,
        "promotion_decision": promotion_decision,
        "predictions": result.predictions,
        "output_dir": destination,
    }
