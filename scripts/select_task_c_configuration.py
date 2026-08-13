#!/usr/bin/env python3
"""只用公开调节细胞选择 Task C 方法设置，不读取最终参考关系。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_method_run import (  # noqa: E402
    MAXIMUM_RAW_PREDICTION_BYTES,
    MAXIMUM_RECORD_BYTES,
    TaskCMethodRunError,
    _capture_file,
    _capture_public_input,
    _capture_synthetic_input,
    _load_fixed_npz,
    _parse_json,
    _validate_status_seal,
    _verify_snapshot,
)
from src.evaluation.task_c_profile_input import (  # noqa: E402
    TaskCProfileInputError,
    validate_task_c_profile_input,
)
from src.evaluation.task_c_tuning import (  # noqa: E402
    CONTROL_LABEL,
    EXCLUDED_LABEL,
    MAXIMUM_TRIALS,
    TaskCTuningError,
    build_tuning_response_edges,
    load_task_c_tuning_config,
    select_task_c_configuration,
    thaw_task_c_json,
)


_SMOKE_TRIAL_FIELDS = frozenset(
    {"schema_version", "trial_index", "method_id", "condition", "profile", "parameters"}
)
_FORMAL_TRIAL_FIELDS = _SMOKE_TRIAL_FIELDS | frozenset(
    {
        "run_identity_sha256",
        "tune_input_sha256",
        "public_manifest_sha256",
        "profile_manifest_sha256",
    }
)
_MAXIMUM_TRIAL_JSON_BYTES = 1024 * 1024
_CONDITIONS = frozenset({"within_environment", "cross_environment"})
_PROFILES = frozenset({"connection", "comprehensive"})
_COMPLETED_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "method_id",
        "status",
        "run_identity_sha256",
        "artifacts",
        "inner_status",
        "status_origin",
        "status_content_sha256",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        "method_id",
        "role",
        "source_kind",
        "training_information",
        "output_semantics",
        "data_status",
        "context_id",
        "seed",
        "min_cells",
        "registry_sha256",
        "input",
        "derived_input_manifest",
        "public_manifest",
        "registered_method",
        "code",
        "assets",
        "command",
        "python",
        "run_identity",
        "run_identity_sha256",
    }
)


class _PlainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法选择 Task C 设置：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _PlainArgumentParser(
        description=(
            "根据公开调节干预的细胞响应，从最多二十组候选设置中选择一组；"
            "该命令没有最终参考关系或留出集入口。"
        )
    )
    parser.add_argument("--tune-npz", required=True, type=Path, help="公开调节细胞文件。")
    parser.add_argument(
        "--trial-dir",
        required=True,
        action="append",
        type=Path,
        help="一次已完成方法运行及其候选设置记录，可重复但最多二十次。",
    )
    parser.add_argument("--output-json", required=True, type=Path, help="新的选择记录文件。")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="固定的调节规则。",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        help="正式公开数据的文件清单；正式选择时必需。",
    )
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        help="若调节文件是公开清单派生的小型版本，提供其来源记录。",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="明确标记仅检查流程的合成数据，不能作为正式结果。",
    )
    return parser


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _snapshot(path: Path, label: str, *, maximum_bytes: int):
    try:
        return _capture_file(
            path,
            label,
            maximum_bytes=maximum_bytes,
            reject_private=True,
            require_single_link=True,
        )
    except TaskCMethodRunError as exc:
        raise TaskCTuningError(str(exc)) from exc


def _strict_json(snapshot: object, label: str) -> dict[str, Any]:
    try:
        return _parse_json(snapshot, label)  # type: ignore[arg-type]
    except TaskCMethodRunError as exc:
        raise TaskCTuningError(str(exc)) from exc


def _safe_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise TaskCTuningError(f"{label} does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCTuningError(f"{label} must not contain a symbolic link")
    if not absolute.is_dir():
        raise TaskCTuningError(f"{label} must be a directory")
    return absolute


def _verify_method_artifacts(
    root: Path, status: Mapping[str, object]
) -> tuple[object, ...]:
    snapshots: list[object] = []
    observed: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"method_status.json", "trial_parameters.json"}:
            continue
        snapshot = _snapshot(
            path,
            f"trial artifact {relative}",
            maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
        )
        snapshots.append(snapshot)
        observed[relative] = {
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size,
        }
    if status.get("artifacts") != observed:
        raise TaskCTuningError("trial artifact inventory or hash changed")
    return tuple(snapshots)


def _read_predictions(snapshot: object, expected_rows: int) -> pd.DataFrame:
    payload = snapshot.payload  # type: ignore[attr-defined]
    if payload.count(b"\n") != expected_rows + 1:
        raise TaskCTuningError("trial prediction row count differs from the complete universe")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise TaskCTuningError("trial predictions are not a valid bounded CSV table") from exc
    if len(frame) != expected_rows:
        raise TaskCTuningError("trial prediction row count differs from the complete universe")
    return frame


def _trial_record(
    trial_dir: Path,
    *,
    synthetic_smoke: bool,
    expected_rows: int,
    evidence_hashes: Mapping[str, str | None],
) -> tuple[
    int,
    Mapping[str, object],
    pd.DataFrame,
    dict[str, object],
    tuple[object, ...],
]:
    root = _safe_directory(trial_dir, "trial directory")
    top = {path.name for path in root.iterdir()}
    required = {"predictions.csv", "trial_parameters.json"}
    if not required.issubset(top):
        raise TaskCTuningError("trial directory lacks predictions or parameter record")
    if synthetic_smoke:
        if top != required:
            raise TaskCTuningError("synthetic trial directory must contain exactly two files")
    elif not {"method_status.json", "environment_manifest.json"}.issubset(top):
        raise TaskCTuningError("formal trial needs completed method status and environment evidence")
    prediction_snapshot = _snapshot(
        root / "predictions.csv",
        "trial predictions",
        maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
    )
    parameter_snapshot = _snapshot(
        root / "trial_parameters.json",
        "trial parameters",
        maximum_bytes=_MAXIMUM_TRIAL_JSON_BYTES,
    )
    parameters = _strict_json(parameter_snapshot, "trial parameters")
    expected_fields = _SMOKE_TRIAL_FIELDS if synthetic_smoke else _FORMAL_TRIAL_FIELDS
    if set(parameters) != expected_fields or parameters.get("schema_version") != "1.0":
        raise TaskCTuningError("trial parameter fields changed")
    trial_index = parameters.get("trial_index")
    method = parameters.get("method_id")
    condition = parameters.get("condition")
    profile = parameters.get("profile")
    if (
        isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or trial_index < 0
        or not isinstance(method, str)
        or not method
        or condition not in _CONDITIONS
        or profile not in _PROFILES
        or not isinstance(parameters.get("parameters"), dict)
    ):
        raise TaskCTuningError(
            "trial index, method, condition, profile, or parameter identity is malformed"
        )
    snapshots: list[object] = [prediction_snapshot, parameter_snapshot]
    if not synthetic_smoke:
        status_snapshot = _snapshot(
            root / "method_status.json",
            "trial method status",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
        )
        environment_snapshot = _snapshot(
            root / "environment_manifest.json",
            "trial environment record",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
        )
        status = _strict_json(status_snapshot, "trial method status")
        environment = _strict_json(environment_snapshot, "trial environment record")
        try:
            _validate_status_seal(status)
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc
        if set(status) != _COMPLETED_STATUS_FIELDS:
            raise TaskCTuningError("completed method status fields changed")
        if set(environment) != _ENVIRONMENT_FIELDS:
            raise TaskCTuningError("completed method environment fields changed")
        run_identity = environment.get("run_identity")
        if not isinstance(run_identity, dict) or environment.get(
            "run_identity_sha256"
        ) != _sha256(_json_bytes(run_identity)):
            raise TaskCTuningError("completed method run identity changed")
        context_id = environment.get("context_id")
        expected_context = (
            context_id in {"k562", "rpe1"}
            if condition == "within_environment"
            else context_id in {"k562_to_rpe1", "rpe1_to_k562"}
        )
        if (
            environment.get("schema_version") != "1.0"
            or status.get("schema_version") != "1.0"
            or status.get("status") != "completed_standardized_output"
            or status.get("method_id") != method
            or environment.get("method_id") != method
            or environment.get("data_status") != "external_benchmark"
            or not expected_context
            or status.get("run_identity_sha256")
            != environment.get("run_identity_sha256")
            or parameters.get("run_identity_sha256")
            != status.get("run_identity_sha256")
        ):
            raise TaskCTuningError("formal trial is not the recorded completed method run")
        for field, expected in evidence_hashes.items():
            if parameters.get(field) != expected:
                raise TaskCTuningError(f"trial {field} differs from the tuning evidence")
        snapshots.extend((status_snapshot, environment_snapshot))
        artifacts = status.get("artifacts")
        if not isinstance(artifacts, dict) or "predictions.csv" not in artifacts:
            raise TaskCTuningError("formal trial status lacks standardized predictions")
        predicted = artifacts["predictions.csv"]
        if (
            not isinstance(predicted, dict)
            or predicted.get("sha256") != prediction_snapshot.sha256
            or predicted.get("size_bytes") != prediction_snapshot.size
        ):
            raise TaskCTuningError("formal trial predictions differ from completed run evidence")
        snapshots.extend(_verify_method_artifacts(root, status))
    frame = _read_predictions(prediction_snapshot, expected_rows)
    identity = {
        "method_id": method,
        "condition": condition,
        "profile": profile,
        "trial_index": trial_index,
        "parameters_sha256": parameter_snapshot.sha256,
        "predictions_sha256": prediction_snapshot.sha256,
        "run_identity_sha256": parameters.get("run_identity_sha256"),
    }
    return (
        trial_index,
        parameters["parameters"],
        frame,
        identity,
        tuple(snapshots),
    )


def _load_tune_input(args: argparse.Namespace):
    if args.synthetic_smoke:
        if args.public_manifest is not None or args.profile_manifest is not None:
            raise TaskCTuningError(
                "synthetic smoke data must not use a public or profile manifest"
            )
        try:
            input_snapshot = _capture_synthetic_input(args.tune_npz)
            expression, labels, genes, environments = _load_fixed_npz(input_snapshot)
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc
        return (
            expression,
            labels,
            genes,
            frozenset(set(labels) - {CONTROL_LABEL, EXCLUDED_LABEL}),
            {
                "tune_input_sha256": input_snapshot.sha256,
                "public_manifest_sha256": None,
                "profile_manifest_sha256": None,
            },
            (input_snapshot,),
            "synthetic_smoke",
        )
    if args.public_manifest is None:
        raise TaskCTuningError("formal selection requires the matching public manifest")
    if args.profile_manifest is None:
        try:
            input_snapshot, public_snapshot, public, relative = _capture_public_input(
                args.tune_npz, args.public_manifest
            )
            expression, labels, genes, environments = _load_fixed_npz(input_snapshot)
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc
        if not (
            relative.startswith("within/") and relative.endswith("/tune.npz")
            or relative.startswith("cross/") and relative.endswith("/source_tune.npz")
        ):
            raise TaskCTuningError("formal tuning input must be the registered tune partition")
        if public.get("min_cells_per_intervention", 0) < 5:
            raise TaskCTuningError("public tuning groups must keep at least five cells")
        sources = frozenset(public["tune_sources"])
        if set(labels) - {CONTROL_LABEL, EXCLUDED_LABEL} != set(sources):
            raise TaskCTuningError(
                "registered tune cells must contain exactly the public tune sources"
            )
        snapshots = (input_snapshot, public_snapshot)
        profile_hash = None
    else:
        try:
            validated = validate_task_c_profile_input(
                input_path=args.tune_npz,
                profile_manifest_path=args.profile_manifest,
                public_manifest_path=args.public_manifest,
            )
        except TaskCProfileInputError as exc:
            raise TaskCTuningError(str(exc)) from exc
        if validated.manifest.get("stage") != "tune":
            raise TaskCTuningError("formal profile input must record the tune stage")
        expression = validated.expression
        labels = validated.interventions
        genes = validated.gene_names
        sources = frozenset(set(labels) - {CONTROL_LABEL, EXCLUDED_LABEL})
        input_snapshot = validated.input_snapshot
        public_snapshot = validated.public_snapshot
        snapshots = (
            input_snapshot,
            validated.manifest_snapshot,
            public_snapshot,
            *validated.parent_snapshots,
        )
        profile_hash = validated.manifest_sha256
    if not sources.issubset(set(labels)):
        raise TaskCTuningError("public tune sources differ from the tuning cell labels")
    return (
        expression,
        labels,
        genes,
        sources,
        {
            "tune_input_sha256": input_snapshot.sha256,
            "public_manifest_sha256": public_snapshot.sha256,
            "profile_manifest_sha256": profile_hash,
        },
        snapshots,
        "external_benchmark",
    )


def _verify_snapshots(snapshots: Sequence[object]) -> None:
    for index, snapshot in enumerate(snapshots):
        try:
            _verify_snapshot(snapshot, f"fixed tuning input {index + 1}")  # type: ignore[arg-type]
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc


def _publish_json(path: Path, payload: bytes) -> None:
    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    parent = destination.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _safe_directory(parent, "output parent")
    if destination.exists() or destination.is_symlink():
        raise TaskCTuningError("output JSON already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise TaskCTuningError("output JSON already exists") from exc
        temporary.unlink()
        published = True
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if len(args.trial_dir) > MAXIMUM_TRIALS:
            raise TaskCTuningError("configuration selection allows at most twenty trials")
        config_snapshot = _snapshot(
            args.config, "tuning configuration", maximum_bytes=MAXIMUM_RECORD_BYTES
        )
        config = load_task_c_tuning_config(config_snapshot.path)
        (
            expression,
            labels,
            genes,
            sources,
            evidence_hashes,
            tune_snapshots,
            data_status,
        ) = _load_tune_input(args)
        tuning_edges = build_tuning_response_edges(
            expression,
            labels,
            genes,
            eligible_sources=sources,
            q_value_threshold=config.q_value_threshold,
        )
        trials = []
        trial_snapshots: list[object] = []
        identities: list[dict[str, object]] = []
        expected_rows = len(genes) * (len(genes) - 1)
        for trial_dir in args.trial_dir:
            index, parameters, frame, identity, snapshots = _trial_record(
                trial_dir,
                synthetic_smoke=args.synthetic_smoke,
                expected_rows=expected_rows,
                evidence_hashes=evidence_hashes,
            )
            trials.append((index, parameters, frame))
            identities.append(identity)
            trial_snapshots.extend(snapshots)
        shared_identity_fields = ("method_id", "condition", "profile")
        if any(
            any(identity[field] != identities[0][field] for field in shared_identity_fields)
            for identity in identities[1:]
        ):
            raise TaskCTuningError(
                "all trials must use the same method, condition, and profile"
            )
        selection = select_task_c_configuration(
            trials,
            tuning_edges=tuning_edges,
            maximum_trials=config.maximum_trials_per_method,
            gene_names=genes,
        )
        code_snapshots = tuple(
            _snapshot(path, "tuning code", maximum_bytes=MAXIMUM_RECORD_BYTES)
            for path in (
                ROOT / "src/evaluation/task_c_tuning.py",
                ROOT / "scripts/select_task_c_configuration.py",
            )
        )
        code_identity = _sha256(
            b"".join(snapshot.payload for snapshot in code_snapshots)
        )
        result = thaw_task_c_json(dict(selection))
        assert isinstance(result, dict)
        result["method_id"] = identities[0]["method_id"]
        result["condition"] = identities[0]["condition"]
        result["profile"] = identities[0]["profile"]
        result["evidence"] = {
            "data_status": data_status,
            **evidence_hashes,
            "config_sha256": config_snapshot.sha256,
            "code_sha256": code_identity,
            "tuning_positive_relation_count": len(tuning_edges),
            "gene_count": len(genes),
            "trials": sorted(identities, key=lambda item: int(item["trial_index"])),
        }
        result["selection_identity_sha256"] = _sha256(_json_bytes(result))
        all_snapshots = (
            config_snapshot,
            *tune_snapshots,
            *trial_snapshots,
            *code_snapshots,
        )
        _verify_snapshots(all_snapshots)
        _publish_json(args.output_json, _json_bytes(result))
    except (
        TaskCTuningError,
        TaskCMethodRunError,
        TaskCProfileInputError,
        OSError,
        UnicodeError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
