"""任务 C 比较方法、来源和可用信息的固定登记表。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class TaskCMethodRegistryError(ValueError):
    """方法登记表缺少来源、输入边界或运行信息。"""


@dataclass(frozen=True)
class TaskCMethodSpec:
    """一种比较方法及其允许使用的研究信息。"""

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
    """固定版本的比较方法清单。"""

    schema_version: str
    methods: Mapping[str, TaskCMethodSpec]
    causalbench: Mapping[str, str]

    def __post_init__(self) -> None:
        methods, causalbench = _freeze_and_validate_registry(
            self.schema_version,
            self.methods,
            self.causalbench,
        )
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "causalbench", causalbench)


_OFFICIAL_RETURN_ORDER = (
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
)
_NO_OUTPUT_ORDER = ("betterboost", "sparse_rc", "catran")
_METHOD_ORDER = _OFFICIAL_RETURN_ORDER + _NO_OUTPUT_ORDER
_MAX_REGISTRY_BYTES = 64 * 1024
_MAX_JSON_NESTING = 32

_CAUSALBENCH = {
    "repository": "https://github.com/causalbench/causalbench.git",
    "commit": "1a2143cffdc85f835b41ce8d52034be1bf903e71",
    "environment": "hypersca-task-c-causalbench",
}

_METHODS: dict[str, dict[str, object]] = {
    "hypersca_c": {
        "role": "candidate",
        "source_kind": "local",
        "training_information": "partial_interventional",
        "command": "local_hypersca_c",
        "required_for_core_rehearsal": True,
    },
    "mean_difference": {
        "role": "simple_baseline",
        "source_kind": "local",
        "training_information": "partial_interventional",
        "command": "local_mean_difference",
        "required_for_core_rehearsal": True,
    },
    "random1000": {
        "role": "null_control",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "random1000",
        "required_for_core_rehearsal": True,
    },
    "grnboost": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "grnboost",
        "required_for_core_rehearsal": True,
    },
    "pc": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "pc",
        "required_for_core_rehearsal": True,
    },
    "ges": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "ges",
        "required_for_core_rehearsal": False,
    },
    "gies": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "partial_interventional",
        "command": "gies",
        "required_for_core_rehearsal": True,
    },
    "gsp": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "gsp",
        "required_for_core_rehearsal": False,
    },
    "igsp": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "partial_interventional",
        "command": "igsp",
        "required_for_core_rehearsal": False,
    },
    "notears_linear": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "notears-lin-sparse",
        "required_for_core_rehearsal": True,
    },
    "dcdi_g": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "partial_interventional",
        "command": "DCDI-G",
        "required_for_core_rehearsal": False,
    },
    "dcdi_dsf": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "partial_interventional",
        "command": "DCDI-DSF",
        "required_for_core_rehearsal": False,
    },
    "dcdfg_linear": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "partial_interventional",
        "command": "DCDFG-LIN",
        "required_for_core_rehearsal": False,
    },
    "dcdfg_mlp": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "partial_interventional",
        "command": "DCDFG-MLP",
        "required_for_core_rehearsal": False,
    },
    "sortnregress": {
        "role": "external_comparator",
        "source_kind": "causalbench",
        "training_information": "observational",
        "command": "sortnregress",
        "required_for_core_rehearsal": False,
    },
    "guanlab_psgrn": {
        "role": "external_comparator",
        "source_kind": "git",
        "training_information": "partial_interventional",
        "command": "psgrn",
        "required_for_core_rehearsal": False,
        "repository": "https://github.com/GuanLab/PSGRN.git",
        "commit": "74aa640f7c472b23a69811f6795bb17678efd344",
        "environment": "hypersca-task-c-psgrn",
    },
    "betterboost": {
        "role": "external_comparator",
        "source_kind": "publication_only",
        "training_information": "partial_interventional",
        "command": None,
        "required_for_core_rehearsal": False,
        "publication": "https://openreview.net/forum?id=gpDOOAOmMe",
    },
    "sparse_rc": {
        "role": "external_comparator",
        "source_kind": "publication_only",
        "training_information": "partial_interventional",
        "command": None,
        "required_for_core_rehearsal": False,
        "publication": "https://openreview.net/forum?id=TOaPl9tXlmD",
    },
    "catran": {
        "role": "external_comparator",
        "source_kind": "publication_only",
        "training_information": "partial_interventional",
        "command": None,
        "required_for_core_rehearsal": False,
        "publication": "https://openreview.net/forum?id=Wf0QRYUkhwV",
    },
}


def _expected_output_semantics(method_id: str) -> str:
    if method_id in _OFFICIAL_RETURN_ORDER:
        return "official_return_order"
    return "no_output"


def _validate_causalbench_values(raw: Mapping[str, object]) -> Mapping[str, str]:
    _require_exact_fields(raw, set(_CAUSALBENCH), "causalbench")
    validated: dict[str, str] = {}
    for field, expected in _CAUSALBENCH.items():
        value = raw[field]
        if not isinstance(value, str) or not value:
            raise TaskCMethodRegistryError(
                f"causalbench {field} must be a non-empty string"
            )
        if value != expected:
            raise TaskCMethodRegistryError(f"causalbench {field} must remain fixed")
        validated[field] = value
    return MappingProxyType(validated)


def _validate_constructed_method(
    method_id: str, spec: object
) -> TaskCMethodSpec:
    if not isinstance(spec, TaskCMethodSpec):
        raise TaskCMethodRegistryError(
            f"{method_id} must be a TaskCMethodSpec record"
        )
    if spec.method_id != method_id:
        raise TaskCMethodRegistryError(f"{method_id} method_id must remain fixed")
    if type(spec.required_for_core_rehearsal) is not bool:
        raise TaskCMethodRegistryError(
            f"{method_id} required_for_core_rehearsal must be a real boolean"
        )

    expected = {
        **_METHODS[method_id],
        "output_semantics": _expected_output_semantics(method_id),
        "repository": _METHODS[method_id].get("repository"),
        "commit": _METHODS[method_id].get("commit"),
        "environment": _METHODS[method_id].get("environment"),
        "publication": _METHODS[method_id].get("publication"),
    }
    for field, expected_value in expected.items():
        if getattr(spec, field) != expected_value:
            raise TaskCMethodRegistryError(
                f"{method_id} {field} must remain {expected_value!r}"
            )
    return spec


def _freeze_and_validate_registry(
    schema_version: object,
    methods_value: object,
    causalbench_value: object,
) -> tuple[Mapping[str, TaskCMethodSpec], Mapping[str, str]]:
    if schema_version != "1.0":
        raise TaskCMethodRegistryError("schema_version must be 1.0")
    if not isinstance(methods_value, Mapping):
        raise TaskCMethodRegistryError("methods must be a mapping")
    if len(methods_value) != len(_METHOD_ORDER) or set(methods_value) != set(
        _METHOD_ORDER
    ):
        raise TaskCMethodRegistryError(
            "method registry must contain the fixed method set"
        )
    if tuple(methods_value) != _METHOD_ORDER:
        raise TaskCMethodRegistryError("method order must remain fixed")
    if not isinstance(causalbench_value, Mapping):
        raise TaskCMethodRegistryError("causalbench must be a mapping")

    methods = {
        method_id: _validate_constructed_method(
            method_id, methods_value[method_id]
        )
        for method_id in _METHOD_ORDER
    }
    causalbench = _validate_causalbench_values(causalbench_value)
    return MappingProxyType(methods), causalbench


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskCMethodRegistryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_excessive_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise TaskCMethodRegistryError(
                    "cannot parse method registry: JSON is too deeply nested"
                )
        elif character in "]}":
            depth -= 1


def _require_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TaskCMethodRegistryError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    raw: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise TaskCMethodRegistryError(
            f"{label} fields must be exactly {sorted(expected)}; "
            f"missing={missing}, extra={extra}"
        )


def _read_payload(path: str | Path) -> Mapping[str, Any]:
    try:
        registry_path = Path(path)
        if registry_path.stat().st_size > _MAX_REGISTRY_BYTES:
            raise TaskCMethodRegistryError(
                f"method registry is too large; maximum is {_MAX_REGISTRY_BYTES} bytes"
            )
        text = registry_path.read_text(encoding="utf-8")
        if len(text.encode("utf-8")) > _MAX_REGISTRY_BYTES:
            raise TaskCMethodRegistryError(
                f"method registry is too large; maximum is {_MAX_REGISTRY_BYTES} bytes"
            )
        _reject_excessive_json_nesting(text)
        payload = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except TaskCMethodRegistryError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise TaskCMethodRegistryError(
            f"cannot parse method registry: {exc}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskCMethodRegistryError(f"cannot read method registry: {exc}") from exc
    return _require_object(payload, "method registry")


def _validate_output_groups(
    raw: object, method_ids: set[str]
) -> dict[str, str]:
    groups = _require_object(raw, "output_semantics")
    _require_exact_fields(
        groups,
        {"official_return_order", "no_output"},
        "output_semantics",
    )
    expected_lengths = {
        "official_return_order": len(_OFFICIAL_RETURN_ORDER),
        "no_output": len(_NO_OUTPUT_ORDER),
    }
    for group_name, expected_length in expected_lengths.items():
        members = groups[group_name]
        if not isinstance(members, list):
            raise TaskCMethodRegistryError(
                f"output_semantics {group_name} must be a list of method names"
            )
        if len(members) != expected_length:
            raise TaskCMethodRegistryError(
                f"{group_name} must contain exactly {expected_length} methods"
            )

    method_semantics: dict[str, str] = {}
    duplicates: list[str] = []
    for group_name in ("official_return_order", "no_output"):
        for method_id in groups[group_name]:
            if not isinstance(method_id, str):
                raise TaskCMethodRegistryError(
                    f"output_semantics {group_name} must be a list of method names"
                )
            if method_id in method_semantics:
                duplicates.append(method_id)
            else:
                method_semantics[method_id] = group_name
    if duplicates:
        raise TaskCMethodRegistryError(
            "methods listed more than once in output_semantics: "
            f"{sorted(set(duplicates))}"
        )
    if set(method_semantics) != method_ids:
        raise TaskCMethodRegistryError(
            "every fixed method must be listed exactly once in output_semantics"
        )
    if tuple(groups["official_return_order"]) != _OFFICIAL_RETURN_ORDER:
        raise TaskCMethodRegistryError("official_return_order must remain fixed")
    if tuple(groups["no_output"]) != _NO_OUTPUT_ORDER:
        raise TaskCMethodRegistryError("no_output order must remain fixed")
    return method_semantics


def _validate_causalbench(raw: object) -> Mapping[str, str]:
    source = _require_object(raw, "causalbench")
    return _validate_causalbench_values(source)


def _validate_method(
    method_id: str, raw_value: object, output_semantics: str
) -> TaskCMethodSpec:
    raw = _require_object(raw_value, f"method {method_id}")
    expected = _METHODS[method_id]
    _require_exact_fields(raw, set(expected), f"fields for {method_id}")

    required = raw["required_for_core_rehearsal"]
    if type(required) is not bool:
        raise TaskCMethodRegistryError(
            f"{method_id} required_for_core_rehearsal must be a real boolean"
        )
    source_kind = raw["source_kind"]
    if source_kind == "publication_only":
        if raw["command"] is not None:
            raise TaskCMethodRegistryError(
                f"publication-only method {method_id} cannot declare a command"
            )
        if required:
            raise TaskCMethodRegistryError(
                f"publication-only method {method_id} cannot be required for rehearsal"
            )
    if source_kind == "git" and (
        not isinstance(raw.get("environment"), str) or not raw.get("environment")
    ):
        raise TaskCMethodRegistryError(
            f"git method {method_id} needs a non-empty environment"
        )

    for field, expected_value in expected.items():
        if raw[field] != expected_value:
            raise TaskCMethodRegistryError(
                f"{method_id} {field} must remain {expected_value!r}"
            )

    return TaskCMethodSpec(
        method_id=method_id,
        role=str(raw["role"]),
        source_kind=str(raw["source_kind"]),
        training_information=str(raw["training_information"]),
        command=raw["command"],
        required_for_core_rehearsal=required,
        output_semantics=output_semantics,
        repository=raw.get("repository"),
        commit=raw.get("commit"),
        environment=raw.get("environment"),
        publication=raw.get("publication"),
    )


def load_task_c_method_registry(path: str | Path) -> TaskCMethodRegistry:
    """读取已审核的比较清单，并拒绝改变方法身份或证据边界的内容。"""

    payload = _read_payload(path)
    _require_exact_fields(
        payload,
        {"schema_version", "causalbench", "output_semantics", "methods"},
        "registry",
    )
    if payload["schema_version"] != "1.0":
        raise TaskCMethodRegistryError("schema_version must be 1.0")

    raw_methods = _require_object(payload["methods"], "methods")
    if set(raw_methods) != set(_METHOD_ORDER):
        raise TaskCMethodRegistryError(
            "method registry must contain the fixed method set"
        )
    semantics = _validate_output_groups(payload["output_semantics"], set(raw_methods))
    causalbench = _validate_causalbench(payload["causalbench"])
    methods = {
        method_id: _validate_method(
            method_id, raw_methods[method_id], semantics[method_id]
        )
        for method_id in _METHOD_ORDER
    }
    return TaskCMethodRegistry(
        schema_version="1.0",
        methods=MappingProxyType(methods),
        causalbench=causalbench,
    )
