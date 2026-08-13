from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistryError,
    load_task_c_method_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/task_c_methods_v1.json"
OFFICIAL_RETURN_ORDER = (
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
NO_OUTPUT_ORDER = ("betterboost", "sparse_rc", "catran")


def _payload() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registry_covers_confirmed_comparison_methods_in_fixed_order() -> None:
    registry = load_task_c_method_registry(REGISTRY_PATH)

    assert tuple(registry.methods) == OFFICIAL_RETURN_ORDER + NO_OUTPUT_ORDER
    assert tuple(
        method_id
        for method_id, method in registry.methods.items()
        if method.output_semantics == "official_return_order"
    ) == OFFICIAL_RETURN_ORDER
    assert tuple(
        method_id
        for method_id, method in registry.methods.items()
        if method.output_semantics == "no_output"
    ) == NO_OUTPUT_ORDER


def test_publication_only_methods_cannot_claim_runnable_code() -> None:
    registry = load_task_c_method_registry(REGISTRY_PATH)

    for method_id in NO_OUTPUT_ORDER:
        method = registry.methods[method_id]
        assert method.source_kind == "publication_only"
        assert method.command is None
        assert method.publication is not None
        assert method.required_for_core_rehearsal is False


def test_every_method_has_explicit_output_semantics() -> None:
    registry = load_task_c_method_registry(REGISTRY_PATH)

    assert all(
        method.output_semantics in {"official_return_order", "no_output"}
        for method in registry.methods.values()
    )
    assert registry.methods["guanlab_psgrn"].output_semantics == (
        "official_return_order"
    )
    assert registry.methods["catran"].output_semantics == "no_output"


def test_loaded_registry_cannot_be_changed_by_a_runner() -> None:
    registry = load_task_c_method_registry(REGISTRY_PATH)

    with pytest.raises(TypeError):
        registry.methods["pc"] = registry.methods["ges"]  # type: ignore[index]
    with pytest.raises(TypeError):
        registry.causalbench["commit"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("method_id", "replacement"),
    [
        ("random1000", "partial_interventional"),
        ("grnboost", "partial_interventional"),
        ("pc", "partial_interventional"),
        ("ges", "partial_interventional"),
        ("gsp", "partial_interventional"),
        ("notears_linear", "partial_interventional"),
        ("sortnregress", "partial_interventional"),
        ("hypersca_c", "observational"),
    ],
)
def test_registry_rejects_changed_training_information(
    tmp_path: Path, method_id: str, replacement: str
) -> None:
    payload = _payload()
    payload["methods"][method_id]["training_information"] = replacement  # type: ignore[index]

    with pytest.raises(
        TaskCMethodRegistryError,
        match=rf"{method_id} training_information must remain",
    ):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_registry_requires_the_fixed_method_set(tmp_path: Path, change: str) -> None:
    payload = _payload()
    methods = payload["methods"]  # type: ignore[index]
    semantics = payload["output_semantics"]  # type: ignore[index]
    if change == "missing":
        del methods["pc"]  # type: ignore[index]
        semantics["official_return_order"].remove("pc")  # type: ignore[index]
    else:
        methods["invented_method"] = methods["pc"]  # type: ignore[index]
        semantics["official_return_order"].append("invented_method")  # type: ignore[index]

    with pytest.raises(TaskCMethodRegistryError, match="fixed method set"):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_group", "output_semantics fields"),
        ("unknown_group", "output_semantics fields"),
        ("duplicate_membership", "listed more than once"),
        ("uncovered", "listed exactly once"),
        ("changed_order", "official_return_order must remain"),
    ],
)
def test_registry_rejects_incomplete_or_changed_output_groups(
    tmp_path: Path, mutation: str, message: str
) -> None:
    payload = _payload()
    groups = payload["output_semantics"]  # type: ignore[index]
    if mutation == "missing_group":
        del groups["no_output"]  # type: ignore[index]
    elif mutation == "unknown_group":
        groups["ranked_edges"] = []  # type: ignore[index]
    elif mutation == "duplicate_membership":
        groups["no_output"].append("pc")  # type: ignore[index]
    elif mutation == "uncovered":
        groups["official_return_order"].remove("pc")  # type: ignore[index]
    else:
        groups["official_return_order"][0:2] = [  # type: ignore[index]
            "mean_difference",
            "hypersca_c",
        ]

    with pytest.raises(TaskCMethodRegistryError, match=message):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("scope", "mutation", "message"),
    [
        ("top", "extra", "registry fields"),
        ("top", "missing", "registry fields"),
        ("method", "extra", "fields for pc"),
        ("method", "missing", "fields for pc"),
        ("causalbench", "extra", "causalbench fields"),
        ("causalbench", "missing", "causalbench fields"),
    ],
)
def test_registry_schema_is_closed(
    tmp_path: Path, scope: str, mutation: str, message: str
) -> None:
    payload = _payload()
    if scope == "top":
        target = payload
        field = "schema_version"
    elif scope == "method":
        target = payload["methods"]["pc"]  # type: ignore[index]
        field = "role"
    else:
        target = payload["causalbench"]  # type: ignore[index]
        field = "repository"
    if mutation == "extra":
        target["unreviewed_field"] = "value"  # type: ignore[index]
    else:
        del target[field]  # type: ignore[index]

    with pytest.raises(TaskCMethodRegistryError, match=message):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


def test_registry_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = REGISTRY_PATH.read_text(encoding="utf-8").replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0", "schema_version": "1.0",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(TaskCMethodRegistryError, match="duplicate JSON key"):
        load_task_c_method_registry(path)


@pytest.mark.parametrize(
    ("method_id", "field", "value", "message"),
    [
        ("pc", "role", "candidate", "pc role must remain"),
        ("pc", "source_kind", "local", "pc source_kind must remain"),
        ("pc", "command", "not-pc", "pc command must remain"),
        ("guanlab_psgrn", "commit", "0" * 40, "guanlab_psgrn commit must remain"),
        ("betterboost", "publication", "", "publication must remain"),
    ],
)
def test_method_identity_and_source_are_fixed(
    tmp_path: Path, method_id: str, field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload["methods"][method_id][field] = value  # type: ignore[index]

    with pytest.raises(TaskCMethodRegistryError, match=message):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("method_id", "field", "value", "message"),
    [
        ("betterboost", "command", "pretend-runner", "cannot declare a command"),
        ("betterboost", "required_for_core_rehearsal", True, "cannot be required"),
        ("guanlab_psgrn", "environment", "", "non-empty environment"),
        ("pc", "repository", "https://example.invalid/code.git", "fields for pc"),
        ("hypersca_c", "publication", "https://example.invalid/paper", "fields for hypersca_c"),
    ],
)
def test_source_specific_boundaries_are_enforced(
    tmp_path: Path, method_id: str, field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload["methods"][method_id][field] = value  # type: ignore[index]

    with pytest.raises(TaskCMethodRegistryError, match=message):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


def test_core_rehearsal_flag_requires_a_real_boolean(tmp_path: Path) -> None:
    payload = _payload()
    payload["methods"]["pc"]["required_for_core_rehearsal"] = 1  # type: ignore[index]

    with pytest.raises(TaskCMethodRegistryError, match="real boolean"):
        load_task_c_method_registry(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("bad_root", [[], "registry", 1, None])
def test_registry_rejects_non_object_documents(
    tmp_path: Path, bad_root: object
) -> None:
    with pytest.raises(TaskCMethodRegistryError, match="JSON object"):
        load_task_c_method_registry(_write_payload(tmp_path, bad_root))
