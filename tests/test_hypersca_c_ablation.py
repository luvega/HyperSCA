from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext, HyperSCACError
from src.evaluation.task_c_data import (
    build_shared_task_c_split,
    load_task_c_dataset,
    materialize_task_c_split,
    sha256_path,
)


ROOT = Path(__file__).resolve().parents[1]
ABLATION_IDS = (
    "primary",
    "shared_only",
    "separate_contexts",
    "observational_only",
    "no_stability_weighting",
    "acyclicity_off",
    "acyclicity_strong",
    "prior_on_secondary",
)
ARTIFACT_NAMES = {
    "raw_predictions.csv",
    "fit_summary.json",
    "method_status.json",
    "run_manifest.json",
}


def test_default_prior_trust_registry_is_fail_closed() -> None:
    payload = json.loads(
        (ROOT / "configs/hypersca_c_prior_trust_v1.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "schema_version": "1.0",
        "relation_fingerprint_schema": "directed_edge_set_v1",
        "intersection_audit_schema": "exact_directed_edge_set_intersection_v1",
        "approved_priors": [],
    }


def _config(**changes: object) -> HyperSCACConfig:
    payload = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    payload.update(changes)
    return HyperSCACConfig.from_mapping(payload)


def _context(context_id: str = "k562") -> HyperSCACContext:
    return HyperSCACContext(
        context_id=context_id,
        expression=np.asarray(
            [
                [0.0, 0.2, 0.4],
                [0.4, 0.8, 0.2],
                [1.0, 0.1, 0.5],
                [1.2, 0.3, 0.6],
                [0.1, 1.0, 0.7],
                [0.2, 1.2, 0.9],
            ],
            dtype=np.float32,
        ),
        interventions=np.asarray(
            ["non-targeting", "non-targeting", "A", "A", "B", "B"]
        ),
        gene_names=("A", "B", "C"),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _registry_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "ablations": {
            "primary": {"mode": "joint", "configuration_changes": {}},
            "shared_only": {
                "mode": "joint",
                "configuration_changes": {"enable_context_adjustments": False},
            },
            "separate_contexts": {
                "mode": "one_context_per_fit",
                "configuration_changes": {},
            },
            "observational_only": {
                "mode": "control_cells_only",
                "configuration_changes": {},
            },
            "no_stability_weighting": {
                "mode": "joint",
                "configuration_changes": {"bootstrap_repeats": 1},
            },
            "acyclicity_off": {
                "mode": "joint",
                "configuration_changes": {"acyclicity_weight": 0.0},
            },
            "acyclicity_strong": {
                "mode": "joint",
                "configuration_changes": {"acyclicity_weight": 0.1},
            },
            "prior_on_secondary": {
                "mode": "joint_with_external_prior",
                "configuration_changes": {"prior_discount": 0.5},
            },
        },
    }


def test_registry_is_exact_ordered_and_deeply_read_only() -> None:
    from src.causal.hypersca_c_ablation import load_hypersca_c_ablations

    registry = load_hypersca_c_ablations(ROOT / "configs/hypersca_c_ablations_v1.json")
    assert isinstance(registry, MappingProxyType)
    assert tuple(registry) == ABLATION_IDS
    assert {key: dict(value) for key, value in registry.items()} == {
        key: {
            "mode": value["mode"],
            "configuration_changes": value["configuration_changes"],
        }
        for key, value in _registry_payload()["ablations"].items()
    }
    with pytest.raises(TypeError):
        registry["primary"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        registry["shared_only"]["configuration_changes"][  # type: ignore[index]
            "enable_context_adjustments"
        ] = True


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":"1.0","schema_version":"1.0","ablations":{}}',
        '{"schema_version":"1.0","ablations":{},"extra":1}',
        '{"schema_version":"1.0","ablations":{"primary":{"mode":"joint",'
        '"configuration_changes":{"acyclicity_weight":NaN}}}}',
    ],
)
def test_registry_rejects_duplicate_unknown_and_nonfinite_json(
    tmp_path: Path, raw: str
) -> None:
    from src.causal.hypersca_c_ablation import load_hypersca_c_ablations

    path = tmp_path / "invalid.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(HyperSCACError):
        load_hypersca_c_ablations(path)


def test_registry_rejects_missing_reordered_or_changed_candidate(
    tmp_path: Path,
) -> None:
    from src.causal.hypersca_c_ablation import load_hypersca_c_ablations

    cases: list[dict[str, object]] = []
    missing = _registry_payload()
    missing["ablations"].pop("shared_only")  # type: ignore[union-attr]
    cases.append(missing)
    reordered = _registry_payload()
    reordered["ablations"] = dict(  # type: ignore[index]
        reversed(tuple(reordered["ablations"].items()))  # type: ignore[union-attr]
    )
    cases.append(reordered)
    changed = _registry_payload()
    changed["ablations"]["acyclicity_strong"]["configuration_changes"][  # type: ignore[index]
        "acyclicity_weight"
    ] = 0.2
    cases.append(changed)
    for index, payload in enumerate(cases):
        path = tmp_path / f"invalid_{index}.json"
        _write_json(path, payload)
        with pytest.raises(HyperSCACError, match="固定|登记|顺序"):
            load_hypersca_c_ablations(path)


@pytest.mark.parametrize(
    "forgery",
    ["mode", "prior", "nested", "duplicate_items", "unknown", "nonfinite"],
)
def test_library_registry_argument_cannot_change_the_fixed_eight_ablations(
    forgery: str,
) -> None:
    from src.causal.hypersca_c_ablation import apply_hypersca_c_ablation

    registry = _registry_payload()["ablations"]
    assert isinstance(registry, dict)
    if forgery == "mode":
        registry["primary"]["mode"] = "control_cells_only"  # type: ignore[index]
    elif forgery == "prior":
        registry["prior_on_secondary"]["configuration_changes"][  # type: ignore[index]
            "prior_discount"
        ] = 0.9
    elif forgery == "nested":
        registry["primary"]["configuration_changes"] = []  # type: ignore[index]
    elif forgery == "duplicate_items":

        class DuplicateItems(dict[str, object]):
            def items(self) -> object:
                return (*super().items(), ("primary", super().__getitem__("primary")))

        registry = DuplicateItems(registry)
    elif forgery == "unknown":
        registry["primary"]["unknown"] = True  # type: ignore[index]
    else:
        registry["acyclicity_strong"]["configuration_changes"][  # type: ignore[index]
            "acyclicity_weight"
        ] = float("nan")
    with pytest.raises(HyperSCACError, match="登记|八项|固定|重复|格式|映射"):
        apply_hypersca_c_ablation(
            (_context(),), _config(), "primary", registry=registry
        )


@pytest.mark.parametrize(
    ("ablation_id", "field", "expected"),
    [
        ("shared_only", "enable_context_adjustments", False),
        ("no_stability_weighting", "bootstrap_repeats", 1),
        ("acyclicity_off", "acyclicity_weight", 0.0),
        ("acyclicity_strong", "acyclicity_weight", 0.1),
        ("prior_on_secondary", "prior_discount", 0.5),
    ],
)
def test_ablation_changes_only_its_registered_setting(
    ablation_id: str, field: str, expected: object
) -> None:
    from src.causal.hypersca_c_ablation import apply_hypersca_c_ablation

    original_context = _context()
    original_config = _config()
    before_config = asdict(original_config)
    before_expression = original_context.expression.copy()
    transformed, changed = apply_hypersca_c_ablation(
        (original_context,), original_config, ablation_id
    )
    expected_config = dict(before_config)
    expected_config[field] = expected
    assert asdict(changed) == expected_config
    assert asdict(original_config) == before_config
    assert transformed == (original_context,)
    assert np.array_equal(original_context.expression, before_expression)
    assert not original_context.expression.flags.writeable


def test_primary_and_separate_contexts_leave_data_and_config_unchanged() -> None:
    from src.causal.hypersca_c_ablation import apply_hypersca_c_ablation

    contexts = (_context("k562"), _context("rpe1"))
    config = _config()
    for ablation_id in ("primary", "separate_contexts"):
        transformed, transformed_config = apply_hypersca_c_ablation(
            contexts, config, ablation_id
        )
        assert transformed == contexts
        assert transformed_config == config


def test_observational_ablation_removes_perturbed_cells_instead_of_relabeling() -> None:
    from src.causal.hypersca_c_ablation import apply_hypersca_c_ablation

    original = _context()
    transformed, transformed_config = apply_hypersca_c_ablation(
        (original,), _config(), "observational_only"
    )
    assert transformed[0].expression.shape == (2, 3)
    assert transformed[0].interventions.tolist() == [
        "non-targeting",
        "non-targeting",
    ]
    assert transformed_config == _config()
    assert original.expression.shape == (6, 3)


def test_separate_contexts_calls_one_fit_per_context_and_returns_complete_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    calls: list[tuple[str, ...]] = []

    def fake_fit_once(
        contexts: tuple[HyperSCACContext, ...] | list[HyperSCACContext],
        config: HyperSCACConfig,
        *,
        seed: int,
        device: str,
        prior_mask: np.ndarray | None = None,
    ) -> object:
        del config, seed, device, prior_mask
        calls.append(tuple(context.context_id for context in contexts))
        context = contexts[0]
        matrix = np.zeros((3, 3), dtype=np.float32)
        matrix[0, 1] = 0.3 if context.context_id == "k562" else 0.5
        return SimpleNamespace(context_adjacencies={context.context_id: matrix})

    monkeypatch.setattr(module, "fit_hypersca_c_once", fake_fit_once)
    result = module.fit_separate_contexts_stable(
        (_context("k562"), _context("rpe1")),
        _config(bootstrap_repeats=2),
        seed=17,
        device="cpu",
    )
    assert calls == [("k562",), ("rpe1",), ("k562",), ("rpe1",)]
    assert len(result.predictions) == 6
    assert {"effect_k562", "effect_rpe1"} <= set(result.predictions)
    assert result.summary["requested_repeats"] == 2


def test_separate_contexts_attempts_later_context_after_an_earlier_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    calls: list[str] = []

    def fake_fit_once(
        contexts: tuple[HyperSCACContext, ...] | list[HyperSCACContext],
        config: HyperSCACConfig,
        *,
        seed: int,
        device: str,
        prior_mask: np.ndarray | None = None,
    ) -> object:
        del config, seed, device, prior_mask
        context = contexts[0]
        calls.append(context.context_id)
        if context.context_id == "k562":
            raise HyperSCACError("k562 独立拟合失败")
        matrix = np.zeros((3, 3), dtype=np.float32)
        return SimpleNamespace(context_adjacencies={context.context_id: matrix})

    monkeypatch.setattr(module, "fit_hypersca_c_once", fake_fit_once)
    result = module.fit_separate_contexts_stable(
        (_context("k562"), _context("rpe1")),
        _config(bootstrap_repeats=1),
        seed=17,
        device="cpu",
    )
    assert calls == ["k562", "rpe1"]
    assert result.summary["successful_repeats"] == 0
    assert result.failures == ("repeat_0:k562:k562 独立拟合失败",)


def test_primary_ablation_delegates_to_the_same_stability_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    sentinel = object()
    calls: list[tuple[object, ...]] = []

    def fake_fit(*args: object, **kwargs: object) -> object:
        calls.append((*args, kwargs))
        return sentinel

    monkeypatch.setattr(module, "fit_stable_hypersca_c", fake_fit)
    contexts = (_context(),)
    config = _config(bootstrap_repeats=1)
    result = module.fit_hypersca_c_ablation(
        contexts,
        config,
        "primary",
        seed=19,
        device="cpu",
    )
    assert result is sentinel
    assert calls == [
        (contexts, config, {"seed": 19, "device": "cpu", "prior_mask": None})
    ]


def _expected_relation_fingerprint(rows: list[tuple[str, str]]) -> str:
    ordered = sorted(set(rows), key=lambda edge: (edge[0].encode(), edge[1].encode()))
    canonical = b"directed_edge_set_v1\0" + json.dumps(
        [[source, target] for source, target in ordered],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _prior_manifest(edges: Path) -> dict[str, object]:
    frame = pd.read_csv(edges, dtype=str, keep_default_na=False)
    rows = list(frame[["source", "target"]].itertuples(index=False, name=None))
    fingerprint = _expected_relation_fingerprint(rows)
    return {
        "schema_version": "1.0",
        "prior_id": "independent-pathway-v1",
        "source_uri": "https://example.org/independent-pathway-v1",
        "source_description": "独立于任务 C 评分关系的预登记通路关系",
        "prior_edges_sha256": sha256_path(edges),
        "relation_fingerprint_schema": "directed_edge_set_v1",
        "relation_fingerprint": fingerprint,
        "scoring_reference_fingerprints": {
            "pooled_essentiality": "sha256:" + "2" * 64,
            "chip_directional_reference": "sha256:" + "3" * 64,
        },
        "independence_attestation": {
            "reuses_pooled_essentiality": False,
            "reuses_chip_directional_reference": False,
        },
    }


def _install_prior_trust(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
    edges: Path,
    source_manifest: Path,
    *,
    audit_status: str = "no_overlap",
) -> Path:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    trust = directory / "hypersca_c_prior_trust_v1.json"
    _write_json(
        trust,
        {
            "schema_version": "1.0",
            "relation_fingerprint_schema": "directed_edge_set_v1",
            "intersection_audit_schema": "exact_directed_edge_set_intersection_v1",
            "approved_priors": [
                {
                    "prior_id": source["prior_id"],
                    "source_uri": source["source_uri"],
                    "prior_source_manifest_sha256": sha256_path(source_manifest),
                    "prior_edges_sha256": sha256_path(edges),
                    "relation_fingerprint": source["relation_fingerprint"],
                    "scoring_reference_fingerprints": source[
                        "scoring_reference_fingerprints"
                    ],
                    "intersection_audit": {
                        "status": audit_status,
                        "pooled_essentiality_overlap_count": 0,
                        "chip_directional_reference_overlap_count": 0,
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(module, "_DEFAULT_PRIOR_TRUST_REGISTRY", trust, raising=False)
    return trust


def test_prior_edges_form_ordered_mask_only_with_independent_registered_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module

    edges = tmp_path / "prior.csv"
    pd.DataFrame(
        [{"source": "C", "target": "A"}, {"source": "A", "target": "B"}]
    ).to_csv(edges, index=False)
    manifest = tmp_path / "prior_manifest.json"
    _write_json(manifest, _prior_manifest(edges))
    trust = _install_prior_trust(module, monkeypatch, tmp_path, edges, manifest)
    mask, record, snapshots = module.load_registered_prior(
        edges, manifest, ("C", "A", "B")
    )
    assert mask.tolist() == [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    assert record["prior_id"] == "independent-pathway-v1"
    assert record["edge_count"] == 2
    assert record["prior_edges_sha256"] == sha256_path(edges)
    assert not mask.flags.writeable
    assert record["prior_trust_registry_path"] == str(trust)
    assert record["intersection_audit"]["status"] == "no_overlap"
    assert len(snapshots) == 3


def test_relation_fingerprint_is_independent_of_csv_row_order(tmp_path: Path) -> None:
    from src.causal.hypersca_c_ablation import _directed_relation_fingerprint

    rows = [("C", "A"), ("A", "B")]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(rows, columns=["source", "target"]).to_csv(first, index=False)
    pd.DataFrame(list(reversed(rows)), columns=["source", "target"]).to_csv(
        second, index=False
    )
    expected = _expected_relation_fingerprint(rows)
    assert _directed_relation_fingerprint(rows) == expected
    assert _directed_relation_fingerprint(list(reversed(rows))) == expected
    assert sha256_path(first) != sha256_path(second)


@pytest.mark.parametrize(
    "case", ["overlap", "self_edge", "unknown_gene", "extra_column"]
)
def test_prior_rejects_scoring_overlap_and_invalid_relations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    import src.causal.hypersca_c_ablation as module

    row: dict[str, str] = {"source": "A", "target": "B"}
    if case == "self_edge":
        row["target"] = "A"
    if case == "unknown_gene":
        row["target"] = "D"
    if case == "extra_column":
        row["weight"] = "1"
    edges = tmp_path / f"{case}.csv"
    pd.DataFrame([row]).to_csv(edges, index=False)
    payload = _prior_manifest(edges)
    if case == "overlap":
        payload["scoring_reference_fingerprints"][  # type: ignore[index]
            "pooled_essentiality"
        ] = payload["relation_fingerprint"]
    manifest = tmp_path / f"{case}.json"
    _write_json(manifest, payload)
    _install_prior_trust(module, monkeypatch, tmp_path, edges, manifest)
    with pytest.raises(HyperSCACError, match="重用|重叠|自身|基因|列"):
        module.load_registered_prior(edges, manifest, ("C", "A", "B"))


def test_unanchored_prior_cannot_self_authorize_with_its_own_manifest(
    prepared_batch: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module

    edges = tmp_path / "self-authorized.csv"
    pd.DataFrame([{"source": "C", "target": "A"}]).to_csv(edges, index=False)
    source_manifest = tmp_path / "self-authorized.json"
    _write_json(source_manifest, _prior_manifest(edges))
    observed: list[str] = []

    def fast_fit(
        contexts: tuple[HyperSCACContext, ...],
        config: HyperSCACConfig,
        ablation_id: str,
        **kwargs: object,
    ) -> object:
        del kwargs
        observed.append(ablation_id)
        return _fast_stability_result(contexts, config)

    monkeypatch.setattr(module, "fit_hypersca_c_ablation", fast_fit)
    kwargs = _batch_kwargs(prepared_batch, output=tmp_path / "unanchored")
    kwargs.update(
        prior_edges_path=edges,
        prior_source_manifest_path=source_manifest,
    )
    statuses = module.run_hypersca_c_ablations(**kwargs)
    assert statuses["prior_on_secondary"] == "official_assets_unavailable"
    assert "prior_on_secondary" not in observed


@pytest.mark.parametrize("private_part", ["edges", "source_manifest", "trust"])
def test_prior_boundary_rejects_any_private_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_part: str
) -> None:
    import src.causal.hypersca_c_ablation as module

    edges = tmp_path / "prior.csv"
    pd.DataFrame([{"source": "C", "target": "A"}]).to_csv(edges, index=False)
    source_manifest = tmp_path / "prior.json"
    _write_json(source_manifest, _prior_manifest(edges))
    trust = _install_prior_trust(module, monkeypatch, tmp_path, edges, source_manifest)
    private = tmp_path / "private"
    private.mkdir()
    chosen_edges, chosen_manifest = edges, source_manifest
    if private_part == "edges":
        chosen_edges = private / edges.name
        chosen_edges.write_bytes(edges.read_bytes())
    elif private_part == "source_manifest":
        chosen_manifest = private / source_manifest.name
        chosen_manifest.write_bytes(source_manifest.read_bytes())
    else:
        private_trust = private / trust.name
        private_trust.write_bytes(trust.read_bytes())
        monkeypatch.setattr(module, "_DEFAULT_PRIOR_TRUST_REGISTRY", private_trust)
    with pytest.raises(HyperSCACError, match="private|公开|路径"):
        module.load_registered_prior(chosen_edges, chosen_manifest, ("C", "A", "B"))


def test_trusted_exact_intersection_audit_must_report_no_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module

    edges = tmp_path / "partly-overlapping.csv"
    pd.DataFrame([{"source": "C", "target": "A"}]).to_csv(edges, index=False)
    source_manifest = tmp_path / "partly-overlapping.json"
    _write_json(source_manifest, _prior_manifest(edges))
    _install_prior_trust(
        module,
        monkeypatch,
        tmp_path,
        edges,
        source_manifest,
        audit_status="overlap_detected",
    )
    with pytest.raises(HyperSCACError, match="审计|重叠|交集"):
        module.load_registered_prior(edges, source_manifest, ("C", "A", "B"))


def _write_raw_context(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = ["non-targeting"] * 10 + [
        source for source in ("A", "B", "C", "D", "E") for _ in range(5)
    ]
    expression = rng.normal(size=(len(labels), 5)).astype(np.float32)
    np.savez(
        path,
        expression_matrix=expression,
        interventions=np.asarray(labels),
        var_names=np.asarray(["A", "B", "C", "D", "E"]),
    )


@pytest.fixture
def prepared_batch(tmp_path: Path) -> dict[str, Path]:
    raw_k562 = tmp_path / "raw_k562.npz"
    raw_rpe1 = tmp_path / "raw_rpe1.npz"
    _write_raw_context(raw_k562, 11)
    _write_raw_context(raw_rpe1, 23)
    k562 = load_task_c_dataset(raw_k562, context_id="k562")
    rpe1 = load_task_c_dataset(raw_rpe1, context_id="rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")

    config = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    config.update(
        {
            "maximum_epochs": 2,
            "early_stopping_patience": 1,
            "bootstrap_repeats": 1,
        }
    )
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    genes = tmp_path / "genes.json"
    _write_json(
        genes,
        {
            "schema_version": "1.0",
            "selection_id": "ablation-check-v1",
            "selection_basis": "预先登记的小型消融核验基因集合",
            "genes": ["C", "A", "B"],
        },
    )
    return {
        "k562": Path(bundle["within"]["k562"]["refit"]),
        "rpe1": Path(bundle["within"]["rpe1"]["refit"]),
        "private": Path(bundle["within"]["k562"]["holdout"]),
        "public_manifest": Path(bundle["public_manifest"]),
        "config": config_path,
        "genes": genes,
        "registry": ROOT / "configs/hypersca_c_ablations_v1.json",
        "output": tmp_path / "ablations",
    }


def _batch_kwargs(
    prepared: dict[str, Path], output: Path | None = None
) -> dict[str, object]:
    return {
        "context_values": [
            f"k562={prepared['k562']}",
            f"rpe1={prepared['rpe1']}",
        ],
        "config_path": prepared["config"],
        "gene_list_path": prepared["genes"],
        "public_manifest_path": prepared["public_manifest"],
        "ablation_registry_path": prepared["registry"],
        "output_root": output or prepared["output"],
        "seed": 11,
        "device": "cpu",
        "prior_edges_path": None,
        "prior_source_manifest_path": None,
    }


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fast_stability_result(
    contexts: tuple[HyperSCACContext, ...] | list[HyperSCACContext],
    config: HyperSCACConfig,
) -> object:
    from src.causal.hypersca_c_stability import (
        HyperSCAStabilityResult,
        build_stability_table,
    )

    genes = contexts[0].gene_names
    matrices = [
        {
            context.context_id: np.zeros((len(genes), len(genes)), dtype=np.float32)
            for context in contexts
        }
        for _ in range(config.bootstrap_repeats)
    ]
    predictions, summary = build_stability_table(
        matrices,
        genes,
        selection_threshold=config.selection_threshold,
        requested_repeats=config.bootstrap_repeats,
        minimum_success_fraction=config.bootstrap_success_fraction,
        source_variance={gene: 1.0 for gene in genes},
        minimum_source_variance=config.minimum_source_variance,
        expected_contexts=tuple(context.context_id for context in contexts),
    )
    return HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=(),
    )


def _install_fast_ablation_fit(module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    def fast_fit(
        contexts: tuple[HyperSCACContext, ...],
        config: HyperSCACConfig,
        ablation_id: str,
        **kwargs: object,
    ) -> object:
        transformed, transformed_config = module.apply_hypersca_c_ablation(  # type: ignore[attr-defined]
            contexts,
            config,
            ablation_id,
            registry=kwargs["registry"],
        )
        return _fast_stability_result(transformed, transformed_config)

    monkeypatch.setattr(module, "fit_hypersca_c_ablation", fast_fit)


def _resign_item_and_batch(root: Path, ablation_id: str) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    item_manifest_path = root / ablation_id / "run_manifest.json"
    item_manifest = json.loads(item_manifest_path.read_text(encoding="utf-8"))
    item_manifest.pop("run_manifest_content_sha256", None)
    item_manifest["run_manifest_content_sha256"] = run_module._payload_sha256(
        item_manifest
    )
    write_json(item_manifest_path, item_manifest)

    batch_path = root / "ablation_batch_manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    for record in batch["ablations"]:
        if record["ablation_id"] == ablation_id:
            record["run_manifest_sha256"] = sha256_path(item_manifest_path)
    batch.pop("batch_manifest_content_sha256", None)
    batch["batch_manifest_content_sha256"] = run_module._payload_sha256(batch)
    write_json(batch_path, batch)


def _resign_changed_status_and_batch(root: Path, ablation_id: str) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    status_path = root / ablation_id / "method_status.json"
    item_path = root / ablation_id / "run_manifest.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    item["artifacts"]["method_status.json"]["sha256"] = sha256_path(status_path)
    item.pop("run_manifest_content_sha256", None)
    item["run_manifest_content_sha256"] = run_module._payload_sha256(item)
    write_json(item_path, item)
    _resign_item_and_batch(root, ablation_id)


def _resign_batch_item_hashes(root: Path) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    batch_path = root / "ablation_batch_manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    for record in batch["ablations"]:
        record["run_manifest_sha256"] = sha256_path(
            root / record["ablation_id"] / "run_manifest.json"
        )
    batch.pop("batch_manifest_content_sha256", None)
    batch["batch_manifest_content_sha256"] = run_module._payload_sha256(batch)
    write_json(batch_path, batch)


def test_batch_runs_all_eight_in_order_discloses_missing_prior_and_reuses_exact_run(
    prepared_batch: dict[str, Path],
) -> None:
    from src.causal.hypersca_c_ablation import run_hypersca_c_ablations

    summary = run_hypersca_c_ablations(**_batch_kwargs(prepared_batch))
    root = prepared_batch["output"]
    assert tuple(summary) == ABLATION_IDS
    assert set(path.name for path in root.iterdir()) == set(ABLATION_IDS) | {
        "ablation_batch_manifest.json"
    }
    for ablation_id in ABLATION_IDS:
        directory = root / ablation_id
        assert {path.name for path in directory.iterdir()} == ARTIFACT_NAMES
        status = json.loads(
            (directory / "method_status.json").read_text(encoding="utf-8")
        )
        assert status["ablation_id"] == ablation_id
        assert status["seed"] == 11
        assert status["condition"] == "within_refit_k562_rpe1"
        assert "configuration_changes" in status
        assert "failures" in status
    unavailable = json.loads(
        (root / "prior_on_secondary/method_status.json").read_text(encoding="utf-8")
    )
    assert unavailable["status"] == "official_assets_unavailable"
    assert unavailable["reason"] == "no_nonoverlapping_preregistered_prior"
    assert unavailable["usable_for_ranking"] is False

    batch = json.loads(
        (root / "ablation_batch_manifest.json").read_text(encoding="utf-8")
    )
    assert batch["ablation_order"] == list(ABLATION_IDS)
    assert [record["ablation_id"] for record in batch["ablations"]] == list(
        ABLATION_IDS
    )
    assert batch["claim_level"] == "ablation_raw_inference_only"
    for field in (
        "ablation_registry_sha256",
        "config_sha256",
        "gene_list_sha256",
        "public_manifest_sha256",
        "code_state_sha256",
    ):
        assert batch["run_identity"][field].startswith("sha256:")
    assert len(batch["run_identity"]["contexts"]) == 2

    before = _tree_snapshot(root)
    reused = run_hypersca_c_ablations(**_batch_kwargs(prepared_batch))
    assert reused == summary
    assert _tree_snapshot(root) == before


def test_batch_registry_change_is_rejected_without_overwriting_existing_results(
    prepared_batch: dict[str, Path], tmp_path: Path
) -> None:
    from src.causal.hypersca_c_ablation import run_hypersca_c_ablations

    run_hypersca_c_ablations(**_batch_kwargs(prepared_batch))
    root = prepared_batch["output"]
    before = _tree_snapshot(root)
    changed_registry = tmp_path / "changed_registry.json"
    raw = prepared_batch["registry"].read_text(encoding="utf-8")
    changed_registry.write_text(raw + "\n", encoding="utf-8")
    kwargs = _batch_kwargs(prepared_batch)
    kwargs["ablation_registry_path"] = changed_registry
    with pytest.raises(HyperSCACError):
        run_hypersca_c_ablations(**kwargs)
    assert _tree_snapshot(root) == before


def test_invalid_and_partial_prior_arguments_are_hash_bound_in_batch_identity(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    _install_fast_ablation_fit(module, monkeypatch)
    edges = tmp_path / "unpaired_prior.csv"
    pd.DataFrame([{"source": "A", "target": "B"}]).to_csv(edges, index=False)
    partial_output = tmp_path / "partial-prior-output"
    partial_kwargs = _batch_kwargs(prepared_batch, output=partial_output)
    partial_kwargs["prior_edges_path"] = edges
    module.run_hypersca_c_ablations(**partial_kwargs)
    partial_manifest = json.loads(
        (partial_output / "ablation_batch_manifest.json").read_text(encoding="utf-8")
    )
    arguments = partial_manifest["run_identity"]["prior_arguments"]
    assert arguments["pair_complete"] is False
    assert arguments["edges"]["path"] == str(edges.resolve())
    assert arguments["edges"]["sha256"] == sha256_path(edges)
    assert arguments["source_manifest"]["provided"] is False

    invalid_manifest = tmp_path / "invalid_prior_manifest.json"
    invalid_payload = _prior_manifest(edges)
    invalid_payload["independence_attestation"][  # type: ignore[index]
        "reuses_pooled_essentiality"
    ] = True
    _write_json(invalid_manifest, invalid_payload)
    invalid_output = tmp_path / "invalid-prior-output"
    invalid_kwargs = _batch_kwargs(prepared_batch, output=invalid_output)
    invalid_kwargs["prior_edges_path"] = edges
    invalid_kwargs["prior_source_manifest_path"] = invalid_manifest
    module.run_hypersca_c_ablations(**invalid_kwargs)
    invalid_batch = json.loads(
        (invalid_output / "ablation_batch_manifest.json").read_text(encoding="utf-8")
    )
    invalid_arguments = invalid_batch["run_identity"]["prior_arguments"]
    assert invalid_arguments["edges"]["sha256"] == sha256_path(edges)
    assert invalid_arguments["source_manifest"]["sha256"] == sha256_path(
        invalid_manifest
    )
    before = _tree_snapshot(invalid_output)
    edges.write_bytes(edges.read_bytes() + b"\n")
    with pytest.raises(HyperSCACError, match="另一组输入|不能覆盖"):
        module.run_hypersca_c_ablations(**invalid_kwargs)
    assert _tree_snapshot(invalid_output) == before


@pytest.mark.parametrize("raised", [RuntimeError("unexpected"), KeyboardInterrupt()])
def test_unexpected_abort_removes_batch_staging_and_preserves_exception(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: BaseException,
) -> None:
    import src.causal.hypersca_c_ablation as module

    def abort(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise raised

    monkeypatch.setattr(module, "fit_hypersca_c_ablation", abort)
    output = tmp_path / "aborted"
    with pytest.raises(type(raised), match="unexpected" if str(raised) else None):
        module.run_hypersca_c_ablations(**_batch_kwargs(prepared_batch, output=output))
    assert not output.exists()
    assert not tuple(tmp_path.glob(".aborted.staging-*"))


def test_batch_rejects_reversed_item_time_before_publication(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    _install_fast_ablation_fit(module, monkeypatch)
    values = iter(
        [
            "2026-08-13T12:00:00Z",
            "2026-08-13T12:00:02Z",
            "2026-08-13T12:00:01Z",
        ]
    )
    last = [2]

    def utc_now() -> str:
        try:
            return next(values)
        except StopIteration:
            last[0] += 1
            return f"2026-08-13T12:00:{last[0]:02d}Z"

    monkeypatch.setattr(module._run, "_utc_now", utc_now)
    output = tmp_path / "reversed-item-time"
    with pytest.raises(HyperSCACError, match="时间|UTC|时长"):
        module.run_hypersca_c_ablations(**_batch_kwargs(prepared_batch, output=output))
    assert not output.exists()
    assert not tuple(tmp_path.glob(".reversed-item-time.staging-*"))


def test_batch_time_validation_normalizes_huge_duration() -> None:
    from src.causal.hypersca_c_ablation import _validate_time_record

    with pytest.raises(HyperSCACError, match="时长|有限"):
        _validate_time_record(
            {
                "started_utc": "2026-08-13T12:00:00Z",
                "completed_utc": "2026-08-13T12:00:01Z",
                "duration_seconds": 10**400,
            },
            "故障注入记录",
        )


def test_batch_rejects_inconsistent_item_written_to_staging(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module
    from src.evaluation.task_c_data import write_json

    _install_fast_ablation_fit(module, monkeypatch)
    original = module._write_item

    def corrupt_one(directory: Path, **kwargs: object) -> object:
        record = original(directory, **kwargs)
        if directory.name == "shared_only":
            path = directory / "method_status.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["ablation_id"] = "primary"
            write_json(path, payload)
        return record

    monkeypatch.setattr(module, "_write_item", corrupt_one)
    output = tmp_path / "inconsistent-staging"
    with pytest.raises(HyperSCACError, match="状态|登记|校验|改变|一致"):
        module.run_hypersca_c_ablations(**_batch_kwargs(prepared_batch, output=output))
    assert not output.exists()


@pytest.mark.parametrize("initial_state", ["missing", "symlink"])
def test_optional_prior_argument_state_is_frozen_until_publication(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_state: str,
) -> None:
    import src.causal.hypersca_c_ablation as module

    edge_argument = tmp_path / "late-prior.csv"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("source,target\nA,B\n", encoding="utf-8")
    second.write_text("source,target\nC,A\n", encoding="utf-8")
    if initial_state == "symlink":
        edge_argument.symlink_to(first)
    changed = False

    def mutate_argument(
        contexts: tuple[HyperSCACContext, ...],
        config: HyperSCACConfig,
        ablation_id: str,
        **kwargs: object,
    ) -> object:
        nonlocal changed
        del ablation_id, kwargs
        if not changed:
            if edge_argument.is_symlink():
                edge_argument.unlink()
                edge_argument.symlink_to(second)
            else:
                edge_argument.write_bytes(first.read_bytes())
            changed = True
        return _fast_stability_result(contexts, config)

    monkeypatch.setattr(module, "fit_hypersca_c_ablation", mutate_argument)
    kwargs = _batch_kwargs(prepared_batch, output=tmp_path / f"state-{initial_state}")
    kwargs["prior_edges_path"] = edge_argument
    with pytest.raises(HyperSCACError, match="先验|变化|状态"):
        module.run_hypersca_c_ablations(**kwargs)
    assert not Path(kwargs["output_root"]).exists()


def test_batch_rejects_staging_file_with_an_external_hardlink(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    _install_fast_ablation_fit(module, monkeypatch)
    original = module._write_item
    outside = tmp_path / "outside-hardlink.csv"

    def link_one(directory: Path, **kwargs: object) -> object:
        record = original(directory, **kwargs)
        if directory.name == "primary":
            os.link(directory / "raw_predictions.csv", outside)
        return record

    monkeypatch.setattr(module, "_write_item", link_one)
    output = tmp_path / "hardlinked-staging"
    with pytest.raises(HyperSCACError, match="硬链接|普通文件|link"):
        module.run_hypersca_c_ablations(**_batch_kwargs(prepared_batch, output=output))
    assert not output.exists()


def test_batch_reuse_rejects_an_external_hardlink(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    _install_fast_ablation_fit(module, monkeypatch)
    kwargs = _batch_kwargs(prepared_batch)
    module.run_hypersca_c_ablations(**kwargs)
    outside = tmp_path / "existing-output-alias.csv"
    os.link(
        prepared_batch["output"] / "primary/raw_predictions.csv",
        outside,
    )
    with pytest.raises(HyperSCACError, match="硬链接|普通文件|link"):
        module.run_hypersca_c_ablations(**kwargs)


def test_reuse_rejects_resigned_item_with_changed_effective_config(
    prepared_batch: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module
    from src.evaluation.task_c_data import write_json

    _install_fast_ablation_fit(module, monkeypatch)
    kwargs = _batch_kwargs(prepared_batch)
    module.run_hypersca_c_ablations(**kwargs)
    root = prepared_batch["output"]
    item_path = root / "primary/run_manifest.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    item["run_identity"]["effective_config"]["learning_rate"] = 0.5
    write_json(item_path, item)
    _resign_item_and_batch(root, "primary")
    before = _tree_snapshot(root)
    with pytest.raises(HyperSCACError, match="设置|身份|config"):
        module.run_hypersca_c_ablations(**kwargs)
    assert _tree_snapshot(root) == before


def test_reuse_rejects_swapped_ablation_directories_even_after_resigning_batch(
    prepared_batch: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module

    _install_fast_ablation_fit(module, monkeypatch)
    kwargs = _batch_kwargs(prepared_batch)
    module.run_hypersca_c_ablations(**kwargs)
    root = prepared_batch["output"]
    temporary = root / "swap-temporary"
    (root / "primary").rename(temporary)
    (root / "shared_only").rename(root / "primary")
    temporary.rename(root / "shared_only")
    _resign_batch_item_hashes(root)
    before = _tree_snapshot(root)
    with pytest.raises(HyperSCACError, match="登记|目录|身份"):
        module.run_hypersca_c_ablations(**kwargs)
    assert _tree_snapshot(root) == before


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("ablation_id", "primary"),
        ("ablation_mode", "joint"),
        ("configuration_changes", {}),
        ("interpretation", "未登记的解释"),
        ("seed", 12),
        ("contexts", ["k562"]),
        ("condition", "unexpected_condition"),
        ("requested_bootstraps", 99),
        ("successful_bootstraps", 1),
        ("failure_count", 99),
        ("coverage", 0.5),
        ("usable_for_ranking", True),
        ("condition_mode", "cross"),
        ("direction", "k562_to_rpe1"),
        ("stage", "train"),
    ],
)
def test_reuse_rejects_resigned_unavailable_static_or_count_change(
    prepared_batch: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered: object,
) -> None:
    import src.causal.hypersca_c_ablation as module
    from src.evaluation.task_c_data import write_json

    _install_fast_ablation_fit(module, monkeypatch)
    kwargs = _batch_kwargs(prepared_batch)
    module.run_hypersca_c_ablations(**kwargs)
    root = prepared_batch["output"]
    status_path = root / "prior_on_secondary/method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status[field] = tampered
    write_json(status_path, status)
    _resign_changed_status_and_batch(root, "prior_on_secondary")
    before = _tree_snapshot(root)
    with pytest.raises(HyperSCACError, match="状态|批次|固定|设置|计数|登记|排序"):
        module.run_hypersca_c_ablations(**kwargs)
    assert _tree_snapshot(root) == before


def test_reuse_rejects_resigned_unavailable_status_with_extra_field(
    prepared_batch: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    _install_fast_ablation_fit(module, monkeypatch)
    kwargs = _batch_kwargs(prepared_batch)
    module.run_hypersca_c_ablations(**kwargs)
    root = prepared_batch["output"]
    status_path = root / "prior_on_secondary/method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["unexpected"] = "同步重签也不应通过"
    write_json(status_path, status)

    item_path = root / "prior_on_secondary/run_manifest.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    item["artifacts"]["method_status.json"]["sha256"] = sha256_path(status_path)
    item.pop("run_manifest_content_sha256", None)
    item["run_manifest_content_sha256"] = run_module._payload_sha256(item)
    write_json(item_path, item)
    _resign_item_and_batch(root, "prior_on_secondary")
    before = _tree_snapshot(root)
    with pytest.raises(HyperSCACError, match="字段|格式|状态"):
        module.run_hypersca_c_ablations(**kwargs)
    assert _tree_snapshot(root) == before


def test_reuse_rejects_resigned_batch_with_extra_field(
    prepared_batch: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    _install_fast_ablation_fit(module, monkeypatch)
    kwargs = _batch_kwargs(prepared_batch)
    module.run_hypersca_c_ablations(**kwargs)
    path = prepared_batch["output"] / "ablation_batch_manifest.json"
    batch = json.loads(path.read_text(encoding="utf-8"))
    batch["unexpected"] = "同步重签也不应通过"
    batch.pop("batch_manifest_content_sha256", None)
    batch["batch_manifest_content_sha256"] = run_module._payload_sha256(batch)
    write_json(path, batch)
    before = _tree_snapshot(prepared_batch["output"])
    with pytest.raises(HyperSCACError, match="字段|格式"):
        module.run_hypersca_c_ablations(**kwargs)
    assert _tree_snapshot(prepared_batch["output"]) == before


def test_legal_prior_runs_only_secondary_item_with_discount_and_mask(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    edges = tmp_path / "prior.csv"
    pd.DataFrame([{"source": "C", "target": "A"}]).to_csv(edges, index=False)
    source_manifest = tmp_path / "prior_manifest.json"
    _write_json(source_manifest, _prior_manifest(edges))
    trust = _install_prior_trust(module, monkeypatch, tmp_path, edges, source_manifest)
    observed: dict[str, tuple[float, np.ndarray | None]] = {}

    def fast_fit(
        contexts: tuple[HyperSCACContext, ...],
        config: HyperSCACConfig,
        ablation_id: str,
        **kwargs: object,
    ) -> object:
        _, transformed_config = module.apply_hypersca_c_ablation(
            contexts,
            config,
            ablation_id,
            registry=kwargs["registry"],
        )
        prior_mask = kwargs["prior_mask"]
        observed[ablation_id] = (
            transformed_config.prior_discount,
            None if prior_mask is None else np.array(prior_mask, copy=True),
        )
        return _fast_stability_result(contexts, transformed_config)

    monkeypatch.setattr(module, "fit_hypersca_c_ablation", fast_fit)
    kwargs = _batch_kwargs(prepared_batch, output=tmp_path / "with-prior")
    kwargs["prior_edges_path"] = edges
    kwargs["prior_source_manifest_path"] = source_manifest
    statuses = module.run_hypersca_c_ablations(**kwargs)
    assert statuses["prior_on_secondary"] == "completed_raw_inference"
    for ablation_id in ABLATION_IDS[:-1]:
        assert observed[ablation_id][1] is None
        assert observed[ablation_id][0] == 0.0
    assert observed["prior_on_secondary"][0] == 0.5
    assert observed["prior_on_secondary"][1].tolist() == [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    batch = json.loads(
        (tmp_path / "with-prior/ablation_batch_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert batch["run_identity"]["registered_prior"][
        "prior_trust_registry_sha256"
    ] == sha256_path(trust)


def test_one_failed_ablation_does_not_hide_other_results(
    prepared_batch: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module

    original = module.fit_hypersca_c_ablation

    def fail_one(*args: object, **kwargs: object) -> object:
        if args[2] == "shared_only":
            raise HyperSCACError("预先登记的单项核验失败")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fit_hypersca_c_ablation", fail_one)
    summary = module.run_hypersca_c_ablations(**_batch_kwargs(prepared_batch))
    assert summary["shared_only"] == "failed_ablation"
    assert summary["primary"] == "completed_raw_inference"
    assert summary["prior_on_secondary"] == "official_assets_unavailable"
    failed = json.loads(
        (prepared_batch["output"] / "shared_only/method_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed["status"] == "failed_ablation"
    assert failed["failures"] == ["预先登记的单项核验失败"]
    assert failed["usable_for_ranking"] is False


def test_transform_failure_does_not_hide_later_ablation_results(
    prepared_batch: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_ablation as module

    original_apply = module.apply_hypersca_c_ablation
    attempted: list[str] = []

    def fail_one_transform(
        contexts: tuple[HyperSCACContext, ...],
        config: HyperSCACConfig,
        ablation_id: str,
        **kwargs: object,
    ) -> object:
        attempted.append(ablation_id)
        if ablation_id == "shared_only":
            raise HyperSCACError("共享成分消融变换失败")
        return original_apply(contexts, config, ablation_id, **kwargs)

    def fast_fit(
        contexts: tuple[HyperSCACContext, ...],
        config: HyperSCACConfig,
        ablation_id: str,
        **kwargs: object,
    ) -> object:
        del ablation_id, kwargs
        return _fast_stability_result(contexts, config)

    monkeypatch.setattr(module, "apply_hypersca_c_ablation", fail_one_transform)
    monkeypatch.setattr(module, "fit_hypersca_c_ablation", fast_fit)
    summary = module.run_hypersca_c_ablations(**_batch_kwargs(prepared_batch))

    assert attempted == list(ABLATION_IDS)
    assert summary["shared_only"] == "failed_ablation"
    assert summary["separate_contexts"] == "completed_raw_inference"
    assert summary["prior_on_secondary"] == "official_assets_unavailable"
    assert {
        path.name for path in prepared_batch["output"].iterdir() if path.is_dir()
    } == set(ABLATION_IDS)
    failed = json.loads(
        (prepared_batch["output"] / "shared_only/method_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed["failures"] == ["共享成分消融变换失败"]


def test_batch_reuses_task_c_public_boundary_and_rejects_private_input_before_fit(
    prepared_batch: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_ablation as module

    monkeypatch.setattr(
        module,
        "fit_hypersca_c_ablation",
        lambda *args, **kwargs: pytest.fail("fit must not be called"),
    )
    kwargs = _batch_kwargs(prepared_batch, output=tmp_path / "must-not-exist")
    kwargs["context_values"] = [f"k562={prepared_batch['private']}"]
    with pytest.raises(HyperSCACError, match="公开|private|登记"):
        module.run_hypersca_c_ablations(**kwargs)
    assert not Path(kwargs["output_root"]).exists()


def test_ablation_cli_requires_complete_public_run_boundary() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_hypersca_c_ablations_script",
        ROOT / "scripts/run_hypersca_c_ablations.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    parser = script.build_parser()
    action_names = {action.dest for action in parser._actions}
    assert {
        "context",
        "config",
        "gene_list",
        "public_manifest",
        "seed",
        "device",
        "output_root",
        "ablation_registry",
        "prior_edges",
        "prior_source_manifest",
    } <= action_names


def test_ablation_cli_runs_with_python_entrypoint(
    prepared_batch: dict[str, Path]
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_hypersca_c_ablations.py"),
            "--context",
            f"k562={prepared_batch['k562']}",
            "--context",
            f"rpe1={prepared_batch['rpe1']}",
            "--config",
            str(prepared_batch["config"]),
            "--gene-list",
            str(prepared_batch["genes"]),
            "--public-manifest",
            str(prepared_batch["public_manifest"]),
            "--ablation-registry",
            str(prepared_batch["registry"]),
            "--output-root",
            str(prepared_batch["output"]),
            "--seed",
            "11",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["prior_on_secondary"] == (
        "official_assets_unavailable"
    )
