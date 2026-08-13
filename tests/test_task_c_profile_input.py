from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pytest

from src.evaluation.task_c_data import (
    build_shared_task_c_split,
    load_task_c_dataset,
    materialize_task_c_split,
)
from src.evaluation.task_c_profile_input import (
    PROFILE_LIMITS,
    TaskCProfileInputError,
    _stratified_cell_indices,
    materialize_task_c_profile_input,
    validate_task_c_profile_input,
)


@pytest.fixture(scope="module")
def large_public_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("task-c-large-profile")
    genes = tuple(f"G{index:03d}" for index in range(270))
    labels = ["non-targeting"] * 1_000 + [
        gene for gene in genes[:90] for _ in range(30)
    ]
    loaded = {}
    for context, seed in (("k562", 11), ("rpe1", 23)):
        path = root / f"raw-{context}.npz"
        rng = np.random.default_rng(seed)
        expression = rng.normal(size=(len(labels), len(genes))).astype(np.float32)
        expression[:, 1] *= 4.0
        expression[:, 2] *= 2.0
        np.savez(
            path,
            expression_matrix=expression,
            interventions=np.asarray(labels),
            var_names=np.asarray(genes),
        )
        loaded[context] = load_task_c_dataset(path, context_id=context)
    split = build_shared_task_c_split(
        loaded["k562"], loaded["rpe1"], seed=11, min_cells=5
    )
    return materialize_task_c_split(
        loaded["k562"], loaded["rpe1"], split, root / "public"
    )


def test_profiles_freeze_gene_and_cell_limits_for_within_and_cross(
    large_public_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    public_manifest = Path(large_public_bundle["public_manifest"])
    connection = materialize_task_c_profile_input(
        public_manifest_path=public_manifest,
        profile="connection",
        condition="within_environment",
        context_id="k562",
        output_dir=tmp_path / "within-connection",
    )
    comprehensive = materialize_task_c_profile_input(
        public_manifest_path=public_manifest,
        profile="comprehensive",
        condition="within_environment",
        context_id="k562",
        output_dir=tmp_path / "within-comprehensive",
    )
    cross = materialize_task_c_profile_input(
        public_manifest_path=public_manifest,
        profile="connection",
        condition="cross_environment",
        direction="k562_to_rpe1",
        output_dir=tmp_path / "cross-connection",
    )

    with np.load(connection["input_npz"], allow_pickle=False) as archive:
        assert set(archive.files) == {
            "expression_matrix",
            "interventions",
            "var_names",
        }
        assert archive["expression_matrix"].shape == (2_000, 64)
    with np.load(comprehensive["input_npz"], allow_pickle=False) as archive:
        assert archive["expression_matrix"].shape == (2_960, 72)
    with np.load(cross["input_npz"], allow_pickle=False) as archive:
        assert set(archive.files) == {
            "expression_matrix",
            "interventions",
            "var_names",
            "environment_labels",
        }
        environments, counts = np.unique(
            archive["environment_labels"], return_counts=True
        )
        assert dict(zip(environments.tolist(), counts.tolist())) == {
            "k562": 2_000,
            "rpe1": 800,
        }
        for context in environments:
            selected = archive["environment_labels"] == context
            controls = archive["interventions"][selected] == "non-targeting"
            np.testing.assert_allclose(
                archive["expression_matrix"][selected][controls].mean(axis=0),
                0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                archive["expression_matrix"][selected][controls].std(
                    axis=0, ddof=0
                ),
                1.0,
                atol=1e-12,
            )

    connection_manifest = json.loads(
        Path(connection["manifest"]).read_text(encoding="utf-8")
    )
    assert connection_manifest["profile_input_schema"] == "task_c_profile_subset_v1"
    assert connection_manifest["profile"] == "connection"
    assert connection_manifest["limits"] == {
        "gene_count": 64,
        "cells_per_context": 2_000,
    }
    assert len(connection_manifest["gene_selection"]["ordered_genes"]) == 64
    assert len(connection_manifest["gene_selection"]["ordered_indices"]) == 64
    selected_genes = set(connection_manifest["gene_selection"]["ordered_genes"])
    with np.load(connection["input_npz"], allow_pickle=False) as archive:
        labels = set(archive["interventions"].tolist())
        assert labels <= {"non-targeting", *selected_genes}
        assert "excluded" not in labels
        assert archive["expression_matrix"].shape[0] <= 2_000
    context_record = connection_manifest["contexts"][0]
    assert context_record["row_filter_rule"] == (
        "retain_control_and_selected_gene_interventions_v1"
    )
    assert context_record["dropped_original_row_count"] > 0
    assert context_record["dropped_original_row_indices"] == sorted(
        context_record["dropped_original_row_indices"]
    )
    assert not selected_genes & set(context_record["dropped_by_label"])
    cross_manifest = json.loads(
        Path(cross["manifest"]).read_text(encoding="utf-8")
    )
    assert cross_manifest["transformation"] == (
        "per_environment_control_zscore_then_row_concatenate_v1"
    )
    assert cross_manifest["environment_labels"] == {
        "ordered_context_ids": ["k562", "rpe1"],
        "cell_counts": {"k562": 2_000, "rpe1": 800},
    }


def test_profile_materialization_is_deterministic_and_validator_recomputes_parents(
    large_public_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    arguments = {
        "public_manifest_path": Path(large_public_bundle["public_manifest"]),
        "profile": "connection",
        "condition": "cross_environment",
        "direction": "rpe1_to_k562",
    }
    first = materialize_task_c_profile_input(
        **arguments, output_dir=tmp_path / "first"
    )
    second = materialize_task_c_profile_input(
        **arguments, output_dir=tmp_path / "second"
    )
    assert Path(first["input_npz"]).read_bytes() == Path(second["input_npz"]).read_bytes()
    assert Path(first["manifest"]).read_bytes() == Path(second["manifest"]).read_bytes()

    validated = validate_task_c_profile_input(
        input_path=Path(first["input_npz"]),
        profile_manifest_path=Path(first["manifest"]),
        public_manifest_path=arguments["public_manifest_path"],
    )
    assert validated.profile == "connection"
    assert validated.direction == "rpe1_to_k562"

    manifest_path = Path(first["manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["contexts"][0]["selected_sorted_indices"][0] += 1
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(TaskCProfileInputError, match="profile|subset|manifest|indices"):
        validate_task_c_profile_input(
            input_path=Path(first["input_npz"]),
            profile_manifest_path=manifest_path,
            public_manifest_path=arguments["public_manifest_path"],
        )


def test_public_inventory_hashes_unique_files_without_retaining_all_bytes(
    large_public_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.task_c_profile_input as profile_module

    original_capture = profile_module._capture
    registered_reads: list[tuple[int, int]] = []

    def recording_capture(*args: object, **kwargs: object) -> object:
        snapshot = original_capture(*args, **kwargs)
        if str(args[1]).startswith("registered public file"):
            registered_reads.append((snapshot.device, snapshot.inode))
        return snapshot

    monkeypatch.setattr(profile_module, "_capture", recording_capture)
    _, _, inventory = profile_module._load_public_manifest(
        Path(large_public_bundle["public_manifest"])
    )

    assert len(registered_reads) == len(set(registered_reads))
    assert all(snapshot.payload is None for snapshot in inventory.values())


def test_profile_rejects_selected_parents_whose_arrays_expand_past_budget(
    large_public_bundle: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.task_c_profile_input as profile_module

    monkeypatch.setattr(profile_module, "MAXIMUM_TOTAL_EXPANDED_PARENT_BYTES", 1)
    with pytest.raises(TaskCProfileInputError, match="expanded arrays"):
        materialize_task_c_profile_input(
            public_manifest_path=Path(large_public_bundle["public_manifest"]),
            profile="connection",
            condition="within_environment",
            context_id="k562",
            output_dir=tmp_path / "too-expanded",
        )


def test_validator_rejects_manifest_replacement_after_initial_capture(
    large_public_bundle: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.task_c_profile_input as profile_module

    created = materialize_task_c_profile_input(
        public_manifest_path=Path(large_public_bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="k562",
        output_dir=tmp_path / "profile",
    )
    manifest_path = Path(created["manifest"])
    original_build = profile_module._build_profile

    def replacing_build(*args: object, **kwargs: object) -> object:
        built = original_build(*args, **kwargs)
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        return built

    monkeypatch.setattr(profile_module, "_build_profile", replacing_build)
    with pytest.raises(TaskCProfileInputError, match="changed|manifest"):
        validate_task_c_profile_input(
            input_path=Path(created["input_npz"]),
            profile_manifest_path=manifest_path,
            public_manifest_path=Path(large_public_bundle["public_manifest"]),
        )


def test_profile_rejects_unknown_profile_and_old_cross_transformation(
    large_public_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    public_manifest = Path(large_public_bundle["public_manifest"])
    with pytest.raises(TaskCProfileInputError, match="profile"):
        materialize_task_c_profile_input(
            public_manifest_path=public_manifest,
            profile="oversized",
            condition="within_environment",
            context_id="k562",
            output_dir=tmp_path / "wrong",
        )

    created = materialize_task_c_profile_input(
        public_manifest_path=public_manifest,
        profile="connection",
        condition="cross_environment",
        direction="k562_to_rpe1",
        output_dir=tmp_path / "cross",
    )
    manifest_path = Path(created["manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["transformation"] = "per_environment_control_center_then_row_concatenate_v1"
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(TaskCProfileInputError, match="transformation|manifest"):
        validate_task_c_profile_input(
            input_path=Path(created["input_npz"]),
            profile_manifest_path=manifest_path,
            public_manifest_path=public_manifest,
        )


def test_comprehensive_stratified_selector_enforces_twenty_thousand_cell_cap() -> None:
    labels = np.asarray(
        ["non-targeting"] * 10_001 + ["G000"] * 10_000,
        dtype=str,
    )
    first = _stratified_cell_indices(labels, limit=PROFILE_LIMITS["comprehensive"][1])
    second = _stratified_cell_indices(labels, limit=20_000)
    assert len(first) == 20_000
    assert np.array_equal(first, second)
    assert np.all(first[:-1] < first[1:])
    assert set(labels[first]) == {"non-targeting", "G000"}


def test_hypersca_uses_the_same_capped_profile_cells(
    large_public_bundle: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.causal import hypersca_c_run as run_module
    from src.causal.hypersca_c_stability import (
        HyperSCAStabilityResult,
        build_stability_table,
    )

    created = materialize_task_c_profile_input(
        public_manifest_path=Path(large_public_bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="k562",
        output_dir=tmp_path / "profile",
    )
    profile_manifest = json.loads(
        Path(created["manifest"]).read_text(encoding="utf-8")
    )
    genes = tuple(profile_manifest["gene_selection"]["ordered_genes"])
    gene_list = tmp_path / "genes.json"
    gene_list.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "selection_id": "connection-profile-v1",
                "selection_basis": "使用统一比较范围记录中的固定基因顺序",
                "genes": list(genes),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/hypersca_c_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config["bootstrap_repeats"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    observed_shapes: list[tuple[int, int]] = []

    def fake_fit(
        contexts: Sequence[object],
        config_value: object,
        *,
        seed: int,
        device: str,
    ) -> HyperSCAStabilityResult:
        del config_value, seed, device
        observed_shapes.extend(context.expression.shape for context in contexts)  # type: ignore[attr-defined]
        predictions, summary = build_stability_table(
            [],
            genes,
            selection_threshold=0.0001,
            requested_repeats=2,
            minimum_success_fraction=0.8,
            source_variance={gene: 1.0 for gene in genes},
            minimum_source_variance=1e-8,
            expected_contexts=("k562",),
        )
        return HyperSCAStabilityResult(
            predictions=predictions,
            summary=summary,
            failures=("repeat_0:failed", "repeat_1:failed"),
        )

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", fake_fit)
    output = tmp_path / "hypersca"
    run_module.run_hypersca_c(
        context_values=(),
        profile_input_path=Path(created["input_npz"]),
        profile_manifest_path=Path(created["manifest"]),
        config_path=config_path,
        gene_list_path=gene_list,
        public_manifest_path=Path(large_public_bundle["public_manifest"]),
        output_dir=output,
        seed=11,
        device="cpu",
    )

    assert observed_shapes == [(2_000, 64)]
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["profile_input"]["manifest_sha256"].startswith("sha256:")
    assert run_manifest["profile_input"]["record"]["contexts"][0][
        "selected_sorted_indices"
    ] == profile_manifest["contexts"][0]["selected_sorted_indices"]


def test_hypersca_cli_accepts_an_explicit_profile_pair(tmp_path: Path) -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts/run_hypersca_c.py"
    spec = importlib.util.spec_from_file_location("profile_hypersca_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.build_parser().parse_args(
        [
            "--profile-input",
            str(tmp_path / "profile.npz"),
            "--profile-manifest",
            str(tmp_path / "profile.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--gene-list",
            str(tmp_path / "genes.json"),
            "--public-manifest",
            str(tmp_path / "public.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--seed",
            "11",
        ]
    )
    assert parsed.context is None
    assert parsed.profile_input == tmp_path / "profile.npz"
    assert parsed.profile_manifest == tmp_path / "profile.json"
