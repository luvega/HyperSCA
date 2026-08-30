from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_causal_pilot_scores_complete_relation_universe_without_holdout() -> None:
    from src.evaluation.methods_causal_pilot import build_causal_pilot_evidence

    genes = ("A", "B", "C")
    hypersca = pd.DataFrame(
        [("A", "B", 3.0), ("B", "C", 2.0)],
        columns=["source", "target", "score"],
    )
    mean = pd.DataFrame(
        [("A", "C", 2.0)], columns=["source", "target", "score"]
    )
    shared = pd.DataFrame(
        [("A", "B", 1.0), ("A", "C", 2.0)],
        columns=["source", "target", "score"],
    )

    evidence = build_causal_pilot_evidence(
        gene_names=genes,
        eligible_sources=("A", "B"),
        tuning_edges=frozenset({("A", "B"), ("B", "C")}),
        hypersca_predictions=hypersca,
        comparator_predictions={
            "mean_difference": mean,
            "hypersca_c_shared_only": shared,
        },
        direction="k562_to_rpe1",
        seed=11,
    )

    assert set(evidence.metrics) == {
        "mean_difference",
        "hypersca_c_shared_only",
    }
    assert evidence.metrics["mean_difference"]["paired_difference"] > 0.0
    assert evidence.metrics["hypersca_c_shared_only"]["paired_difference"] > 0.0
    assert len(evidence.relations) == 2 * len(genes) * (len(genes) - 1)
    assert set(evidence.relations["source"]) == set(genes)
    assert evidence.promotion_eligible is False
    assert evidence.data_scopes == ("train", "tune")


def test_causal_pilot_splits_only_verified_environment_labels() -> None:
    from src.evaluation.methods_causal_pilot import split_hypersca_contexts

    expression = np.arange(24, dtype=np.float32).reshape(6, 4)
    interventions = np.asarray(
        ["non-targeting", "A", "non-targeting", "B", "A", "B"]
    )
    environments = np.asarray(["k562", "k562", "k562", "rpe1", "rpe1", "rpe1"])
    contexts = split_hypersca_contexts(
        expression=expression,
        interventions=interventions,
        gene_names=("A", "B", "C", "D"),
        environment_labels=environments,
    )

    assert tuple(context.context_id for context in contexts) == ("k562", "rpe1")
    assert tuple(len(context.expression) for context in contexts) == (3, 3)
    expression[0, 0] = -999.0
    assert contexts[0].expression[0, 0] != -999.0


def test_causal_pilot_rejects_missing_or_extra_environment() -> None:
    import pytest

    from src.evaluation.methods_causal_pilot import split_hypersca_contexts

    with pytest.raises(ValueError, match="exactly k562 and rpe1"):
        split_hypersca_contexts(
            expression=np.ones((4, 3), dtype=np.float32),
            interventions=np.asarray(["non-targeting", "A", "A", "A"]),
            gene_names=("A", "B", "C"),
            environment_labels=np.asarray(["k562", "k562", "rpe1", "other"]),
        )


def test_causal_pilot_runner_has_no_refit_or_private_input() -> None:
    import inspect

    from src.evaluation.methods_causal_pilot import run_causalbench_pilot_run

    parameters = set(inspect.signature(run_causalbench_pilot_run).parameters)
    assert parameters == {
        "public_manifest_path",
        "direction",
        "output_dir",
        "seed",
        "hypersca_config_path",
        "ablation_registry_path",
        "protocol_path",
        "protocol_identity_value",
        "device",
    }
    assert not any("private" in name or "holdout" in name or "refit" in name for name in parameters)


def test_causal_pilot_requires_one_fixed_public_data_split(tmp_path: Path) -> None:
    import json
    import pytest

    from src.evaluation.methods_causal_pilot import validate_causal_pilot_public_split

    accepted_path = tmp_path / "accepted.json"
    rejected_path = tmp_path / "rejected.json"
    accepted_path.write_text(
        json.dumps(
            {
                "seed": 11,
                "split_id": "C-context-intervention-holdout-v1-seed-11",
            }
        ),
        encoding="utf-8",
    )
    rejected_path.write_text(
        json.dumps(
            {
                "seed": 23,
                "split_id": "C-context-intervention-holdout-v1-seed-23",
            }
        ),
        encoding="utf-8",
    )
    accepted = validate_causal_pilot_public_split(accepted_path)
    assert accepted["seed"] == 11
    with pytest.raises(ValueError, match="fixed data split seed 11"):
        validate_causal_pilot_public_split(rejected_path)


def test_causal_runner_publishes_one_replayable_fixed_split_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.causal import hypersca_c_ablation
    from src.evaluation.methods_causal_pilot import run_causalbench_pilot_run
    from src.evaluation.run_evidence_publisher import verify_run_evidence_bundle
    from src.evaluation.task_c_data import (
        build_shared_task_c_split,
        load_task_c_dataset,
        materialize_task_c_split,
    )

    genes = tuple(f"G{index:03d}" for index in range(70))
    interventions = ["non-targeting"] * 300 + [
        gene for gene in genes[:30] for _ in range(20)
    ]
    loaded = {}
    for context, seed in (("k562", 11), ("rpe1", 23)):
        rng = np.random.default_rng(seed)
        expression = rng.normal(size=(len(interventions), len(genes))).astype(
            np.float32
        )
        for row, intervention in enumerate(interventions):
            if intervention != "non-targeting":
                source_index = genes.index(intervention)
                expression[row, (source_index + 1) % len(genes)] += 8.0
        raw = tmp_path / f"raw-{context}.npz"
        np.savez(
            raw,
            expression_matrix=expression,
            interventions=np.asarray(interventions),
            var_names=np.asarray(genes),
        )
        loaded[context] = load_task_c_dataset(raw, context_id=context)
    split = build_shared_task_c_split(
        loaded["k562"], loaded["rpe1"], seed=11, min_cells=5
    )
    prepared = materialize_task_c_split(
        loaded["k562"], loaded["rpe1"], split, tmp_path / "prepared"
    )

    def fake_fit(contexts, config, ablation_id, **kwargs):
        selected_genes = contexts[0].gene_names
        if ablation_id == "primary":
            rows = [
                (source, selected_genes[(index + 1) % len(selected_genes)], 10.0)
                for index, source in enumerate(selected_genes)
            ]
        else:
            rows = [
                (source, selected_genes[(index + 2) % len(selected_genes)], 1.0)
                for index, source in enumerate(selected_genes)
            ]
        return SimpleNamespace(
            predictions=pd.DataFrame(rows, columns=["source", "target", "score"]),
            summary={"requested_bootstraps": 20, "successful_bootstraps": 20},
            failures=(),
        )

    monkeypatch.setattr(hypersca_c_ablation, "fit_hypersca_c_ablation", fake_fit)
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            (
                "protocol_version: hypersca-methods-v2.1",
                "execution:",
                "  pilot_scopes: [train, tune]",
                "comparators:",
                "  causal:",
                "    confirmatory: mean_difference",
                "    attribution: hypersca_c_shared_only",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pilot"
    record = run_causalbench_pilot_run(
        public_manifest_path=Path(prepared["public_manifest"]),
        direction="k562_to_rpe1",
        output_dir=output,
        seed=23,
        hypersca_config_path=Path("configs/hypersca_c_v1.json"),
        ablation_registry_path=Path("configs/hypersca_c_ablations_v1.json"),
        protocol_path=protocol,
        protocol_identity_value="a" * 64,
        device="cpu",
    )

    verified = verify_run_evidence_bundle(output)
    assert record["promotion_eligible"] is False
    assert verified.identity.data_split_seed == 11
    assert verified.identity.model_seed == 23
    assert verified.identity.statistical_unit_schema == (
        "causalbench_direction_source_relation_v1"
    )
    assert verified.identity.evidence_role == "pilot_audit_only"
    assert verified.statistical_unit_record["direction"] == "k562_to_rpe1"
    assert verified.statistical_unit_record["eligible_sources"]
    assert verified.statistical_unit_record["relation_universe_sha256"]
    summary_text = repr(verified.summary)
    assert "private" not in summary_text
    assert "refit" not in summary_text

    second_output = tmp_path / "pilot-seed-11"
    run_causalbench_pilot_run(
        public_manifest_path=Path(prepared["public_manifest"]),
        direction="k562_to_rpe1",
        output_dir=second_output,
        seed=11,
        hypersca_config_path=Path("configs/hypersca_c_v1.json"),
        ablation_registry_path=Path("configs/hypersca_c_ablations_v1.json"),
        protocol_path=protocol,
        protocol_identity_value="a" * 64,
        device="cpu",
    )
    second = verify_run_evidence_bundle(second_output)
    assert second.identity.data_split_seed == second.identity.model_seed == 11
    assert (
        second.identity.data_split_identity_sha256
        == verified.identity.data_split_identity_sha256
    )
    assert (
        second.identity.statistical_unit_identity_sha256
        == verified.identity.statistical_unit_identity_sha256
    )
    assert second.identity.run_identity_sha256 != verified.identity.run_identity_sha256
