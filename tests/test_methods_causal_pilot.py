from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
