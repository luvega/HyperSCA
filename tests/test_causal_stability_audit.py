from __future__ import annotations

import json

import numpy as np
import pytest


def test_edge_stability_table_marks_proxy_edges_conservatively():
    from src.causal.stability_audit import build_edge_stability_table

    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    bootstrap = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.7],
            [0.1, 0.2, 0.0],
        ]
    )
    stability_freqs = [
        bootstrap,
        np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.0, 0.1, 0.0]]),
    ]
    null_freqs = [
        np.array([[0.0, 0.2, 0.1], [0.1, 0.0, 0.2], [0.0, 0.1, 0.0]])
        for _ in range(100)
    ]

    table = build_edge_stability_table(
        adjacency=adjacency,
        bootstrap_freq=bootstrap,
        node_labels=["A", "B", "C"],
        type_mapping={"A": "CAF", "B": "TAM", "C": "CD8T"},
        stability_freqs=stability_freqs,
        null_freqs=null_freqs,
        group_freqs={"g1": bootstrap, "g2": stability_freqs[1]},
        causal_input_metadata={"observation_unit": "gene_proxy"},
        threshold=0.5,
    )

    assert {"source_node", "target_node", "stability_class", "evidence_level"}.issubset(table.columns)
    ab = table[(table["source_node"] == "A") & (table["target_node"] == "B")].iloc[0]
    assert ab["base_edge"] is True
    assert ab["stability_class"] == "stable_candidate"
    assert ab["negative_control_pass"] is True
    assert ab["evidence_level"] == "exploratory_cluster_graph"


def test_edge_stability_requires_null_controls_to_pass_negative_control():
    from src.causal.stability_audit import build_edge_stability_table, build_negative_control_report

    adjacency = np.array([[0.0, 1.0], [0.0, 0.0]])
    bootstrap = np.array([[0.0, 0.9], [0.1, 0.0]])

    table = build_edge_stability_table(
        adjacency=adjacency,
        bootstrap_freq=bootstrap,
        node_labels=["CAF", "TAM"],
        threshold=0.5,
    )
    edge = table[(table["source_node"] == "CAF") & (table["target_node"] == "TAM")].iloc[0]
    report = build_negative_control_report(table)

    assert edge["negative_control_pass"] is False
    assert edge["negative_control_status"] == "not_run"
    assert edge["stability_class"] == "not_controlled_candidate"
    assert "No null controls were supplied" in report


def test_group_consistency_summary_separates_consensus_and_specific_edges():
    from src.causal.stability_audit import summarize_group_consistency

    group_freqs = {
        "patient_a": np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]]),
        "patient_b": np.array([[0.0, 0.9, 0.0], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]]),
    }

    summary = summarize_group_consistency(group_freqs, ["A", "B", "C"], threshold=0.5)

    assert summary["n_groups"] == 2
    assert {"source": "A", "target": "B"}.items() <= summary["consensus_edges"][0].items()
    assert any(edge["source"] == "B" and edge["target"] == "C" for edge in summary["group_specific_edges"])
    assert any(edge["source"] == "C" and edge["target"] == "A" for edge in summary["group_specific_edges"])


def test_load_step2_audit_inputs_accepts_standard_step2_outputs(tmp_path):
    from src.causal.stability_audit import load_step2_audit_inputs

    np.save(tmp_path / "causal_adjacency.npy", np.array([[0.0, 1.0], [0.0, 0.0]]))
    np.save(tmp_path / "bootstrap_freq_matrix.npy", np.array([[0.0, 0.8], [0.1, 0.0]]))
    (tmp_path / "node_info.json").write_text(
        json.dumps({"node_labels": ["CAF", "TAM"], "type_mapping": {"CAF": "CAF", "TAM": "TAM"}}),
        encoding="utf-8",
    )
    (tmp_path / "causal_input_metadata.json").write_text(
        json.dumps({"observation_unit": "gene_proxy"}),
        encoding="utf-8",
    )

    inputs = load_step2_audit_inputs(tmp_path)

    assert inputs["node_labels"] == ["CAF", "TAM"]
    assert inputs["type_mapping"]["CAF"] == "CAF"
    assert inputs["causal_input_metadata"]["observation_unit"] == "gene_proxy"
    assert inputs["bootstrap_freq"].shape == (2, 2)


def test_generated_null_matrices_are_reproducible_and_seed_sensitive():
    from src.causal.stability_audit import build_null_frequency_matrices

    bootstrap = np.arange(16, dtype=float).reshape(4, 4) / 16.0

    first = build_null_frequency_matrices(
        bootstrap,
        n_controls=6,
        null_modes=(
            "matrix_permutation",
            "node_label_shuffle",
            "outgoing_weight_permutation",
        ),
        random_seed=17,
    )
    repeated = build_null_frequency_matrices(
        bootstrap,
        n_controls=6,
        null_modes=(
            "matrix_permutation",
            "node_label_shuffle",
            "outgoing_weight_permutation",
        ),
        random_seed=17,
    )
    changed = build_null_frequency_matrices(
        bootstrap,
        n_controls=6,
        null_modes=(
            "matrix_permutation",
            "node_label_shuffle",
            "outgoing_weight_permutation",
        ),
        random_seed=18,
    )

    assert len(first) == 6
    assert all(np.array_equal(left, right) for left, right in zip(first, repeated))
    assert any(not np.array_equal(left, right) for left, right in zip(first, changed))
    assert all(np.diag(matrix).tolist() == [0.0] * 4 for matrix in first)


def test_outgoing_weight_null_preserves_each_source_weight_multiset():
    from src.causal.stability_audit import build_null_frequency_matrices

    bootstrap = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.0, 0.5, 0.6],
            [0.7, 0.8, 0.0, 0.9],
            [1.0, 0.15, 0.25, 0.0],
        ]
    )

    generated = build_null_frequency_matrices(
        bootstrap,
        n_controls=1,
        null_modes=("outgoing_weight_permutation",),
        random_seed=3,
    )[0]

    for row in range(len(bootstrap)):
        source = np.delete(bootstrap[row], row)
        null = np.delete(generated[row], row)
        assert np.array_equal(np.sort(source), np.sort(null))


def test_null_generator_rejects_invalid_requests():
    from src.causal.stability_audit import build_null_frequency_matrices

    bootstrap = np.eye(2)
    with pytest.raises(ValueError, match="n_controls"):
        build_null_frequency_matrices(bootstrap, n_controls=-1)
    with pytest.raises(ValueError, match="null_modes"):
        build_null_frequency_matrices(bootstrap, n_controls=1, null_modes=())
    with pytest.raises(ValueError, match="unsupported causal null mode"):
        build_null_frequency_matrices(
            bootstrap,
            n_controls=1,
            null_modes=("unknown",),
        )
    with pytest.raises(ValueError, match="finite"):
        build_null_frequency_matrices(
            np.array([[0.0, np.nan], [0.1, 0.0]]),
            n_controls=1,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_null_frequency_matrices(
            np.array([[0.0, 1.1], [0.1, 0.0]]),
            n_controls=1,
        )


def test_fewer_than_ten_nulls_never_pass_negative_control():
    from src.causal.stability_audit import build_edge_stability_table

    adjacency = np.array([[0.0, 1.0], [0.0, 0.0]])
    bootstrap = np.array([[0.0, 0.9], [0.1, 0.0]])
    low_null = np.array([[0.0, 0.1], [0.0, 0.0]])

    table = build_edge_stability_table(
        adjacency=adjacency,
        bootstrap_freq=bootstrap,
        node_labels=["CAF", "TAM"],
        null_freqs=[low_null] * 9,
        threshold=0.5,
    )
    edge = table[(table["source_node"] == "CAF") & (table["target_node"] == "TAM")].iloc[0]

    assert edge["negative_control_pass"] is False
    assert edge["negative_control_status"] == "failed_insufficient_nulls_for_fdr"


def test_edge_stability_rejects_null_matrices_with_wrong_shape():
    from src.causal.stability_audit import build_edge_stability_table

    with pytest.raises(ValueError, match="same shape"):
        build_edge_stability_table(
            adjacency=np.zeros((2, 2)),
            bootstrap_freq=np.zeros((2, 2)),
            node_labels=["A", "B"],
            null_freqs=[np.zeros((3, 3))],
        )


def test_run_causal_stability_audit_generates_manifested_null_controls(tmp_path):
    from src.causal.stability_audit import run_causal_stability_audit

    np.save(tmp_path / "causal_adjacency.npy", np.array([[0.0, 1.0], [0.0, 0.0]]))
    np.save(tmp_path / "bootstrap_freq.npy", np.array([[0.0, 0.9], [0.1, 0.0]]))
    (tmp_path / "node_info.json").write_text(
        json.dumps(
            {
                "node_labels": ["CAF", "TAM"],
                "type_mapping": {"CAF": "CAF", "TAM": "TAM"},
            }
        ),
        encoding="utf-8",
    )

    result = run_causal_stability_audit(
        step2_dir=tmp_path,
        output_dir=tmp_path / "audit",
        threshold=0.5,
        n_null_controls=20,
        null_modes=(
            "matrix_permutation",
            "node_label_shuffle",
            "outgoing_weight_permutation",
        ),
        random_seed=11,
    )

    edge_table = result["edge_stability"]
    manifest = json.loads(
        (tmp_path / "audit" / "null_control_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (tmp_path / "audit" / "causal_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert int(edge_table["null_control_count"].max()) == 20
    assert set(edge_table["negative_control_status"]) <= {"passed", "failed"}
    assert manifest["random_seed"] == 11
    assert manifest["n_generated"] == 20
    assert sum(manifest["mode_counts"].values()) == 20
    assert manifest["scope"] == "bootstrap_frequency_matrix_surrogate"
    assert len(manifest["null_control_sha256"]) == 64
    assert summary["n_null_control_matrices"] == 20
    assert summary["null_control"]["scope"] == manifest["scope"]


def test_zero_requested_nulls_preserves_not_run_behavior(tmp_path):
    from src.causal.stability_audit import run_causal_stability_audit

    np.save(tmp_path / "causal_adjacency.npy", np.array([[0.0, 1.0], [0.0, 0.0]]))
    np.save(tmp_path / "bootstrap_freq.npy", np.array([[0.0, 0.9], [0.1, 0.0]]))

    result = run_causal_stability_audit(
        step2_dir=tmp_path,
        output_dir=tmp_path / "audit",
    )

    assert set(result["edge_stability"]["negative_control_status"]) == {"not_run"}
    assert result["null_freqs"] == []
    assert "No null controls were supplied" in result["negative_control_report"]


def test_causal_stability_cli_exposes_reproducible_null_configuration():
    from scripts.run_causal_stability_audit import build_parser

    args = build_parser().parse_args(
        [
            "--n-null-controls",
            "25",
            "--null-modes",
            "matrix_permutation,node_label_shuffle",
            "--random-seed",
            "19",
        ]
    )

    assert args.n_null_controls == 25
    assert args.null_modes == "matrix_permutation,node_label_shuffle"
    assert args.random_seed == 19
