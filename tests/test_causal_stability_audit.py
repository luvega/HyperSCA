from __future__ import annotations

import json

import numpy as np


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
        np.array([[0.0, 0.2, 0.1], [0.1, 0.0, 0.2], [0.0, 0.1, 0.0]]),
        np.array([[0.0, 0.3, 0.1], [0.0, 0.0, 0.1], [0.1, 0.0, 0.0]]),
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
