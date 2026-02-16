"""阶段 2 集成测试

覆盖:
- PC 算法基本功能
- Bootstrap 聚合
- CausalCellGraph 创建与导出
- DisentangleModel 前向传播
- 已知轴评估
- 信号流推断
"""
import numpy as np
import torch
import pytest
import tempfile
from pathlib import Path


# =========================================================================
# PC 算法
# =========================================================================

def test_pc_algorithm_chain():
    """PC 算法: 链式因果 A→B→C"""
    from src.causal.cmi_pruning import pc_algorithm

    rng = np.random.default_rng(42)
    N = 500
    A = rng.normal(0, 1, N)
    B = 0.8 * A + rng.normal(0, 0.3, N)
    C = 0.8 * B + rng.normal(0, 0.3, N)
    data = np.column_stack([A, B, C])

    dag = pc_algorithm(data, alpha=0.05, max_cond_set=2)
    # 应发现 A-B 和 B-C 之间有边
    assert dag.shape == (3, 3)
    # A 与 C 条件独立于 B → 无直接边
    assert dag[0, 2] == 0 or dag[2, 0] == 0


def test_pc_algorithm_independent():
    """PC 算法: 独立变量无边"""
    from src.causal.cmi_pruning import pc_algorithm

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (500, 3))
    dag = pc_algorithm(data, alpha=0.01)
    assert dag.sum() <= 2  # 可能有少量 false positive


def test_bootstrap_aggregation():
    """Bootstrap 聚合: 输出形状正确且频率在 [0,1]"""
    from src.causal.cmi_pruning import bootstrap_causal_discovery

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (100, 4))
    freq = bootstrap_causal_discovery(
        data, n_bootstraps=5, alpha=0.05, verbose=False
    )
    assert freq.shape == (4, 4)
    assert freq.min() >= 0
    assert freq.max() <= 1


def test_threshold_pruning():
    """阈值剪枝"""
    from src.causal.cmi_pruning import threshold_pruning

    freq = np.array([[0, 0.8, 0.3], [0.1, 0, 0.6], [0.4, 0.2, 0]])
    adj, pruned = threshold_pruning(freq, threshold=0.5)
    assert adj[0, 1] == 1  # 0.8 >= 0.5
    assert adj[0, 2] == 0  # 0.3 < 0.5
    assert adj[1, 2] == 1  # 0.6 >= 0.5
    assert pruned[0, 2] == 0


# =========================================================================
# DisentangleModel
# =========================================================================

def test_disentangle_model_forward():
    """DisentangleModel 前向传播"""
    from src.causal.disentangle import DisentangleModel

    N, G = 10, 50
    x = torch.randn(N, G)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    model = DisentangleModel(input_dim=G, z_dim=8, hidden_dims=[32, 16])
    out = model(x, edge_index)

    assert out["z_int"].shape == (N, 8)
    assert out["z_ext"].shape == (N, 8)
    assert out["x_hat"].shape == (N, G)


def test_disentangle_model_embeddings():
    """DisentangleModel 嵌入提取"""
    from src.causal.disentangle import DisentangleModel

    N, G = 10, 50
    x = torch.randn(N, G)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    model = DisentangleModel(input_dim=G, z_dim=8)
    z_int, z_ext = model.get_embeddings(x, edge_index)

    assert z_int.shape == (N, 8)
    assert z_ext.shape == (N, 8)
    assert isinstance(z_int, np.ndarray)


def test_hsic_torch():
    """HSIC torch 版: 独立变量低、相关变量高"""
    from src.causal.disentangle import compute_hsic_torch

    torch.manual_seed(42)
    X = torch.randn(50, 5)
    Y_ind = torch.randn(50, 5)
    Y_dep = X + torch.randn(50, 5) * 0.1

    hsic_ind = compute_hsic_torch(X, Y_ind).item()
    hsic_dep = compute_hsic_torch(X, Y_dep).item()

    assert hsic_dep > hsic_ind


# =========================================================================
# CausalCellGraph
# =========================================================================

def test_causal_graph_creation():
    """CausalCellGraph 创建"""
    from src.causal.causal_graph import CausalCellGraph

    K = 4
    adj = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    labels = ["A", "B", "C", "D"]
    graph = CausalCellGraph(adj, labels)

    assert graph.K == 4
    stats = graph.summary_stats()
    assert stats["n_edges"] == 3
    assert stats["is_dag"] is True


def test_causal_graph_from_freq():
    """CausalCellGraph 从频率矩阵创建"""
    from src.causal.causal_graph import CausalCellGraph

    freq = np.array([[0, 0.8, 0.3], [0.1, 0, 0.6], [0.4, 0.2, 0]])
    graph = CausalCellGraph.from_freq_matrix(
        freq, ["X", "Y", "Z"], threshold=0.5
    )
    assert graph.summary_stats()["n_edges"] == 2  # 0.8 and 0.6


def test_causal_graph_export_graphml():
    """CausalCellGraph GraphML 导出"""
    from src.causal.causal_graph import CausalCellGraph

    adj = np.array([[0, 1], [0, 0]])
    graph = CausalCellGraph(adj, ["A", "B"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.graphml"
        graph.to_graphml(path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_causal_graph_known_axes():
    """已知轴评估: 匹配与不匹配"""
    from src.causal.causal_graph import CausalCellGraph

    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    labels = ["CAF", "TAM", "Treg"]
    graph = CausalCellGraph(adj, labels)

    axes = [
        {"name": "test_axis", "source_type": "CAF", "target_type": "TAM"},
        {"name": "test_missing", "source_type": "NK", "target_type": "DC"},
    ]
    results = graph.evaluate_known_axes(
        known_axes=axes,
        type_mapping={"CAF": "CAF", "TAM": "TAM", "Treg": "Treg"},
    )
    assert results["known_axis_recall"] == 1.0  # CAF→TAM found
    assert results["n_axes_tested"] == 1  # NK/DC not found → not tested


# =========================================================================
# 信号流
# =========================================================================

def test_signaling_flow_basic():
    """信号流推断基本功能"""
    from src.causal.signaling_flow import infer_signaling_flow, summarize_signaling_flows

    adj = np.array([[0, 1], [0, 0]])
    labels = ["CAF", "TAM"]
    type_map = {"CAF": "CAF", "TAM": "TAM"}

    edges = infer_signaling_flow(
        causal_graph_adj=adj,
        node_labels=labels,
        type_mapping=type_map,
    )
    assert len(edges) > 0

    summary = summarize_signaling_flows(edges)
    assert "n_total_flow_edges" in summary
    assert summary["n_total_flow_edges"] > 0


def test_signaling_flow_no_match():
    """信号流: 无匹配返回空"""
    from src.causal.signaling_flow import infer_signaling_flow

    adj = np.array([[0, 1], [0, 0]])
    labels = ["X", "Y"]
    type_map = {"X": "Unknown1", "Y": "Unknown2"}

    edges = infer_signaling_flow(
        causal_graph_adj=adj,
        node_labels=labels,
        type_mapping=type_map,
    )
    assert len(edges) == 0
