"""阶段 2 因果评估指标单元测试

覆盖:
- HSIC 计算
- 图稀疏度
- Bootstrap 频率统计
- 邻居组成预测
- 一站式评估
"""
import numpy as np
import pytest


# =========================================================================
# HSIC
# =========================================================================

def test_hsic_independent():
    """独立变量的 HSIC 应趋近 0"""
    from src.evaluation.causal_metrics import compute_hsic

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 5))
    Y = rng.normal(0, 1, (200, 5))
    hsic = compute_hsic(X, Y)
    assert hsic < 0.05, f"HSIC of independent vars should be ~0, got {hsic}"


def test_hsic_dependent():
    """强相关变量的 HSIC 应显著 > 0"""
    from src.evaluation.causal_metrics import compute_hsic

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 5))
    Y = X + rng.normal(0, 0.1, (200, 5))  # 强相关
    hsic = compute_hsic(X, Y)
    assert hsic > 0.01, f"HSIC of dependent vars should be > 0, got {hsic}"


def test_hsic_small_sample():
    """小样本 HSIC 不崩溃"""
    from src.evaluation.causal_metrics import compute_hsic

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0, 6.0], [7.0, 8.0]])
    hsic = compute_hsic(X, Y)
    assert isinstance(hsic, float)


# =========================================================================
# 图结构指标
# =========================================================================

def test_graph_sparsity():
    """图稀疏度计算"""
    from src.evaluation.causal_metrics import compute_graph_sparsity

    K = 5
    adj = np.zeros((K, K))
    adj[0, 1] = 1
    adj[1, 2] = 1
    sparsity = compute_graph_sparsity(adj)
    expected = 2 / (5 * 4)
    assert abs(sparsity - expected) < 1e-6


def test_graph_sparsity_empty():
    """空图稀疏度 = 0"""
    from src.evaluation.causal_metrics import compute_graph_sparsity

    adj = np.zeros((5, 5))
    assert compute_graph_sparsity(adj) == 0.0


def test_bootstrap_freq_stats():
    """Bootstrap 频率统计"""
    from src.evaluation.causal_metrics import compute_bootstrap_freq_stats

    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    freq = np.array([[0, 0.8, 0], [0, 0, 0.6], [0, 0, 0]])
    stats = compute_bootstrap_freq_stats(freq, adj)
    assert stats["mean"] == pytest.approx(0.7, abs=1e-6)
    assert stats["min"] == pytest.approx(0.6, abs=1e-6)
    assert stats["max"] == pytest.approx(0.8, abs=1e-6)


# =========================================================================
# 邻居组成预测
# =========================================================================

def test_neighbor_composition():
    """邻居组成计算"""
    from src.evaluation.causal_metrics import compute_neighbor_composition

    labels = np.array([0, 0, 1, 1])
    adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ], dtype=float)
    comp = compute_neighbor_composition(labels, adj)
    assert comp.shape == (4, 2)
    # 节点0: 邻居是1(type0), 2(type1) → [0.5, 0.5]
    assert comp[0, 0] == pytest.approx(0.5, abs=0.01)
    assert comp[0, 1] == pytest.approx(0.5, abs=0.01)


def test_neighbor_predictivity():
    """邻居预测力"""
    from src.evaluation.causal_metrics import compute_neighbor_predictivity

    rng = np.random.default_rng(42)
    z = rng.normal(0, 1, (20, 5))
    comp = rng.dirichlet([1, 1, 1], size=20)
    r2 = compute_neighbor_predictivity(z, comp)
    assert isinstance(r2, float)
    assert -1 <= r2 <= 1


# =========================================================================
# 一站式评估
# =========================================================================

def test_evaluate_causal_basic():
    """一站式评估不崩溃"""
    from src.evaluation.causal_metrics import evaluate_causal

    K = 5
    adj = np.eye(K, k=1)
    freq = adj * 0.7
    metrics = evaluate_causal(adjacency=adj, bootstrap_freq=freq)
    assert "graph_sparsity" in metrics
    assert "mean_bootstrap_freq" in metrics
    assert "n_edges" in metrics
    assert metrics["n_edges"] == 4


def test_evaluate_causal_with_disentangle():
    """一站式评估含解缠指标"""
    from src.evaluation.causal_metrics import evaluate_causal

    rng = np.random.default_rng(42)
    K = 10
    adj = (rng.random((K, K)) > 0.7).astype(float)
    np.fill_diagonal(adj, 0)
    freq = adj * rng.random((K, K))
    z_int = rng.normal(0, 1, (K, 5))
    z_ext = rng.normal(0, 1, (K, 5))

    metrics = evaluate_causal(
        adjacency=adj,
        bootstrap_freq=freq,
        z_int=z_int,
        z_ext=z_ext,
    )
    assert "hsic_z_int_z_ext" in metrics
