"""阶段 3 集成与单元测试

覆盖:
- LatentArithmetic: 扰动向量估计、施加、端到端 KO
- spatial_propagation: BFS 层传播、距离衰减、收敛
- cf_metrics: R² / PCC / MSE / Marker Direction / DEG Overlap
- spatial_metrics: Moran's I / Gradient Decay / Propagation Depth
- PerturbationPipeline 内部方法
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# =========================================================================
# LatentArithmetic
# =========================================================================

def test_latent_arithmetic_compute_perturbation_vector():
    """扰动向量估计: delta 形状正确且非全零。"""
    from src.perturbation.latent_arithmetic import LatentArithmetic
    from src.models.hyperbolic.lorentz import lorentz_origin, exp_map

    la = LatentArithmetic(device="cpu")
    dim = 8
    o = lorentz_origin(dim, batch_size=1)

    # 构造处理组和对照组: 沿不同方向平移
    rng = np.random.default_rng(42)
    v_ctrl = torch.zeros(20, dim + 1)
    v_ctrl[:, 1] = torch.tensor(rng.normal(0.1, 0.01, 20).astype(np.float32))
    ctrl_z = exp_map(v_ctrl, o.expand(20, -1)).detach().numpy()

    v_treat = torch.zeros(20, dim + 1)
    v_treat[:, 1] = torch.tensor(rng.normal(0.5, 0.01, 20).astype(np.float32))
    treat_z = exp_map(v_treat, o.expand(20, -1)).detach().numpy()

    delta = la.compute_perturbation_vector(treat_z, ctrl_z)
    assert delta.shape == (dim + 1,)
    assert np.linalg.norm(delta) > 0.01


def test_latent_arithmetic_apply_perturbation():
    """施加扰动后嵌入仍在 Lorentz 流形上。"""
    from src.perturbation.latent_arithmetic import LatentArithmetic
    from src.models.hyperbolic.lorentz import lorentz_origin, lorentzian_inner

    la = LatentArithmetic(device="cpu")
    dim = 4
    o = lorentz_origin(dim, batch_size=10).numpy()
    delta = np.zeros(dim + 1, dtype=np.float32)
    delta[1] = 0.3

    z_cf = la.apply_perturbation(o, delta, direction=-1.0)
    assert z_cf.shape == (10, dim + 1)

    # 检查 <z_cf, z_cf>_L ≈ -1
    z_t = torch.as_tensor(z_cf, dtype=torch.float32)
    inner = lorentzian_inner(z_t, z_t, keepdim=False)
    np.testing.assert_allclose(inner.numpy(), -1.0, atol=1e-4)


def test_latent_arithmetic_virtual_knockout():
    """端到端虚拟敲除: 返回所有必要字段。"""
    from src.perturbation.latent_arithmetic import LatentArithmetic
    from src.models.hyperbolic.lorentz import lorentz_origin, exp_map

    la = LatentArithmetic(device="cpu")
    dim = 4
    o = lorentz_origin(dim, batch_size=1)
    rng = np.random.default_rng(42)

    def make_z(n, shift):
        v = torch.zeros(n, dim + 1)
        v[:, 1] = shift
        return exp_map(v, o.expand(n, -1)).detach().numpy()

    ctrl = make_z(15, 0.1)
    treat = make_z(15, 0.5)
    obs = make_z(10, 0.3)

    result = la.virtual_knockout(obs, treat, ctrl)
    assert "delta" in result
    assert "z_cf" in result
    assert result["z_cf"].shape == obs.shape


# =========================================================================
# Spatial Propagation
# =========================================================================

def test_propagation_chain():
    """链式因果图: A→B→C，效应逐层衰减。"""
    from src.perturbation.spatial_propagation import propagate_perturbation

    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    delta = np.array([1.0, 0.0, 0.0])
    result = propagate_perturbation(
        causal_adj=adj, source_nodes=[0], source_delta=delta, max_depth=5
    )

    assert result["effect"][0] == 1.0
    # B 应接收到来自 A 的效应
    assert result["effect"][1] > 0
    # C 应接收到来自 B 的效应
    assert result["effect"][2] > 0
    # BFS 层数应 >= 2（source + 2 hops）
    assert len(result["bfs_layers"]) >= 2


def test_propagation_with_spatial_decay():
    """空间距离衰减: 远距离节点效应更小。"""
    from src.perturbation.spatial_propagation import propagate_perturbation

    # A→B, A→C; B 近、C 远
    adj = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
    delta = np.array([1.0, 0.0, 0.0])
    coords = np.array([[0, 0], [10, 0], [1000, 0]], dtype=float)

    result = propagate_perturbation(
        causal_adj=adj,
        source_nodes=[0],
        source_delta=delta,
        spatial_coords=coords,
        decay_length=50.0,
        max_depth=3,
    )
    # B（近）效应应大于 C（远）
    assert abs(result["effect"][1]) > abs(result["effect"][2])


def test_propagation_convergence():
    """传播在小效应时收敛停止。"""
    from src.perturbation.spatial_propagation import propagate_perturbation

    # 长链 A→B→C→D→E
    K = 5
    adj = np.zeros((K, K), dtype=float)
    for i in range(K - 1):
        adj[i, i + 1] = 0.1  # 弱边
    delta = np.zeros(K)
    delta[0] = 0.05

    result = propagate_perturbation(
        causal_adj=adj,
        source_nodes=[0],
        source_delta=delta,
        max_depth=10,
        convergence_tol=0.01,
    )
    # 由于边权弱且初始效应小，应早于 max_depth 收敛
    assert len(result["bfs_layers"]) <= K


def test_propagation_bfs_stats_structure():
    """BFS 层统计结构正确。"""
    from src.perturbation.spatial_propagation import propagate_perturbation

    adj = np.array([[0, 1], [0, 0]], dtype=float)
    delta = np.array([1.0, 0.0])
    result = propagate_perturbation(causal_adj=adj, source_nodes=[0], source_delta=delta)

    for layer in result["bfs_layers"]:
        assert "hop" in layer
        assert "nodes" in layer
        assert "mean_effect" in layer
        assert "n_cells" in layer
        assert isinstance(layer["nodes"], list)


# =========================================================================
# CF Metrics
# =========================================================================

def test_r2_mean_perfect():
    """完全相同的观测与反事实 → R²=1。"""
    from src.evaluation.cf_metrics import r2_mean

    x = np.random.default_rng(42).normal(0, 1, (50, 20))
    assert r2_mean(x, x) == pytest.approx(1.0, abs=1e-6)


def test_r2_var_nontrivial():
    """有噪声的反事实 → R² < 1。"""
    from src.evaluation.cf_metrics import r2_var

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (50, 20))
    x_noisy = x + rng.normal(0, 0.5, x.shape)
    r2 = r2_var(x, x_noisy)
    assert r2 < 1.0


def test_pcc_median_identical():
    """完全相同 → PCC ≈ 1。"""
    from src.evaluation.cf_metrics import pcc_median

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (50, 10))
    assert pcc_median(x, x) == pytest.approx(1.0, abs=1e-6)


def test_mse_zero():
    """完全相同 → MSE = 0。"""
    from src.evaluation.cf_metrics import mse

    x = np.ones((10, 5))
    assert mse(x, x) == pytest.approx(0.0, abs=1e-12)


def test_marker_direction_accuracy():
    """方向准确率: 2/3 正确。"""
    from src.evaluation.cf_metrics import marker_direction_accuracy

    obs = np.array([[1.0, 2.0, 3.0]])
    cf = np.array([[0.5, 3.0, 2.5]])  # A 下调, B 上调, C 下调
    expected = {"A": -1, "B": 1, "C": 1}  # C 预期上调但实际下调
    genes = ["A", "B", "C"]

    acc = marker_direction_accuracy(obs, cf, expected, genes)
    assert acc == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_deg_overlap_jaccard():
    """DEG Overlap: 有交集。"""
    from src.evaluation.cf_metrics import deg_overlap_jaccard

    rng = np.random.default_rng(42)
    obs = rng.normal(0, 1, (50, 100))
    cf = obs.copy()
    # 制造几个明确的 DEG
    cf[:, 0] += 5.0
    cf[:, 1] += 4.0
    cf[:, 2] += 3.0
    gene_names = [f"G{i}" for i in range(100)]
    ref_degs = {"G0", "G1", "G2", "G10"}

    j = deg_overlap_jaccard(obs, cf, gene_names, ref_degs, top_k=5)
    assert j > 0  # 至少有部分交集


def test_evaluate_counterfactual_all_keys():
    """一站式评估包含必要字段。"""
    from src.evaluation.cf_metrics import evaluate_counterfactual

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (30, 10))
    x_cf = x + rng.normal(0, 0.1, x.shape)
    genes = [f"G{i}" for i in range(10)]

    result = evaluate_counterfactual(
        x, x_cf, gene_names=genes,
        expected_directions={"G0": -1},
        reference_degs={"G0", "G1"},
    )
    for key in ("r2_mean", "r2_var", "pcc_median", "mse",
                "marker_direction_accuracy", "deg_overlap_jaccard"):
        assert key in result


# =========================================================================
# Spatial Metrics
# =========================================================================

def test_morans_i_clustered():
    """空间聚集模式 → Moran's I > 0。"""
    from src.evaluation.spatial_metrics import morans_i

    rng = np.random.default_rng(42)
    n = 100
    coords = rng.uniform(0, 100, (n, 2))
    # 左边区域高表达，右边低
    values = np.where(coords[:, 0] < 50, 5.0, 0.0) + rng.normal(0, 0.1, n)
    I = morans_i(values, coords, k=6)
    assert I > 0


def test_delta_morans_i():
    """ΔMoran's I 结构正确。"""
    from src.evaluation.spatial_metrics import delta_morans_i

    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, (50, 2))
    obs = rng.normal(0, 1, 50)
    cf = obs + rng.normal(0, 0.1, 50)

    result = delta_morans_i(obs, cf, coords)
    assert "morans_i_obs" in result
    assert "morans_i_cf" in result
    assert "delta_morans_i" in result


def test_gradient_decay_r2_fit():
    """梯度衰减 R² 拟合: 完美指数衰减 → 高 R²。"""
    from src.evaluation.spatial_metrics import gradient_decay_r2

    distances = np.linspace(0, 100, 50)
    effects = 2.0 * np.exp(-distances / 30.0) + 1e-6

    result = gradient_decay_r2(effects, distances)
    assert result["gradient_decay_r2"] > 0.9
    assert result["characteristic_length"] > 0


def test_propagation_depth_threshold():
    """传播深度与阈值匹配。"""
    from src.evaluation.spatial_metrics import propagation_depth

    layers = [
        {"hop": 0, "mean_effect": 1.0},
        {"hop": 1, "mean_effect": 0.5},
        {"hop": 2, "mean_effect": 0.05},
        {"hop": 3, "mean_effect": 0.005},
    ]
    assert propagation_depth(layers, threshold=0.01) == 2
    assert propagation_depth(layers, threshold=0.001) == 3


def test_evaluate_spatial_all_keys():
    """一站式空间评估包含必要字段。"""
    from src.evaluation.spatial_metrics import evaluate_spatial_propagation

    rng = np.random.default_rng(42)
    n = 30
    coords = rng.uniform(0, 100, (n, 2))
    effects = rng.uniform(0, 1, n)
    dists = rng.uniform(0, 100, n)
    bfs = [{"hop": 0, "mean_effect": 1.0, "nodes": [0]}, {"hop": 1, "mean_effect": 0.5, "nodes": [1, 2]}]
    adj = rng.choice([0, 1], (n, n), p=[0.9, 0.1]).astype(float)
    obs = rng.normal(0, 1, n)
    cf = obs + rng.normal(0, 0.1, n)

    result = evaluate_spatial_propagation(
        coords=coords,
        effect_magnitudes=effects,
        source_distances=dists,
        bfs_layers=bfs,
        causal_adj=adj,
        observed_expr=obs,
        counterfactual_expr=cf,
    )
    assert "gradient_decay_r2" in result
    assert "propagation_depth" in result
    assert "morans_i_obs" in result


# =========================================================================
# Pipeline 内部方法
# =========================================================================

def test_pipeline_expression_ko_cf_reduces_target():
    """Pipeline expression KO: 靶基因表达下降。"""
    from src.pipeline.config import HyperSCAConfig
    from src.pipeline.step3_perturbation import PerturbationPipeline

    config = HyperSCAConfig()
    config.step3_latent_ko_scale = 0.5
    pipeline = PerturbationPipeline.__new__(PerturbationPipeline)
    pipeline.config = config

    genes = ["GENE_A", "REC_A", "GENE_B"]
    obs = pd.DataFrame([[2.0, 1.5, 1.0], [1.8, 1.3, 0.9]], columns=genes, index=["n1", "n2"])
    cf = pipeline._expression_knockout_cf(
        observed_expr=obs,
        target_gene="GENE_A",
        flow_edges=[],
        node_to_type={},
    )
    assert cf["GENE_A"].mean() < obs["GENE_A"].mean()
    # 其他基因无 flow edge → 不变
    assert cf["GENE_B"].equals(obs["GENE_B"])


def test_pipeline_legacy_latent_arithmetic_name_is_rejected():
    """The old latent_arithmetic label should not silently run expression KO."""
    from src.pipeline.config import HyperSCAConfig
    from src.pipeline.step3_perturbation import PerturbationPipeline

    pipeline = PerturbationPipeline.__new__(PerturbationPipeline)
    pipeline.config = HyperSCAConfig(step3_method="latent_arithmetic")
    obs = pd.DataFrame([[2.0]], columns=["GENE_A"], index=["n1"])

    with pytest.raises(ValueError, match="expression_ko"):
        pipeline._counterfactual_for_method(
            observed_expr=obs,
            target_gene="GENE_A",
            flow_edges=[],
            node_to_type={},
            gene_names=["GENE_A"],
        )


def test_pipeline_hyperbolic_latent_ko_requires_artifacts():
    """Hyperbolic latent KO must fail explicitly when Step1 artifacts are missing."""
    from src.pipeline.config import HyperSCAConfig
    from src.pipeline.step3_perturbation import PerturbationPipeline

    pipeline = PerturbationPipeline.__new__(PerturbationPipeline)
    pipeline.config = HyperSCAConfig(step3_method="hyperbolic_latent_ko")
    pipeline.step1_dir = Path("missing_step1")
    obs = pd.DataFrame([[2.0]], columns=["GENE_A"], index=["n1"])

    with pytest.raises(NotImplementedError, match="requires Step1 H-VAE"):
        pipeline._counterfactual_for_method(
            observed_expr=obs,
            target_gene="GENE_A",
            flow_edges=[],
            node_to_type={},
            gene_names=["GENE_A"],
        )


def test_pipeline_hyperbolic_latent_ko_loads_hvae_artifacts(tmp_path: Path):
    """Hyperbolic latent KO should load Step1 H-VAE artifacts and decode CF expression."""
    from src.models.hyperbolic.hvae import HyperbolicVAE
    from src.models.hyperbolic.lorentz import exp_map, lorentz_origin
    from src.pipeline.config import HyperSCAConfig
    from src.pipeline.step3_perturbation import PerturbationPipeline

    step1 = tmp_path / "step1"
    step1.mkdir()

    model = HyperbolicVAE(
        input_dim=3,
        latent_dim=2,
        encoder_layers=[4],
        decoder_layers=[4],
        gcn_layers=1,
        dropout=0.0,
    )
    torch.save(model.state_dict(), step1 / "hvae_model.pt")
    (step1 / "config.json").write_text(
        json.dumps(
            {
                "hvae_latent_dim": 2,
                "hvae_encoder_layers": [4],
                "hvae_decoder_layers": [4],
                "hvae_gcn_layers": 1,
                "hvae_use_zinb": False,
                "hvae_dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )

    origin = lorentz_origin(2, batch_size=1)
    tangent = torch.zeros(4, 3)
    tangent[:, 1] = torch.tensor([0.1, 0.2, 0.8, 0.9])
    z = exp_map(tangent, origin.expand(4, -1)).detach().numpy()
    np.save(step1 / "embeddings_lorentz.npy", z)

    obs = pd.DataFrame(
        [[0.2, 1.0, 0.4], [0.3, 1.1, 0.5], [2.0, 1.2, 0.6], [2.2, 1.3, 0.7]],
        columns=["GENE_A", "REC_A", "GENE_B"],
        index=["n1", "n2", "n3", "n4"],
    )
    pipeline = PerturbationPipeline.__new__(PerturbationPipeline)
    pipeline.config = HyperSCAConfig(step3_method="hyperbolic_latent_ko", device="cpu")
    pipeline.step1_dir = step1

    cf = pipeline._counterfactual_for_method(
        observed_expr=obs,
        target_gene="GENE_A",
        flow_edges=[],
        node_to_type={},
        gene_names=list(obs.columns),
    )

    assert list(cf.columns) == list(obs.columns)
    assert list(cf.index) == list(obs.index)
    assert cf.shape == obs.shape
    assert np.isfinite(cf.values).all()
    assert pipeline._last_hyperbolic_latent_metadata["decoder_source"] == str(step1 / "hvae_model.pt")


def test_pipeline_default_targets_are_data_driven_from_flow_edges():
    from src.pipeline.config import HyperSCAConfig
    from src.pipeline.step3_perturbation import PerturbationPipeline

    pipeline = PerturbationPipeline.__new__(PerturbationPipeline)
    pipeline.config = HyperSCAConfig(step3_target_top_k=3)
    obs = pd.DataFrame(
        [[1.0, 2.0, 5.0], [1.5, 2.2, 5.5]],
        columns=["GENE_A", "REC_A", "GENE_BACKGROUND"],
        index=["n1", "n2"],
    )
    flow_edges = [{"source": "GENE_A", "target": "REC_A", "weight": 0.9}]

    targets = pipeline._resolve_target_genes(obs, flow_edges, list(obs.columns))

    assert HyperSCAConfig().step3_target_genes == []
    assert targets[:2] == ["GENE_A", "REC_A"]


def test_pipeline_build_gene_causal_mask():
    """基因因果掩码构建。"""
    from src.pipeline.step3_perturbation import PerturbationPipeline

    flow_edges = [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
    ]
    genes = ["A", "B", "C", "D"]
    mask = PerturbationPipeline._build_gene_causal_mask(flow_edges, genes)
    assert mask.shape == (4, 4)
    assert mask[0, 1] == 1.0  # A → B
    assert mask[1, 2] == 1.0  # B → C
    assert mask[0, 2] == 0.0  # A 不直接 → C
    assert mask[3, :].sum() == 0  # D 无出边
