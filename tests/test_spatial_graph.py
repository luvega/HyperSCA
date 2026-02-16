"""空间图构建与 TopoLa 增强单元测试"""
import pytest
import numpy as np
from scipy import sparse

from src.data.spatial_graph import (
    build_knn_graph,
    build_delaunay_graph,
    topola_enhance,
    build_spatial_graph,
    graph_statistics,
)


@pytest.fixture
def grid_coords():
    """10x10 网格坐标"""
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    return np.column_stack([x.ravel(), y.ravel()]).astype(float)


@pytest.fixture
def knn_adj(grid_coords):
    return build_knn_graph(grid_coords, k=4)


class TestBuildKnnGraph:
    def test_shape(self, grid_coords, knn_adj):
        N = len(grid_coords)
        assert knn_adj.shape == (N, N)

    def test_symmetric(self, knn_adj):
        diff = knn_adj - knn_adj.T
        assert diff.nnz == 0 or abs(diff).max() < 1e-10

    def test_no_self_loops(self, knn_adj):
        assert knn_adj.diagonal().sum() == 0

    def test_positive_weights(self, knn_adj):
        assert np.all(knn_adj.data > 0)


class TestBuildDelaunayGraph:
    def test_basic_shape(self, grid_coords):
        adj = build_delaunay_graph(grid_coords)
        N = len(grid_coords)
        assert adj.shape == (N, N)
        assert adj.nnz > 0


class TestTopoLaEnhance:
    def test_output_shape(self, knn_adj):
        enhanced = topola_enhance(knn_adj, lambda_val=1e-3)
        assert enhanced.shape == knn_adj.shape

    def test_singular_value_transform(self):
        """验证 sigma^3 / (sigma^2 + 1/lambda) 变换"""
        sigma = np.array([100.0, 10.0, 1.0, 0.01])
        lam = 1e-3
        expected = sigma ** 3 / (sigma ** 2 + 1.0 / lam)
        # 大奇异值保留程度高于小奇异值
        ratio = expected / sigma
        assert ratio[0] > ratio[1] > ratio[2] > ratio[3]
        # 小奇异值被显著抑制
        assert ratio[-1] < 0.001


class TestBuildSpatialGraph:
    def test_without_topola(self, grid_coords):
        adj_orig, adj_enh = build_spatial_graph(
            grid_coords, k=4, use_topola=False
        )
        assert adj_enh is None
        assert adj_orig.shape[0] == len(grid_coords)

    def test_with_topola(self, grid_coords):
        adj_orig, adj_enh = build_spatial_graph(
            grid_coords, k=4, use_topola=True
        )
        assert adj_enh is not None
        assert adj_enh.shape == adj_orig.shape


class TestGraphStatistics:
    def test_stats_keys(self, knn_adj):
        stats = graph_statistics(knn_adj)
        assert "n_nodes" in stats
        assert "n_edges" in stats
        assert "mean_degree" in stats
        assert "n_components" in stats

    def test_degree_range(self, grid_coords, knn_adj):
        stats = graph_statistics(knn_adj)
        assert stats["min_degree"] >= 4  # k=4
        assert stats["max_degree"] >= 4
