"""Lorentz 双曲空间工具单元测试"""
import pytest
import torch

from src.models.hyperbolic.lorentz import (
    lorentzian_inner,
    lorentz_distance,
    exp_map,
    log_map,
    parallel_transport,
    lorentz_to_poincare,
    poincare_to_lorentz,
    polar_project,
    project_to_lorentz,
    lorentz_origin,
)


@pytest.fixture
def origin():
    return lorentz_origin(4, batch_size=8)


@pytest.fixture
def random_points():
    """生成随机 Lorentz 流形上的点"""
    x = torch.randn(8, 4) * 0.5
    return polar_project(x)


class TestLorentzianInner:
    def test_origin_self_inner(self, origin):
        """原点 <o, o>_L = -1"""
        inner = lorentzian_inner(origin, origin, keepdim=False)
        assert torch.allclose(inner, torch.tensor(-1.0).expand(8), atol=1e-6)

    def test_points_on_manifold(self, random_points):
        """流形上的点 <p, p>_L = -1"""
        inner = lorentzian_inner(random_points, random_points, keepdim=False)
        assert torch.allclose(inner, torch.tensor(-1.0).expand(8), atol=1e-5)


class TestLorentzDistance:
    def test_self_distance_zero(self, random_points):
        """d(x, x) ~ 0 (within numerical tolerance)"""
        d = lorentz_distance(random_points, random_points)
        assert torch.all(d < 0.01)

    def test_distance_nonnegative(self, origin, random_points):
        """d(o, p) >= 0"""
        d = lorentz_distance(origin, random_points)
        assert torch.all(d >= 0)

    def test_distance_symmetric(self, random_points):
        """d(x, y) = d(y, x)"""
        x = random_points[:4]
        y = random_points[4:]
        d_xy = lorentz_distance(x, y)
        d_yx = lorentz_distance(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-5)


class TestExpLogMap:
    def test_exp_log_roundtrip(self, origin):
        """ExpMap -> LogMap 互逆"""
        v = torch.randn(8, 5) * 0.3
        v[:, 0] = 0  # 切向量时间分量为 0
        p = exp_map(v, origin)
        v_back = log_map(p, origin)
        assert torch.allclose(v, v_back, atol=1e-4)

    def test_exp_stays_on_manifold(self, origin):
        """ExpMap 输出在流形上"""
        v = torch.randn(8, 5) * 0.5
        v[:, 0] = 0
        p = exp_map(v, origin)
        inner = lorentzian_inner(p, p, keepdim=False)
        assert torch.allclose(inner, torch.tensor(-1.0).expand(8), atol=1e-4)


class TestParallelTransport:
    def test_preserves_norm(self, origin, random_points):
        """平行传输保持切向量范数"""
        v = torch.randn(8, 5) * 0.3
        v[:, 0] = 0
        v_transported = parallel_transport(v, origin, random_points)

        norm_orig = lorentzian_inner(v, v, keepdim=False)
        norm_transported = lorentzian_inner(v_transported, v_transported, keepdim=False)
        assert torch.allclose(norm_orig, norm_transported, atol=1e-4)


class TestCoordinateConversion:
    def test_lorentz_poincare_roundtrip(self, random_points):
        """Lorentz -> Poincare -> Lorentz 往返"""
        poincare = lorentz_to_poincare(random_points)
        lorentz_back = poincare_to_lorentz(poincare)
        assert torch.allclose(random_points, lorentz_back, atol=1e-4)

    def test_poincare_norm_less_than_one(self, random_points):
        """Poincare 坐标范数 < 1"""
        poincare = lorentz_to_poincare(random_points)
        norms = torch.norm(poincare, dim=-1)
        assert torch.all(norms < 1.0)


class TestPolarProject:
    def test_output_on_manifold(self):
        """极坐标投影输出在 Lorentz 流形上"""
        x = torch.randn(16, 8) * 1.0
        z = polar_project(x)
        inner = lorentzian_inner(z, z, keepdim=False)
        assert torch.allclose(inner, torch.tensor(-1.0).expand(16), atol=5e-4)

    def test_output_dimension(self):
        """输出维度 = 输入维度 + 1"""
        x = torch.randn(4, 10)
        z = polar_project(x)
        assert z.shape == (4, 11)
