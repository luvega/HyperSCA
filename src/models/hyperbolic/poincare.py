"""Poincare 球模型工具函数

在 Poincare 球 B^d_c = {x in R^d : c*||x||^2 < 1} 上实现:
- Poincare 距离
- 球内投影（数值安全）
- Mobius 加法

Adapter 模式: 参考 references/scDHMap/src/scDHMap/lorentzian_helper.py,
              禁止直接 import。
"""
from __future__ import annotations

import torch

EPS = 1e-7


def poincare_distance(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """Poincare 球距离

    d_B(x,y) = (1/sqrt(c)) * arccosh(1 + 2c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))

    Parameters
    ----------
    x, y : (..., d)
    c : float
        曲率参数 c = -K > 0

    Returns
    -------
    (...,) 非负距离
    """
    diff_sq = ((x - y) ** 2).sum(dim=-1)
    x_sq = (x ** 2).sum(dim=-1)
    y_sq = (y ** 2).sum(dim=-1)

    denom = (1.0 - c * x_sq) * (1.0 - c * y_sq)
    denom = torch.clamp(denom, min=EPS)

    arg = 1.0 + 2.0 * c * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + EPS)

    return torch.acosh(arg) / (c ** 0.5)


def project_to_ball(
    x: torch.Tensor, c: float = 1.0, max_norm: float = 1.0 - 1e-5
) -> torch.Tensor:
    """投影到 Poincare 球内: 确保 sqrt(c)*||x|| < max_norm

    Parameters
    ----------
    x : (..., d)
    c : float
    max_norm : float
        球内最大范数（< 1/sqrt(c)）

    Returns
    -------
    (..., d) 投影后的坐标
    """
    radius = max_norm / (c ** 0.5)
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    cond = x_norm > radius
    projected = x / x_norm * radius
    return torch.where(cond, projected, x)


def mobius_add(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """Mobius 加法 (Poincare 球上的加法运算)

    x (+)_c y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y)
                / (1 + 2c<x,y> + c^2 ||x||^2 ||y||^2)

    Parameters
    ----------
    x, y : (..., d)
    c : float

    Returns
    -------
    (..., d) Poincare 球上的点
    """
    xy = (x * y).sum(dim=-1, keepdim=True)
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)

    num = (1.0 + 2.0 * c * xy + c * y_sq) * x + (1.0 - c * x_sq) * y
    denom = 1.0 + 2.0 * c * xy + c ** 2 * x_sq * y_sq
    denom = torch.clamp(denom, min=EPS)

    return project_to_ball(num / denom, c=c)
