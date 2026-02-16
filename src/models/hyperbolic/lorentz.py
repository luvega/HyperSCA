"""Lorentz (Hyperboloid) 模型工具函数

在 Lorentz 模型 L^d = {x in R^{d+1}: <x,x>_L = -1/K} 上实现:
- Minkowski 内积
- 双曲距离
- 指数映射 (ExpMap)
- 对数映射 (LogMap)
- 平行传输 (Parallel Transport)
- Lorentz <-> Poincare 坐标转换
- 投影至流形（数值安全）

Adapter 模式: 参考 references/scDHMap/src/scDHMap/lorentzian_helper.py,
              禁止直接 import，完全重写。

所有操作支持 batch 维度，使用 clamp 防止数值溢出。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7
MAX_NORM = 50.0  # 防止 cosh/sinh 溢出的最大范数


# =========================================================================
# 基础运算
# =========================================================================

def lorentzian_inner(
    x: torch.Tensor, y: torch.Tensor, keepdim: bool = True
) -> torch.Tensor:
    """Minkowski 内积: <x, y>_L = -x_0*y_0 + sum(x_i*y_i)

    Parameters
    ----------
    x, y : (..., d+1) Lorentz 坐标
    keepdim : bool
        是否保留最后一维

    Returns
    -------
    (..., 1) or (...,) Minkowski 内积
    """
    # 分离时间分量与空间分量
    x0 = x.narrow(-1, 0, 1)   # (..., 1)
    x_rest = x.narrow(-1, 1, x.shape[-1] - 1)
    y0 = y.narrow(-1, 0, 1)
    y_rest = y.narrow(-1, 1, y.shape[-1] - 1)

    # -x0*y0 + sum(xi*yi)
    inner = -x0 * y0 + (x_rest * y_rest).sum(dim=-1, keepdim=True)

    if not keepdim:
        inner = inner.squeeze(-1)
    return inner


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentz 双曲距离: d_L(x,y) = arccosh(-<x,y>_L)

    Parameters
    ----------
    x, y : (..., d+1)

    Returns
    -------
    (...,) 非负距离
    """
    inner = lorentzian_inner(x, y, keepdim=False)
    # -<x,y>_L >= 1 on the hyperboloid; clamp for numerical safety
    return torch.acosh(torch.clamp(-inner, min=1.0 + EPS))


def lorentz_norm(v: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """切向量的 Lorentz 范数: ||v||_L = sqrt(<v,v>_L)

    对于切向量 <v,v>_L >= 0。
    """
    sq = lorentzian_inner(v, v, keepdim=keepdim)
    return torch.sqrt(torch.clamp(sq, min=EPS))


# =========================================================================
# 指数映射 & 对数映射
# =========================================================================

def exp_map(v: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """指数映射: T_base H^d -> H^d

    exp_base(v) = cosh(||v||)*base + sinh(||v||)*v/||v||

    Parameters
    ----------
    v : (..., d+1) 切向量（满足 <v, base>_L = 0）
    base : (..., d+1) 流形上的基点

    Returns
    -------
    (..., d+1) 流形上的点
    """
    v_norm = lorentz_norm(v, keepdim=True)  # (..., 1)
    v_norm_clamped = torch.clamp(v_norm, max=MAX_NORM)

    # 单位方向
    v_unit = v / torch.clamp(v_norm, min=EPS)

    result = (
        torch.cosh(v_norm_clamped) * base
        + torch.sinh(v_norm_clamped) * v_unit
    )
    return project_to_lorentz(result)


def log_map(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """对数映射: H^d -> T_base H^d

    log_base(x) = d(base,x) / sqrt(alpha^2 - 1) * (x - alpha*base)
    其中 alpha = -<base, x>_L

    Parameters
    ----------
    x : (..., d+1) 流形上的点
    base : (..., d+1) 流形上的基点

    Returns
    -------
    (..., d+1) 切向量
    """
    alpha = -lorentzian_inner(base, x, keepdim=True)  # >= 1
    alpha = torch.clamp(alpha, min=1.0 + EPS)

    dist = torch.acosh(alpha)  # (..., 1)
    denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=EPS))

    # 方向: x - alpha * base（投影掉 base 方向分量）
    direction = x - alpha * base
    result = dist / denom * direction

    return result


# =========================================================================
# 平行传输
# =========================================================================

def parallel_transport(
    v: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor
) -> torch.Tensor:
    """平行传输: T_src H^d -> T_tgt H^d

    PT_{src->tgt}(v) = v + <tgt, v>_L / (alpha + 1) * (src + tgt)
    其中 alpha = -<src, tgt>_L

    Parameters
    ----------
    v : (..., d+1) 在 src 处的切向量
    src : (..., d+1) 源点
    tgt : (..., d+1) 目标点

    Returns
    -------
    (..., d+1) 在 tgt 处的切向量
    """
    alpha = -lorentzian_inner(src, tgt, keepdim=True)
    coeff = lorentzian_inner(tgt, v, keepdim=True) / (alpha + 1.0)
    return v + coeff * (src + tgt)


# =========================================================================
# 坐标转换
# =========================================================================

def lorentz_to_poincare(x: torch.Tensor) -> torch.Tensor:
    """Lorentz -> Poincare 球: phi(x) = x_{1:d} / (x_0 + 1)

    Parameters
    ----------
    x : (..., d+1) Lorentz 坐标（x_0 为时间分量）

    Returns
    -------
    (..., d) Poincare 球坐标
    """
    d = x.shape[-1] - 1
    spatial = x.narrow(-1, 1, d)
    time = x.narrow(-1, 0, 1)
    return spatial / (time + 1.0)


def poincare_to_lorentz(x: torch.Tensor) -> torch.Tensor:
    """Poincare 球 -> Lorentz: x -> (1+||x||^2, 2x) / (1-||x||^2)

    Parameters
    ----------
    x : (..., d) Poincare 球坐标

    Returns
    -------
    (..., d+1) Lorentz 坐标
    """
    x_sq_norm = (x * x).sum(dim=-1, keepdim=True)
    denom = 1.0 - x_sq_norm + EPS
    time = (1.0 + x_sq_norm) / denom
    spatial = 2.0 * x / denom
    return torch.cat([time, spatial], dim=-1)


# =========================================================================
# 投影（数值安全）
# =========================================================================

def project_to_lorentz(x: torch.Tensor) -> torch.Tensor:
    """将点投影回 Lorentz 流形: 修正 x_0 使 <x,x>_L = -1

    x_0 = sqrt(1 + ||x_{1:d}||^2)

    Parameters
    ----------
    x : (..., d+1)

    Returns
    -------
    (..., d+1) 满足 Lorentz 约束
    """
    d = x.shape[-1] - 1
    spatial = x.narrow(-1, 1, d)
    sq_norm = (spatial * spatial).sum(dim=-1, keepdim=True)
    time = torch.sqrt(torch.clamp(1.0 + sq_norm, min=EPS))
    return torch.cat([time, spatial], dim=-1)


def lorentz_origin(dim: int, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
    """Lorentz 原点: (1, 0, 0, ..., 0)

    Parameters
    ----------
    dim : int
        空间维度 d（总维度 d+1）
    batch_size : int
    device : str

    Returns
    -------
    (batch_size, d+1)
    """
    o = torch.zeros(batch_size, dim + 1, device=device)
    o[:, 0] = 1.0
    return o


# =========================================================================
# 极坐标投影（Encoder 输出 -> Lorentz）
# =========================================================================

def polar_project(x: torch.Tensor) -> torch.Tensor:
    """极坐标投影: R^d -> L^d

    将欧氏 d 维向量转为 Lorentz (d+1) 维坐标:
    z = (cosh(r), sinh(r) * x/||x||)  其中 r = ||x||

    Parameters
    ----------
    x : (..., d) 欧氏空间向量

    Returns
    -------
    (..., d+1) Lorentz 坐标
    """
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_unit = x / torch.clamp(x_norm, min=EPS)
    x_norm = torch.clamp(x_norm, max=MAX_NORM)

    time = torch.cosh(x_norm)
    spatial = torch.sinh(x_norm) * x_unit
    return torch.cat([time, spatial], dim=-1)
