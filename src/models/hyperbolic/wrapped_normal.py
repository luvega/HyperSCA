"""Wrapped Normal 分布 (双曲空间上的正态分布)

在 Lorentz 模型上定义 Wrapped Normal 分布:
- 采样: 切空间正态 -> 平行传输 -> 指数映射
- 对数概率: 对数映射 -> 平行传输回原点 -> 基分布 log_prob - 体积校正
- KL 散度: 与标准先验的 KL（Monte Carlo 估计）

Adapter 模式: 参考 references/scDHMap/src/scDHMap/wrapped_normal.py,
              禁止直接 import。
"""
from __future__ import annotations

import torch
import torch.distributions as dist

from src.models.hyperbolic.lorentz import (
    EPS,
    exp_map,
    log_map,
    parallel_transport,
    lorentzian_inner,
    lorentz_origin,
    project_to_lorentz,
)


class WrappedNormal(dist.Distribution):
    """双曲 Wrapped Normal 分布

    在 Lorentz 流形上定义的正态分布。均值 loc 在流形上，
    方差 scale 定义在 loc 处的切空间中。

    Parameters
    ----------
    loc : (batch, d+1)
        Lorentz 流形上的均值点
    scale : (batch, d)
        切空间中各维度的标准差（空间维度，不含时间分量）
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool = False,
    ):
        self._loc = loc          # (batch, d+1)
        self._scale = scale      # (batch, d)
        self._dim = loc.shape[-1] - 1  # d

        # 基分布: d 维独立正态
        self._base_dist = dist.Normal(
            torch.zeros_like(scale), scale
        )

        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args)

    @property
    def loc(self) -> torch.Tensor:
        return self._loc

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    def _origin(self) -> torch.Tensor:
        """Lorentz 原点 (1, 0, ..., 0)，与 loc 同 shape"""
        return lorentz_origin(
            self._dim,
            batch_size=self._loc.shape[0],
            device=self._loc.device,
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """可微采样（重参数化）

        1. 从欧氏正态 N(0, scale) 采样 v_0 in R^d
        2. 构造切向量 (0, v_0) in T_origin
        3. 平行传输: T_origin -> T_loc
        4. 指数映射: T_loc -> H^d

        Returns
        -------
        (..., d+1) Lorentz 流形上的采样点
        """
        # Step 1: 切空间采样
        v_spatial = self._base_dist.rsample(sample_shape)  # (..., d)

        # Step 2: 构造切向量 at origin (时间分量 = 0)
        v_at_origin = torch.cat(
            [torch.zeros_like(v_spatial[..., :1]), v_spatial],
            dim=-1,
        )  # (..., d+1)

        # Step 3: 平行传输到 loc
        origin = self._origin()
        # 处理 sample_shape: 扩展 origin 和 loc
        if len(sample_shape) > 0:
            origin = origin.unsqueeze(0).expand_as(v_at_origin)
            loc_expanded = self._loc.unsqueeze(0).expand_as(v_at_origin)
        else:
            loc_expanded = self._loc

        v_at_loc = parallel_transport(v_at_origin, origin, loc_expanded)

        # Step 4: 指数映射
        z = exp_map(v_at_loc, loc_expanded)
        return z

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """对数概率密度

        1. log_loc(x) -> 切向量 v
        2. ||v||_L -> 体积校正
        3. PT v from loc to origin
        4. 基分布 log_prob(v_spatial) - volume_correction

        Parameters
        ----------
        x : (..., d+1) 流形上的点

        Returns
        -------
        (...,) 对数概率
        """
        # Step 1: 对数映射到切空间
        v = log_map(x, self._loc)

        # Step 2: 切向量范数（用于体积校正）
        v_sq = lorentzian_inner(v, v, keepdim=False)
        v_norm = torch.sqrt(torch.clamp(v_sq, min=EPS))

        # 体积校正: (d-1) * log(sinh(||v||) / ||v||)
        # sinh(r)/r >= 1 对 r >= 0
        sinh_ratio = torch.sinh(v_norm) / torch.clamp(v_norm, min=EPS)
        volume_correction = (self._dim - 1.0) * torch.log(
            torch.clamp(sinh_ratio, min=EPS)
        )

        # Step 3: 平行传输回原点
        origin = self._origin()
        if x.dim() > self._loc.dim():
            origin = origin.unsqueeze(0).expand_as(v)
            loc_expanded = self._loc.unsqueeze(0).expand_as(v)
        else:
            loc_expanded = self._loc

        v_at_origin = parallel_transport(v, loc_expanded, origin)

        # Step 4: 取空间分量，计算基分布 log_prob
        v_spatial = v_at_origin[..., 1:]  # (..., d)
        base_log_prob = self._base_dist.log_prob(v_spatial).sum(dim=-1)

        return base_log_prob - volume_correction

    def kl_divergence(self, n_samples: int = 10) -> torch.Tensor:
        """与标准先验 WN(origin, I) 的 KL 散度（Monte Carlo 估计）

        KL(q || p) = E_q[log q(z) - log p(z)]

        Parameters
        ----------
        n_samples : int
            MC 采样数

        Returns
        -------
        (batch,) KL 散度
        """
        # 先验: WN(origin, I)
        origin = self._origin()
        ones = torch.ones_like(self._scale)
        prior = WrappedNormal(origin, ones)

        # MC 估计
        z = self.rsample(torch.Size([n_samples]))  # (n_samples, batch, d+1)
        log_q = self.log_prob(z)   # (n_samples, batch)
        log_p = prior.log_prob(z)  # (n_samples, batch)

        kl = (log_q - log_p).mean(dim=0)  # (batch,)
        return kl
