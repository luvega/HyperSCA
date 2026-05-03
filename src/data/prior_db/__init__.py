"""
src.data.prior_db — 先验信号通路调控数据库下载与整合
=====================================================

提供 OmniPath / NicheNet / LIANA 三大先验数据库的一键下载、本地缓存、
格式统一与整合功能。

Public API
----------
download_all(force=False)   一键下载全部先验数据库
load_lr_interactions()      加载配体-受体互作表
load_tf_targets()           加载 TF-靶基因调控表
load_signaling_network()    加载信号通路 PPI 网络
load_nichenet_prior()       加载 NicheNet 先验网络
load_liana_resource()       加载 LIANA 共识配受体资源
get_manifest()              获取下载清单与统计摘要
"""

from src.data.prior_db._config import PRIOR_DB_DIR
from src.data.prior_db._download import download_all
from src.data.prior_db._integrate import (
    load_lr_interactions,
    load_tf_targets,
    load_signaling_network,
    load_nichenet_prior,
    load_liana_resource,
    get_manifest,
)

__all__ = [
    "PRIOR_DB_DIR",
    "download_all",
    "load_lr_interactions",
    "load_tf_targets",
    "load_signaling_network",
    "load_nichenet_prior",
    "load_liana_resource",
    "get_manifest",
]
