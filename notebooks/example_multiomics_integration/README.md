# HyperSCA Multi-omics Integration Example

本目录包含展示 HyperSCA **多组学整合分析**能力的 notebook 示例。
所有结果基于预计算的集成分析输出，靶点发现**完全数据驱动**（无预设 anchor 基因）。

## Notebook 列表

| # | Notebook | 内容 |
|---|---------|------|
| 00 | `00_data_landscape.ipynb` | 多组学数据概览（scRNA×3 + Visium + CosMx + VisiumHD） |
| 01 | `01_hyperbolic_vs_euclidean_embedding.ipynb` | 双曲 vs 欧氏嵌入对比：Silhouette +70%，层级相关 1.0 vs −0.57 |
| 02 | `02_multiscale_spatial_niche.ipynb` | 多尺度 niche（micro→macro），跨样本一致性 >0.83 |
| 03 | `03_causal_network_spatial_advantage.ipynb` | 空间约束因果网络、DoWhy 证伪、信号流完整性 |
| 04 | `04_data_driven_target_discovery.ipynb` | 5,873 候选基因 → 5 维证据排名，纯数据驱动 |
| 05 | `05_integration_summary.ipynb` | 靶点-Niche 关联、信号流可视化、最终对比总结 |

## 数据规模

- **485,362** spots/cells，跨 **3 个空间平台**（Visium / CosMx / VisiumHD）
- **3 个 scRNA-seq 队列**（scCRC_Neu / scCRC_ICB / scCRC_IFNG）
- **18** 统一 niche 定义，**4** 个尺度（micro / small / medium / macro）

## 核心发现

- **双曲嵌入** niche Silhouette 比欧氏提升 **70%**
- **层级相关性** 1.0（双曲）vs −0.569（欧氏）
- 空间约束增加了 **2 个独立证据维度**（spatial + niche）
- 靶点排名完全数据驱动，传统 anchor 基因自然浮现

## 运行环境

- Python 3.10+, conda env `hypersca`
- 依赖：numpy, pandas, matplotlib, nbformat, scanpy
- 所有 notebook 已预嵌入图表，可直接在 GitHub 上查看
