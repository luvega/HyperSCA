# HyperSCA 全量复现数据存在性检查

检查日期：2026-06-12
项目根目录：`E:/Codex_Projects/HyperSCA`

## 结论

- 当前已经实现的 `distance-constrained LR communication OT/flow proxy` 可以在本地全量复现：`hyperbolic` 与 `euclidean` 两套 cluster-level 输入均存在，且已可由 `scripts/reproduce_spatial_communication_ot.py` 生成复现结果。
- 严格意义上的 spot/cell-level COMMOT 全量复现所需核心数据也基本存在：Visium、ST_CRC_MSS、CosMx、Xenium、VisiumHD 的表达矩阵与空间坐标资源均可在本地找到。
- 现阶段缺口不是原始数据缺失，而是缺一个 COMMOT-ready 的锁定输入契约：表达矩阵、空间坐标、质量/距离代价、LR prior、平台级抽样策略、输出 manifest 与依赖版本需要统一。
- 当前 shell 未发现 AGENTS 中记录的 `E:/ProgramData/Anaconda3/envs/hypersca/python.exe`；`conda env list` 仅显示 `base` 与 `r45-qs`。系统 `C:/Python313/python.exe` 可导入 `pandas`、`anndata` 与 `pyarrow`，但若要按仓库文档做一键复现，需要先恢复或重新指定 `hypersca` 环境路径。

## 当前 OT/Flow Proxy 复现输入

| mode | 文件 | 存在 | 复现作用 |
|---|---|---|---|
| hyperbolic | `results/integration/discovery/hyperbolic/step2/cluster_expr.csv` | yes | cluster-level expression matrix；检查到 20 x 15,610 |
| hyperbolic | `results/integration/discovery/hyperbolic/step2/node_info.json` | yes | cluster node metadata |
| hyperbolic | `results/integration/discovery/hyperbolic/step2/causal_adjacency.npy` | yes | Step2 因果邻接，用于方向一致率分层 |
| hyperbolic | `results/integration/discovery/hyperbolic/geometry/distance.npy` | yes | 几何距离约束 |
| hyperbolic | `results/integration/discovery/hyperbolic/geometry/blended.npy` | yes | 缺少持久化 `spatial_adjacency.npy` 时的兼容空间约束 |
| euclidean | `results/integration/discovery/euclidean/step2/cluster_expr.csv` | yes | cluster-level expression matrix；检查到 20 x 15,610 |
| euclidean | `results/integration/discovery/euclidean/step2/node_info.json` | yes | cluster node metadata |
| euclidean | `results/integration/discovery/euclidean/step2/causal_adjacency.npy` | yes | Step2 因果邻接，用于方向一致率分层 |
| euclidean | `results/integration/discovery/euclidean/geometry/distance.npy` | yes | 几何距离约束 |
| euclidean | `results/integration/discovery/euclidean/geometry/blended.npy` | yes | 缺少持久化 `spatial_adjacency.npy` 时的兼容空间约束 |

## Spot/Cell-Level 空间数据

| 数据集 | 本地入口 | 存在 | 检查到的规模/字段 | 用途 |
|---|---|---|---|---|
| Visium_HumanColon_Oliveira processed h5ad | `data/Visium_HumanColon_Oliveira/expression.h5ad` | yes | 4,269 spots x 18,085 genes；`obsm['spatial']` 存在 | spot-level 空间转录组复现输入 |
| ST_CRC_MSS processed h5ad | `data/ST/ST_CRC_MSS/expression.h5ad` | yes | 178,980 spots x 43,535 genes；`obsm['spatial']` 与 `obsm['rctd_freq']` 存在 | 大规模 ST 队列与反卷积元数据 |
| scCRC_IFNG_CosMx processed h5ad | `data/ST/scCRC_IFNG_CosMx/expression.h5ad` | yes | 744,816 cells/spots x 1,010 genes；`obsm['spatial']` 存在 | CosMx 风格单细胞空间输入 |
| Xenium processed h5ad | `data/Xenium_HumanColon_Oliveira/expression.h5ad` | yes | 340,837 cells x 422 genes；`obsm['spatial']` 存在 | cell-level in situ 空间输入 |
| VisiumHD square 008um | `data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_008um/` | yes | matrix 545,913 bins x 18,085 features；positions 702,244 rows | 高分辨率 binned spatial expression，需 adapter |
| VisiumHD square 016um | `data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/` | yes | matrix 137,051 bins x 18,085 features；positions 175,561 rows | 可作为 VisiumHD 首个 COMMOT 适配对象 |
| VisiumHD segmented cell matrix | `data/VisiumHD_HumanColon_Oliveira/segmented_outputs/filtered_feature_cell_matrix.h5` | yes | 220,703 cells x 18,132 features | cell-segmented spatial expression，需坐标 join |
| Xenium raw parquet stack | `data/Xenium_HumanColon_Oliveira/` | yes | cells 340,837 rows；transcripts 66,248,117 rows；cell boundaries 8,509,589 rows | cell-level 原始空间证据 |

## 单细胞参考与多源整合输入

| 数据集 | 本地入口 | 存在 | 检查到的规模/字段 | 用途 |
|---|---|---|---|---|
| Chromium_HumanColon_Oliveira | `data/Chromium_HumanColon_Oliveira/expression.h5ad` | yes | 279,609 cells x 18,082 genes | scRNA reference / Step1-3 标准流水线输入 |
| scCRC_ICB subset | `data/scRNA/scCRC_ICB/expression.h5ad` | yes | 200,000 cells x 36,027 genes；metadata 与 DEG tables 存在 | ICB reference 与 target-discovery evidence layer |
| scCRC_Neu | `data/scRNA/scCRC_Neu/expression.h5ad` | yes | 126,991 cells x 19,793 genes；`X_scVI` 与 `X_umap` 存在 | 多源 scRNA 外部队列 |
| ICB reference model adata | `data/ref/mappings/icb_reference/v1/reference_adata.h5ad` | yes | 100,000 cells x 3,000 genes；`X_scvi` 存在 | reference mapping object |

## 先验资源

| 资源 | 本地入口 | 存在 | 检查到的行数 | 用途 |
|---|---|---|---|---|
| LIANA consensus LR | `data/prior_db/liana/consensus_lr_resource.tsv` | yes | 4,624 | ligand-receptor prior |
| NicheNet LR network | `data/prior_db/nichenet/lr_network.tsv` | yes | 4,986 | ligand-receptor prior |
| NicheNet GR network | `data/prior_db/nichenet/gr_network.tsv` | yes | 5,870,450 | LR->TF->target 机制链路中的基因调控层 |
| OmniPath/Dorothea TF-target | `data/prior_db/omnipath/dorothea_tf_target.tsv` | yes | 15,267 | TF-target evidence layer |

## 需要注意的复现边界

1. 旧的 integration 预计算目录没有持久化 `spatial_adjacency.npy`，当前复现脚本使用 `geometry/blended.npy` 作为兼容的空间约束输入。由于本地存在 spot/cell 坐标，后续可以从坐标重新生成平台特异的空间邻接。
2. VisiumHD 的 8um/16um bin 与 segmented cell matrix 原始文件存在，但尚未统一转换成 COMMOT/AnnData 风格的单一输入对象；需要 adapter 或 on-disk loader。
3. 对严格 COMMOT 复现，建议先固定一个小到中等规模平台作为验证集，例如 Visium 或 VisiumHD 16um，再扩展到 CosMx/Xenium/VisiumHD 8um。
4. 所有通信流、OT、方向一致率结果仍应表述为机制假设、计算优先级或实验设计依据，不能写成因果证明。

## 可直接执行的当前复现命令

```powershell
python scripts\reproduce_spatial_communication_ot.py
```

该命令写出：`results/integration/discovery/communication_ot_reproduction/`，并同步摘要图和报告到 `docs/research/`。
