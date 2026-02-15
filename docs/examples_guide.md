# HyperSCA Examples 运行指南

*版本: v2.0 | 日期: 2026-02-15*

---

## 快速开始

```bash
# 激活环境
conda activate hypersca

# 或直接使用完整路径
# E:\ProgramData\Anaconda3\envs\hypersca\python.exe scripts/run_examples_all.py

# 一键运行全部 4 个 Example
python scripts/run_examples_all.py

# 或单独运行
python scripts/run_example_01.py   # Chromium 元数据 QC（增强版）
python scripts/run_example_02.py   # Visium 空间图构建（增强版）
python scripts/run_example_03.py   # VisiumHD 分割统计（增强版）
python scripts/run_example_04.py   # Xenium 基因面板摘要（增强版）
```

## 前置条件

- Python 3.10（conda 环境 `hypersca`，路径 `E:\ProgramData\Anaconda3\envs\hypersca`）
- 基础依赖：`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`（均已含在 `requirements-core.txt`）
- 不需要 GPU、不需要 `.h5ad` 表达矩阵文件
- 数据目录 `data/` 下需存在以下文件：

| Example | 必需文件 |
|---------|----------|
| 01 | `data/Chromium_HumanColon_Oliveira/cell_metadata.csv` |
| 02 | `data/Visium_HumanColon_Oliveira/outs/spatial/tissue_positions.csv` |
| 02 | `data/Visium_HumanColon_Oliveira/outs/spatial/scalefactors_json.json` |
| 03 | `data/VisiumHD_HumanColon_Oliveira/segmented_outputs/cell_segmentations.geojson` |
| 03 | `data/VisiumHD_HumanColon_Oliveira/segmented_outputs/nucleus_segmentations.geojson` |
| 04 | `data/Xenium_HumanColon_Oliveira/experiment.xenium` |
| 04 | `data/Xenium_HumanColon_Oliveira/gene_panel.json` |

## 输出产物

运行后在 `results/examples/` 下生成：

```
results/examples/
├── run_log.txt                          # 批量运行日志
├── example01/
│   ├── summary.csv                      # 细胞类型统计（Level1/Level2, count, fraction）
│   ├── patient_summary.csv              # 患者分组统计
│   ├── fig_celltype.png                 # Level1 分布条形图（基础版）
│   ├── fig_celltype.meta.json           # 图元信息（config + timestamp）
│   ├── fig_celltype_nested.png          # [新] Level1×Level2 嵌套分组条形图
│   ├── fig_celltype_nested.meta.json
│   ├── fig_celltype_sunburst.png        # [新] Level1→Level2 Sunburst 环形层级图
│   ├── fig_celltype_sunburst.meta.json
│   ├── fig_patient_qc.png              # [新] 患者 QC 保留/丢弃 堆叠条形图
│   └── fig_patient_qc.meta.json
├── example02/
│   ├── knn_edges.csv                    # k-NN 边表（source, target, distance）
│   ├── graph_stats.json                 # [新] 图基础统计
│   ├── spatial_graph.png                # 空间散点 + 图覆盖（基础版）
│   ├── spatial_graph.meta.json
│   ├── spatial_graph_region.png         # [新] 按组织区域着色的空间图
│   ├── spatial_graph_region.meta.json
│   ├── edge_distance_dist.png           # [新] 边距离分布直方图 + KDE
│   └── edge_distance_dist.meta.json
├── example03/
│   ├── cell_areas.csv                   # 细胞多边形面积
│   ├── nucleus_areas.csv                # 细胞核多边形面积
│   ├── segmentation_stats.json          # 汇总统计（含核质比）
│   ├── area_hist.png                    # 面积分布直方图（基础版）
│   ├── area_hist.meta.json
│   ├── segmentation_quality.png         # [新] 3-panel 分割质量 Dashboard
│   ├── segmentation_quality.meta.json   #       (Cell Area + Nucleus Area + NC Ratio)
│   ├── area_scatter.png                 # [新] Cell vs Nucleus 联合散点图
│   └── area_scatter.meta.json
└── example04/
    ├── panel_summary.csv                # 基因面板 targets 列表
    ├── experiment_info.json             # 实验元信息
    ├── experiment_report.md             # 可读 Markdown 报告
    ├── descriptor_donut.png             # [新] Descriptor 构成环形图
    ├── descriptor_donut.meta.json
    ├── source_bar.png                   # [新] Source panel 来源条形图
    ├── source_bar.meta.json
    ├── panel_composition.png            # [新] 面板组成综合 Dashboard
    └── panel_composition.meta.json
```

### 图元信息 (`.meta.json`)

每张图自动附带同名 `.meta.json` 文件，记录可复现元信息：

```json
{
  "figure": "fig_celltype.png",
  "created": "2026-02-15T18:00:00",
  "dpi": 200,
  "data_version": "",
  "model_version": "",
  "seed": null,
  "config": {"chart": "celltype_bar_level1"}
}
```

### 后续阶段产物目录（预留）

```
results/figures/
├── step1/     # Phase 1 双曲嵌入可视化
├── step2/     # Phase 2 因果图可视化
├── step3/     # Phase 3 反事实扰动可视化
└── dashboard/ # Phase 4 综合 Dashboard
```

## Notebook 使用

Notebook 位于 `notebooks/` 目录，已升级为增强版，包含 Phase 0 新增图型：

| Notebook | 对应 Script | 新增图型 |
|----------|-------------|----------|
| `example_01_metadata.ipynb` | `scripts/run_example_01.py` | 嵌套条形、Sunburst、患者 QC 图 |
| `example_02_spatial_graph.ipynb` | `scripts/run_example_02.py` | 区域着色空间图、边距离分布 |
| `example_03_segmentation.ipynb` | `scripts/run_example_03.py` | 3-panel 质量图、联合散点 |
| `example_04_xenium.ipynb` | `scripts/run_example_04.py` | Descriptor 环形、Source 条形、综合 Dashboard |

使用 Jupyter 打开时需确保 kernel 为 `hypersca` 环境。

## 验收检查清单

### Phase 0 基础验收

- [ ] `scripts/run_examples_all.py` 退出码为 0（4/4 通过）
- [ ] `results/examples/run_log.txt` 显示 "Passed: 4/4"

### Example 01 (Chromium 元数据 QC)

- [ ] `summary.csv` 非空，包含 Level1、Level2、count、fraction 列
- [ ] `fig_celltype.png` 已生成（基础版 Level1 条形图）
- [ ] `fig_celltype_nested.png` 已生成（嵌套分组条形图）
- [ ] `fig_celltype_sunburst.png` 已生成（Sunburst 环形图）
- [ ] `fig_patient_qc.png` 已生成（患者 QC 堆叠条形图）

### Example 02 (Visium 空间图)

- [ ] `knn_edges.csv` 行数 = in_tissue_spots × k
- [ ] `graph_stats.json` 包含 n_nodes、n_edges、mean_distance
- [ ] `spatial_graph.png` 已生成（基础版）
- [ ] `spatial_graph_region.png` 已生成（区域着色版）
- [ ] `edge_distance_dist.png` 已生成（边距离分布）

### Example 03 (VisiumHD 分割)

- [ ] `segmentation_stats.json` 中 `n_cells > 0`
- [ ] `area_hist.png` 已生成（基础版直方图）
- [ ] `segmentation_quality.png` 已生成（3-panel 质量 Dashboard）
- [ ] `area_scatter.png` 已生成（联合散点图）

### Example 04 (Xenium 面板)

- [ ] `panel_summary.csv` 行数 > 0
- [ ] `experiment_report.md` 已生成
- [ ] `descriptor_donut.png` 已生成（Descriptor 环形图）
- [ ] `source_bar.png` 已生成（Source 条形图）
- [ ] `panel_composition.png` 已生成（综合 Dashboard）

### 统一绘图风格

- [ ] 所有 `.png` 附带对应 `.meta.json`
- [ ] 图表使用 HyperSCA 统一色板（colorblind-friendly）
- [ ] 图表右下角含 HyperSCA 水印（alpha=0.08）

## 代码结构

```
src/
├── data/
│   ├── loaders.py              # 统一数据加载（4 个 load_* 函数）
│   └── validators.py           # 字段校验
├── examples/
│   ├── config.py               # 公共路径与参数配置
│   ├── metadata_qc.py          # Example01 分析 + 可视化（含嵌套条形/Sunburst/患者QC）
│   ├── spatial_graph.py        # Example02 分析 + 可视化（含着色空间图/热图/距离分布）
│   ├── segmentation_stats.py   # Example03 分析 + 可视化（含3-panel质量图/散点）
│   └── gene_panel_summary.py   # Example04 分析 + 可视化（含环形图/条形/Dashboard）
├── utils/
│   ├── __init__.py
│   └── plot_style.py           # [新] 统一绘图风格（色板、rcParams、save_figure）
├── visualization/              # [新] 全阶段可视化模块（Phase 1-3 skeleton）
│   ├── __init__.py
│   ├── hyperbolic.py           # Phase 1: Poincaré 圆盘/Hyperboloid 3D/径向分支/指标
│   ├── causal.py               # Phase 2: DAG/信号流 Sankey/邻接热图/证据卡/指标
│   └── perturbation.py         # Phase 3: 扰动对比/空间热图/传播梯度/多靶点/指标
scripts/
├── run_example_01.py ~ run_example_04.py  # 单 example 入口（增强版）
└── run_examples_all.py                     # 批量运行
notebooks/
└── example_01 ~ example_04                 # 交互式版本（增强版）
```

## 可视化模块 (Phase 1-3 Skeleton)

Phase 1-3 的可视化函数已创建为 skeleton 模块，位于 `src/visualization/`。
当前传入 `None` 数据时返回带占位说明的 Figure；数据就绪后传入真实数组即可产出正式图。

| 模块 | 图类型 | 状态 |
|------|--------|------|
| `hyperbolic.py` | Poincaré 圆盘 2D、Hyperboloid 3D、径向分支、Baseline 对照、指标 Dashboard | Skeleton（待数据） |
| `causal.py` | 因果 DAG、信号流 Sankey、邻接热图、关键轴证据卡、指标 Dashboard | Skeleton（待数据） |
| `perturbation.py` | 扰动对比、反事实空间热图、传播梯度、多靶点热图、指标 Dashboard | Skeleton（待数据） |

## 后续衔接

当真实表达矩阵（`.h5ad`）补齐后，将追加：
- Example 05: Step1 双曲嵌入最小版
- Example 06: Step2 因果网络最小版
- Example 07: Step3 虚拟敲除最小版

本轮代码（数据加载、空间图、统一绘图风格、Phase 0-3 可视化模板）将作为后续模型 example 的底座。
