# HyperSCA 项目目录、文件与结果说明

> 快照日期：2026-05-24  
> 本地工作目录：`F:\HyperSCA`

本文用于界定当前项目中的四类内容：项目提交代码、验证代码、本地数据、运行结果。`data/`、`results/`、`references/`、`.venv/`、`.Trash/` 均为本地资产，默认不进入 Git。

## 1. 仓库边界

| 路径 | 类型 | Git 策略 | 作用 |
|---|---|---|---|
| `src/` | 项目提交代码 | 跟踪 | HyperSCA 核心包：数据处理、双曲模型、因果推断、扰动分析、靶点发现、评估和可视化。 |
| `scripts/` | 项目入口脚本 | 跟踪 | 数据入库、示例、Step1-Step4、靶点发现、图表生成、环境验证等可运行入口。 |
| `tests/` | 验证代码 | 跟踪 | 几何、因果、扰动、靶点发现、MSI 推断、环境验证等测试。 |
| `notebooks/` | 可复现实例 | 跟踪 | 多组学整合示例和 scCRC ICB 分步示例。 |
| `docs/` | 文档和静态图 | 跟踪 | 研究方案、技术路线、项目目录说明、图示和开发规格。 |
| `requirements.txt` | 核心 Python 依赖 | 跟踪 | 主线运行依赖；不再包含 `scgen`。 |
| `requirements-optional-baselines.txt` | 可选 baseline 依赖 | 跟踪 | 历史扰动 baseline 复现依赖，例如 `scgen`。 |
| `environment-r.yml` | R 环境说明 | 跟踪 | 辅助 R 工作流的环境记录。 |
| `.gitignore` | 仓库边界规则 | 跟踪 | 排除本地数据、结果、参考仓库、虚拟环境、缓存和测试临时目录。 |
| `data/` | 本地数据 | 忽略 | 原始、处理后、参考和元数据输入。当前快照约 81 个文件，15.9 GB。 |
| `results/` | 本地运行结果 | 忽略 | 分析输出、图表、报告、模型、嵌入和阶段产物。当前快照约 6,572 个文件，7.4 GB。 |
| `references/` | 本地参考仓库 | 忽略 | 只读外部方法仓库，用于实现参考，不直接 import。当前快照约 1,543 个文件，1.5 GB。 |
| `.venv/` | 本地 Python 环境 | 忽略 | Python 3.13 实验验证环境。当前快照约 6.4 GB。 |
| `.Trash/` | 本地清理归档 | 忽略 | 清理出的草稿、旧工作树、缓存和测试临时产物。 |

## 2. 源码目录说明

| 包 | 主要职责 |
|---|---|
| `src/data/` | 数据加载、预处理、空间图构建、先验数据库下载/整合、实验回写输入。 |
| `src/models/hyperbolic/` | Lorentz/Poincare 运算、wrapped normal 分布、双曲 VAE 组件。 |
| `src/causal/` | 因果图发现、CMI 剪枝、潜变量解缠、信号流、baseline 通讯比较、时序因果工具。 |
| `src/perturbation/` | 潜空间扰动、扩散反事实骨架、空间传播、靶点排序、假阳性过滤、剂量和 PK/PD 工具。 |
| `src/pipeline/` | Step1 嵌入、Step2 因果推断、Step3 扰动、Step4 动态干预、roundtrip 更新和共享配置。 |
| `src/discovery/target_discovery/` | 模块化靶点发现：配置、stage 协议、artifact writer、候选池、表达/空间输入、几何、因果 wrapper、扰动 wrapper、评分、生态位、报告和图表。 |
| `src/evaluation/` | 嵌入、因果、反事实、空间、跨样本和 MSI 推断指标。 |
| `src/examples/` | 轻量示例辅助函数：metadata QC、gene panel、segmentation stats、spatial graph。 |
| `src/visualization/` | 双曲、因果和扰动相关可视化。 |
| `src/utils/` | 共享绘图风格工具。 |

## 3. 脚本文件说明

| 脚本组 | 文件 | 作用 |
|---|---|---|
| 环境与资源 | `scripts/validate_env.py`, `scripts/download_prior_db.py` | 验证核心包、GPU/PyG 扩展、数据可读性，并下载本地先验资源。 |
| 数据入库 | `scripts/run_data_onboarding.py`, `scripts/build_canonical_schema.py`, `scripts/build_icb_reference.py`, `scripts/prepare_h5ad.py` | 标准化外部队列，构建 schema 表，准备 AnnData 和参考映射产物。 |
| 主流程 | `scripts/run_step1.py`, `scripts/run_step2.py`, `scripts/run_step3.py`, `scripts/run_step4.py` | 执行嵌入、因果推断、反事实扰动、动态干预和 roundtrip 更新。 |
| 靶点发现 | `scripts/run_target_discovery.py` | thin CLI，只构造 `TargetDiscoveryConfig` 并调用 `src/discovery/target_discovery/`。 |
| 示例 | `scripts/run_example_01.py` 到 `scripts/run_example_04.py`, `scripts/run_examples_all.py`, `scripts/run_platform_niche_analysis.py` | 生成演示输出和平台/生态位分析。 |
| 图表 | `scripts/generate_step1_figures.py`, `scripts/generate_step2_figures.py`, `scripts/generate_step3_figures.py`, `scripts/generate_step4_figures.py`, `scripts/generate_spatial_comm_figures.py`, `scripts/generate_spatial_combo_comm_figures.py` | 生成图表和 `.meta.json` 溯源信息。 |
| MSI 推断 | `scripts/infer_msi_status.py` | 基于 marker 覆盖度和表达方向推断 MSI/MSS 信号状态。 |

## 4. 验证代码说明

验证代码集中在 `tests/`，与 pytest 运行时产生的临时目录分离。

| 测试范围 | 文件 |
|---|---|
| 靶点发现管线 | `tests/discovery/test_target_discovery_*.py` |
| 双曲几何与 HVAE | `tests/test_lorentz.py`, `tests/test_hvae.py`, `tests/test_step1_vs_umap.py` |
| 数据标准化与参考整合 | `tests/test_data_standardization.py`, `tests/test_icb_reference_integration.py` |
| 因果与空间逻辑 | `tests/test_causal_metrics.py`, `tests/test_spatial_graph.py`, `tests/test_step2_pipeline.py`, `tests/test_step2_spatial_causal_advantage.py`, `tests/test_temporal_causal.py` |
| 扰动与动态干预 | `tests/test_perturbation.py`, `tests/test_step3_pipeline.py`, `tests/test_step3_false_positive_reduction.py`, `tests/test_step4_dynamic.py`, `tests/test_roundtrip_update.py` |
| 环境验证行为 | `tests/test_validate_env.py` |
| MSI 推断 | `tests/test_msi_inference.py` |

当前 Python 3.13 实验环境下，全量测试结果为 `138 passed, 137 warnings`。warning 主要来自依赖包弃用提示或小型测试数据导致的预期数值提示，不是失败。

## 5. 本地数据目录

`data/` 已被忽略，用于保存本地分析输入。它是复现本地结果所需的数据层，但不是项目提交代码。

| 路径 | 作用 |
|---|---|
| `data/metadata/` | 统一 schema 表、mapping rules、onboarding manifest、roundtrip 实验元数据。 |
| `data/prior_db/` | LIANA、NicheNet、OmniPath 等配体-受体、信号、TF-target 和基因调控先验资源。 |
| `data/ref/` | ICB reference mapping 产物和模型文件。 |
| `data/scRNA/` | scRNA 队列和 DEG 表，包括 scCRC ICB、IFNG、Neu 来源。 |
| `data/ST/` | 空间转录组 AnnData 输入，包括 `ST_CRC_MSS`。 |
| `data/Chromium_HumanColon_Oliveira/` | 10x Chromium human colon 参考数据。 |
| `data/Visium_HumanColon_Oliveira/` | Visium human colon 数据、图像和空间坐标。 |
| `data/VisiumHD_HumanColon_Oliveira/` | Visium HD binned 和 segmented 输出。 |
| `data/Xenium_HumanColon_Oliveira/` | Xenium cell、transcript、boundary 和 expression 资产。 |

## 6. 本地结果目录

`results/` 已被忽略，用于保存运行产物。它们是分析结果，不是提交代码。

| 路径 | 作用 |
|---|---|
| `results/integration/audit/` | 多源数据格式审计和数据就绪情况说明。 |
| `results/integration/schema/` | 生成的 `sample_table`、`entity_table`、`feature_table`、`measure_table`。 |
| `results/integration/msi_inference/` | MSI/MSS 推断输出。当前 `ST_CRC_MSS` 推断为 MSS，置信度 medium，marker 覆盖度 1.0。 |
| `results/integration/discovery/` | 旧版/预计算靶点发现输出，用于 notebook 和 README 展示：候选池、证据矩阵、几何比较、因果输出、Step3 汇总、生态位上下文、报告和图。 |
| `results/discovery/target_discovery/<run_id>/` | 模块化靶点发现 CLI 的默认输出路径。新运行会写入 manifest、迁移说明、候选池、表达/空间输入、几何、因果、扰动、评分、生态位、报告和图表。该路径在尚未执行新模块化 run 时可不存在。 |
| `results/examples/` | 数据概览、segmentation、panel summary 和 scCRC ICB 分步示例输出。 |
| `results/figures/` | 图表输出和 `.meta.json` 溯源文件。 |
| `results/step1/` | demo 嵌入输出：embedded AnnData、邻接矩阵、Lorentz/Poincare 嵌入、HVAE 模型、训练 loss。 |
| `results/step1_st_crc_mss_full_stable/` | ST CRC MSS 工作流的稳定 full Step1 输出。 |
| `results/step2/` | demo 因果输出：因果图、bootstrap frequency、解缠模型/loss、baseline 比较、信号流汇总。 |
| `results/step3/` | Step3 反事实输出：CF expression、interaction targets、propagation、metrics；当前默认靶点由 flow edges 或表达矩阵数据驱动解析。 |
| `results/step4/` | demo 动态干预输出：combination ranking、PK/PD summary、temporal causal arrays、roundtrip update report。 |

## 7. 当前结果摘要

当前本地结果说明项目已经超过框架搭建阶段，具备可运行 stage 产物、模块化靶点发现、notebook 展示图表和测试覆盖。

| 方向 | 当前证据 |
|---|---|
| 多平台生态位结构 | `results/integration/discovery/niche/niche_hierarchy_metrics.json` 显示 485,362 个 spots、18 个 selected niches，hyperbolic silhouette 0.710 vs Euclidean 0.417，hierarchy correlation 1.000 vs -0.569。 |
| 数据驱动靶点发现 | target discovery 候选池、扰动筛选、hub 保留和展示图均不注入人工候选锚点；排序由跨队列一致性、表达、因果、空间和 niche 证据共同决定。 |
| Top candidates | 当前候选排序以实际输入表和评分矩阵为准；README/notebook 展示不再把任何基因标记为预设 anchor。 |
| 几何比较 | integrated discovery mode comparison 显示 hyperbolic separation 2.073 vs Euclidean 2.065，在当前预计算 run 中双曲几何略优。 |
| MSI/MSS 推断 | `results/integration/msi_inference/ST_CRC_MSS_msi_inference.json` 推断为 MSS，confidence 为 medium，score -0.686，marker gene coverage 1.0。 |
| Step2 demo 因果图 | `results/step2/step2_metrics.json` 显示 14 个节点、8 条因果边、graph sparsity 0.044；相比 baseline communication graph 更稀疏（8 vs 37 edges）。 |
| Step3 demo 扰动 | Step3 支持 `expression_ko`、`hyperbolic_latent_ko` 与 `diffusion_cf`；`hyperbolic_latent_ko` 会加载 Step1 H-VAE artifacts 并从 Lorentz 潜空间解码反事实表达。 |
| Step4 demo 干预 | Step4 支持从 Step3 输出或 cluster expression 数据驱动解析 targets，并输出 PK/PD、时空传播、组合排序和 roundtrip 更新结果。 |

## 8. 历次更新结果

| 日期 | 更新 | 结果 |
|---|---|---|
| 2026-05-03 | 靶点发现管线模块化 | 将臃肿 CLI 拆为 `scripts/run_target_discovery.py` thin CLI，并将实现迁移到 `src/discovery/target_discovery/`；新增 stage 协议、artifact writer、geometry/scoring helpers、lightweight/heavy stages、reporting 和 discovery tests。 |
| 2026-05-03 | MSI 推断与 ICB reference 整合 | 新增 MSI inference utility 和 ICB reference integration tests；在合并远端 `main` 后完成安全发布。 |
| 2026-05-24 | Python 3.13 实验与依赖边界 | 建立 Python 3.13 `.venv` 实验环境并验证 Torch CUDA/PyG，同时主线仍保留 Python 3.10；将 `scgen` 从核心依赖移至 `requirements-optional-baselines.txt`，`validate_env.py` 将其作为 optional warning。 |
| 2026-05-24 | 本地目录梳理与清理 | 明确 Git、本地数据、运行结果、参考仓库和环境边界；新增 pytest 临时目录忽略规则；将根目录下 pytest 临时运行目录移动到 `.Trash/cleanup-20260524-local-run-artifacts/`。 |
| 2026-05-24 | v0.5 anchor-free 更新 | 移除默认候选 anchor 注入；Step3/Step4 改为数据驱动解析 targets；新增中文 Pipeline Flowchart v0.5；全量测试更新为 `138 passed`。 |

## 9. 当前项目进度和效果

- 主线运行环境仍定位为 Python 3.10；Python 3.13 `.venv` 是本地实验验证环境。
- Step1-Step4 已有可执行脚本、源码模块、本地 demo 输出和测试。
- 靶点发现已经模块化，新 run 默认输出到 `results/discovery/target_discovery/<run_id>/`；旧版/预计算 notebook 展示结果保留在 `results/integration/discovery/`。
- `scgen` 仅保留为历史 baseline 复现的可选依赖；仓库源码没有直接导入它。
- 本地数据和结果体积较大，已明确排除在 Git 外。新 clone 需要重新执行数据入库或挂载本地数据后才能复现当前结果快照。
- 当前 Python 3.13 实验环境下，全量测试通过，环境验证通过；`scgen` 缺失仅作为 optional warning。
