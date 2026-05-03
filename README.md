<p align="center">
  <img src="docs/Logo_high%20res.png" alt="HyperSCA Logo" width="280" />
</p>

HyperSCA (Hyperbolic Spatiotemporal Causal Analysis) 是一个面向空间组学与单细胞组学联合分析的多组学计算框架。该框架集成双曲几何嵌入、因果图发现和反事实扰动分析，支持 scRNA-seq、空间转录组及临床/表型分层数据的联合建模，用于机制推断与可干预靶点评估。除肿瘤免疫场景外，也可用于自身免疫、慢性炎症、感染及组织损伤修复等疾病环境。

## Project and Algorithm Overview

HyperSCA 的研究完整版流程由五个连续阶段构成，可按具体队列与研究问题灵活裁剪：

- Phase D0（Data Onboarding）：四项目标准化入库与字段校验。
- Stage 1（Embedding）：在 Lorentz/Poincare 双曲流形上学习细胞状态表示。
- Stage 2（Causal）：在去缠结潜变量上执行因果结构发现与信号流推断。
- Stage 3（Counterfactual）：在潜空间做基因扰动并模拟空间传播，完成靶点排序与去假阳性过滤。
- Stage 4（Dynamic Intervention）：在 PK/PD 约束下执行时序传播与联靶组合干预评估，并支持实验回写后的 roundtrip 更新。

## Pipeline Flowchart

![HyperSCA Analysis Framework](docs/HyperSCA分析框架示意图.jpg)

## Core Algorithms

### 1) Hyperbolic Embedding

- 关键模块：`src/models/hyperbolic/lorentz.py`, `src/models/hyperbolic/poincare.py`, `src/models/hyperbolic/wrapped_normal.py`, `src/models/hyperbolic/hvae.py`
- 目的：在双曲空间中更好地保持细胞层级结构与远近关系，降低欧氏空间下的几何失真。

### 2) Causal Discovery and Signaling Flow

- 关键模块：`src/causal/disentangle.py`, `src/causal/cmi_pruning.py`, `src/causal/causal_graph.py`, `src/causal/signaling_flow.py`
- 方法要点：`z_int/z_ext` 去缠结 + PC 条件独立检验 + bootstrap 稳定性 + DoWhy 结构验证 + L-R-TF-Target 多层流。

### 3) Counterfactual Perturbation and Spatial Propagation

- 关键模块：`src/perturbation/latent_arithmetic.py`, `src/perturbation/spatial_propagation.py`, `src/perturbation/diffusion_cf.py`, `src/perturbation/target_ranking.py`
- 方法要点：潜空间虚拟敲除、因果图约束扩散、空间梯度衰减拟合、靶点可干预性排序。

### 4) Niche and Cross-sample Stratification

- 关键模块：`src/evaluation/cross_sample_metrics.py`
- 方法要点：生态位聚类（niche clustering）、跨样本边一致性、临床/表型分层差异检验，纳入最终证据矩阵。

### 5) Modular Target Discovery

- 入口脚本：`scripts/run_target_discovery.py` 现在是 thin CLI，只负责解析参数、构造 `TargetDiscoveryConfig` 并调用 pipeline。
- 核心包：`src/discovery/target_discovery/`
  - `config.py`、`pipeline.py`、`stage.py`、`artifacts.py` 定义配置、编排、stage 协议与 run manifest。
  - `loaders.py`、`candidates.py`、`expression.py`、`spatial.py` 构建轻量数据输入。
  - `geometry.py`、`causal_stage.py`、`perturbation_stage.py`、`scoring.py`、`niche.py`、`reporting.py`、`figures.py` 负责双几何比较、Step2/Step3 wrapper、证据排序、生态位映射、报告和图。
- 输出根目录：默认写入 `results/discovery/target_discovery/<run_id>/`，按 `candidates/`、`expression/`、`spatial/`、`geometry/{mode}/`、`causal/{mode}/`、`perturbation/{mode}/`、`scoring/`、`niche/`、`reports/`、`figures/` 分区，并生成 `manifest.json` 与 `reports/migration_notes.md`。

## Example Data Samples

项目常用示例输入（路径为本地外部数据目录，不纳入版本控制；以下为脱敏占位路径）：

- `<PATH_TO_scRNA_REFERENCE>`  
  - 代表文件：`*-NormalizedCounts.tsv`, `*-DE_result.tsv`
  - 用途：构建 cluster-level 表达矩阵、候选差异基因池与细胞状态先验。
- `<PATH_TO_SPATIAL_OMICS>`  
  - 代表文件：`STmetadata_*.csv`, `spot_annotations.*`
  - 用途：空间反卷积、细胞共定位邻接、传播梯度与生态位结构评估。
- `<PATH_TO_CLINICAL_OR_PHENOTYPE>`  
  - 代表文件：`sample_clinical_mapping.csv`, `group_labels.csv`
  - 用途：临床/表型分层（如免疫亚型、疾病分期、治疗反应）及跨样本差异分析。

统一标准化输出（示例）：

- `results/integration/schema/sample_table.csv`
- `results/integration/schema/entity_table.csv`
- `results/integration/schema/feature_table.csv`
- `results/integration/schema/measure_table.csv`

## Installation Guide

### 1) Create Conda Environment

```bash
conda create -n hypersca python=3.10 -y
conda activate hypersca
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

CUDA 12.4 推荐安装：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 3) Validate Runtime Environment

```bash
python scripts/validate_env.py
```

## Quick Start

### Multi-omics Integration Example (推荐)

展示 HyperSCA **多组学整合分析**核心能力的完整示例（6 个 notebook，含嵌入图表）：

- `notebooks/example_multiomics_integration/README.md`
- `00_data_landscape` → `01_hyperbolic_vs_euclidean` → `02_multiscale_niche` → `03_causal_network` → `04_target_discovery` → `05_summary`

核心对比结果：

| 指标 | scRNA-only + Euclidean | Multi-omics + Hyperbolic | 提升 |
|------|----------------------|-------------------------|------|
| Niche Silhouette | 0.417 | **0.710** | **+70%** |
| Hierarchy Correlation | −0.569 | **+1.000** | 反转→完美 |
| 证据维度 | 3 | **5** (+spatial, +niche) | +2 独立维度 |

数据规模：485K spots × 3 空间平台 + 3 scRNA-seq 队列，**靶点发现完全数据驱动**（无预设 anchor）。

### Step-by-step scCRC_ICB (单细胞基础流程)

如需仅基于 scRNA-seq 数据按主流程逐步运行：

- `notebooks/example_sccrc_icb_step_by_step/README.md`
- `notebooks/example_sccrc_icb_step_by_step/00_environment_and_data_check.ipynb` 到 `05_step4_dynamic_intervention_and_summary.ipynb`

### A. Build Canonical Schema

```bash
python scripts/build_canonical_schema.py
```

### A0. Onboard Multi-cohort Data to `/data`

说明：脚本参数名保留历史命名（`icb/neu/st/ifng`），但可映射到任意疾病场景的数据根目录。

```bash
python scripts/run_data_onboarding.py \
  --icb-root <PATH_TO_COHORT_A> \
  --neu-root <PATH_TO_COHORT_B> \
  --st-root <PATH_TO_SPATIAL_OMICS> \
  --ifng-root <PATH_TO_COHORT_D>
```

### B. Run Step1 (Hyperbolic Embedding)

```bash
python scripts/run_step1.py \
  --data-dir data/ST/<YOUR_SPATIAL_PROJECT> \
  --modality visium \
  --output-dir results/step1
```

### C. Run Step2 (Spatial Causal Inference)

```bash
python scripts/run_step2.py \
  --input-dir results/step1 \
  --output-dir results/step2
```

### D. Run Step3 (Counterfactual Perturbation)

```bash
python scripts/run_step3.py \
  --input-step1 results/step1 \
  --input-step2 results/step2 \
  --output-dir results/step3
```

### E. Run Target Discovery and Hub Retention

```bash
python scripts/run_target_discovery.py \
  --run-id demo_target_discovery \
  --max-perturb 10 \
  --geometry-k 4 \
  --geometry-blend 0.30 \
  --platform all \
  --skip-figures
```

默认输出位于 `results/discovery/target_discovery/<run_id>/`。旧版展示口径中的预计算发现结果仍保留在 `results/integration/discovery/`，用于 notebook 和 README 中的历史图表展示。

### F. Run Dynamic Intervention (Step4) and Roundtrip Update

```bash
python scripts/run_step4.py --with-roundtrip \
  --experiment-file data/metadata/experiment_roundtrip.csv
```

### G. Generate CNS-style Figures (Step1/2/3)

```bash
python scripts/generate_step1_figures.py
python scripts/generate_step2_figures.py
python scripts/generate_step3_figures.py
```

## Key Outputs

- Canonical schema and metadata: `data/metadata/`, `results/integration/schema/`
- Step1 outputs: `results/step1/` (`adata_embedded.h5ad`, `embedding_benchmark.json`)
- Step2 outputs: `results/step2/`（因果图、稳定性指标、baseline 对比）
- Step3 outputs: `results/step3/`（扰动结果、去假阳性靶点与组合）
- Target discovery runs: `results/discovery/target_discovery/<run_id>/`（run manifest、候选池、几何比较、Step2/Step3 wrapper 产物、评分表、生态位映射、报告、迁移说明）
- Legacy/precomputed discovery reports for notebooks: `results/integration/discovery/`
- Step4 outputs: `results/step4/`（`pkpd_summary.json`, `combination_ranking.csv`, `roundtrip_update_report.json`）
- CNS figure outputs: `results/figures/step1/`, `results/figures/step2/`, `results/figures/step3/`

## Testing

```bash
pytest tests/ -v
pytest tests/discovery -q
```

## License

MIT License.
