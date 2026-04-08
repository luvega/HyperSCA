# scCRC_ICB Step-by-step Notebooks

本目录提供 HyperSCA 主流程的分步 Notebook 示例，面向 `CRC_ICB` 示例数据。

## 执行顺序

按文件名前缀顺序执行：

1. `00_environment_and_data_check.ipynb`
2. `01_step1_hyperbolic_embedding.ipynb`
3. `02_step2_causal_network.ipynb`
4. `03_data_driven_target_selection.ipynb`
5. `04_step3_counterfactual_from_data_driven_targets.ipynb`
6. `05_step4_dynamic_intervention_and_summary.ipynb`


## 数据与输出路径

- 输入数据：`data/scRNA/scCRC_ICB/`
- 主要输出：`results/examples/sccrc_icb/step1~step4/`
- 可视化输出：`results/figures/examples/sccrc_icb/step3/`

## 预期运行时长（参考）

实际耗时与 CPU/GPU、内存、磁盘 IO 有关，以下为常见范围：

- `00`：< 1 分钟（环境与路径检查）
- `01`：5-30 分钟（取决于 epochs、设备与样本规模）
- `02`：10-60 分钟（Bootstrap 次数影响最大）
- `03`：< 5 分钟（DEG 聚合与打分）
- `04`：10-40 分钟（反事实推演，目标数越多越慢）
- `05`：5-20 分钟（动态干预与汇总可视化）

## 常见报错与处理建议

### 1) 缺少依赖包（例如 `ModuleNotFoundError: No module named 'scipy'`）

- 安装依赖：`pip install -r requirements.txt`
- 或补装单包：`pip install scipy`

### 2) 数据文件缺失（`FileNotFoundError`）

- 先运行 `00` 检查数据路径是否存在
- 确认 `data/scRNA/scCRC_ICB/expression.h5ad` 与 `deg_tables/*.csv` 已准备
- 如未入库，可在 `00` 中将 `RUN_ONBOARDING=True` 后再执行

### 3) 显存/内存不足（OOM）

- 在 `01/02/04` 中优先使用 `--device cpu`
- 下调参数：`epochs`、`bootstrap-n`、`target_top_n`
- 必要时先做小规模验证，再放大配置

### 4) 输出为空或候选数过少

- 检查 `03` 的 DEG 表是否包含有效列：`gene`, `avg_log2FC`, `p_val_adj`
- 放宽筛选后再观察稳定性（不要直接预设固定锚点）

## 复现建议

- 固定随机种子并记录运行参数
- 每步执行后保留 `config` 与 `metrics` 文件
- 建议将每次实验输出写入独立目录，避免覆盖
