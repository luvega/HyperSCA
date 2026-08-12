# Task S Simple Spatial Baselines v1

## 目的

Task S 首先检验空间传播是否比同一个非空间 own-effect 输入提供更多 neighbor-effect 信息。为避免上游模型质量混入空间增益，`own_only` 和 `fixed_distance_decay` 必须读取完全相同的 `own_effect_prediction`，使用相同 holdout 行和相同拒答掩码。

这两个方法是 `hypersca-csd-benchmark-v1` 的强制简单基线，不是空间因果机制模型，也不能自行 promotion。

## Canonical holdout 表

输入 CSV 一行对应一个 `unit × perturbation × feature` 终点，必须包含：

| 字段 | 含义 |
|---|---|
| `unit_id` | 细胞或 spot ID |
| `sample_id` | 外部 slice/sample ID |
| `spatial_block` | 不与训练块相邻的 holdout block |
| `perturbation_id` | 干预 ID |
| `feature_id` | 基因或预注册 gene-program ID |
| `distance` | 到被扰动源的非负距离；own unit 必须为 0 |
| `is_perturbed` | own endpoint 为 true，neighbor endpoint 为 false |
| `own_effect_prediction` | 只由训练数据产生的非空间 own-effect 预测 |
| `observed_effect` | 空间干预 holdout 中的实测效应 |

`own_effect_prediction` 缺失时两个基线对同一行同时拒答，因此 coverage 和 abstention 可直接比较。实测 holdout 结果不得用于生成 own effect 或确定长度尺度。

## 两个固定基线

`own_only`：

```text
prediction = own_effect_prediction   if is_perturbed
prediction = 0                       otherwise
```

`fixed_distance_decay`：

```text
prediction = own_effect_prediction * exp(-distance / length_scale)
```

`length_scale` 必须来自训练 slice/graph 的版本化 artifact；运行时同时记录其 source ID 与 SHA-256。禁止在 holdout outcome 上拟合长度尺度。

## 终点与声明边界

主终点是 neighbor-effect RMSE。own-effect RMSE、neighbor PCC、distance-binned calibration error、coverage 和 abstention 分开报告。own 与 neighbor 不合并为一个平均分，防止大量 own unit 掩盖空间传播失败。

即使固定距离衰减表现良好，也只能说明一个简单空间规律能解释部分 holdout 效应；不能据此识别 ligand-receptor 路径、药物机制或因果传播。候选方法必须在相同输入、切分和预算下同时超过 `own_only` 与 `fixed_distance_decay`，并通过坐标/邻域负控后才可能 promotion。

## 使用

```bash
python scripts/run_task_s_baseline.py \
  --input-csv /path/to/external_holdout.csv \
  --baseline-id fixed_distance_decay \
  --dataset-id spatial_perturb_seq_v1 \
  --dataset-source https://example.org/versioned-record \
  --data-status external_benchmark \
  --own-effect-source-id task_c_model_seed_11 \
  --own-effect-source /path/to/own_effect_manifest.json \
  --length-scale 42.0 \
  --length-scale-source-id training_slice_nn_distance_v1 \
  --length-scale-source /path/to/training_length_scale.json \
  --attest-own-effect-train-only \
  --attest-nonadjacent-spatial-blocks \
  --output-dir results/benchmarks/task_s/fixed_distance_decay/seed_11
```

外部 benchmark 若缺少两项 leakage attestation 会 fail closed。合成 smoke 可以执行，但 artifact 始终带 `synthetic_smoke=true`，不能进入科学 promotion。
