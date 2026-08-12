# 任务 S 空间简单对照方法（baselines）v1

## 这份文件解决什么问题

任务 S 要判断空间信息是否在同一份非空间自身效应预测之外，提供了额外的邻近细胞效应信息。两个方法必须读取相同的自身效应输入，才能把空间增益与上游模型质量分开。

## 通俗解释

第一个方法只预测被干预细胞自身的效应（own effect），把邻近细胞效应（neighbor effect）设为 0。第二个方法使用相同的自身效应，再按距离作固定指数衰减。两者在未参与模型建立的独立验证数据（holdout）上使用相同记录和相同暂不判断范围。

## 不能据此得出什么结论

这两个方法是规则强制要求的比较参照，不是空间因果机制模型，也不能自行进入下一证据阶段。固定距离衰减表现良好，只能说明一个简单的空间规律可解释部分独立验证效应。它不能识别配体—受体通路、药物机制或因果传播；较低误差也不等于已经确定生物学机制。

## 机器读取名称

- 两个方法：`own_only`、`fixed_distance_decay`
- 共用输入：`own_effect_prediction`
- 主要终点：`neighbor-effect RMSE`
- 其他终点：`own-effect RMSE`、`neighbor PCC`、`distance-binned calibration error`
- 运行判断：`synthetic_smoke=true`

代码名、字段和指标名保持不变。

## 独立验证数据表

输入 CSV 一行对应一个 `unit × perturbation × feature` 终点，必须包含：

| 字段 | 含义 |
|---|---|
| `unit_id` | 细胞或 spot ID |
| `sample_id` | 外部切片或样本 ID |
| `spatial_block` | 不与训练空间分区相邻的验证分区 |
| `perturbation_id` | 干预 ID |
| `feature_id` | 基因或预先登记的 gene-program ID |
| `distance` | 到被扰动来源的非负距离；被干预单元必须为 0 |
| `is_perturbed` | 自身终点为 true，邻近终点为 false |
| `own_effect_prediction` | 只用训练数据产生的非空间自身效应预测 |
| `observed_effect` | 空间干预独立验证数据中的实测效应 |

自身效应预测缺失时，两个方法对同一行同时暂不判断，因此可作出判断的比例（coverage）和暂不判断的比例（abstention）可以直接比较。实测验证结果不得用于生成自身效应或确定长度尺度。

## 两个固定方法

`own_only`：

```text
prediction = own_effect_prediction   if is_perturbed
prediction = 0                       otherwise
```

`fixed_distance_decay`：

```text
prediction = own_effect_prediction * exp(-distance / length_scale)
```

`length_scale` 必须来自训练切片或训练图的版本化分析输出文件（artifact）。运行时同时记录来源 ID 和 SHA-256，禁止在验证结果上拟合该值。

## 终点与比较边界

主要终点是邻近效应 RMSE。自身效应 RMSE、邻近效应 PCC、按距离分组的校准误差、可判断比例和暂不判断比例分别报告。自身与邻近终点不能合成一个平均分，以免大量自身单元掩盖空间传播失败。

候选方法必须在相同输入、数据划分和参数尝试上限下，同时超过 `own_only` 和 `fixed_distance_decay`，并通过坐标与邻域零效应对照，之后才可能进入下一证据阶段。

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

外部对照评估若缺少两项防止信息泄漏的确认，会在条件不全时停止并判为未通过（fail closed）。合成数据可以执行最小运行检查，但其输出始终带有 `synthetic_smoke=true`，不能进入科学证据升级。
