# Task C Mean-Difference Baseline v1

## 状态

该实现是 `hypersca-csd-benchmark-v1` 强制要求的简单干预基线，不是 HyperSCA 候选算法，也不参与自身 promotion。合成数据只能用于 smoke test；真实结论必须使用外部 Perturb-seq 数据、声明的参考网络和契约规定的跨 context 评估。

## 与 CausalBench 的接口关系

CausalBench 的公开模型接口接收 `expression_matrix`、逐细胞 `interventions`、`gene_names`、training regime 和 seed，并以 `non-targeting` 表示对照。官方 `CreateDataset` 缓存 NPZ 使用键：

- `expression_matrix`
- `interventions`
- `var_names`

本地实现直接兼容这三个数组和模型调用签名，不依赖、不 vendoring CausalBench。接口核对基于官方仓库 commit `1a2143cffdc85f835b41ce8d52034be1bf903e71`。数据下载和预处理仍应使用 CausalBench 官方流程，以保留其许可、过滤和标准化记录。

官方资料：

- CausalBench repository: <https://github.com/causalbench/causalbench>
- Published benchmark: <https://doi.org/10.1038/s42003-025-07764-y>

## 算法

对每个满足最小细胞数且存在于表达特征中的干预源基因 `s`，计算每个靶基因 `t`：

```text
effect(s, t) = mean(X_t | intervention=s) - mean(X_t | non-targeting)
score(s, t)  = abs(effect(s, t))
```

删除 `s → s` 自环；每个源内按 score 降序、target 名升序稳定排序。该方法只利用干预标签和训练输入表达；声明的参考边仅在评分阶段使用。它是刻意简单的效应量基线，不建立调控机制、传递路径或无混杂识别。

## 输出与可靠性边界

运行会写出 benchmark contract 要求的五个 artifact：

- `run_manifest.json`
- `input_summary.json`
- `metrics.json`
- `predictions.csv`
- `promotion_decision.json`

数据和参考边文件都以 SHA-256 绑定到 manifest。参考网络（例如 ChIP、STRING 或 CORUM 派生网络）是不完整评价证据，不是完整因果真值。单 context 的一次运行始终保留 `external_holdout_passed=false`；Task C promotion 需要 K562/RPE1 等外部 context 聚合、五个预注册 seed、负控和候选方法相对基线的配对区间。

## 使用

先按 CausalBench 官方流程生成 `dataset_k562.npz` 或 `dataset_rpe1.npz`，然后运行：

```bash
python scripts/run_task_c_mean_difference.py \
  --input-npz /path/to/dataset_k562.npz \
  --dataset-id weissmann_k562 \
  --dataset-source https://plus.figshare.com/ndownloader/files/35773219 \
  --context-id K562 \
  --data-status external_benchmark \
  --reference-edges /path/to/directed_reference.csv \
  --reference-id chipseq_k562_versioned \
  --output-dir results/benchmarks/task_c/k562/mean_difference/seed_11
```

参考边 CSV 默认需要 `source,target` 两列。未提供参考边时仍生成完整 score artifact，但 `metrics.status` 保持 `not_evaluated_no_reference`。
