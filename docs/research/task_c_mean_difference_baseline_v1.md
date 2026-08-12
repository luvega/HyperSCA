# 任务 C 均值差简单对照方法（baseline）v1

## 这份文件解决什么问题

任务 C 需要判断复杂因果网络方法是否真正利用了干预信息。均值差方法直接比较某个基因受到干预时与未定向干预对照时的表达差异，为复杂方法提供一个刻意简单但使用干预标签的比较下限。

## 通俗解释

对每个干预源基因，该方法计算其他基因在干预组和对照组之间的平均表达差，并按绝对差值排序。参考网络只在结果评分时使用，不参与预测，因此可以检查复杂方法是否超过直接利用干预标签的做法。

## 不能据此得出什么结论

均值差方法是规则强制要求的比较参照，不是 HyperSCA 候选算法，也不对它本身作证据升级判断。它不建立调控机制、信号传递路径或无混杂因果识别。参考网络是不完整的评价证据，不是完整因果真值。合成数据上的最小运行检查（smoke test）只用于检查程序能否执行。

最小运行检查只证明分析流程可以运行，不代表方法已在真实数据中得到验证。

## 机器读取名称

- 预先固定的比较规则标识：`hypersca-csd-benchmark-v1`
- 输入数组：`expression_matrix`、`interventions`、`var_names`
- 对照标签：`non-targeting`
- 五个可复查的分析输出文件（artifacts）：`run_manifest.json`、`input_summary.json`、`metrics.json`、`predictions.csv`、`promotion_decision.json`
- 单次情境运行标记：`external_holdout_passed=false`
- 无参考网络时的状态：`metrics.status=not_evaluated_no_reference`

这些数组键、标签、字段和值保持不变。数据和参考边文件的 SHA-256 写入分析记录清单（manifest）。

## 与 CausalBench 的接口关系

CausalBench 的公开模型接口接收表达矩阵、逐细胞干预标签、基因名、训练方式和随机种子，并用 `non-targeting` 表示对照。官方 `CreateDataset` 缓存的 NPZ 使用前述三个数组键。

本地实现直接兼容这些数组和模型调用形式，不复制 CausalBench 代码。接口核对基于官方仓库 commit `1a2143cffdc85f835b41ce8d52034be1bf903e71`。数据下载和预处理仍应使用 CausalBench 官方流程，以保留许可、过滤和标准化记录。

官方资料：

- CausalBench repository: <https://github.com/causalbench/causalbench>
- Published benchmark: <https://doi.org/10.1038/s42003-025-07764-y>

## 计算方法

对每个满足最小细胞数且存在于表达特征中的干预源基因 `s`，计算每个靶基因 `t`：

```text
effect(s, t) = mean(X_t | intervention=s) - mean(X_t | non-targeting)
score(s, t)  = abs(effect(s, t))
```

删除 `s → s` 自环；每个来源内部按分数降序、靶基因名称升序稳定排序。该方法只使用干预标签和训练输入表达。

## 输出与可靠性边界

单一生物学情境的一次运行始终保留 `external_holdout_passed=false`。真实结论必须使用外部 Perturb-seq 数据、已声明的参考网络和跨情境评价。任务 C 若要进入下一证据阶段，需要汇总 K562、RPE1 等外部情境，完成五个预先登记的随机种子和零效应对照，并报告候选方法相对该简单对照的配对区间。

参考网络可来自 ChIP、STRING 或 CORUM 等版本化来源。它们可能漏掉真实关系，也可能包含不适用于当前细胞情境的关系，因此只能作为评价依据之一。

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

参考边 CSV 默认需要 `source,target` 两列。未提供参考边时仍生成完整分数输出，但结果保持前述未评价状态。
