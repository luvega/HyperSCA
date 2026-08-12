# HyperSCA 任务 C/S/D 预先固定的比较规则 v1

状态：`frozen`

## 这份文件解决什么问题

复杂方法未必超过简单方法，空间混杂也可能产生看似可信的关系。本文件在查看新结果前固定任务、数据划分、指标、参数尝试上限和通过条件，使 HyperSCA 与现有方法的比较可以复查，并减少根据结果临时改变标准的风险。

## 通俗解释

预先固定的比较规则（benchmark contract）要求候选方法和简单对照方法在相同数据、相同评价指标和相同计算机会下比较。任务 C 检查干预网络，任务 S 分开检查细胞自身效应与邻近细胞效应，任务 D 检查药物反应关联以及暴露和靶点结合证据。

## 不能据此得出什么结论

达到进入下一证据阶段的条件，只表示该任务在预先固定的对照评估中得到支持。任务 C、S、D 分别作出判断，一个任务的结果不能替代另外两个任务的证据。

即使达到进入下一证据阶段的条件，也不能据此声称临床疗效、已验证药物机制或普适算法优势。

## 机器读取名称

- 机器规则：`configs/benchmark_contract_v1.json`
- 契约标识和文件内容指纹字段：`contract_id`、`contract_sha256`
- 数据划分与版本字段：`split_id`、`code_revision`、`input_digest`
- 判断状态字段：`promotion_status`
- 每次比较要求的分析输出文件（artifacts）：`run_manifest.json`、`input_summary.json`、`metrics.json`、`predictions.csv`、`promotion_decision.json`

这些字段和文件名保持不变。修改机器规则会改变 SHA-256；已有运行不得在没有记录的情况下沿用新内容。

## 固定的三个任务

| 任务 | 独立研究对象 | 独立验证数据（holdout） | 主要指标 | 强制简单对照方法（baseline） |
|---|---|---|---|---|
| C | 干预源基因对靶基因的有向平均表达效应 | 生物学情境 + 干预靶点 | average precision | mean difference |
| S | own effect 与额外 neighbor effect，分开报告 | 外部切片 + 不相邻空间分区 | neighbor-effect RMSE | own-only；fixed distance decay |
| D | niche-dose-time 分层下药物与实测响应的关联，要求暴露与 target engagement 字段 | MOA/scaffold + 药物 + 生物学情境 | response Spearman correlation | non-spatial signature reversal |

任务 S 不得把自身效应与邻近效应合成一个分数，从而掩盖邻近效应失败。任务 D 的表达反转分数只是简单对照；缺少真实暴露或 `target engagement` 时不能进入下一证据阶段。

## 所有方法共用的比较条件

- 固定随机种子：`11, 23, 47, 71, 97`。
- 每个方法最多尝试 20 组参数；所有方法使用相同目标、训练来源特征和停止规则。
- 禁止使用测试数据或独立验证数据选择特征、先验或参数。
- 采用 95% 配对分组重复抽样区间报告不确定性。
- 每项任务必须运行规则中列出的零效应对照，并报告可作出判断的比例（coverage）、暂不判断的比例（abstention）、失败的随机种子、计算成本和失败案例。
- 每次运行必须绑定规则的 SHA-256、数据划分标识、代码版本、输入文件内容指纹和预先登记的随机种子。

## 进入下一证据阶段的条件（promotion gate）

任务只有同时满足下列条件，机器状态才返回 `promoted`：

1. `coverage ≥ 0.80`，`abstention rate ≤ 0.20`；
2. 至少 80% 的预先登记随机种子成功；
3. 外部独立验证和任务零效应对照均通过；
4. 候选方法与简单对照使用相同的数据划分、特征集合和参数尝试上限；
5. 对每个强制简单对照，方向统一后的配对改进 95% 置信区间下界严格大于 0。

任一条件失败，状态即为 `not_promoted / hypothesis_only`。不得删除失败的随机种子后重新计算，也不得用次要指标掩盖主要指标失败。

## 输出和登记要求

每次真实对照评估必须产生以下文件：

- `run_manifest.json`：记录规则、数据划分、方法、代码、随机种子和输入内容指纹；
- `input_summary.json`：记录样本和特征数量、分层统计、缺失率及信息泄漏检查；
- `metrics.json`：记录逐随机种子指标、配对差值、置信区间、可判断/暂不判断比例和成本；
- `predictions.csv`：记录逐分析单元的预测、真值、置信度和暂不判断标志；
- `promotion_decision.json`：记录每项条件的结果和允许使用的结论等级。

规则快照不是方法比较结果。初始 `task_registry.json` 必须保持 `not_evaluated`，直到真实外部数据和强制简单对照全部完成。

## 使用

```bash
python scripts/validate_benchmark_contract.py
python scripts/validate_benchmark_contract.py \
  --output-dir results/benchmarks/preregistration_v1
```

实质性修改应创建新的 `contract_id` 和版本，不应覆盖 v1。

## Bear 报告对应关系

- E9 PerturBench：统一数据、划分和指标后再比较方法。
- E16：PCA 等简单表示仍可能具有竞争力。
- E17/E18：药物类别内泛化和跨环境可迁移性需要外部独立验证。
- E19：空间坐标和邻域零效应对照用于检查空间混杂。
- E20：可判断比例、校准和暂不判断进入可靠性终点。
- E22 CausalBench：任务 C 必须证明模型确实利用干预信息，并与简单干预方法比较。

完整证据与限定语见 `reports/research/bear_hypersca_spatial_causal_20260810/report.md`。
