# HyperSCA Tasks C/S/D Benchmark Contract v1

状态：`frozen`  
机器可读契约：`configs/benchmark_contract_v1.json`

## 目的与声明边界

本契约在运行新模型前固定比较条件，响应 Bear 研究报告中最直接的风险：复杂表示未必超过 mean/PCA 等简单基线，空间混杂可制造伪因果，跨环境效果可能不可迁移，未校准不确定性会夸大可用范围。当前相对可辩护的贡献是统一、可撤回的评价框架，而不是仅凭模型复杂度主张算法优越。

即使通过 promotion，也只能写作“在预注册 benchmark 上获得支持的候选方法”。不得据此声称临床疗效、已验证药物机制、普适因果识别或 SOTA。Task C、S、D 分别晋级；一个任务的成功不能替代另外两个任务的证据。

## 冻结的任务

| 任务 | 独立 estimand | 外部切分 | 主指标 | 强制简单基线 |
|---|---|---|---|---|
| C | 干预源基因对靶基因的有向平均表达效应 | biological context + perturbation target holdout | average precision | mean difference |
| S | own effect 与额外 neighbor effect，分开报告 | external slice + non-adjacent spatial block holdout | neighbor-effect RMSE | own-only；fixed distance decay |
| D | niche-dose-time 分层下药物与实测响应的关联，要求暴露与 target engagement 字段 | MOA/scaffold + drug + biological context holdout | response Spearman correlation | non-spatial signature reversal |

Task S 的 own 与 neighbor 终点不得合成后掩盖 neighbor 失败。Task D 的表达反转分数只是基线；缺少真实暴露或 target engagement 时不得 promotion。

## 共同比较条件

- 固定随机种子：`11, 23, 47, 71, 97`。
- 每个方法最多 20 次调参；所有方法使用同一目标、训练来源特征和停止规则。
- 禁止使用 test/holdout 结果选择特征、先验或超参数。
- 95% 配对 cluster bootstrap 区间作为不确定性报告。
- 每项任务必须运行契约所列负控，并报告 coverage、abstention、失败 seed、计算成本和失败案例。
- 每个 run 必须绑定 contract SHA-256、split ID、代码 revision、输入 artifact digest 和预注册 seed。

## Promotion gate

任务只有同时满足下列条件才返回 `promoted`：

1. coverage ≥ 0.80，abstention rate ≤ 0.20；
2. 至少 80% 的预注册 seed 成功；
3. 外部 holdout 和任务负控均通过；
4. 候选方法与基线使用同一 split、feature set 和 tuning budget；
5. 对每个强制简单基线，方向统一后的配对改进 95% CI 下界严格大于 0。

任一条件失败即 `not_promoted / hypothesis_only`。不得删除失败 seed 后重算，也不得用次指标覆盖主指标失败。

## Artifact contract

每个实际 benchmark run 必须产出：

- `run_manifest.json`：契约、split、方法、代码、seed 与输入 digest；
- `input_summary.json`：样本/特征数量、分层统计、缺失率和泄漏检查；
- `metrics.json`：逐 seed 指标、配对差值、置信区间、coverage/abstention 与成本；
- `predictions.csv`：逐分析单元的预测、真值、置信度和拒答标志；
- `promotion_decision.json`：逐条 gate 结果和允许的 claim level。

契约快照不等于 benchmark 结果。生成的初始 `task_registry.json` 必须保持 `not_evaluated`，直到真实外部数据与强制基线完成。

## 使用

```bash
python scripts/validate_benchmark_contract.py
python scripts/validate_benchmark_contract.py \
  --output-dir results/benchmarks/preregistration_v1
```

修改机器契约会改变 SHA-256，因此既有 run 不能静默沿用。实质性修改应创建新 contract ID 和版本，而不是覆盖 v1。

## Bear 报告映射

- E9 PerturBench：统一数据、切分与指标后再比较方法。
- E16：PCA 等简单表示仍可能具有竞争力。
- E17/E18：药物类别内泛化和跨环境 transportability 需要外部 holdout。
- E19：空间坐标和邻域负控用于审计空间混杂。
- E20：coverage、calibration 与 abstention 进入可靠性终点。
- E22 CausalBench：Task C 必须证明模型真正利用干预信息，并与简单干预基线比较。

完整证据与限定语见 `reports/research/bear_hypersca_spatial_causal_20260810/report.md`。
