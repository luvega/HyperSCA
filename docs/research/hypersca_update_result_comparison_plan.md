# HyperSCA 更新结果对比与图表展示计划

## 对比边界

本计划把“之前版本”定义为当前分支 `HEAD`，把“当前更新”定义为工作区中新增的短期可信度增强实现。当前对比先做 **结果契约与能力面比较**，不把 sidecar 计算结果写成新的生物学结论，也不替代后续真实 run 的数值复核。

## 本轮可直接展示的结果

- 表 1：更新前后输出契约、证据、风险和下一步验证矩阵：`docs/research/hypersca_update_comparison_matrix.csv`
- 表 2：新增模块、stage、artifact family 和测试覆盖计数：`docs/research/hypersca_update_output_coverage.csv`
- 图 1：能力/就绪度对比：`docs/research/figures/hypersca_update_capability_scores.png`
- 图 2：输出与测试覆盖对比：`docs/research/figures/hypersca_update_output_coverage.png`

## 图表设计

1. **能力分组条形图**：横向 grouped bar 展示 `previous_HEAD` 与 `current_update`。分数不是生物学性能分，而是基于输出契约、审计可见性、测试护栏和解释边界的 contract-level readiness score。
2. **输出覆盖条形图**：展示新增 sidecar module、CLI、默认 pipeline stage、artifact family 和测试项数量。该图适合放在汇报中解释“更新产生了哪些可检查结果”。
3. **主对照表**：按四个实施目标加两个工程护栏组织，每行同时写出收益、风险和下一步真实验证，避免只展示“新增了文件”。

## 主对照表

| comparison_axis          | previous_version                                                                                   | current_update                                                                                                                                                                      | remaining_risk                                                                  | next_validation                                                                                           |
|:-------------------------|:---------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| 因果图稳定性与负控审计              | 仅有 bootstrap_freq、graph sparsity、DoWhy proxy skip 与基础 signaling flow 指标，缺独立审计产物。                   | 新增 append-only stability audit sidecar，声明 edge_stability.csv、negative_control_report.md、platform_consistency.json、causal_audit_summary.json。                                        | 真实 platform/patient/niche 分层仍依赖可选 group-specific 输入；默认 CLI 不重跑完整多 seed Step2。   | 用合成链式图、独立随机矩阵、表达置换和双 group 一致性完成 smoke run；再接真实 Step2 run。                                                |
| LR→TF→target 机制证据分解      | flow_edges/flow_summary 可展示路径，scoring 主要保留 s_actionability 与 flow hit，缺逐证据列拆分。                     | 新增 mechanism_evidence sidecar，声明 mechanism_evidence_matrix.csv、mechanism_summary.json、mechanism_evidence_report.md、target_ranking_with_mechanism.csv。                               | s_mechanism 仅作解释列，不并入 final_score；TF-target prior 仍以内置 curated chain 为主。        | 用 CAF→TAM POSTN→ITGAV→SRC→CD163 合成链验证链路重建、低表达惩罚和 relaxed mode 标记。                                         |
| 空间通信 OT/flow 对照          | spatial_adjacency、geometry distance 和 baseline communication 分散存在，缺 directed LR communication 对照层。 | 新增 distance-constrained LR communication sidecar，声明 communication/{mode}/lr_flow_edges.csv、flow_matrix.npy、pathway_summary.csv、direction_consistency.json、baseline_comparison.json。 | 当前不是完整 COMMOT；forward/reverse/ambiguous/unresolved 只能作为方向一致性证据，不能写成因果证明。        | 用 forward/reverse/bidirectional/no-edge 四类合成状态测试方向分层，并在真实 run 上检查缺 LR gene 与空 ST metadata 的 warning/skip。 |
| target discovery 默认流程可见性 | 默认 stage 以 candidate→expression→spatial→geometry→causal→perturbation→scoring→niche→report 为主。      | 默认 stage 在 causal 后追加 communication_flow，在 scoring 后追加 mechanism_evidence，保持 append-only，不改原排序列。                                                                                    | 默认流程仍假设 Euclidean/Hyperbolic mode 的上游 artifact contract 稳定；真实 run 需检查 manifest。 | 跑一个 --skip-figures 的小 run，比较 manifest 新旧 stage/artifact 列表，不比较生物学结论。                                      |
| 文档状态一致性                  | technical_roadmap 顶部已更新，但后文仍有“当前仓库状态”与“待实现/Skeleton”混写，容易和 README/project_inventory 冲突。            | 将相关段落改为“历史设计快照/早期实现计划”，并声明当前状态以文档顶部、README 和 project_inventory 为准。                                                                                                                  | 本轮没有清理 README 与 project_inventory 的所有历史展示口径；只修 technical_roadmap。               | 汇报前执行 rg '当前仓库状态|待实现|Skeleton|规划中'，保留命中必须明确属于历史设计语境。                                                      |
| 测试与可复现性护栏                | 无针对三类可信度 sidecar 的专项测试；已有 Step2/discovery 测试不能覆盖新增输出契约。                                            | 新增 8 个专项测试入口；相关回归 tests/discovery + causal/Step2 测试 52 passed；compileall 通过。                                                                                                        | 本地计划指定的 hypersca conda Python 路径不存在，本轮用系统 Python 3.13 验证；需在用户机器 conda 环境复跑。     | 恢复/确认 E:\ProgramData\Anaconda3\envs\hypersca\python.exe 后按计划命令复跑。                                         |

## 覆盖计数表

| metric                           |   previous_HEAD |   current_update | note                                                     |
|:---------------------------------|----------------:|-----------------:|:---------------------------------------------------------|
| Sidecar modules                  |               0 |                3 | stability_audit, mechanism_evidence, communication_flow  |
| CLI entrypoints                  |               0 |                1 | run_causal_stability_audit.py                            |
| Default pipeline sidecar stages  |               0 |                2 | communication_flow + mechanism_evidence                  |
| New output artifact families     |               0 |               13 | 4 causal audit + 4 mechanism + 5 communication artifacts |
| Targeted tests collected         |               0 |                8 | new sidecar and stage-order tests                        |
| Relevant regression tests passed |               0 |               52 | tests/discovery + causal audit + causal metrics + Step2  |

## 后续验证计划

### 阶段 A：已完成的静态/契约对比
- 对比 `HEAD` 与当前工作区是否存在新增 sidecar 模块。
- 检查默认 target discovery stage 是否 append-only 插入通信 flow 与机制证据分解。
- 用新增测试收集和相关回归命令证明输出契约有基础保护。

### 阶段 B：轻量 smoke run
- 因果审计：合成链式图、随机独立图、表达置换和双 group 一致性。
- 机制证据：CAF→TAM POSTN→ITGAV→SRC→CD163 合成链。
- 通信 flow：forward/reverse/bidirectional/no-edge 四类方向状态。

### 阶段 C：真实 run 结果对比
- 选择一个小型 `target_discovery` run id，分别在 `HEAD` 和当前更新上输出 manifest/artifact 列表。
- 只比较新增 sidecar 的 presence、row count、warning/skip、direction consistency 分层和 mechanism evidence summary。
- 不比较或宣称 wet-lab 治疗有效性；所有 target/flow/因果输出只作为机制假设和实验优先级。

### 阶段 D：论文/汇报图表包
- 一张 workflow schematic：显示三个 append-only sidecar 的插入点。
- 一张 capability score 图：解释更新前后可信度可见性差异。
- 一张 artifact coverage 图：解释新增可追溯结果。
- 一张 evidence matrix 表：连接 HyperSCA 模块、输出文件、收益、风险和验证计划。

## 口径要求

- 因果审计输出统一写为 `exploratory_cluster_graph` 或 `causal_candidate`，不写成已验证因果机制。
- 空间通信 flow 与 OT 对照统一写为方向一致性或通信优先级，不写成因果证明。
- `s_mechanism` 是解释列，MVP 不并入 `final_score`，避免改变既有 target ranking 语义。
- 当前图 1 的 readiness score 只用于项目管理和展示，不作为模型性能指标。
