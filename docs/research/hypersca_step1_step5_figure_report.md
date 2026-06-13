# HyperSCA Step1-Step5 更新前后主图对比

## Material Passport

- Artifact type: figure pack and output-difference table
- Scope: current branch `HEAD` vs current working-tree update
- Source status: local result artifacts plus static output-contract comparison
- Verification status: ANALYZED; no full biological rerun performed in this step
- Interpretation boundary: 所有新增 sidecar 结果只作为审计、机制假设、通信优先级或实验设计依据，不写成已验证治疗结论。

## 生成文件

- 总览主图：`docs/research/figures/step1_step5_comparison/fig_step1_to_step5_academic_overview.png` / `.pdf` / `.svg`
- Step1 主图：`docs/research/figures/step1_step5_comparison/fig_step1_embedding_before_after.png` / `.pdf` / `.svg`
- Step2 主图：`docs/research/figures/step1_step5_comparison/fig_step2_causal_audit_before_after.png` / `.pdf` / `.svg`
- Step3 主图：`docs/research/figures/step1_step5_comparison/fig_step3_perturbation_mechanism_before_after.png` / `.pdf` / `.svg`
- Step4 主图：`docs/research/figures/step1_step5_comparison/fig_step4_dynamic_context_before_after.png` / `.pdf` / `.svg`
- Step5 主图：`docs/research/figures/step1_step5_comparison/fig_step5_behavior_grammar_before_after.png` / `.pdf` / `.svg`
- 对照表：`docs/research/hypersca_step1_step5_output_difference_table.csv`
- 图表 manifest：`docs/research/hypersca_step1_step5_figure_manifest.json`

## Step1-Step5 输出差异表

| step | previous_main_outputs | current_main_outputs | major_difference | evidence_metric | interpretation_boundary |
| --- | --- | --- | --- | --- | --- |
| Step1 Embedding | H-VAE embeddings, benchmark metrics, topology graph | Core output unchanged; used as upstream representation context | No new Step1 artifact; comparison preserves existing hyperbolic-vs-UMAP metrics | Hyperbolic silhouette 0.235; hierarchy corr 0.713 | Representation quality context only; not rerun in this update |
| Step2 Causal graph | Causal adjacency, bootstrap frequency, signaling flow, baseline comparison | + causal stability audit; + negative controls; + LR communication flow sidecar | Auditability expands from graph metrics to stability, controls and direction consistency contracts | n_edges 8; mean bootstrap 0.638; added artifact families 9 | Exploratory cluster graph / causal candidate; not causal proof |
| Step3 Perturbation | Latent arithmetic counterfactuals, propagation reports, target ranking proxies | + LR-TF-target mechanism evidence matrix and report | Interpretability expands from target score/proxy to mechanism evidence columns | mean CF R2 1.000; added mechanism artifacts 4 | Mechanism hypothesis and prioritization only; s_mechanism not merged into final_score |
| Step4 Dynamic intervention | PK/PD summaries, dynamic effects, combination ranking, roundtrip report | Core output unchanged; now exposed as Step5 context | No scoring change; Step5 reads Step4 summary and combination ranking for behavior simulation context | n_targets 3; n_combos 6; top effect 1.00 | Dynamic/combination in silico priority; not wet-lab validation |
| Step5 Behavior grammar | Absent in HEAD | Behavior rules, virtual tissue trajectory, QoI sensitivity, simulation report | New append-only behavior grammar sidecar turns ranked hypotheses into transparent cell-behavior rules | rules 5; cell types 4; time steps 8 | Toy virtual tissue hypothesis sandbox; not a calibrated tissue digital twin |

## 图注草案

| figure | file | caption |
| --- | --- | --- |
| Figure 1 / Overview | docs/research/figures/step1_step5_comparison/fig_step1_to_step5_academic_overview.png | Step1-Step5 更新前后主输出对比。A 显示 HEAD 只覆盖 Step1-Step4，当前更新新增 Step5 behavior grammar；B 显示新增 artifact family 集中在 Step2、Step3 和 Step5；C-F 分别展示保留的 Step1/Step4 主干结果和新增可信度/行为语法输出。 |
| Step1 main | docs/research/figures/step1_step5_comparison/fig_step1_embedding_before_after.png | Step1 嵌入主图。当前更新没有重算 Step1，而是保留已有 H-VAE/UMAP 指标作为上游表示质量上下文。 |
| Step2 main | docs/research/figures/step1_step5_comparison/fig_step2_causal_audit_before_after.png | Step2 因果图主图。左侧展示已有因果图和空间一致性指标，右侧展示当前更新新增的稳定性审计、负控、平台一致性和 LR flow 方向一致性输出契约。 |
| Step3 main | docs/research/figures/step1_step5_comparison/fig_step3_perturbation_mechanism_before_after.png | Step3 扰动主图。左侧保留反事实质量指标，右侧展示当前更新新增的 LR-TF-target 机制证据分解输出。 |
| Step4 main | docs/research/figures/step1_step5_comparison/fig_step4_dynamic_context_before_after.png | Step4 动态干预主图。当前更新不改变 Step4 scoring，但将 Step4 summary 和组合排序作为 Step5 behavior grammar 的上下文。 |
| Step5 main | docs/research/figures/step1_step5_comparison/fig_step5_behavior_grammar_before_after.png | Step5 行为语法主图。HEAD 中没有该阶段；当前更新输出规则、虚拟组织轨迹和 QoI 敏感性，用于透明机制假设模拟。 |

## 读图口径

1. Step1 与 Step4 的核心结果在本次更新中没有重算；图中保留它们是为了说明端到端主干上下文。
2. Step2 的主要差异是可信度审计和通信方向一致性输出，而不是把因果图写成已验证机制。
3. Step3 的主要差异是机制证据分解，`s_mechanism` 仍是解释列，不改既有 `final_score`。
4. Step5 是新增 append-only behavior grammar sidecar，用规则、轨迹和敏感性把候选机制转成可计算假设。
