# HyperSCA 项目进度评估与因果推断、空间药物作用机制研究版图

**报告日期：** 2026-08-10
**评估对象：** HyperSCA 本地工作树与公开 GitHub 仓库
**检索窗口：** 以 2023-01-01 至 2026-08-10 为主，保留 2022 年 Perturb-map 作为空间功能基因组学奠基工作
**报告性质：** 定向范围综述（scoping review）与工程/科学进度审计；不是系统综述或荟萃分析
**结论置信度：** 工程状态为高（本地命令实测）；已发表论文的方法描述为中高；预印本与跨数据集迁移判断为中低
**Bear 证据附录：** [Markdown](bear_hypersca_spatial_causal_20260810/report.md) · [交互 HTML](bear_hypersca_spatial_causal_20260810/report.html) · [证据台账](bear_hypersca_spatial_causal_20260810/evidence_ledger.json) · [BibTeX](bear_hypersca_spatial_causal_20260810/references.bib)

## 摘要

HyperSCA 已形成覆盖单细胞、空间组学、因果图、反事实生成、时空传播、组合干预和靶点发现的研究型原型，但尚未达到可发布的机制验证平台。2026-08-10 的本地复核显示，显式设置 `PYTHONPATH=.` 后 283 项测试全部通过；然而默认 `pytest` 仍在收集阶段出现 19 个导入错误，`validate_env.py` 因 Python 3.13/目标 Python 3.10 不一致及 7 个必需依赖缺失而失败。GPU 路径当前可用：CUDA 13.0、RTX 4090 与 PyG 扩展测试均通过。当前分支相对 `origin/main` 有 1 个仅本地提交和 4 个尚未纳入的主线提交，另有 44 个已跟踪文件与大量未跟踪文件处于变更状态，因此“代码存在”不能等同于“已合并且可复现”。科学上，空间/双曲 v3 和 celcomen-inspired energy 已完成正式审计，但 target-rank delta 仍为 0，celcomen alignment Spearman 仅 0.0095 且置信区间跨 0，合理状态仍是 `audit_only_no_promotion`。

Bear/SciMaster 的 15 次检索返回 255 条记录，其中 22 条进入证据台账。检索未发现与完整 Task C/S/D 加分层 promotion gate 同构的直接撞车，但找到了 3 个方法孪生和 6 个问题孪生；空间反事实、扰动传播和空间药物排序已经拥挤。因此，现阶段可辩护的创新定位是“候选框架/评价贡献”：用统一 artifact contract 联结因果网络、空间 own/neighbor 效应与剂量—时间药理证据；不能据此声称算法首创、SOTA 或优越性。升级应先建立真实干预基准：以 CausalBench、Pertpy/CINEMA-OT 检验单细胞干预效应和因果网络，以 Spatial Perturb-Seq 与 Perturb-map 检验细胞自主效应、邻域效应和距离衰减；Celcomen、SpatialProp 与 CONCERT 仅在相同切分、特征和调参预算下作为比较器。只有外部 holdout、多 seed 置信区间、负控、校准和失败案例共同支持时，才讨论算法或靶点 promotion。

**关键词：** HyperSCA；单细胞因果推断；空间转录组；空间扰动；药物重定位；靶点发现；反事实生成

## 1. 研究问题与判定框架

### 1.1 主问题

截至 2026-08-10，哪些公开方法、数据与 GitHub 工程可以补足 HyperSCA 从“审计型研究原型”推进到“可外部验证的空间因果药物机制平台”所缺少的证据？

### 1.2 子问题

1. HyperSCA 的工程实现、可复现性、科学验证和版本治理分别进展到什么程度？
2. 单细胞因果效应、因果网络和反事实生成的近期方法中，哪些适合直接纳入基准，哪些只适合并行复现？
3. 哪些空间扰动数据能够为细胞自主效应、邻域传播和组织尺度反事实提供接近真值的验证？
4. 空间药物重定位与药物作用机制研究中，哪些结果是候选生成证据，哪些接近干预验证？
5. HyperSCA 接下来 1、3、6 和 12 个月应设置什么 promotion gate？

### 1.3 因果证据语言

本报告区分四个层级：

| 层级 | 含义 | 允许的表述 |
|---|---|---|
| E3：外部干预支持 | 真实遗传/药物干预，具有对照与外部数据验证 | “干预支持”“与因果效应一致” |
| E2：模型可识别或反事实支持 | 在明确假设、模拟或部分干预条件下可识别 | “模型推断”“假设条件下的反事实” |
| E1：稳定关联/机制一致 | 负控、稳定性、空间或通路一致，但无直接干预真值 | “候选因果”“机制一致” |
| E0：探索性候选 | 表示、相关性、知识库或 LLM 辅助排序 | “候选”“假设生成” |

HyperSCA 当前的因果图、空间传播和机制链主体仍位于 E0-E1；代码中的 `causal`、`counterfactual` 或 `mechanism` 命名本身不提升证据等级。

## 2. 方法与检索策略

### 2.1 本地审计

审计读取了项目说明、现有方法评审、benchmark progress、PR 队列以及 `src/causal/`、`src/perturbation/`、`src/discovery/target_discovery/` 和 `src/evaluation/`。主要复核命令如下：

```bash
git status --short --branch
git rev-list --left-right --count HEAD...origin/main
PYTHONPATH=. pytest tests -q -p no:cacheprovider
pytest tests -q -p no:cacheprovider
python scripts/validate_env.py
```

本地判断以 2026-08-10 的工作树为准，不把未跟踪文件视为已经进入远程主线。历史结果主要来自 [benchmark progress](../../docs/research/hypersca_benchmark_progress_20260622.md)、[当前算法 ARS 评审](../methodology/hypersca_current_version_algorithm_ars_review_20260622.md)、[hyperbolic-spatial v3 评审](../methodology/hypersca_hyperbolic_spatial_v3_progress_review_20260622.md) 与 [post-merge PR queue](../../docs/research/hypersca_post_merge_pr_queue_20260623.md)。

### 2.2 外部检索

检索源包括 Nature Portfolio、ICLR Proceedings、PubMed/PMC、bioRxiv、论文代码可用性段落和 GitHub 连接器。关键词族包括：

```text
(single-cell OR spatial transcriptomics) AND
(causal inference OR causal discovery OR counterfactual OR perturbation) AND
(drug mechanism OR drug repurposing OR spatial propagation OR microenvironment)
```

GitHub 定向检索覆盖 `pertpy`、`causalbench`、`CINEMA-OT`、`celcomen`、`spatialperturbseq`、`CONCERT`、`spatial-prop`、`CausCell`、`HALO`、`SOAR` 与 `STDrug`。纳入标准是：直接对应研究问题；有正式论文或可核验预印本；方法、数据或代码至少一项可复用；能够明确说明限制。排除仅有宣传材料、无法核验的方法名称，以及只做空间聚类但不涉及扰动、因果或药物机制的工作。

最终核心证据集包含 15 篇论文/综述，其中 12 篇为同行评议论文或会议论文，3 篇为预印本；另核验 11 个论文配套仓库。由于检索采用定向搜索且网页结果动态，本报告不虚构数据库总命中数，也不宣称 PRISMA 意义上的穷尽性。

**分布偏斜提示：** 13/15（86.7%）核心来源发表于 2025-2026 年，12/15（80.0%）属于计算方法、基准、综述或数据资源。该偏斜符合“最近研究”的任务范围，但也意味着真实空间药物干预证据明显少于计算模型证据。

### 2.2.1 Bear/SciMaster 定向压力测试

为检验“统一因果—空间—药物机制平台”的新颖性边界，又使用本机 `sci` 0.3.15 执行 Bear `bear-propose` 流程。共运行 15 个预先编号查询：q01-q06 采用 `ultra_low` 作宽检索，q07-q15 采用 `low` 作排序、安静区支撑与反方检索；未使用 `high` 或更高成本模式。15 次检索均返回有效 JSON 和 BibTeX，共 255 条返回记录，去重、逐条审阅后有 22 条进入证据台账。检索前后 quota 从 9582.00 降至 9488.44，观察消耗 93.56。

该流程不是新的系统综述样本框，也不与上文 15 篇核心集直接相加；它承担的是新颖性压力测试和挑战证据补充。全部查询、原始 JSON/BibTeX、纳入理由、证据角色和保留摘要见 [Bear 报告](bear_hypersca_spatial_causal_20260810/report.md)、[query manifest](bear_hypersca_spatial_causal_20260810/query_manifest.tsv) 与 [evidence ledger](bear_hypersca_spatial_causal_20260810/evidence_ledger.json)。因此，下文“未见直接撞车”只表示在本次查询与索引范围内未检出，不表示不存在未索引、术语不同或正在进行的工作。

### 2.3 证据评级

本报告采用适合计算生物学的三档实用评级：

- **A：** 同行评议，代码/数据可获得，且包含真实干预或明确外部基准。
- **B：** 同行评议，代码可获得或方法可复现，但因果结论依赖较强假设或缺少直接干预真值。
- **C：** 预印本或资源可得性不完整；可做探索性比较，不可作为 promotion 的唯一依据。

GitHub stars 只作为社区可见度的弱信号，不作为科学质量或因果有效性的证据。

## 3. HyperSCA 当前进度评估

### 3.1 一句话判断

**HyperSCA 已经是“模块完整、审计意识较强的研究型原型”，但仍是 audit-stage，而不是 release-ready 或 mechanism-validated 平台。**

### 3.2 工程与治理状态

| 维度 | 2026-08-10 证据 | 状态 | 判断 |
|---|---|---|---|
| 模块覆盖 | 104 个 Python 源文件、54 个测试文件、53 个顶层脚本；覆盖因果、反事实、时空传播、靶点发现和空间评估 | 🟢 | 研究功能面完整，已超出概念验证最小骨架 |
| 测试 | `PYTHONPATH=. pytest ...` 为 **283 passed, 309 warnings** | 🟢/🟡 | 当前测试逻辑通过，但入口依赖环境变量 |
| 默认可执行性 | 不设置 `PYTHONPATH` 时出现 19 个 collection errors | 🔴 | 包安装/导入契约不稳，CI 和用户入口不可依赖隐式路径 |
| 环境验证 | 活跃解释器 Python 3.13，而项目要求 3.10；`validate_env.py` exit 1；CUDA 13.0/RTX 4090 与 PyG 扩展通过，但 `squidpy`、`econml`、`pgmpy`、`pingouin`、`diffusers`、`torch_cluster`、`torch_spline_conv` 缺失，`scgen` 为可选缺失 | 🔴 | GPU 基础路径可用，但当前机器仍不能证明官方完整环境可复现 |
| 版本收敛 | 当前分支相对 `origin/main`：1 个本地侧提交、4 个主线侧提交；44 个 tracked 文件变更，3023 行新增、7425 行删除，并有大量 untracked 文件 | 🔴 | 尚无可审查的稳定候选版本；大规模删除需要拆分审阅 |
| 远程治理 | 公开仓库已有 5 个合并 PR；现有 PR queue 已提出排名门控、v3 sidecar、prior DB 与 causal null controls | 🟡 | 方向明确，但队列文档不等于实现已合并 |
| 文档一致性 | 2026-06 报告记录 281 passed/2 failed；本次复核为 283 passed | 🟡 | 旧结论已被后续修改修复，但报告与运行状态需要版本化绑定 |

测试通过说明核心行为没有在当前工作树中明显回归；它不消除默认导入失败、依赖漂移、未合并文件或大规模脚本删除带来的发布风险。

### 3.3 科学模块成熟度

| 科学模块 | 已完成 | 尚缺证据 | 成熟度 |
|---|---|---|---|
| 空间注释与上下文 | RCTD/cell2location 路径、545,913 行 abundance 校验、dominant-grid concordance 0.827、空间注释门控 | 外部样本/平台复现；T/ILC 低一致区域的人工/组织学核验 | 中：可作为上下文层 |
| 双曲/层级表示 | v3、3 seeds、loss ablation、GPU 正式运行；kNN purity 约 0.67 | label AUC 仍低于 SCimilarity；prototype AUC 接近随机；缺外部功能收益 | 中低：仅 sidecar |
| celcomen-inspired energy | endpoint 可计算，energy AUC 0.7157，对照 0.7136 | 增益仅 0.0021；alignment Spearman 0.0095 且 CI 跨 0；不是完整 Celcomen | 低：审计指标 |
| 因果图 | PC skeleton/orientation、bootstrap、DoWhy arrow strength/structure refutation、负控与稳定性审计 | 真实 interventional edge recovery、跨细胞系泛化、反馈环与隐混杂处理 | 中低：E1 候选因果 |
| 反事实 | latent arithmetic、diffusion counterfactual、dose-response、PK、组合干预与 Bliss 指标 | 真实干预后表达、剂量时间曲线、未见条件泛化、校准与失败案例 | 低：E0-E1 假设生成 |
| 空间传播 | BFS/邻接传播、temporal-spatial simulation、Moran's I、gradient decay、propagation depth | own/neighbor perturbation ground truth；组织切片/动物 holdout；邻域干预基线 | 低：代理评估 |
| 靶点发现 | 分层非加权证据、admission gate、机制链与主排名隔离 | target-rank delta 仍为 0；无外部靶点 enrichment 或药理验证 | 中低：排序框架成熟、功能收益未证实 |
| 空间药物机制 | PK/PD、药物组合与通路/通信链可表达 | 缺直接药物空间扰动、target engagement、毒性、剂量和临床结局 | 低：候选机制阶段 |

关键区别是：HyperSCA 已有许多“能够生成机制解释的结构”，但还没有证明这些结构比简单基线更准确地恢复真实干预效应。现有代码也已经明确把 graph-propagation 结果称为 proxy，并要求 public Perturb-seq 或 spatial perturbation data；这一保守接口应继续保留。

### 3.4 当前最强证据与最弱环节

当前最强的成果不是“因果机制已经成立”，而是三项工程科学资产：

1. **证据门控正确。** 主排名不因 sidecar 的小幅表示改进自动变化，避免了指标挑选和循环论证。
2. **空间上下文已可审计。** RCTD/cell2location、丰度规模检查与 concordance 为后续干预 benchmark 提供了数据入口。
3. **因果/传播模块有明确接口。** PC、DoWhy、稳定性负控、反事实和传播指标可被替换或加入外部基线。

最弱环节则是“从代理指标到真实干预”的断层。target-rank delta 为 0 不是单一模型失败，而是在提醒：当前 embedding、causal graph、spatial propagation 与药物排序尚未形成可被外部干预数据证伪的统一评价任务。

## 4. 近期单细胞因果推断研究

两篇近期综述把领域核心问题概括为：表示学习、因果推断、机制发现和扰动外推必须根据数据生成机制与可识别假设选择，不能把预测精度直接解释为因果机制。Tejada-Lapuerta 等的 Perspective 系统讨论了单细胞因果机器学习的泛化、动态与可解释性；Dimitrov 等进一步强调简单线性基线、未见条件外推和 biological hallucination 的评估风险（[Tejada-Lapuerta et al., 2025](https://www.nature.com/articles/s41588-025-02124-2)<!--ref:tejada2025causal--><!--anchor:section:Abstract-->；[Dimitrov et al., 2026](https://www.nature.com/articles/s41576-025-00920-4)<!--ref:dimitrov2026interpretation--><!--anchor:section:Abstract-->）。

### 4.1 方法与基准矩阵

| 工作 | 核心问题与方法 | 直接价值 | 关键限制 | 评级/建议 |
|---|---|---|---|---|
| [CINEMA-OT](https://www.nature.com/articles/s41592-023-02040-5) | 用 ICA 和函数依赖过滤分离处理相关因子与混杂因子，再以加权最优传输匹配反事实细胞；支持 individual treatment effect、response cluster、attribution 和 synergy（Dong et al., 2023）<!--ref:dong2023cinema--><!--anchor:section:Abstract--> | 可作为 HyperSCA latent arithmetic/diffusion 的非神经基线；输出可直接转为 per-cell effect matrix | 依赖独立成分与混杂可分假设；差异丰度会破坏未加权匹配 | **A；P0 立即基准**，优先用 Pertpy 实现 |
| [Pertpy](https://www.nature.com/articles/s41592-025-02909-7) | scverse/AnnData 兼容的扰动分析框架，统一数据、元数据、距离、差异与多种方法；包含实验性 CINEMA-OT（Heumos et al., 2026）<!--ref:heumos2026pertpy--><!--anchor:section:Abstract--> | 与 HyperSCA 数据结构相容；可先做 adapter 而非复制算法 | 是分析框架，不自动提供因果真值；超大数据仍需 out-of-core 优化 | **A；P0 依赖/基准层** |
| [CausalBench](https://www.nature.com/articles/s42003-025-07764-y) | 两个大型 CRISPRi scRNA 数据集、超过 20 万干预样本，比较观察与干预网络方法（Chevalley et al., 2025）<!--ref:chevalley2025causalbench--><!--anchor:section:Abstract--> | 为 PC/DoWhy/stability 提供 K562、RPE1 真实干预评估和标准基线 | 生物知识网络不是真正完整 GRN；反馈环、未观测混杂和 dropout 仍存在 | **A；P0 最高优先级** |
| [Celcomen](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f8d8d09728100b1947d6add8ec625d56-Abstract-Conference.html) | 从空间基因相关约束推导生成式 GNN，分解 intra/inter-cellular regulation 并生成 spatial counterfactual（Megas et al., 2025）<!--ref:megas2025celcomen--><!--anchor:section:Abstract--> | 与 HyperSCA 空间因果主线最接近；适合验证“完整模型”与本地 energy proxy 的差距 | 观察数据只能恢复 Markov 等价类；依赖 causal sufficiency；DAG 不含反馈环；无 cell-type-specific forces；只建模 pairwise forces | **B；P1 并行复现**，不得把 energy endpoint 称为 Celcomen 复现 |
| [CausCell](https://www.nature.com/articles/s41467-025-62008-1) | 将预给定概念因果结构与 diffusion 结合，学习可控的因果解耦表示和反事实（Gao et al., 2025）<!--ref:gao2025causcell--><!--anchor:section:Abstract--> | 可作为 HyperSCA diffusion counterfactual 的表示级比较 | 需要概念标签和给定因果结构；不是 de novo gene causal discovery；存在作者更正 | **B；P1/P2 容器化比较** |
| [HALO](https://www.nature.com/articles/s41467-025-63921-1) | 在 representation 与 gene-peak 两层建模 RNA/ATAC 的 coupled/decoupled temporal relations（Mao et al., 2025）<!--ref:mao2025halo--><!--anchor:section:Abstract--> | 若 HyperSCA 纳入 multiome/ATAC，可补充调控方向与时序证据 | 依赖 co-assayed multiome 和时间/latent time；不能解决当前无 ATAC 数据的验证缺口 | **B；P2 数据就绪后** |

### 4.2 对 HyperSCA 最重要的反直觉结果

CausalBench 的结果不支持“模型越复杂，因果网络越好”。Mean Difference 与 Guanlab 在统计和生物评估上表现突出，而许多 PC、GES、NOTEARS/DCDI 类方法提取的有效干预信息有限；GRNBoost 以低精度换取高召回。论文同时指出，知识库构造的 biological ground truth 不完整，真正真值仍需前瞻性干预（Chevalley et al., 2025）<!--ref:chevalley2025causalbench--><!--anchor:section:Discussion-->。因此，HyperSCA 的首要对手不是另一个深模型，而是 **Mean Difference、Guanlab、GRNBoost 与简单观察/干预基线**。

Celcomen 则提供了另一项边界条件：它的 identifiability 是在三个模型假设下建立，实际论文承认无干预数据时只能接近 Markov 等价类，且 causal sufficiency、DAG、无细胞类型特异相互作用和无基因协同都是限制（Megas et al., 2025, pp. 22-23）<!--ref:megas2025celcomen--><!--anchor:page:22-23-->。HyperSCA 现有 celcomen-inspired energy 只是球面表达邻域能量，不能继承这些可识别性结论；反过来，即便未来复现完整 Celcomen，也必须在 Spatial Perturb-Seq 或 Perturb-map 上验证。

## 5. 空间扰动与空间药物作用机制研究

### 5.1 从“空间相关”到“空间干预”的证据阶梯

| 工作 | 实验/模型 | 关键证据 | 对 HyperSCA 的用途 | 评级/建议 |
|---|---|---|---|---|
| [Spatial Perturb-Seq](https://www.nature.com/articles/s41467-026-69677-6) | 小鼠脑内 18 个基因/位点的 pooled CRISPR，Stereo-seq 与 Xenium 读取 | Stereo-seq 获得 229,775 个细胞；跨平台 own DEG Spearman 0.63、neighbor DEG 0.47；247-gene Xenium panel 仅 9/18 perturbations 检出上调 DEG（Shen et al., 2026）<!--ref:shen2026spatialperturbseq--><!--anchor:section:Results--> | 最直接的外部空间传播真值；可验证 own/neighbor、距离、细胞类型和 targeted-panel gate | **A；P0 第一空间基准** |
| [Perturb-map](https://pmc.ncbi.nlm.nih.gov/articles/PMC8992964/) | 小鼠肺癌体内 35-gene KO，Pro-Code、成像与空间转录组 | 解析肿瘤生长、组织学和免疫组成；Tgfbr2 KO 与 T-cell exclusion/fibro-mucinous TME 同位（Dhainaut et al., 2022）<!--ref:dhainaut2022perturbmap--><!--anchor:section:Summary--> | 肿瘤场景更接近 HyperSCA；适合检验 niche-specific 与邻近免疫变化 | **A；P0 第二空间基准** |
| [CONCERT](https://www.biorxiv.org/content/10.1101/2025.11.08.686890v2) | GP-VAE 学习空间 kernel 与 perturbation context，做 niche-aware counterfactual | 在 seen/unseen spots、niche perturbation 和 Perturb-map 上比较 kNN、scGen、BioLORD、CPA 等（Lin et al., 2025）<!--ref:lin2025concert--><!--anchor:section:Abstract--> | 可作为 HyperSCA diffusion/spatial propagation 的 niche-aware comparator | **C；P1 观察/复现，等待同行评议** |
| [SpatialProp](https://pubmed.ncbi.nlm.nih.gov/41573962/) | GNN 从组织微环境预测多基因、多细胞类型扰动传播；含校准与 CausalInteractionBench | 强调从单细胞 perturbation 向整个 tissue 的传播，但当前为预印本（Buendia et al., 2025）<!--ref:buendia2025spatialprop--><!--anchor:section:Abstract--> | 与 `src/perturbation/spatial_propagation.py` 任务高度同构；适合作为结构和 calibration 基线 | **C；P1 比较，不作为真值** |
| [SOAR](https://doi.org/10.1126/sciadv.adt7450) | 统一处理 3,461 个样本、13 个物种、42 个组织、19 类 ST 技术的资源，连接 CMap/PPI 与药物发现 | 提出 sirolimus、trichostatin A 和 JAK/STAT inhibitor 等候选（Li et al., 2025）<!--ref:li2025soar--><!--anchor:section:Abstract--> | 可作为空间疾病 signature、通路和候选药物先验 | **B；P1 数据资源，证据层 E0** |
| [STDrug](https://www.biorxiv.org/content/10.64898/2026.04.03.715101v1) | GCN+CPD 对齐病/对照空间域，以 L1000、Tahoe-100M、SIDER、GDSC 和 GPT-4o 文献权重排序药物 | 预印本报告 HCC/前列腺癌 AUC 0.81-0.82，并有 EHR 与体外验证（Yang et al., 2026）<!--ref:yang2026stdrug--><!--anchor:section:Abstract--> | 可做 drug-ranking comparator，帮助拆分 reversal、toxicity 与 spatial-domain scores | **C；P2 候选层**；LLM 派生权重有循环验证风险，且代码仓库当前不可访问 |
| [Multiplex pharmacotranscriptomics](https://www.nature.com/articles/s41589-024-01761-8) | 96-plex scRNA-seq，45 个药物、13 种 MOA、288 个处理样本、约 36,000 个细胞 | 揭示 PI3K-AKT-mTOR inhibitor 诱导 CAV1/EGFR feedback，并用联合抑制缓解（Dini et al., 2025）<!--ref:dini2025pharmaco--><!--anchor:section:Abstract--> | 非空间，但提供真实药物反应与 MOA 正控；适合作为药物机制轴的外部验证 | **A；P1 药物机制基准** |

### 5.2 关键综合判断

近期空间研究正在把“邻域相关”推进到“已知扰动在邻域中的效应”，但遗传扰动证据明显领先于药物扰动。Spatial Perturb-Seq 和 Perturb-map 有明确 perturbation identity；SOAR 是空间资源与候选发现平台，STDrug 是药物排序模型，二者都不能单独证明药物在特定空间生态位中的作用机制。

Spatial Perturb-Seq 的跨平台结果也直接支持 HyperSCA 已经采用的 panel-aware gate：247 基因 Xenium panel 只在一半 perturbations 中恢复上调 DEG，说明 targeted panel 的“未检出”不能解释为“无效应”。但该结果来自小鼠脑遗传扰动，迁移到人肿瘤药物机制时仍需重新校准。

药物机制至少需要四个相互独立的证据面：

1. **处理身份与暴露：** 化合物、剂量、时间、PK/PD 或 target engagement；
2. **细胞自主效应：** 处理细胞内的表达、通路和表型改变；
3. **空间非自主效应：** 邻域细胞类型、距离衰减、通信或组织结构变化；
4. **可证伪验证：** 独立数据、已知阳性/阴性药物、体外或体内实验。

HyperSCA 当前覆盖了第 2-3 项的计算代理和第 1 项的 PK/剂量函数接口，但没有形成四项同时满足的外部验证链。

## 6. GitHub 工程版图与集成决策

GitHub 连接器在 2026-08-10 确认下列仓库均为公开且未归档：`scverse/pertpy`、`causalbench/causalbench`、`Teichlab/celcomen`、`vandijklab/CINEMA-OT`、`kimberle9/spatialperturbseq`、`mims-harvard/CONCERT`、`abuendia/spatial-prop`、`luoyuanlab/SOAR`、`bm2-lab/CausCell` 和 `benoslab/HALO`。连接器对论文/PyPI 指向的 `akiyiwen/STdrug` 返回 404，因此其可复现性需降级处理。

| 决策层 | GitHub 项目 | 理由 | 具体动作 |
|---|---|---|---|
| **现在纳入基准** | [scverse/pertpy](https://github.com/scverse/pertpy) | MIT、标准安装、AnnData/scverse、tests/docs 完整；适合统一扰动分析 | 新建 adapter，只交换 AnnData 与结果表，不复制其内部代码 |
| **现在纳入基准** | [causalbench/causalbench](https://github.com/causalbench/causalbench) | Apache-2.0；提供两个干预数据集、基线与模型接口 | 将 HyperSCA causal adjacency 输出适配为 benchmark edge list |
| **现在纳入数据层** | [kimberle9/spatialperturbseq](https://github.com/kimberle9/spatialperturbseq) | 论文配套分析包，含数据与空间扰动处理入口 | 先做 schema/fixture adapter，再做全量运行 |
| **并行复现** | [Teichlab/celcomen](https://github.com/Teichlab/celcomen) 与 [reproducibility repo](https://github.com/stathismegas/celcomen_reproducibility) | 与本项目概念最接近，但完整模型和本地 energy proxy 差异大 | 在隔离环境重现论文数据；不直接并入主排名 |
| **通过 Pertpy 使用** | [vandijklab/CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) | 原始实现可核验，但论文已说明实验性新实现位于 Pertpy | 首选 Pertpy；原仓库只用于结果交叉检查 |
| **预印本比较** | [mims-harvard/CONCERT](https://github.com/mims-harvard/CONCERT)、[abuendia/spatial-prop](https://github.com/abuendia/spatial-prop) | 与 niche counterfactual/propagation 强相关，但尚未同行评议 | 固定 commit、容器和数据切分，输出 comparator artifact |
| **数据/先验资源** | [luoyuanlab/SOAR](https://github.com/luoyuanlab/SOAR) | 空间 atlas 和药物候选资源，而非因果预测真值 | 只进入 prior/evidence 层，记录数据版本与来源 |
| **延后** | [bm2-lab/CausCell](https://github.com/bm2-lab/CausCell)、[benoslab/HALO](https://github.com/benoslab/HALO) | 分别需要给定概念因果图或 multiome/时间信息 | 数据和明确 RQ 就绪后做独立 benchmark |
| **阻塞** | `akiyiwen/STdrug` | 论文与 PyPI 可见，但 GitHub connector 返回 404 | 不自动执行 PyPI 中引用的远程 shell；等待仓库恢复并核验 license/commit |

应避免以 star 数决定集成顺序。Pertpy 的成熟度有利于工程接入，但 Celcomen/SpatialProp 的科学价值必须由相同数据切分、简单基线和真实干预终点决定。

## 7. Bear 新颖性压力测试与可靠性设计

### 7.1 撞车分层：创新空间在哪里

Bear 检索没有发现同时覆盖 Task C/S/D、真实药理暴露、空间 own/neighbor 效应和 promotion gate 的完整同构平台；但组件层已经明显拥挤，不能把“整合了多个模块”自动等同于新算法。

| 层级 | Bear 证据 | 对 HyperSCA 的含义 |
|---|---|---|
| 直接撞车 | 本次检索为 0 | 仅说明查询范围内未检出；不能写“首个”或“唯一” |
| 方法孪生 | [Counterfactual Diffusion](https://doi.org/10.3390/biology15141097)、[Celcomen](https://doi.org/10.48550/arXiv.2409.05804)、[DynPerturb](https://doi.org/10.1101/2025.09.15.676236)（E1、E2、E5） | 空间反事实、因果解耦与时空动态本身不是充分创新点 |
| 问题孪生 | SpatialProp、CONCERT、SpaRx、STDrug、GBM atlas、spatial pharmaco-multiomics（E3、E4、E6-E8、E11） | 传播、niche response 与空间药物问题已有直接竞争者；必须同任务比较 |
| 邻近工作 | PerturBench、CINEMA-OT、药理读出、校准、空间混杂与 CausalBench（E9、E10、E12-E16、E19-E22） | 提供基线与可靠性约束，而不是可忽略的背景引用 |

模型层的最近竞争者应在正文中正面呈现，而不是只在限制部分列出。其中，Counterfactual Diffusion 与 HyperSCA 的空间反事实主线最接近；Celcomen、SpatialProp 和 CONCERT 分别覆盖空间因果解耦、组织尺度传播和 niche-aware prediction。由此，创新论证必须落到任务契约、证据分层或经过公平比较后确证的新模块，而不能落到方法名称。

### 7.2 相对安静区：可主张什么

本次检索支持两个“相对安静区”，但不支持“研究空白”或“首次提出”的绝对表述。

1. **统一证据契约。** 把 Task C（干预因果网络）、Task S（空间传播）和 Task D（空间药物机制）放入版本化 artifact contract，并以 E0-E3 证据等级控制 promotion。现有工作通常只覆盖其中一个任务，因此这更可能形成框架或评价贡献。
2. **同组织药理链。** 联合剂量/时间、药物空间分布、target engagement、细胞自主效应和邻域效应。2026 年的 [spatial pharmaco-multiomics](https://doi.org/10.64898/2026.01.25.701559) 预印本（E11）已经非常接近这一方向，所以这里不是空白；HyperSCA 的可辨别贡献只能来自跨数据集标准化、可证伪的 benchmark 或方法在外部数据上的净收益。

因此，现阶段允许的表述是“候选框架创新”“候选评价创新”和“待验证的算法模块”。不允许写“新型空间因果算法优于现有方法”“首次统一空间药理机制”或“SOTA”，除非后续比较与消融提供直接证据。

### 7.3 挑战证据转化为设计约束

| 主要威胁 | 挑战证据 | 必须加入的验证 |
|---|---|---|
| 复杂表示不一定优于简单基线 | [PCA benchmark](https://doi.org/10.48550/arXiv.2410.13956) 与 CausalBench（E16、E22） | mean、PCA、nearest neighbor、线性/简单干预法必须同预算比较 |
| 未见药物可能只在同类内泛化 | [drug-blind response study](https://doi.org/10.1101/2025.06.16.659838)（E17，预印本） | drug-class、MOA、scaffold holdout；报告拒判率和校准 |
| 跨环境效应不可直接迁移 | [Mechanisms Matter](https://doi.org/10.64898/2026.05.08.723625)（E18，预印本） | 跨细胞系、组织、切片/动物 holdout；报告 effect transport failure |
| 空间位置可制造伪因果 | [SpaCE](https://doi.org/10.48550/arXiv.2312.00710)（E19，预印本） | 坐标置换、空间负控、环境分层与敏感性分析 |
| 点预测掩盖不确定性 | [TISSUE](https://doi.org/10.1038/s41592-024-02184-y)（E20） | coverage、calibration error、拒判率进入主终点 |

这些挑战不是报告末尾的免责声明，而是 benchmark contract 的组成部分。任何模型只有在强基线、外部 holdout、负控、校准和失败案例上形成一致证据，才可能由 sidecar 晋级为主运行时。

### 7.4 升级—比较配对矩阵

| 任务 | HyperSCA 升级目标 | 必须比较的方法 | 公平性约束 | 预注册主终点 | 当前状态 |
|---|---|---|---|---|---|
| Task C | 干预因果网络 adapter | Mean Difference、Guanlab、GRNBoost、PC/DoWhy | 相同基因、干预、edge budget 与调参预算 | PR-AUC、precision-recall、intervention utility | 尚未 benchmark |
| single-cell counterfactual | per-cell treatment-effect adapter | mean、nearest neighbor、CINEMA-OT、PerturBench 方法 | 相同特征、切分、coverage rule 与调参预算 | per-cell effect、population distance、coverage、calibration | 尚未 benchmark |
| Task S | 联合 own/neighbor propagation benchmark | own-only、distance decay、Celcomen、SpatialProp、CONCERT | 相同组织图、cell type、panel gate 与 perturbation identity | own/neighbor effect、distance-bin、niche、calibration | 仅有代理指标 |
| Task D | dose-time spatial pharmacology evidence chain | signature reversal、SpaRx、STDrug、真实药理正控 | 遗传/药物证据分离；相同 dose、time、toxicity 规则 | known MOA、target engagement、dose-time、external response | 仅候选层 |

完整机器可读版本见 [comparison matrix](bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv)。所有方法应共享数据切分、输入特征、预处理、超参数搜索预算和停止规则；同时报告绝对表现、相对差值、置信区间、计算成本和失败案例。缺失公开实现的方法可以标为不可复现，但不能静默移除最强竞争者。

### 7.5 创新主张登记表

| 主张 | 最近工作/对照 | 所需证据 | 当前判定 | 允许措辞 |
|---|---|---|---|---|
| 统一 Task C/S/D 与 E0-E3 promotion gate | PerturBench、SpatialProp、STDrug 等分散任务 | 版本化产物、端到端 adapter 演示、gate/rollback 测试 | 候选 | 候选框架或整合贡献 |
| 联合 own/neighbor/distance/niche/pharmacology 的可靠性评价 | SpatialProp、CONCERT、空间药理研究 | metric validation、null、calibration、panel/section holdout | 候选 | 候选评价贡献 |
| 空间因果反事实模型优于现有方法 | Celcomen、SpatialProp、CONCERT、Counterfactual Diffusion | 同任务外部比较、多 seed CI、消融、负控、校准与失败案例 | **未证明** | 算法候选；不得声称 superiority |
| 双曲/energy sidecar 改善靶点排序 | SCimilarity、Euclidean/simple ranking、Celcomen | target-rank delta 与外部 target enrichment | **未证明；delta=0** | 审计型 sidecar；不得声称算法创新 |

该登记表是版本化的“可撤回主张”：新实验只能把具体主张从未证明推进到支持，不能用一个辅助指标替代相应主终点。机器可读版本见 [innovation claim register](bear_hypersca_spatial_causal_20260810/innovation_claim_register.tsv)。

## 8. HyperSCA 缺口到外部证据的映射

| HyperSCA 现有资产 | 当前缺口 | 外部补充 | 建议新增产物 |
|---|---|---|---|
| PC + DoWhy + stability audit | 只有观察/稳定性证据，缺 interventional edge utility | CausalBench | `benchmarks/causal_interventional/{dataset}/metrics.json`；含 Mean Difference/Guanlab/GRNBoost/PC |
| latent arithmetic + diffusion CF | 反事实是否对应真实处理后细胞未知 | Pertpy/CINEMA-OT；真实 Perturb-seq | per-cell effect、population distance、coverage、calibration 与 null-treatment 报告 |
| `spatial_propagation.py` + spatial metrics | Moran/gradient/depth 是代理，无 own/neighbor 真值 | Spatial Perturb-Seq、Perturb-map | own/neighbor DEG、distance-bin effect、cell-type/niche stratification、section holdout |
| celcomen-inspired energy | 并非完整 intra/inter gene-force model | Celcomen | exact-reproduction sidecar；模型假设清单；与 energy proxy 的差异报告 |
| mechanism evidence chain | 机制分数不改主排名，缺真实 perturbation chain | Spatial Perturb-Seq、Dini pharmacotranscriptomics | `mechanism_evidence_tier` 字段：association / interventional / pharmacological |
| PK、Hill、Bliss | 无真实浓度时间和 combination response | 药物扰动/剂量数据 | dose-time surface、Bliss/HSA/ZIP 多指标、target engagement 与 toxicity 字段 |
| prior DB/通信流 | 先验可能形成确认偏差 | CausalBench nulls、SOAR 版本化资源 | prior-on/off ablation、source/version/license、知识库泄漏审计 |

### 8.1 建议的统一任务定义

应把当前松散模块收敛为三个可外部评分的任务：

1. **Task C：Interventional causal network。** 输入观察与部分干预 scRNA；输出有方向 edge ranking；评估统计干预距离、精确率-召回率、干预利用率和跨细胞系稳健性。
2. **Task S：Spatial perturbation propagation。** 输入组织图、cell type、perturbation identity 和基线表达；输出 target cell 与邻域的 effect matrix；评估 own/neighbor、距离、niche、section/animal holdout 和不确定性校准。
3. **Task D：Spatial drug mechanism。** 输入疾病空间样本、药物签名、剂量/时间和先验；输出候选药物、作用细胞类型、空间生态位和机制链；评估真实 drug-response、已知 MOA、毒性与外部验证，而不是只评估文献命中。

这三个任务共享同一 artifact contract，但应分别 promotion。Task C 或 Task S 的通过不能自动证明 Task D。

## 9. 优先路线图与验收门槛

### P0：0-4 周，建立可信基线

1. **收敛工作树。** 以 `origin/main` 为基线，将现有变更按 PR queue 拆成排名策略、v3 sidecar、benchmark report、prior DB、causal null controls；大规模脚本删除单独审查。
2. **修复可复现入口。** 锁定 Python 3.10；使 `pip install -e .` 或等价安装后默认 `pytest` 通过；CPU-only `validate_env.py` 应区分必需与 GPU 可选项。
3. **冻结 benchmark contract。** 预先声明 estimand、primary metric、data split、seed、null、coverage、拒判与 promotion rule；把 [comparison matrix](bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv) 版本化，避免看结果后调整阈值。
4. **先实现简单基线。** Task C 至少包含 Mean Difference；counterfactual 至少包含 mean/nearest/PCA；Task S 至少包含 own-only 与固定 distance decay；Task D 至少包含不使用空间信息的 signature reversal。
5. **接入三项外部资产。** Pertpy/CINEMA-OT、CausalBench、Spatial Perturb-Seq；先做小型 synthetic/fixture，再做全量。所有 adapter 只负责数据与 artifact 契约，避免直接耦合第三方训练代码。

**P0 gate：** clean reviewable branch；Python 3.10 CI；默认测试入口全通过；`validate_env.py` CPU 模式成功；三类外部 adapter 能生成 manifest、输入摘要和基线结果；主终点与数据切分在模型比较前冻结。

### P1：1-3 个月，完成真实干预评估

1. 在 CausalBench K562/RPE1 上比较 HyperSCA、Mean Difference、Guanlab、GRNBoost 和 PC；所有方法使用相同基因集与干预切分。
2. 在 Spatial Perturb-Seq 上评估 own/neighbor effect、距离衰减、细胞类型和 Stereo-seq/Xenium panel sensitivity；在 Perturb-map 上加入肿瘤/免疫 niche 任务。
3. 用 Pertpy 统一处理 effect distance、DE、guide assignment 与对照；CINEMA-OT 作为单细胞 effect 基线。
4. 用 Dini 等的 pharmacotranscriptomic 数据设置已知药物/MOA 正控，检验 HyperSCA 的 drug mechanism score；增加 drug-class/MOA holdout，不能只做随机样本切分。
5. 主表同时报告绝对分数、相对简单基线差值、95% CI、coverage/calibration、运行成本与预先定义的失败案例；不得只选择有利指标。

**P1 gate：** 至少两个 interventional datasets；至少 3 seeds 或 bootstrap CI；预注册 primary metric；相对简单基线的改善在置信区间层面为正且绝对表现达到预设最低值；null treatment、label shuffle、coordinate permutation 和 prior-off 均不过度报阳性。

### P2：3-6 个月，比较空间反事实模型

1. 隔离复现 Celcomen，并明确其完整 gene-force model 与本地 energy endpoint 的区别。
2. 固定版本比较 Counterfactual Diffusion、SpatialProp 与 CONCERT；DynPerturb 仅在具备有效时间标签时纳入。统一 section/animal/niche holdout，禁止同切片邻域泄漏。
3. 将机制证据拆为 association、genetic-interventional、pharmacological 三层；药物候选生成可参考 SOAR/STDrug，但不进入主排名 promotion。
4. 加入 calibration、uncertainty、failure case 与 compute/memory 报告。

**P2 gate：** 空间反事实在 held-out tissue/animal 上优于 mean、kNN 和图传播基线；own 与 neighbor 方向均可复现；新增复杂度带来明确净收益；失败场景有可诊断边界。

### P3：6-12 个月，空间药物机制验证

1. 选择 2-3 个有空间表达、已知 MOA 和可获得药物的候选机制，预先锁定 primary endpoint。
2. 设计剂量×时间×细胞类型/生态位验证，至少包含 vehicle、阳性、阴性与 pathway rescue/combination control。
3. 如使用患者级空间组学或受限数据，补齐伦理、隐私、批次和外部队列验证。
4. 只有在遗传干预、药物干预和空间邻域效应三条证据链至少两条收敛时，才允许“空间药物作用机制”表述。

**P3 gate：** 候选排名在独立药物数据中保留；有 dose-response/target engagement；空间效应在独立样本复现；毒性与非特异效应被量化；至少一项体外或体内验证支持方向。

## 10. Promotion policy 建议

| 对象 | 最低 promotion 条件 | 明确不能作为充分条件的指标 |
|---|---|---|
| 因果图 | 两个真实干预数据集；优于简单干预基线；负控校准；跨细胞系方向稳定 | bootstrap frequency、DoWhy refutation 单独通过 |
| 空间传播 | own/neighbor 真值；animal/section holdout；距离与 niche 分层；不确定性校准 | Moran's I、gradient R²、propagation depth 单独改善 |
| 双曲/Celcomen 模块 | 外部功能终点改善；相对 Euclidean/SCimilarity/graph baseline 有稳定净收益 | label AUC 或 energy AUC 的微小提升 |
| 靶点排序 | target enrichment 或干预验证非零；排名变化可解释；prior-off 后仍稳健 | target-rank delta 非零本身 |
| 药物机制 | 真实 drug perturbation、剂量/时间、空间邻域效应及外部验证 | CMap reversal、SOAR/STDrug 命中、LLM 文献支持 |
| 框架/评价创新 | 可复现的 Task C/S/D adapter；版本化 artifact/gate；能发现基线、泄漏或校准差异 | 仅把现有模块放在同一仓库或统一命名 |
| 核心算法创新 | 在预注册主终点上相对强/简单基线有稳定净收益；外部 holdout、多 seed CI、消融与失败分析一致 | 单个内部数据集、辅助指标、无同预算比较的提升 |

现有 `audit_only_no_promotion` 不应被视为失败状态，而应视为保证项目科学可信度的核心设计。只有预先定义的外部功能 gate 通过时，才应解除。

## 11. 风险、反方论证与限制

### 11.1 Devil's Advocate Checkpoint 2：检索与综合风险

- **新颖性偏差：** 2025-2026 方法占 86.7%，可能高估预印本和新框架，低估稳定但较旧的统计方法。缓解方式是强制加入 Mean Difference、kNN、线性/PC 基线。
- **代码可得不等于可复现：** 公开仓库可能缺数据、锁文件或论文版本。每个外部方法必须固定 commit、环境和输入摘要。
- **因果术语膨胀：** Celcomen、CausCell、SpatialProp 的“causal”对应不同 estimand 和假设，不能放在单一 AUC 下无差别比较。
- **领域迁移：** 小鼠脑遗传扰动不等同于人类肿瘤药物反应；Perturb-map 更接近肿瘤，但技术与读出不同。
- **知识库泄漏：** CausalBench biological metrics、SOAR/CMap 和 STDrug 的 GPT-4o 文献权重可能与候选先验重叠，必须报告 prior-on/off 和纯统计 endpoint。
- **安静区的查询敏感性：** 15 次 Bear 查询未见直接撞车不等于不存在同构系统；同义词、近期未索引预印本和进行中项目都可能被漏检。创新表述必须使用“本次检索中的相对安静区”。
- **排序与纳入偏差：** SciMaster 返回顺序、模式和人工去重会影响 22 条证据集；原始结果和纳入理由已保留，但没有建立独立双人筛选或数据库穷尽性保证。

**Checkpoint 2 verdict：PASS with major caveats。** 证据足以制定路线图，不足以宣布任何外部方法会提升 HyperSCA。

### 11.2 Devil's Advocate Checkpoint 3：对路线图的最强反驳

最强反驳是：HyperSCA 已经存在大量模块和未合并代码，继续加入 Pertpy、Celcomen、SpatialProp 或 CONCERT 可能只增加依赖与维护成本，而不会改变 target ranking。该反驳与现有 target-rank delta=0 完全一致。因此路线图把“导入方法”改为“先以统一 adapter 和外部 benchmark 比较”；只有在简单基线之上产生可复现净收益的模型才进入主运行时。

**Checkpoint 3 verdict：PASS。** 没有发现需要阻止报告交付的完整性问题；但应维持所有模型的非促销状态，直到外部干预 gate 通过。

### 11.3 报告限制

1. 这是定向范围综述，不是穷尽性系统综述；没有进行效应量合并。
2. 未克隆和执行外部仓库，GitHub 判断限于公开元数据、README、论文代码可用性段落和可访问状态。
3. 三项纳入研究为预印本：CONCERT、SpatialProp、STDrug；其结论可能在同行评议后变化。
4. STDrug 的 PyPI 页面仍引用 GitHub 原始文件，但连接器对仓库返回 404；本报告未执行其远程安装/下载脚本。
5. 本地工作树包含用户尚未提交的变更；本文不会把这些变更归因于特定作者，也不判断其最终合并意图。
6. 本报告评估计算证据，不构成临床治疗建议或药物推荐。
7. Bear 证据来自单一检索服务的一次会话；虽然保留了 15 组原始 JSON/BibTeX 和完整摘要，但没有进行双数据库复核、前向/后向引文追踪或独立双人筛选。

## 12. 结论

HyperSCA 的主要进展是建立了跨因果、空间、反事实与靶点发现的完整接口，并且已有保守的证据门控；主要瓶颈不是再缺一个模型，而是缺少统一的外部干预任务。Bear 压力测试进一步表明，空间反事实、扰动传播和药物排序的模型层已经拥挤；目前最稳妥的创新定位是把 Task C/S/D、own/neighbor 药理终点和 E0-E3 promotion gate 组织成可复核、可撤回的框架/评价贡献，而不是预先宣称算法优越。

最优近期路线是：先让仓库在 Python 3.10 下可安装、默认测试和环境验证可重复；随后以相同数据切分、特征和调参预算比较简单基线、CINEMA-OT、Celcomen、Counterfactual Diffusion、SpatialProp 与 CONCERT；再把真实剂量—时间、target engagement 和空间邻域证据接入 Task D。SOAR、SpaRx 与 STDrug 可扩展候选发现，但不能替代真实药物扰动。

按这一标准，当前项目应被定义为：**工程模块较完整、科学证据处于 audit-stage、候选框架/评价创新可辨识，但核心算法创新和外部机制有效性尚未证明的研究型原型。** 下一次里程碑不应是“更多模块已实现”，而应是“在预注册外部干预 benchmark 上，至少一个核心模块稳定优于简单与强基线，并改变可复核的生物学决策”。

## 13. 主要主张—证据映射

| 主张 | 证据 | 状态 |
|---|---|---|
| 当前工作树测试逻辑通过，但默认安装/环境不可复现 | 283 tests with `PYTHONPATH=.`；默认 pytest 19 collection errors；validate_env exit 1 | **支持**，本地实测 |
| v3/celcomen-energy 不应 promotion | target-rank delta 0；energy AUC 增益 0.0021；alignment Spearman 0.0095，CI 跨 0 | **支持**，本地 benchmark |
| 因果模块需要真实干预基准 | CausalBench 对复杂因果方法与简单干预方法的比较 | **支持**，同行评议 |
| 空间传播可获得 own/neighbor 外部真值 | Spatial Perturb-Seq、Perturb-map | **支持**，同行评议 |
| targeted panel 未检出不能视为无效应 | Xenium 247-gene panel 仅 9/18 perturbations 检出 upregulated DEG | **支持**，单一研究，需跨组织复现 |
| SOAR/STDrug 不能单独证明空间药物机制 | 二者主体分别为资源/候选生成与计算排序，缺同一组织内直接药物空间干预真值 | **支持**，方法学推论 |
| 增加模型可能无净收益 | 本项目 target ranking 不变；近期综述和基准强调简单基线与外推风险 | **支持**，本地+外部收敛 |
| 完整 Task C/S/D 证据门控平台在本次检索中处于相对安静区 | 15 个 Bear 查询、255 条返回、22 条入选证据；直接撞车 0、方法孪生 3、问题孪生 6 | **有限支持**，受查询与索引范围约束 |
| 空间反事实、传播与药物排序的模型层已拥挤 | Counterfactual Diffusion、Celcomen、DynPerturb、SpatialProp、CONCERT、SpaRx、STDrug | **支持**，含预印本，成熟度不一 |
| 当前可定位为候选框架/评价创新 | 统一任务、证据等级、artifact/promotion contract；最近工作多为分散任务 | **候选主张**，尚需端到端演示和评价有效性验证 |
| HyperSCA 核心算法优于现有方法 | 尚无同切分、同预算外部比较；target-rank delta=0 | **不支持**，不得写 superiority/SOTA |

## 14. 自审与修订记录

### 14.1 五维对抗性自审

| 维度 | 问题 | 结论 |
|---|---|---|
| 贡献 | 是否给出了超过论文列表的项目决策？ | 部分通过：形成三任务框架、配对比较和 promotion gates；算法贡献仍需新实验 |
| 写作清晰度 | 是否区分实现、合并、可复现与科学验证？ | 通过：四者分别列示 |
| 实证强度 | 是否把代理指标或预印本当作确定结论？ | 写作通过；科学状态为需新实验：代理和预印本均已降级 |
| 评价完整性 | 是否包含简单基线、负控、外部 holdout 和失败案例？ | 设计通过、执行未完成：均已进入 P0-P2 gate 和比较矩阵 |
| 方法合理性 | 新增复杂度是否有退出机制？ | 设计通过：统一 adapter，外部 benchmark 不通过则不进入主运行时 |

### 14.2 修订记录

| # | 严重度 | 初稿风险 | 修订 |
|---|---|---|---|
| 1 | Major | 容易把 celcomen-inspired endpoint 与完整 Celcomen 混同 | 明确模型、可识别假设和本地 proxy 的边界 |
| 2 | Major | 可能将 STDrug 作为可立即集成仓库 | 核验到 GitHub 404，降为阻塞并保留 PyPI/预印本证据 |
| 3 | Major | 可能以内部 AUC 改善代表靶点收益 | 将 target-rank delta、外部干预与预注册 gate 设为必要证据 |
| 4 | Minor | 旧报告测试状态与当前不一致 | 同时报告历史 281/2 与当前 283 pass，并要求版本绑定 |
| 5 | Minor | “最近研究”造成时间和方法偏斜 | 显式给出 distributional skew advisory |
| 6 | Major | “没有直接撞车”可能被误写成绝对新颖 | 增加 15-query Bear 压力测试、撞车分层与查询敏感性限制 |
| 7 | Major | 升级路线可能继续堆模型而无公平比较 | 建立逐任务升级—比较矩阵，固定切分、特征、预算、主终点与退出条件 |
| 8 | Major | 框架创新与算法优越性容易混同 | 新增创新主张登记表；算法 superiority 标记为未证明 |
| 9 | Major | 可靠性只作为附加分析 | 将外部 holdout、多 seed CI、负控、校准、失败案例和成本纳入 promotion gate |

**未解决问题：** 外部仓库尚未在 HyperSCA 环境实际运行；CausalBench 与 Spatial Perturb-Seq 的完整数据下载成本、GPU/CPU 预算和许可证组合需在实施计划中单独核验。

## 15. 伦理、可复现性与 AI 披露

**伦理审查结论：CLEARED（计算研究范围）。** 本报告未处理个人可识别数据、未执行患者级分析、未提供临床用药建议，也未包含显著降低生物伤害门槛的操作细节。未来若纳入患者级空间数据或开展药物实验，应由项目团队根据数据来源、机构要求和研究设计重新判断 IRB/伦理审查与隐私义务。

**利益冲突：** 本报告未发现项目方披露的商业利益。部分外部论文作者与企业存在任职或咨询关系，已发表论文的 competing interests 应在正式方法选择时单独复核；本报告不以作者机构或 GitHub 热度调整方法结论。

**可复现性声明：** 检索日期、关键词、纳入标准、本地命令、项目文档和外部链接均已列出。外部 GitHub 元数据为 2026-08-10 快照；仓库状态可能变化。Bear 附录保留 15 个查询的 manifest、原始 JSON/BibTeX、22 条完整证据记录、比较矩阵和创新主张登记表。建议实施时记录 repository URL、commit SHA、license、environment lock、data checksum 和运行 manifest。

**AI Disclosure：** 本报告由 OpenAI Codex（2026-08-10 会话）辅助完成，用于本地仓库检查、GitHub 元数据检索、文献搜索、证据综合、Markdown 起草与自审；新颖性压力测试使用本机 SciMaster CLI 0.3.15 和 Bear `bear-propose`，执行 6 次 `ultra_low` 与 9 次 `low` 查询，未使用 `high` 模式。AI 未替代人类对科学结论、外部代码执行、临床意义或实验设计的最终判断；引用均链接至可核验的论文、预印本或公开仓库，但项目团队仍应在正式研究/投稿前人工阅读原文并复核数据与许可证。

## 参考文献

Buendia, A., Brunet, A., & Zou, J. (2025). *SpatialProp: Tissue perturbation modeling with spatially resolved single-cell transcriptomics* [Preprint]. https://doi.org/10.64898/2025.11.30.691355

Chevalley, M., Roohani, Y. H., Mehrjou, A., Leskovec, J., & Schwab, P. (2025). A large-scale benchmark for network inference from single-cell perturbation data. *Communications Biology, 8*, 412. https://doi.org/10.1038/s42003-025-07764-y

Dhainaut, M., Rose, S. A., Akturk, G., et al. (2022). Spatial CRISPR genomics identifies regulators of the tumor microenvironment. *Cell, 185*, 1223-1239.e20. https://doi.org/10.1016/j.cell.2022.02.015

Dimitrov, D., Schrod, S., Rohbeck, M., & Stegle, O. (2026). Interpretation, extrapolation and perturbation of single cells. *Nature Reviews Genetics, 27*, 349-370. https://doi.org/10.1038/s41576-025-00920-4

Dini, A., Barker, H., Piki, E., et al. (2025). A multiplex single-cell RNA-Seq pharmacotranscriptomics pipeline for drug discovery. *Nature Chemical Biology, 21*, 432-442. https://doi.org/10.1038/s41589-024-01761-8

Dong, M., Wang, B., Wei, J., et al. (2023). Causal identification of single-cell experimental perturbation effects with CINEMA-OT. *Nature Methods, 20*, 1769-1779. https://doi.org/10.1038/s41592-023-02040-5

Gao, Y., Dong, K., Shan, C., et al. (2025). Causal disentanglement for single-cell representations and controllable counterfactual generation. *Nature Communications, 16*, 6775. https://doi.org/10.1038/s41467-025-62008-1

Heumos, L., Ji, Y., May, L., et al. (2026). Pertpy: An end-to-end framework for perturbation analysis. *Nature Methods, 23*, 350-359. https://doi.org/10.1038/s41592-025-02909-7

Lin, X., Kong, Z., Ghosh, S., Kellis, M., & Zitnik, M. (2025). *CONCERT predicts niche-aware perturbation responses in spatial transcriptomics* [Preprint]. https://doi.org/10.1101/2025.11.08.686890

Mao, H., Jia, M., Di, M., et al. (2025). HALO: Hierarchical causal modeling for single cell multi-omics data. *Nature Communications, 16*, 8892. https://doi.org/10.1038/s41467-025-63921-1

Megas, S., Chen, D., Polanski, K., Eliasof, M., Schönlieb, C.-B., & Teichmann, S. (2025). Estimation of single-cell and tissue perturbation effect in spatial transcriptomics via spatial causal disentanglement. *International Conference on Learning Representations*. https://proceedings.iclr.cc/paper_files/paper/2025/hash/f8d8d09728100b1947d6add8ec625d56-Abstract-Conference.html

Shen, K., Seow, W. Y., Keng, C. T., et al. (2026). Spatial Perturb-Seq: Single-cell functional genomics within intact tissue architecture. *Nature Communications, 17*, 3018. https://doi.org/10.1038/s41467-026-69677-6

Tejada-Lapuerta, A., Bertin, P., Bauer, S., et al. (2025). Causal machine learning for single-cell genomics. *Nature Genetics, 57*, 797-808. https://doi.org/10.1038/s41588-025-02124-2

Yang, Y., Unjitwattana, T., Zhou, S., et al. (2026). *STDrug enables spatially informed personalized drug repurposing from spatial transcriptomics* [Preprint]. https://doi.org/10.64898/2026.04.03.715101

Li, Y., Ding, Y., Dennis, S., et al. (2025). SOAR elucidates biological insights and empowers drug discovery through spatial transcriptomics. *Science Advances, 11*, eadt7450. https://doi.org/10.1126/sciadv.adt7450

Bendidi, I., Whitfield, S., Kenyon-Dean, K., et al. (2024). *Benchmarking transcriptomics foundation models for perturbation analysis: One PCA still rules them all* [Preprint]. https://doi.org/10.48550/arXiv.2410.13956

Ding, W., Luo, Z., & Xiong, Y. (2026). Counterfactual diffusion modeling enables spatially targeted reprogramming of tissue microenvironments. *Biology*. https://doi.org/10.3390/biology15141097

Herbert, W. G., Chia, N., Jensen, P. A., & Walther-Antonio, M. R. S. (2025). *Monotherapy cancer drug-blind response prediction is limited to intraclass generalization* [Preprint]. https://doi.org/10.1101/2025.06.16.659838

Qi, S.-a., & Chapfuwa, P. (2026). *Mechanisms matter: Transportability of cellular perturbation effects* [Preprint]. https://doi.org/10.64898/2026.05.08.723625

Qin, H., Zhang, Y., Guo, Z., et al. (2025). *DynPerturb: Dynamic perturbation modeling for spatiotemporal single-cell systems* [Preprint]. https://doi.org/10.1101/2025.09.15.676236

Sun, E. D., Ma, R., Navarro Negredo, P., Brunet, A., & Zou, J. (2024). TISSUE: Uncertainty-calibrated prediction of single-cell spatial transcriptomics improves downstream analyses. *Nature Methods*. https://doi.org/10.1038/s41592-024-02184-y

Tang, Z., Liu, X., Li, Z., et al. (2023). SpaRx: Elucidate single-cell spatial heterogeneity of drug responses for personalized treatment. *Briefings in Bioinformatics*. https://doi.org/10.1093/bib/bbad338

Tec, M., Trisovic, A., Audirac, M., et al. (2023). *SpaCE: The spatial confounding environment* [Preprint]. https://doi.org/10.48550/arXiv.2312.00710

Vo, T., Cui, C. S., Benedicto, A., et al. (2026). *Spatial pharmaco-multiomics reveals drug distribution, metabolic niches, and spatially constrained resistance in medulloblastoma* [Preprint]. https://doi.org/10.64898/2026.01.25.701559

Wu, Y., Wershof, E., Schmon, S. M., et al. (2024). *PerturBench: Benchmarking machine learning models for cellular perturbation analysis* [Preprint]. https://doi.org/10.48550/arXiv.2408.10609

---

**篇幅：** 36,349 个字符、3,454 个空白分词（含表格与参考文献）。
**报告版本：** v1.1-bear-enhanced
