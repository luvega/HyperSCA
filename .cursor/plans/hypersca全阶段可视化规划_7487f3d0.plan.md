---
name: HyperSCA全阶段可视化规划
overview: 基于当前 HyperSCA 文档、示例结果和参考项目，先明确“当前数据是否足够”的结论，再给出覆盖 Step1/2/3 的可视化体系、数据补齐优先级与落地里程碑。计划强调先用现有数据构建可视化底座，再随表达矩阵与先验知识补齐逐步升级到因果与反事实层。
todos:
  - id: gap-audit
    content: 整理并冻结“可视化所需最小数据规范”（AnnData 字段、空间坐标、注释、真值边集）并与现有 data/ 差异对照。
    status: in_progress
  - id: phase0-template
    content: 基于 src/examples 扩展 Phase0 图模板（层级细胞图、空间着色图、分割质量图、panel 组成图），形成统一风格。
    status: pending
  - id: step1-vis-spec
    content: 定义 Step1 双曲嵌入图与基线对照图规范，明确输入输出与评估图联动。
    status: pending
  - id: step2-vis-spec
    content: 定义 Step2 因果图与多层信号流图规范，包含边权编码、关键轴证据卡和可追溯导出格式。
    status: pending
  - id: step3-vis-spec
    content: 定义 Step3 反事实与空间传播图规范，串联表达变化、空间热图、传播深度/梯度指标。
    status: pending
  - id: dashboard-prototype
    content: 在 notebooks 构建跨阶段 Dashboard 原型，支持样本筛选与图间联动浏览。
    status: pending
isProject: false
---

# HyperSCA 全阶段可视化计划

## 现状判断（数据是否足够）

- 结论：**当前数据对“全阶段设计目标”仍不足**，但足以支撑可视化底座和基线图谱。
- 现有可直接使用的数据主要是元数据与空间结构信息（`Chromium` 元数据、`Visium` 坐标、`VisiumHD` 分割、`Xenium` panel），可见 [docs/examples_guide.md](docs/examples_guide.md) 与 [src/data/loaders.py](src/data/loaders.py)。
- 当前 `examples` 已验证 4/4 通过，仅覆盖 QC/空间图/分割统计/panel 摘要，见 [results/examples/run_log.txt](results/examples/run_log.txt)。
- 但 Step1/2/3 需要的关键输入（表达矩阵 `.h5ad`、细胞状态建模输入、已知信号轴真值边集、可用于反事实评估的对照数据）在当前仓库尚未形成完整闭环，目标要求见 [docs/technical_roadmap.md](docs/technical_roadmap.md)、[docs/engineering_blueprint.md](docs/engineering_blueprint.md)、[docs/evaluation_suite.md](docs/evaluation_suite.md)。

## 设计目标

- 构建一条统一视觉叙事链：**单细胞参考图谱 → 空间转录组结构 → 靶点空间分布 → 靶点间相互作用与信息流 → 反事实与空间传播**。
- 每一阶段同时产出：
  - 解释型主图（面向科研叙事）
  - 评估型指标图（面向模型验收）
  - 可复用图函数（面向工程流水线）

## 分阶段可视化路线

### Phase 0（立即执行）：数据基线与可视化底座

- 目标：把已有 Example 产物升级为“可复用模板库”。
- 图型：
  - 细胞类型层级图（Level1/Level2 sunburst 或嵌套条形）
  - 空间图增强版（按病人/组织区/潜在细胞群着色）
  - 分割质量图（面积分布 + 核质比）
  - Xenium panel 组成图（source/descriptor 构成）
- 依托文件：`src/examples/*.py`、[docs/examples_guide.md](docs/examples_guide.md)。

### Phase 1（Step1）：单细胞参考图谱与双曲嵌入

- 目标：支持 Poincare/Lorentz 嵌入可视化与分支结构验证。
- 主图：
  - 双曲嵌入图（Poincare 圆盘 2D + 可选 3D 投影）
  - 细胞状态分支与径向梯度图（验证 H1.3）
  - 欧氏 baseline 对照图（PCA/UMAP vs Hyperbolic）
- 指标图：Distortion、ARI/NMI、Silhouette、Branch Purity。
- 对齐文档：[docs/technical_roadmap.md](docs/technical_roadmap.md)、[docs/evaluation_suite.md](docs/evaluation_suite.md)。
- 参考范式：`scDHMap`（双曲嵌入/分支）、`TopoLa`（拓扑增强）。

### Phase 2（Step2）：因果图与多层信息流

- 目标：把“靶点间相互作用和信息传递”可视化成可审查网络证据。
- 主图：
  - 因果有向图（边粗细=bootstrap 频率，颜色=arrow strength）
  - 多层信号流图（Ligand→Receptor→TF→Target，Sankey/分层网络）
  - 关键轴证据卡（如 CAF→TAM/Treg）
- 指标图：Falsification p-value、Graph Sparsity、Known Axis Recovery、Direction Accuracy。
- 对齐文档：[docs/engineering_blueprint.md](docs/engineering_blueprint.md)、[docs/evaluation_suite.md](docs/evaluation_suite.md)。
- 参考范式：`FlowSig`（flow network）、`DoWhy`（因果图/邻接热图）。

### Phase 3（Step3）：反事实扰动与空间传播

- 目标：展示“干预后会发生什么”及其空间扩散路径。
- 主图：
  - 干预前后表达对比（目标基因与 marker）
  - 反事实空间热图（局部与远端效应）
  - 传播深度与梯度衰减图（BFS 分层 + 距离衰减）
- 指标图：R²/PCC/MSE、Marker Direction、Moran’s I 变化、Propagation Depth。
- 对齐文档：[docs/technical_roadmap.md](docs/technical_roadmap.md)、[docs/evaluation_suite.md](docs/evaluation_suite.md)。
- 参考范式：`CPA`/`scgen`（扰动评估）、`DynPerturb`（时空传播）。

### Phase 4（跨阶段整合）：综合 Dashboard

- 目标：一页串联 Step1→Step2→Step3 的证据链与结论。
- 组件：
  - 左侧：样本/细胞群筛选
  - 中部：嵌入/空间/因果网络联动
  - 右侧：干预模拟与关键指标
- 承接位置：`notebooks/`（后续可迁移 Streamlit），对应 [docs/engineering_blueprint.md](docs/engineering_blueprint.md) 的 P2.7。

## 数据补齐优先级（决定计划上限）

- P0（必须先补）：
  - `scRNA/ST` 表达矩阵与标准化 `AnnData`（含 cell/spot 注释）
  - 跨模态映射字段（cell type、patient、region）
- P1（Step2 必需）：
  - 配体-受体数据库版本冻结（CellChatDB/CellPhoneDB/LIANA）4
  - 已知信号轴真值边集（CSV/JSON）用于 Known Axis Recovery
- P2（Step3 强烈建议）：
  - 扰动相关先验或外部验证数据（文献 marker 集、公开 perturbation 结果）

## 产物组织建议

- 输出目录建议：
  - `results/figures/step1/`
  - `results/figures/step2/`
  - `results/figures/step3/`
  - `results/figures/dashboard/`
- 统一元信息：每张图附 `config + data_version + model_version + seed`，保障可复现。

## 里程碑（建议）

- M1（1 周）：完成 Phase0 基线图模板与统一绘图风格。
- M2（2-3 周）：完成 Step1/Step2 主图与指标图原型。
- M3（4-6 周）：接入 Step3 空间传播可视化并形成 Dashboard 初版。

## 可视化总流程

```mermaid
flowchart LR
    dataBaseline["DataBaselineExamples"] --> step1Vis["Step1_HyperbolicAtlas"]
    step1Vis --> step2Vis["Step2_CausalFlow"]
    step2Vis --> step3Vis["Step3_CounterfactualPropagation"]
    step3Vis --> dashboardVis["IntegratedDashboard"]
```



## 参考资源

- OSTA（空间组学分析与可视化组织框架）：[https://lmweber.org/OSTA/](https://lmweber.org/OSTA/)
- 项目内参考实现：`references/scDHMap/`, `references/TopoLa/`, `references/flowsig/`, `references/dowhy/`, `references/CPA/`, `references/scgen/`, `references/DynPerturb/`。

