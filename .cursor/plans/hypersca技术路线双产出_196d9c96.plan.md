---
name: HyperSCA技术路线双产出
overview: 基于你现有的三阶段 HyperSCA 草案，输出两套互补交付物：一套用于开题/论文的方法学文档，一套用于研发落地的工程实施计划（含里程碑、接口、评估与风险控制）。计划将严格对齐你当前仓库结构与已安装依赖环境。
todos:
  - id: draft-academic-doc
    content: 产出学术化技术路线文档（术语统一、公式对象统一、三阶段细化）
    status: completed
  - id: draft-engineering-blueprint
    content: 产出工程落地蓝图（模块目录、接口、P0/P1/P2里程碑与验收）
    status: completed
  - id: define-evaluation-suite
    content: 固化评估指标清单（因果可信度、反事实质量、空间传播一致性）
    status: completed
  - id: finalize-priority-roadmap
    content: 给出可执行优先级与风险预案（先文档冻结再实现）
    status: completed
isProject: false
---

# HyperSCA 技术路线双产出计划

## 目标与范围

- 基于现有草案文件 [e:/HyperSCA/HyperSCA技术路线-v0.1.md](e:/HyperSCA/HyperSCA技术路线-v0.1.md)，产出两份内容：
  - 学术化技术路线文档（中文为主，保留关键 English terms）
  - 工程落地计划（模块拆解、里程碑、评估指标、风险与验收）
- 与项目现状保持一致：主仓库说明 [e:/HyperSCA/HyperSCA/README.md](e:/HyperSCA/HyperSCA/README.md)，环境与依赖基线 [e:/HyperSCA/SETUP_PROGRESS_2026-02-14.md](e:/HyperSCA/SETUP_PROGRESS_2026-02-14.md)。

## 已完成调研要点（用于指导产出）

- 第一步（Embedding）可优先复用：`TopoLa`（拓扑增强邻接）+ `scDHMap`（双曲 VAE/双曲距离工具）。
- 第二、三步（Causal + Counterfactual）建议主干：`FlowSig` 做空间因果结构学习，`DoWhy` 做识别与可证伪验证，`CPA/scGen` 做扰动预测，`CausCell/Squidiff/DynPerturb` 作为增强层。
- 当前环境已满足 P0/P1 的主要依赖（PyTorch、scanpy、geoopt、dowhy、scgen、diffusers 等）。

## 交付物 A：学术化技术路线文档

- 重写结构为“问题定义 → 方法总览 → 三阶段技术细节 → 可证伪假设与生物学解释 → 方法优势与局限”。
- 每一阶段统一模板：
  - 输入/输出定义（Input/Output）
  - 核心算法与数学对象（如 Lorentz/Poincaré、SCM、CMI）
  - 与参考框架的关系（TopoLa/scDHMap/FlowSig/CPA 等）
  - 可验证生物学命题（如 CAF→TAM/Treg 轴）
- 增补术语规范与符号一致性（例如 `Z_int`, `Z_ext`, `do(POSTN=low)`）。

## 交付物 B：工程落地计划

- 规划目标目录（建议，不改 `references/`）：
  - `src/data`（多模态读入与空间图）
  - `src/models/hyperbolic`（双曲嵌入）
  - `src/causal`（SCM/CMI/网络剪枝）
  - `src/perturbation`（latent arithmetic + diffusion CF）
  - `src/pipeline`（端到端编排）
  - `scripts/`（step-wise 运行脚本）
- 分阶段实施：
  - P0：空间图 + 双曲嵌入 + 基础因果图 + 单基因虚拟敲除
  - P1：DoWhy 验证、CMI 稀疏化、多层 signaling flow、空间一致性评估
  - P2：扩散反事实、时空传播与远端 EMT/药敏联动模拟
- 统一接口草案（高层 API）：
  - `learn_spatial_causal_network()`
  - `generate_counterfactual()`
  - `evaluate_causal_and_spatial_consistency()`

## 评估与验收框架

- 因果边可信度：bootstrap 频率、falsification、arrow strength。
- 反事实质量：R2/PCC/MSE + marker-level consistency。
- 空间传播一致性：Moran's I、传播梯度衰减、空间距离-因果距离耦合性。
- 工程验收：每阶段均要求最小可运行脚本与固定输入样例可复现。

## 技术流程图（用于文档与研发对齐）

```mermaid
flowchart LR
    rawData[scRNA_ST_MultimodalData] --> graphBuild[SpatialNeighborGraph_TopoLa]
    graphBuild --> hyperVAE[HyperbolicVAE_LorentzPoincare]
    hyperVAE --> disentangle[CausalDisentanglement_Zint_Zext]
    disentangle --> cmiPrune[CMIPruning_CausalCellGraph]
    cmiPrune --> signaling[MultilayerSignalingFlow]
    signaling --> perturb[LatentSpacePerturbation_CPA_scGen]
    perturb --> diffCF[DiffusionCounterfactual_CausCell_Squidiff]
    diffCF --> spatialProp[SpatialPropagation_DynPerturb]
    spatialProp --> eval[Evaluation_Causal_CF_Spatial]
```



## 风险与预案

- 参考实现版本差异：先封装 adapter 层，避免直接耦合上游代码结构。
- 因果可识别性不足：增加先验图约束与独立性检验组合策略。
- 扰动结果生物学可解释性偏弱：引入 marker/pathway 双重验证与专家规则。

## 预计执行顺序

- 第 1 轮：完成文档版（A）并冻结术语与符号。
- 第 2 轮：完成工程版（B）并冻结 P0 接口与验收指标。
- 第 3 轮：按 P0→P1→P2 迭代实现与评估。

