# HyperSCA 执行优先级与风险预案

*版本: v1.0 | 日期: 2026-02-15*

---

## 1. 总体执行节奏

```text
第 1 轮 (Week 0)     ── 文档冻结 ──────────────────────────────────────
  ✓ 学术化技术路线文档 (docs/technical_roadmap.md) — 冻结术语与符号
  ✓ 工程落地蓝图 (docs/engineering_blueprint.md) — 冻结 P0 接口与验收标准
  ✓ 评估指标体系 (docs/evaluation_suite.md) — 冻结指标定义与优先级

第 2 轮 (Week 1-4)   ── P0 基础端到端 ──────────────────────────────────
  数据加载 → 空间图 → H-VAE → 因果解缠 → 基础因果图 → 单基因 KO

第 3 轮 (Week 5-8)   ── P1 验证与精炼 ──────────────────────────────────
  DoWhy 验证 → CMI 精炼 → 信号流 → 全套评估 → 多基因 KO → 可视化

第 4 轮 (Week 9-14)  ── P2 高级扩展 ──────────────────────────────────
  扩散反事实 → 空间传播 → 远端 EMT 联动 → 综合 Dashboard
```

---

## 2. P0 任务详细优先级排序

按**依赖拓扑序 + 风险前置**排列，高风险任务提前以便预留缓冲：

| 序号 | 任务 | 预计工时 | 依赖 | 风险等级 | 说明 |
|------|------|----------|------|----------|------|
| P0.1 | `src/data/loaders.py` — 数据加载 | 0.5 天 | 无 | 低 | scanpy I/O 成熟 |
| P0.2 | `src/data/preprocessing.py` — 预处理 | 1 天 | P0.1 | 低 | 参考 scDHMap preprocess |
| P0.3 | `src/data/spatial_graph.py` — 空间图 + TopoLa | 1 天 | P0.2 | 低 | SVD 变换简单，核心 ~30 行 |
| P0.5 | `src/models/hyperbolic/lorentz.py` 等工具 | 1.5 天 | 无 | **中** | ExpMap/LogMap 数值稳定性需仔细处理 |
| P0.4 | `src/models/hyperbolic/hvae.py` — H-VAE | 3 天 | P0.3, P0.5 | **高** | 核心模型，需适配弃用 API + geoopt 集成 |
| P0.6 | `src/causal/disentangle.py` — 因果解缠 | 2 天 | P0.4 | **中** | G2G 层 + HSIC 损失 |
| P0.7 | `src/causal/cmi_pruning.py` — 基础因果图 | 2 天 | P0.6 | **中** | UT-IGSP 调用 + bootstrap |
| P0.8 | `src/perturbation/latent_arithmetic.py` — 单基因 KO | 1.5 天 | P0.4 | 低 | 双曲 PT 为关键点 |
| P0.9 | `scripts/run_step*.py` — 运行脚本 | 1 天 | P0.1-P0.8 | 低 | CLI 封装 |

**关键路径**: P0.1 → P0.2 → P0.3 → P0.4 → P0.6 → P0.7（最长依赖链 ~10 天）

**建议并行化**:
- P0.5（双曲工具）与 P0.1-P0.3（数据管线）并行开发
- P0.8（扰动）在 P0.4 完成后可与 P0.6-P0.7 并行

---

## 3. 风险登记表

### R1：双曲空间数值溢出

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.4 / P0.5（H-VAE 训练） |
| **风险描述** | 高曲率区域的 ExpMap/LogMap 运算导致 NaN / Inf，尤其当嵌入点接近 Poincaré 球边界（norm → 1）时 |
| **概率** | 高（几乎必然遇到） |
| **影响** | 训练失败或损失爆炸 |
| **预案** | 1) 使用 `geoopt` 内置的 `math.artanh_clamp` / `math.tanh_clamp` 进行数值截断；2) 限制 Poincaré 范数 < 1 - epsilon（epsilon = 1e-5）；3) 采用 Lorentz 模型（数值更稳定）作为计算主模型，仅在可视化时转 Poincaré；4) 使用 float64 进行关键运算 |
| **监控信号** | 训练过程中 `max(norm(z))` 趋势；loss 突然跳变 |

### R2：scDHMap PyTorch 2.6 兼容性

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.4（H-VAE 模型适配） |
| **风险描述** | scDHMap 使用 `torch.autograd.Variable`（已弃用）和旧版 API |
| **概率** | 中（功能可用但有 DeprecationWarning） |
| **影响** | 运行时警告；极端情况下未来版本可能移除 |
| **预案** | 重写时直接使用 `torch.Tensor`，不 import scDHMap 原代码；通过 adapter 模式封装核心逻辑 |

### R3：因果可识别性不足

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.7 / P1.1（因果图学习与验证） |
| **风险描述** | SCM 的 faithfulness / causal sufficiency 假设在 TME 中可能不成立（未观测混杂、selection bias） |
| **概率** | 中 |
| **影响** | 因果图包含错误边或遗漏真实边 |
| **预案** | 1) P0 阶段使用保守阈值（$\tau = 0.7$）仅保留高置信边；2) P1 引入 DoWhy falsification 多重检验（局部 Markov、conditional independence）；3) 对无法识别方向的边标记为"未定向"而非强制定向；4) 引入 LR-database 先验作为软约束，辅助定向 |

### R4：FlowSig / UT-IGSP 依赖兼容

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.7（CMI 剪枝） |
| **风险描述** | FlowSig 依赖的 `causaldag` / `utigsp` 包可能与当前 Python 3.10 + 最新 numpy/scipy 版本冲突 |
| **概率** | 中 |
| **影响** | import 失败或运行时错误 |
| **预案** | 1) 优先检查 `causaldag` 是否可安装（`pip install causaldag`）；2) 若不兼容，提取 UT-IGSP 核心算法手动实现（~200 行）；3) 备选方案：使用 `pgmpy`（已安装）的 PC / GES 算法替代 |

### R5：反事实外推可靠性

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.8 / P2.1（扰动预测） |
| **风险描述** | 纯观测数据训练的模型在外推（out-of-distribution）干预条件下预测不可靠 |
| **概率** | 高 |
| **影响** | 反事实预测在强扰动下偏差增大 |
| **预案** | 1) 限制扰动幅度——首先验证温和扰动（如 50% 下调而非完全 KO）的生物学一致性；2) 引入 CPA 的不确定性量化，对高不确定性预测标记警告；3) P2 阶段引入扩散模型提供更强的非线性建模能力；4) 对关键预测交叉验证（CPA vs scGen vs Diffusion，取共识） |

### R6：计算资源瓶颈

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.4（H-VAE 训练）、P0.7（Bootstrap x100）、P2.1（Diffusion 采样） |
| **风险描述** | RTX 3070 (8GB VRAM) 可能不足以处理大规模 ST 数据 + 高维扩散模型 |
| **概率** | 中 |
| **影响** | OOM 或训练时间过长 |
| **预案** | 1) 数据分块：ST 数据按空间区域分 patch 训练；2) 混合精度训练（AMP）；3) Bootstrap 可 CPU 并行（CMI 计算不需 GPU）；4) P2 扩散模型使用低维潜空间（latent diffusion）而非全基因空间 |

### R7：数据质量与格式问题

| 属性 | 内容 |
|------|------|
| **影响阶段** | P0.1-P0.2（数据加载） |
| **风险描述** | 不同平台（Chromium / Visium / VisiumHD / Xenium）数据格式差异，可能存在空间坐标缺失、基因名不一致等问题 |
| **概率** | 中 |
| **影响** | 预处理流水线需要针对性适配 |
| **预案** | 1) P0 先仅支持 Visium 格式，后续按需扩展；2) 使用 scanpy / squidpy 标准读取接口；3) 添加数据完整性 assertion（坐标非空、基因名唯一、无全零行/列） |

---

## 4. 里程碑验收清单（总览）

### P0 验收 Gate（Week 4 末）

- [ ] `python scripts/run_step1.py --config default` 成功输出嵌入（results/step1/）
- [ ] `python scripts/run_step2.py --config default` 成功输出因果图（results/step2/）
- [ ] `python scripts/run_step3.py --config default --gene INHBA` 成功输出反事实
- [ ] Poincaré 嵌入可视化呈现合理的细胞类型分离
- [ ] 因果图非空且稀疏度 < 0.1
- [ ] INHBA KO 后 CD163 表达方向正确（下调）
- [ ] `pytest tests/` 全部通过
- [ ] 无未处理的 NaN / Inf

### P1 验收 Gate（Week 8 末）

- [ ] DoWhy falsification p > 0.05（至少 2/3 检验通过）
- [ ] 至少识别 1 条已知信号轴（如 CAF→TAM）
- [ ] 至少推断 1 条完整 4 层信号流
- [ ] 评估报告 `results/evaluation_report.json` 自动生成
- [ ] 嵌入 ARI > 0.5
- [ ] 反事实 R²(mean) > 0.7
- [ ] 可视化完整（嵌入图 + 因果网络图 + 空间热图）

### P2 验收 Gate（Week 14 末）

- [ ] 扩散反事实可运行且与 latent arithmetic 在 top-50 marker 方向 PCC > 0.7
- [ ] 空间传播 3-hop 内收敛
- [ ] 综合 notebook 可一键运行
- [ ] 综合评估报告包含全部 3 阶段指标

---

## 5. 决策点与备选路径

### D1：双曲几何模型选择

- **默认路径**: Lorentz 模型（数值稳定） + Poincaré 球（可视化）
- **备选路径**: 若 geoopt Lorentz 支持不足，退回纯 Poincaré 球模型（scDHMap 原生支持）
- **决策时机**: P0.5 完成时

### D2：因果发现算法

- **默认路径**: UT-IGSP（FlowSig 验证过的空间因果算法）
- **备选路径 A**: PC 算法 + 空间约束（pgmpy 已安装）
- **备选路径 B**: NOTEARS + DAG 约束（连续优化，GPU 友好）
- **决策时机**: P0.7 依赖安装检查时

### D3：扰动预测主方法

- **默认路径**: Latent arithmetic（CPA 风格，简单高效）
- **备选路径**: scGen（跨细胞类型泛化更强，但需足够 batch 多样性）
- **决策时机**: P0.8 首次预测结果评估时

### D4：扩散模型架构（P2）

- **默认路径**: Latent Diffusion（在低维潜空间扩散，适配 RTX 3070 显存）
- **备选路径**: 全基因空间 DDPM（表达力更强，但需更大 GPU / 分块训练）
- **决策时机**: P2.1 初步实验时

---

## 6. 沟通与文档维护

| 节点 | 动作 |
|------|------|
| 每个子任务完成 | 更新 `SETUP_PROGRESS_*.md` 记录 |
| 每个 P 阶段验收 | 生成 `results/evaluation_report.json` + 简要总结 |
| 重大技术决策 | 在 `docs/decisions/` 记录 ADR (Architecture Decision Record) |
| 术语/符号变更 | 同步更新 `docs/technical_roadmap.md` 符号表 |
| 依赖变更 | 更新 `requirements-core.txt` 或 `requirements-research.txt` + `pip freeze` 归档 |
