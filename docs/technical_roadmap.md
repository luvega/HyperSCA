# HyperSCA 技术路线文档

**Hyperbolic Spatiotemporal Causal Analysis (HyperSCA)**
*版本: v1.0 | 日期: 2026-02-15*

---

## 符号与术语规范

| 符号 | 含义 |
|:-----|------|
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | 空间邻域图，节点集 $\mathcal{V}$（细胞/spot），边集 $\mathcal{E}$ |
| $\mathbf{A}$ | 邻接矩阵（原始），$\tilde{\mathbf{A}}$ 为 TopoLa 增强后的邻接矩阵 |
| $\mathbf{X} \in \mathbb{R}^{N \times G}$ | 基因表达矩阵，$N$ 个细胞，$G$ 个基因 |
| $\mathbf{S} \in \mathbb{R}^{N \times 2}$ | 空间坐标矩阵 |
| $\mathbb{H}^d_K$ | $d$ 维、曲率 $K < 0$ 的双曲空间 |
| $\mathbb{L}^d$ | Lorentz 模型（Hyperboloid model） |
| $\mathbb{B}^d_c$ | Poincaré 球模型，$c = -K$ |
| $\mathbf{z}_i \in \mathbb{H}^d_K$ | 细胞 $i$ 在双曲潜空间中的嵌入 |
| $\mathbf{z}^{\text{int}}_i$ | 细胞 $i$ 的内源性状态潜变量（intrinsic） |
| $\mathbf{z}^{\text{ext}}_i$ | 细胞 $i$ 的外源性微环境影响潜变量（extrinsic） |
| $\mathcal{N}(i)$ | 细胞 $i$ 的空间邻居集合 |
| $\text{CMI}(X; Y \mid Z)$ | 条件互信息（Conditional Mutual Information） |
| $do(\cdot)$ | Pearl 因果干预算子 |
| $\delta_{\text{pert}}$ | 扰动向量（perturbation vector） |
| $\text{NB}(\mu, \theta)$ | 负二项分布，均值 $\mu$，逆离散参数 $\theta$ |

---

## 1. 问题定义

### 1.1 生物学背景

结直肠癌（Colorectal Cancer, CRC）免疫治疗的应答率受限于复杂的肿瘤免疫微环境（Tumor Microenvironment, TME）。TME 中多种细胞亚群——特别是肿瘤相关成纤维细胞（Cancer-Associated Fibroblasts, CAFs）、肿瘤相关巨噬细胞（Tumor-Associated Macrophages, TAMs）、调节性 T 细胞（Tregs）——形成免疫抑制生态位（immunosuppressive niche），通过配受体信号网络协同构建物理与化学屏障，阻碍效应 T 细胞浸润与功能。

### 1.2 计算学挑战

1. **层级畸变**：欧氏空间无法忠实嵌入细胞谱系树（如 naive → effector → exhausted T 细胞），导致拓扑信息丢失。
2. **伪相关混淆**：空间共定位（spatial co-localization）不等于因果通讯；共享微环境可诱导虚假关联。
3. **干预不可及**：湿实验成本高、通量低，需要可靠的 in silico perturbation 工具预测多靶点干预后果。

### 1.3 总体目标

HyperSCA 旨在：

- 在双曲流形上忠实表征细胞发育层级与空间拓扑；
- 解缠（disentangle）细胞内源状态与外源微环境影响，构建定向因果通讯网络；
- 实现基因级虚拟敲除与空间传播模拟，为免疫治疗靶点筛选提供计算证据。

---

## 2. 方法总览

HyperSCA 框架分为三个递进阶段，每一阶段的输出构成下一阶段的输入：

```
阶段 1: 双曲流形嵌入与多模态数据融合
  输入: scRNA-seq (X) + ST 空间坐标 (S) + 基因表达 (X_st)
  输出: 双曲潜变量 z ∈ H^d_K, 增强邻接矩阵 Ã

阶段 2: 空间约束下的因果通讯网络构建
  输入: z, Ã, 配受体数据库 (CellChatDB / CellPhoneDB)
  输出: 因果细胞图 G_causal, 多层信号流

阶段 3: 微环境演化的扰动模拟与反事实预测
  输入: G_causal, 训练后的 H-VAE 或 Diffusion 模型, 靶基因列表
  输出: 反事实基因表达谱, 空间传播预测图
```

---

## 3. 阶段 1：双曲流形嵌入与多模态数据融合

### 3.1 概述

**目标**：克服欧氏空间在表征细胞发育层级（Hierarchies）和复杂拓扑结构时的畸变问题，构建保真度更高的低维潜在空间，整合 scRNA-seq 转录组特征与 ST 空间邻域信息。

**输入**：
- 基因表达矩阵 $\mathbf{X} \in \mathbb{R}^{N \times G}$（scRNA-seq 或 ST 计数矩阵）
- 空间坐标矩阵 $\mathbf{S} \in \mathbb{R}^{N \times 2}$（ST spots 的物理位置）

**输出**：
- 双曲嵌入 $\{\mathbf{z}_i\}_{i=1}^N \subset \mathbb{H}^d_K$
- TopoLa 增强邻接矩阵 $\tilde{\mathbf{A}}$

### 3.2 空间邻域图构建与拓扑增强

#### 3.2.1 基础图构建

利用 ST 数据的物理坐标 $\mathbf{S}$，以 k-nearest neighbors（$k = 6 \sim 15$）或 Delaunay 三角剖分构建空间邻域图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，初始邻接矩阵为 $\mathbf{A}$。

#### 3.2.2 TopoLa 拓扑编码增强

为避免仅基于距离的伪连接，引入 TopoLa（Topology-encoding distance）策略。其核心思想是通过加权偶数跳路径（even-hop paths）量化细胞间的几何结构相似性，从而将拓扑信息编码至邻接矩阵中。

具体而言，对原始邻接矩阵 $\mathbf{A}$ 进行奇异值分解（SVD）：

$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

对奇异值 $\sigma_j$ 施加非线性变换：

$$\tilde{\sigma}_j = \frac{\sigma_j^3}{\sigma_j^2 + \lambda^{-1}}$$

其中 $\lambda > 0$ 为正则化参数（默认 $\lambda = 10^{-3}$），控制对弱连接的惩罚程度。增强后的邻接矩阵为：

$$\tilde{\mathbf{A}} = \mathbf{U} \tilde{\boldsymbol{\Sigma}} \mathbf{V}^T$$

该变换增强高阶拓扑相似的边权、抑制拓扑不一致的伪连接。

**参考实现**：`TopoLa` 框架（`references/TopoLa/`），核心函数 `TopoLa(A, lambda_val)` 位于 `utils_TopoLa.py`。

### 3.3 双曲变分自编码器 (Hyperbolic VAE)

#### 3.3.1 双曲空间选择

采用 **Lorentz 模型** $\mathbb{L}^d = \{ \mathbf{x} \in \mathbb{R}^{d+1} : \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -1/K \}$ 作为潜空间的主计算模型（数值更稳定），通过微分同胚映射至 **Poincaré 球** $\mathbb{B}^d_c$ 用于可视化：

$$\phi_{\mathbb{L} \to \mathbb{B}}(\mathbf{x}) = \frac{(x_1, \dots, x_d)}{x_0 + 1/\sqrt{-K}}$$

双曲空间的关键性质——面积/体积随离原点距离指数增长——使其天然适合嵌入树状分支结构（如 T 细胞分化谱系），且低维即可保持低畸变。

#### 3.3.2 编码器 (Encoder)

编码器由两条通路组成：

1. **图卷积通路 (Graph Convolution Branch)**：使用 GCN 层处理增强邻接矩阵 $\tilde{\mathbf{A}}$ 和节点特征，提取多尺度空间上下文：
   $$\mathbf{H}^{(l+1)} = \sigma\left(\hat{\mathbf{D}}^{-1/2} \hat{\tilde{\mathbf{A}}} \hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$
   其中 $\hat{\tilde{\mathbf{A}}} = \tilde{\mathbf{A}} + \mathbf{I}$，$\hat{\mathbf{D}}$ 为度矩阵。

2. **表达通路 (Expression Branch)**：全连接层逐层降维基因表达 $\mathbf{X}$。

两条通路在低维层拼接后，通过指数映射（Exponential Map）投影至双曲空间：

$$\text{Exp}_{\mathbf{o}}(\mathbf{v}) = \cosh(\|\mathbf{v}\|_{\mathcal{L}}) \cdot \mathbf{o} + \sinh(\|\mathbf{v}\|_{\mathcal{L}}) \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}$$

其中 $\mathbf{o}$ 为 Lorentz 模型原点。编码器输出双曲潜变量的参数 $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i)$，采用 **Wrapped Normal** 分布 $\mathcal{W}\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 进行重参数化采样。

#### 3.3.3 解码器 (Decoder)

解码器从双曲潜变量 $\mathbf{z}_i$ 出发，先通过对数映射（Logarithmic Map）回到切空间，再经全连接层恢复基因表达谱。

采用负二项分布（Negative Binomial, NB）作为似然函数以适配 scRNA-seq 数据的稀疏性和过离散性：

$$p(x_{ig} \mid \mathbf{z}_i) = \text{NB}(x_{ig}; \mu_{ig}(\mathbf{z}_i), \theta_g)$$

其中 $\mu_{ig}$ 由解码器网络输出，$\theta_g$ 为基因特异性逆离散参数。

#### 3.3.4 损失函数

$$\mathcal{L}_{\text{H-VAE}} = \underbrace{-\mathbb{E}_{q(\mathbf{z}|\mathbf{X})}[\log p(\mathbf{X}|\mathbf{z})]}_{\text{NB 重建损失}} + \beta \cdot \underbrace{D_{\text{KL}}(q(\mathbf{z}|\mathbf{X}) \| p(\mathbf{z}))}_{\text{双曲 KL 散度}} + \gamma \cdot \underbrace{\mathcal{L}_{\text{topo}}(\tilde{\mathbf{A}}, \mathbf{z})}_{\text{拓扑正则化}}$$

其中：
- 重建损失采用 NB 对数似然；
- KL 散度在双曲空间中计算（Wrapped Normal 先验 vs. 后验）；
- 拓扑正则化 $\mathcal{L}_{\text{topo}}$ 鼓励双曲空间中的距离结构与增强邻接矩阵 $\tilde{\mathbf{A}}$ 所描述的拓扑结构一致（如 Cauchy 核吸引力 + 斥力项）。

**参考实现**：`scDHMap` 框架（`references/scDHMap/`），核心类 `scDHMap`，双曲工具 `lorentzian_helper.py` / `wrapped_normal.py`。

### 3.4 可验证命题（阶段 1）

| 编号 | 命题 | 验证方式 |
|------|------|----------|
| H1.1 | 双曲嵌入在保持细胞谱系层级方面优于欧氏嵌入（如 PCA/UMAP） | 对比 Distortion Score ($\delta$-hyperbolicity) 与分支纯度 |
| H1.2 | TopoLa 增强的邻接矩阵去除伪连接后，下游聚类 ARI 优于原始 k-NN 图 | 以已知细胞类型注释计算 ARI/NMI |
| H1.3 | Poincaré 嵌入中，从 naive T → effector → exhausted T 的分化路径呈径向展开结构 | 可视化检查 + 径向梯度相关性 |

---

## 4. 阶段 2：空间约束下的因果通讯网络构建

### 4.1 概述

**目标**：在双曲潜空间中，剥离细胞内在调控与外在通讯信号，构建定向的因果通讯网络。

**输入**：
- 双曲嵌入 $\{\mathbf{z}_i\}$ 与增强邻接矩阵 $\tilde{\mathbf{A}}$（来自阶段 1）
- 配受体数据库（CellChatDB / CellPhoneDB / LIANA+）

**输出**：
- 因果细胞图 $\mathcal{G}_{\text{causal}}$：节点为细胞亚群或单细胞，有向边表示因果通讯
- 多层信号流：配体 → 受体 → 转录因子 → 靶基因

### 4.2 因果解缠 (Causal Disentanglement)

参考 Celcomen 算法，假设细胞 $i$ 的基因表达潜变量可分解为两个因果不可约分量：

$$\mathbf{z}_i = f(\mathbf{z}^{\text{int}}_i, \mathbf{z}^{\text{ext}}_i)$$

其中：
- $\mathbf{z}^{\text{int}}_i$：**内源性状态**，由细胞自身的转录调控程序决定（如细胞周期、基础代谢）；
- $\mathbf{z}^{\text{ext}}_i$：**外源性影响**，由空间邻居 $\mathcal{N}(i)$ 通过配受体相互作用因果决定。

建立结构因果模型（Structural Causal Model, SCM）：

$$\mathbf{z}^{\text{ext}}_i = g\left(\{\mathbf{z}_j\}_{j \in \mathcal{N}(i)}, \boldsymbol{\epsilon}_i\right)$$

其中 $g(\cdot)$ 为 GCN 消息传递函数（G2G layer），$\boldsymbol{\epsilon}_i$ 为外生噪声。

解缠的训练目标：最大化 $\mathbf{z}^{\text{int}}_i$ 与 $\mathbf{z}^{\text{ext}}_i$ 的条件独立性，同时要求两者联合可重建观测表达：

$$\mathcal{L}_{\text{disentangle}} = \mathcal{L}_{\text{recon}}(\mathbf{X} \mid \mathbf{z}^{\text{int}}, \mathbf{z}^{\text{ext}}) + \alpha \cdot \text{HSIC}(\mathbf{z}^{\text{int}}, \mathbf{z}^{\text{ext}})$$

其中 HSIC（Hilbert-Schmidt Independence Criterion）惩罚两个分量之间的统计依赖性。

**参考实现**：`Celcomen` 框架（`references/celcomen/`），核心模型类 `celcomen`，G2G 消息传递层。

### 4.3 条件互信息剪枝 (CMI Pruning)

为消除由共享微环境、batch effect 或混杂因素引起的伪因果边：

1. **计算 CMI**：对每对候选因果边 $(i, j)$，计算：
   $$\text{CMI}(\mathbf{z}^{\text{ext}}_i; \mathbf{z}_j \mid \mathbf{z}_{\mathcal{N}(i) \setminus j})$$
   即在给定所有其他邻居信息条件下，细胞 $j$ 对细胞 $i$ 外源性状态的独立贡献。

2. **Bootstrap 聚合**：采用 block bootstrap（尊重空间相关结构）重复 $B$ 次采样，每次运行约束因果发现算法（如 UT-IGSP / GSP），对所有 bootstrap 样本的因果有向无环图（CPDAG）进行边频率聚合：
   $$w_{ij} = \frac{1}{B}\sum_{b=1}^B \mathbb{1}[i \to j \in \hat{\mathcal{G}}^{(b)}]$$

3. **阈值剪枝**：剔除 $w_{ij} < \tau$（建议 $\tau = 0.5 \sim 0.8$）的边，保留稀疏且鲁棒的因果细胞图 $\mathcal{G}_{\text{causal}}$。

**参考实现**：`FlowSig` 框架（`references/flowsig/`），核心函数 `learn_intercellular_flows()`、`run_utigsp()`。

### 4.4 多层信号流推断 (Multilayer Signaling Flow)

在确立因果细胞图后，进一步解析因果边所承载的分子机制，构建完整信号流：

$$\text{配体 (Ligand)} \xrightarrow{\text{分泌}} \text{受体 (Receptor)} \xrightarrow{\text{转导}} \text{转录因子 (TF)} \xrightarrow{\text{调控}} \text{靶基因 (Target)}$$

具体步骤：

1. **配受体对筛选**：利用 CellChatDB / CellPhoneDB 数据库，在因果边的源与靶细胞亚群中识别显著高表达的配受体对。
2. **下游 TF 推断**：结合 NicheNet / DoRothEA 等工具的先验网络，推断受体激活后下游最可能被激活的转录因子。
3. **信号流强度量化**：整合配体表达水平、受体亲和力先验、TF 活性得分（如 SCENIC/AUCell）以及靶基因表达变化，为每条信号通路赋予定量的通量（flow score）。

**生物学示例**：
- **MFAP2** (CAF 分泌) → **Integrin** $\alpha 5 \beta 1$ (TAM 受体) → **FAK/Src** → 下游 EMT 转录程序
- **INHBA** (CAF 分泌) → **ACVR1B/ACVR2A** (Treg 受体) → **SMAD2/3** → Foxp3 转录调控

**参考实现**：`FlowSig`（`references/flowsig/`）的 GEM (Gene Expression Mixture) 聚合 + `SigXTalk` 策略。

### 4.5 可验证命题（阶段 2）

| 编号 | 命题 | 验证方式 |
|------|------|----------|
| H2.1 | 因果解缠后的 $\mathbf{z}^{\text{ext}}$ 能预测邻居组成，而 $\mathbf{z}^{\text{int}}$ 不能 | 以邻居细胞类型比例为标签回归/分类 |
| H2.2 | CMI 剪枝后的因果图识别 CAF→TAM/Treg 轴的定向性与文献报道一致 | 与已知信号轴（如 POSTN/MFAP2→M2 TAM 极化）对照 |
| H2.3 | 因果图中 CAF 分泌的 POSTN/MFAP2 边直接指向 TAM 的 CD163/MRC1 上调，而非空间共定位 artifact | 通过 DoWhy `refute_causal_structure()` 的 falsification p-value 验证 |
| H2.4 | 多层信号流可追溯 INHBA→SMAD2/3→Foxp3 的完整转导链路 | 信号流 TF 活性与 SCENIC AUCell 得分一致性检验 |

---

## 5. 阶段 3：微环境演化的扰动模拟与反事实预测

### 5.1 概述

**目标**：利用训练好的模型进行虚拟敲除（Virtual Knockout），预测靶向干预后 TME 的动态重塑。

**输入**：
- 因果细胞图 $\mathcal{G}_{\text{causal}}$ 与多层信号流（来自阶段 2）
- 训练后的 H-VAE 解码器（来自阶段 1）
- 靶基因列表（如 $\{$INHBA, POSTN, MFAP2$\}$）

**输出**：
- 反事实基因表达谱 $\hat{\mathbf{X}}^{\text{CF}}$
- 空间传播预测图（扰动效应在组织空间上的扩散分布）

### 5.2 扰动潜变量算术 (Latent Space Arithmetic)

参考 CPA / scGen 的逻辑，在双曲潜空间中学习基因特异性的扰动向量。

对于靶基因 $g$（如 INHBA），学习扰动向量 $\delta_g \in T_{\mathbf{o}}\mathbb{H}^d_K$（原点处的切向量）：

$$\delta_g = \mathbb{E}_{\text{treated}}[\text{Log}_{\mathbf{o}}(\mathbf{z})] - \mathbb{E}_{\text{control}}[\text{Log}_{\mathbf{o}}(\mathbf{z})]$$

虚拟敲除模拟（以 INHBA 为例）：

$$\mathbf{z}^{\text{pred}}_i = \text{Exp}_{\mathbf{z}^{\text{obs}}_i}\left(\text{PT}_{\mathbf{o} \to \mathbf{z}^{\text{obs}}_i}(-\delta_{\text{INHBA}})\right)$$

其中 $\text{PT}$ 为平行传输（Parallel Transport），确保扰动向量在双曲流形上的几何一致性。解码器将 $\mathbf{z}^{\text{pred}}_i$ 映射回基因表达空间：

$$\hat{\mathbf{x}}^{\text{CF}}_i = \text{Decoder}(\mathbf{z}^{\text{pred}}_i)$$

**参考实现**：`CPA`（`references/CPA/`）的 `ComPertAPI.predict()` + `scGen`（`references/scgen/`）的 `SCGEN.predict()`。

### 5.3 基于扩散模型的反事实生成 (Diffusion-based Counterfactual)

为捕捉更复杂的非线性细胞状态转变，采用条件扩散模型（Conditional Diffusion Model），以因果图作为结构约束：

1. **前向扩散**：将观测细胞状态 $\mathbf{x}_0$ 逐步加噪至高斯噪声 $\mathbf{x}_T$。
2. **条件反向去噪**：以干预条件 $c = do(\text{POSTN}=\text{low})$ 与因果图约束为条件，逐步去噪生成反事实状态：
   $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, c, \mathcal{G}_{\text{causal}}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t, c), \sigma^2_t \mathbf{I})$$

3. **因果一致性约束**：反事实生成过程中，要求干预的下游效应遵循 $\mathcal{G}_{\text{causal}}$ 的拓扑序（topological ordering）——只有因果图中干预节点的后继节点才被修改，非后继节点保持不变。

**参考实现**：`CausCell`（`references/CausCell/`）的因果解耦 + 扩散生成框架，`Squidiff`（`references/Squidiff/`）的 `GaussianDiffusion` 模型。

### 5.4 空间传播预测 (Spatial Propagation)

虚拟敲除不仅改变靶细胞自身状态，还将通过空间因果网络向邻近区域传播。

模拟策略：

1. **源节点扰动**：对因果图中靶基因高表达的源细胞亚群施加 $do(\cdot)$ 干预。
2. **沿因果边传播**：按 $\mathcal{G}_{\text{causal}}$ 的拓扑序，逐层计算下游细胞的反事实状态——每个下游节点的 $\mathbf{z}^{\text{ext}}$ 根据其上游邻居的新状态重新计算。
3. **空间衰减建模**：引入距离衰减核 $\kappa(d_{ij}) = \exp(-d_{ij}^2 / 2\ell^2)$，其中 $d_{ij}$ 为空间距离，$\ell$ 为特征衰减尺度，模拟信号分子的扩散物理约束。
4. **多轮迭代**：重复传播直到收敛或达到最大传播深度。

**生物学预期**：
- 敲除 POSTN → 基质硬度降低 → 物理屏障减弱 → T 细胞向瘤内再浸润（空间梯度逆转）
- 敲除 INHBA → SMAD2/3 通路下调 → Treg 分化受阻 → 局部免疫抑制减轻
- 敲除 MFAP2 → Integrin $\alpha 5 \beta 1$/FAK 信号减弱 → 远端肿瘤细胞 EMT 减缓

**参考实现**：`DynPerturb`（`references/DynPerturb/`）的时空嵌入传播机制。

### 5.5 可验证命题（阶段 3）

| 编号 | 命题 | 验证方式 |
|------|------|----------|
| H3.1 | INHBA 虚拟敲除后，SPP1+ TAM 的 M2 标志物（CD163, MRC1）下调 | 反事实 vs 观测的 DEG 检验 + marker 方向一致性 |
| H3.2 | POSTN 虚拟敲除后，空间上 T 细胞浸润深度增加（靠近肿瘤核心） | 反事实空间图中 T 细胞-肿瘤距离分布对比 |
| H3.3 | 扰动效应沿因果图传播，且符合空间距离衰减规律 | Moran's I 变化 + 传播梯度分析 |
| H3.4 | 扩散模型生成的反事实状态与 CPA/scGen 预测在关键 marker 方向上一致 | 跨方法 PCC/cosine similarity |

---

## 6. 方法优势与局限

### 6.1 优势

1. **几何忠实性**：双曲嵌入保留树状层级结构，避免"crowding problem"——欧氏空间中远距离节点在低维投影时被迫靠近。
2. **因果而非相关**：SCM + CMI 剪枝将空间共定位与真正的因果通讯区分开来，减少 false positive 信号轴。
3. **端到端可扰动**：双曲潜空间中的扰动算术 + 扩散反事实提供两种互补的 in silico perturbation 手段，兼顾效率（前者）与非线性表达力（后者）。
4. **空间传播可模拟**：将虚拟敲除的效应沿因果图在物理空间中传播，而非仅限于单细胞层面。

### 6.2 局限与未来方向

1. **因果可识别性假设**：SCM 的可识别性依赖于充分性假设（如 faithfulness、causal sufficiency），TME 中未观测的混杂因素可能违反这些假设——后续可引入 latent confounder 建模（如 DoWhy 的 IV 方法）。
2. **双曲空间数值稳定性**：高曲率区域的指数/对数映射可能出现数值溢出——需结合 `geoopt` 的数值截断策略。
3. **训练数据依赖**：扰动预测的可靠性取决于训练数据是否覆盖足够的扰动条件多样性——纯观测数据的外推能力有限。
4. **跨患者泛化**：当前框架在单患者数据上训练——多患者联合训练或 meta-learning 是重要扩展方向。
5. **实验验证闭环**：计算预测最终需要湿实验（如 Perturb-seq、CRISPR 屏幕）验证——建议选择少量高置信度预测进行实验验证。

---

## 附录 A：参考框架映射表

| 阶段 | 组件 | 主要参考 | 仓库路径 |
|------|------|----------|----------|
| 1 | 拓扑增强邻接 | TopoLa | `references/TopoLa/` |
| 1 | 双曲 VAE + 双曲工具 | scDHMap | `references/scDHMap/` |
| 2 | 因果解缠 (SCM + G2G) | Celcomen | `references/celcomen/` |
| 2 | 空间因果结构学习 (UT-IGSP) | FlowSig | `references/flowsig/` |
| 2 | 因果识别与可证伪验证 | DoWhy | `references/dowhy/` |
| 3 | 扰动潜变量算术 | CPA / scGen | `references/CPA/`, `references/scgen/` |
| 3 | 扩散反事实生成 | CausCell / Squidiff | `references/CausCell/`, `references/Squidiff/` |
| 3 | 时空传播模拟 | DynPerturb | `references/DynPerturb/` |

## 附录 B：核心依赖版本基线

| 包 | 版本 | 用途 |
|---|------|------|
| Python | 3.10.19 | 运行环境 |
| PyTorch | 2.6.0+cu124 | 深度学习框架 |
| torch-geometric | 2.7.0 | 图神经网络 |
| geoopt | 0.5.1 | 双曲几何优化 |
| scanpy | 1.11.5 | 单细胞分析 |
| squidpy | 1.6.5 | 空间转录组工具 |
| scvi-tools | 1.3.3 | 变分推断框架 |
| dowhy | 0.14 | 因果推断 |
| pgmpy | 1.0.0 | 概率图模型 |
| scgen | 2.1.1 | 扰动预测 |
| diffusers | 0.36.0 | 扩散模型 |
| GPU | RTX 3070, CUDA 12.4 | 硬件加速 |
