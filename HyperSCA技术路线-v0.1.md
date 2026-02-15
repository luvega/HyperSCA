**“双曲几何时空因果解析（**Hyper**bolic **S**patiotemporal **C**ausal **A**nalysis, HyperSCA）”**技术路线

融合深度生成模型、双曲几何嵌入与因果结构学习，旨在系统性地解构结直肠癌肿瘤免疫治疗响应免疫抑制微环境（TME）构建机制，识别的多细胞间相互作用靶点和及其调控机制。

\--------------------------------------------------------------------------------

**第一步：双曲流形嵌入与多模态数据融合 (Hyperbolic Manifold Embedding)**

**目标：** 克服欧氏空间在表征细胞发育层级（Hierarchies）和复杂拓扑结构时的畸变问题，构建保真度更高的低维潜在空间，以整合 scRNA-seq 的转录组特征与 Spatial Transcriptomics (ST) 的空间邻域信息。

1. **构建空间邻域图 (Spatial Neighbor Graph Construction):**

  ◦ 利用 ST 数据的物理坐标，构建空间邻域图 *G*=(*V*,*E*)。为避免仅基于距离的伪连接，引入**拓扑编码（Topology-encoding）**策略。参考 **TopoLa** 框架，计算节点间的 TopoLa 距离（TLd），通过加权偶数跳路径（even-hop paths）量化细胞间的几何结构相似性，从而优化邻接矩阵 *A*，保留具有生物学意义的拓扑连接。

2. **双曲变分自编码器 (Hyperbolic VAE) 建模:**

  ◦ 采用 **HyperDiffuseNet** 或 **scDHMap** 的架构设计，构建基于双曲几何的变分自编码器。

  ◦ **编码器 (Encoder):** 使用图卷积网络 (GCN) 处理空间邻域图，提取多尺度空间特征；同时处理基因表达矩阵。将特征映射到双曲潜空间（Latent Hyperbolic Space），通常采用 **Lorentz 模型**或 **Poincaré 球模型**，以利用其负曲率特性高效嵌入树状发育轨迹和层级结构（如从 naive T 细胞到耗竭 T 细胞的分化路径）。

  ◦ **解码器 (Decoder):** 利用负二项分布（Negative Binomial）重建损失函数，从双曲潜变量恢复基因表达谱，确保数据的稀疏性和离散性得到妥善处理。

**第二步：空间约束下的因果通讯网络构建 (Spatial Causal Inference)**

**目标：** 在双曲潜空间中，剥离细胞内在调控与外在通讯信号，构建定向的因果通讯网络（例如CAF (POSTN/MFAP2) → TAM/Treg (INHBA)）。

1. **因果解缠 (Causal Disentanglement):**

  ◦ 参考 **Celcomen** 算法，将细胞的基因表达潜变量分解为**“细胞内源性状态 (*Z**in**t*)”**和**“微环境外源性影响 (*Z**e**x**t*)”**。

  ◦ 利用空间邻域信息，建立结构因果模型（SCM）。假设细胞 *i* 的 *Z**e**x**t* 是由其空间邻居 *N*(*i*) 的状态通过配受体相互作用因果地决定的。通过最大化条件独立性（Conditional Independence），去除仅由共享微环境引起的伪相关，识别真实的细胞间通讯流。

2. **条件互信息剪枝 (CMI Pruning):**

  ◦ 为进一步精炼网络，引入 **CASCAT** 或 **FlowSig** 的策略。计算条件互信息（Conditional Mutual Information, CMI）来量化两个节点（如 *POSTN* 高表达的 CAF 与 *SPP*1+ TAM）在给定其他邻居条件下的依赖性。

  ◦ 剔除 CMI 值低于阈值的冗余边缘，构建稀疏且鲁棒的**因果细胞图 (Causal Cell Graph)**。例如这将可以确认 CAF 分泌的 POSTN/MFAP2 是否直接因果导向了 TAM 的 M2 极化特征（如 *C*D163, *MRC*1 的上调），而非仅仅是空间共定位。

3. **多层信号流推断 (Multilayer Signaling Flow):**

  ◦ 参考 **SigXTalk** 或 **FlowSig**，构建“配体→ 受体→ 下游转录因子 → 靶基因”的完整信号流。例如， **MFAP2** 通过 **Integrin** *α*5*β*1**/FAK** 轴和 **INHBA** 通过 **SMAD2/3** 轴的信号传递路径，量化信号传导强度。

**第三步：微环境演化的扰动模拟与反事实预测 (In Silico Perturbation & Counterfactual Prediction)**

**目标：** 利用训练好的模型进行“虚拟敲除（Virtual Knockout）”，预测靶向干预后的 TME 动态重塑。

1. **扰动潜变量算术 (Latent Space Arithmetic):**

  ◦ 参考 **CPA (Compositional Perturbation Autoencoder)** 或 **scGen** 的逻辑，在双曲潜空间中学习特定基因（如 *INHBA*）的扰动向量 *δ**p**er**t*。

  ◦ 对于 *INHBA* 敲除模拟，将目标细胞（如 *SPP*1+ TAM）的潜变量 *z**o**b**s* 移动为 *z**p**re**d*=*z**o**b**s*−*δ**I**N**H**B**A*，然后解码生成预测的基因表达谱。

2. **基于扩散模型的反事实生成 (Diffusion-based Counterfactual Generation):**

  ◦ 为捕捉非线性动态，可采用 **CausCell** 或 **Squidiff** 框架。利用条件扩散模型（Conditional Diffusion Model），以因果图为约束，生成在“干预 *d**o*(*POSTN*=*l**o**w*)”条件下的反事实细胞状态。

  ◦ 这将不仅预测目标细胞自身的表型变化，还能通过空间因果网络传播，模拟邻近细胞的变化，比如T细胞因物理屏障（如POSTN 介导的基质硬度）移除后的再浸润过程，以及 TAM 极化状态的逆转。

3. **空间传播预测 (Spatial Propagation):**

  ◦ 借鉴 **SpatialProp** 或 **DynPerturb**，模拟扰动效应在组织空间上的扩散。重点考察敲除 **MFAP2** 后，其对基质物理结构的改变如何通过机械信号转导（Mechanotransduction）影响远端肿瘤细胞的 EMT 进程及药物敏感性。