# 拟采取的研究方案

本项目（HyperSCA）旨在建立面向“空间组学 + 单细胞组学”多组学数据的通用几何-因果动态分析范式，并与生物学实验形成协同闭环。框架适用于肿瘤免疫、自身免疫、慢性炎症、感染与组织损伤修复等疾病环境。以下为拟采取的具体研究方案，包含四大模块及相应的图版展示与补充规划。

## 模块一：数据整合与双曲图谱构建

### 步骤1：临床样本获取与高分辨率空间多组学预处理
**方案详述**：
系统收集目标疾病队列的临床样本，获取单细胞转录组（如10x Chromium，提供深度的细胞状态刻画）与高分辨率空间多组学数据（如Visium宏观空间、CosMx单细胞级与VisiumHD亚细胞级空间转录组）。
计算流程中，将多模态原始数据统一转换为`.h5ad`格式标准Schema。为克服跨模态对齐难题，利用基于单细胞数据的Pseudo-bulk NormalizedCounts与空间反卷积元数据构建Cluster级或Single-cell级的输入表达矩阵。进而，通过k-NN与Delaunay三角剖分构建空间邻接图，并引入TopoLa SVD进行拓扑增强谱分析，以高保真地保留微环境复杂的物理空间约束。

**图版展示与建议**：
*   **选用现有图表**：提取 `results/figures/spatial_comm/cosmx/` 或 `visium/` 目录下的 **空间生态位分布图（Spatial Niche Distribution）**（如 `k=13`, `silhouette=1.000`的CosMx聚类结果）与 **基因空间热图（Spatial Gene Heatmap）**，用于直观展示免疫排斥生态位的原位分布。
*   **补充绘图建议**：建议使用绘图软件绘制一张 **多源数据整合与Schema构建流程示意图**，展现从临床切片到统一分析格式转换的全景流程。

### 步骤2：基于Lorentz流形的H-VAE算法开发
**方案详述**：
复杂疾病微环境中的细胞分化演进本质呈指数级扩张的树状拓扑，将开发基于Lorentz流形的双曲变分自编码器（H-VAE）以解决欧式降维导致的失真（Distortion）与过渡态细胞重叠问题。
模型采用GCN（捕捉空间拓扑）与MLP（处理表达特征）双分支编码器，将数据映射至Lorentz双曲流形；解码器采用ZINB（零膨胀负二项）分布拟合组学数据的过度离散性。创新性地引入Wrapped Normal分布用于双曲潜空间的重参数化采样，并增加Cauchy核拓扑正则项（$\mathcal{L}_{topo}$），强制保留细胞间的高维层级谱系距离。

**关键公式**：
*   **Lorentz内积与变分下界（ELBO）**：
    $$ \langle x, y \rangle_\mathcal{L} = -x_0 y_0 + \sum_{i=1}^n x_i y_i $$
    $$ \mathcal{L}_{H-VAE} = \mathbb{E}_{q}[\log p(X|Z_{hyp})] - \beta D_{KL}(q(Z_{hyp}|X) || p_{prior}(Z_{hyp})) + \lambda \mathcal{L}_{topo} $$

**图版展示与建议**：
*   **选用现有图表**：提取 `results/figures/integration/` 下的 **Poincaré盘双曲嵌入图（Poincaré Embedding）**，可视化展示T细胞、巨噬细胞、成纤维细胞等亚群的层级演化轨迹。
*   **选用现有图表**：提取 `results/figures/step1/compare/` 下的 **双曲 vs 欧式几何对比图（如Distortion/Silhouette对比雷达图）**，量化论证底层算法改进的必要性。

---

## 模块二：方向性因果发现

### 步骤3：基于HSIC的细胞内外源特征解缠
**方案详述**：
现有推断工具缺乏对混杂因子的控制。本步骤构建因果解缠模型（DisentangleModel），以剥离复杂的生态位混杂效应。具体而言，利用图卷积网络（GCN）从空间图中提取受邻域微环境影响的“外源特征”（$Z_{ext}$），并利用多层感知机（MLP）提取由细胞内源状态决定的“内源特征”（$Z_{int}$）。
通过在损失函数中引入Hilbert-Schmidt独立性准则（HSIC）惩罚项，强制$Z_{int}$与$Z_{ext}$在统计学上相互独立，从而获得纯净的本征潜变量，消除共表达网络中的假阳性边。

**关键公式**：
*   **HSIC独立性惩罚**：
    $$ \mathcal{L}_{HSIC} = \frac{1}{(n-1)^2} \text{Tr}(K_{int} H K_{ext} H) $$
    *(其中$K_{int}$、$K_{ext}$为核矩阵，$H$为中心化矩阵)*

### 步骤4：结合空间物理约束的因果网络重构
**方案详述**：
基于解缠后的潜空间特征，并以物理空间扩散定律（如指数衰减函数 $e^{-d^2/2\ell^2}$）为先验约束，采用改进的PC算法结合偏相关条件独立性（CI）检验重构因果骨架。为提升稳定性，引入Block Bootstrap聚合策略进行阈值剪枝，并整合OmniPath、LIANA等先验知识库补全跨谱系信号。
聚焦于前期锁定的POSTN、MFAP2、INHBA等靶点，推断其在微环境内的真实因果层级与上下游信号流向（Ligand→Receptor→TF→Target），精确定位免疫排斥屏障的调控中枢。

**关键公式**：
*   **偏相关条件独立性检验**：
    $$ X \perp\!\!\!\perp Y | \mathbf{Z} \iff \rho_{XY\cdot\mathbf{Z}} = 0 $$

**图版展示与建议**：
*   **选用现有图表**：提取 `results/figures/integration/` 下的 **生态位核心因果DAG图（Causal DAG）** 与 **多层信号流向图（Signaling Flow）**（展示如Integrin-FAK、Activin-SMAD等完整通路）。
*   **选用现有图表**：提取 `results/figures/integration/target_niche_heatmap.png` **靶点生态位热图** 与 `target_niche_dotplot.png` **靶点生态位点图**，展示MFAP2/POSTN/INHBA在免疫排斥生态位中的特异性富集。

---

## 模块三：时空传播与虚拟推演

### 步骤5：双曲切空间潜变量反事实算术计算
**方案详述**：
针对静态快照数据无法推演动态级联反应的瓶颈，在Lorentz流形上执行虚拟基因干预（Virtual Knockout）。对于选定的枢纽基因（如POSTN），利用对数映射（LogMap）将未受干预细胞的基态映射至原点切空间并提取扰动向量$\Delta v$；利用平行传输（Parallel Transport）将$\Delta v$沿流形测地线无损平移至靶向细胞的切空间；最终通过指数映射（ExpMap）投射回双曲流形并解码，生成“反事实（Counterfactual）”表达谱。结合步骤4的因果邻接矩阵及空间距离衰减核（BFS因果传播+梯度衰减拟合），精确模拟敲除该基因后引起的局部乃至全局基因表达微扰。

**关键公式**：
*   **反事实潜空间算术（Latent Arithmetic）**：
    $$ z_{cf} = \exp_{z}\left( \alpha \cdot PT_{\mu \to z}(\Delta v) \right) $$

**图版展示与建议**：
*   **选用现有图表**：提取 `results/figures/step3/` 下的 **扰动空间传播梯度衰减图（Spatial Propagation Gradient Decay）**，直观展现干预信号随空间距离的指数级辐射衰减规律。

### 步骤6：耦合PK/PD模型的动态干预模拟
**方案详述**：
将药理学动力学方程原生嵌入分析框架。集成一室口服药代动力学（PK）模型以模拟药物在目标组织微环境中的浓度-时间动态，同时结合Hill方程（药效动力学PD模型）计算药物针对POSTN、MFAP2等靶点的动态占有率与效应抑制。
在此基础上，针对多靶点联合干预策略，采用Bliss独立性模型量化计算不同组合方案的协同阻断效应（Synergy Score），从而在计算系统中前置演算多靶点联合用药诱导的微环境重塑轨迹。

**关键公式**：
*   **药效Hill方程与Bliss协同评分**：
    $$ E = E_{max} \frac{C^\gamma}{C^\gamma + EC_{50}^\gamma} $$
    $$ Synergy_{Bliss} = E_{AB} - (E_A + E_B - E_A E_B) $$

**图版展示与建议**：
*   **选用现有图表**：提取 `results/figures/step4/` 下的 **PK/PD浓度与响应动态曲线（PK/PD Curves）** 和 **动态靶点响应轨迹图（Dynamic Target Trajectories）**。
*   **选用现有图表**：提取 `results/figures/integration/combo_niche_sankey.png` **组合干预生态位流向Sankey图**，展示联合给药方案对靶点表达调控的流向与协同结果。

---

## 模块四：闭环校准与策略优选

### 步骤7：微环境重塑的类器官与PDX模型干预实验
**方案详述**：
从纯计算推演迈向生物学实体验证，建立与疾病场景匹配的验证体系（如患者来源类器官、免疫共培养体系、或人源化/疾病模型动物实验）。基于计算系统输出的优选靶点组合排序，施加干预策略（如中和抗体、小分子抑制剂或联合免疫调节方案）。
利用流式细胞术与多重免疫荧光（mIF），定量测定干预后的核心表型数据，包括细胞外基质（ECM）重塑程度、病理细胞负荷变化率、以及CD8+ T细胞等细胞毒性免疫细胞在物理屏障内的浸润深度，获取真实的干预效应证据（Ground Truth）。

**图版展示与建议**：
*   **补充绘图建议**：这是本项目连接计算与真实世界的关键湿实验环节。强烈建议使用绘图工具（如BioRender）设计一张 **微环境干预实验与参数闭环系统架构图（Wet-lab & Human-in-the-loop Schematic）**。图中需包含：类器官/PDX建模 $\rightarrow$ 联合靶向给药 $\rightarrow$ 多模态表型数据采集（mIF/FACS） $\rightarrow$ 数据流向计算系统。

### 步骤8：贝叶斯驱动的模型参数反向校准
**方案详述**：
将步骤7获取的实测干预验证数据，通过实验表格式导入并回填至计算框架。依托贝叶斯优化算法进行反向校准（Roundtrip Update）：将观测到的真实细胞比例变化及杀伤率作为似然函数（Likelihood），以反向迭代更新步骤6中的PK/PD动力学参数及步骤5中的空间传播衰减系数（$\ell$）等先验设定。
通过实验室在环机制促使计算模型参数收敛，基于校准后的模型重启时空推演。由此彻底筛选出重塑免疫排斥生态位、逆转免疫耐药的最佳多靶点联合用药方案，及最优给药时序设计（Dosing Schedule）。

**关键公式**：
*   **贝叶斯后验参数更新**：
    $$ P(\theta | D_{exp}) \propto P(D_{exp} | \theta_{prior}) P(\theta_{prior}) $$
    *(式中 $\theta$ 包含计算系统中的 $EC_{50}$、空间衰减系数等动力学参数，$D_{exp}$ 为类器官或PDX验证数据)*

**图版展示与建议**：
*   **选用现有图表**：提取 `results/figures/step4/roundtrip_before_after.png` **Roundtrip闭环校准前后对比报告图**，强有力地证明实测数据回填后，模型预测准度的显著提升，突显计算-实验闭环设计的优越性。