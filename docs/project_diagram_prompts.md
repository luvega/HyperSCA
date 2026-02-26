# HyperSCA 项目示意图 AI 作图 Prompts

本文档提供了一组经过设计的 Prompt，用于在 Midjourney、Stable Diffusion (DALL·E 3) 等工具中生成 HyperSCA 的项目示意图。

## 1. 核心语义元素 (Core Elements)

所有 Prompt 均基于以下核心逻辑构建，确保技术准确性：

*   **Input**: Multi-modal data (scRNA-seq + Spatial Transcriptomics + Clinical Metadata).
*   **Stage 1 (Geometry)**: Hyperbolic Manifold (Lorentz/Poincare disk), curved space embedding, hierarchical tree structures.
*   **Stage 2 (Causal)**: Disentangled latent factors, directed acyclic graphs (DAGs), signaling flow arrows.
*   **Stage 3 (Intervention)**: Counterfactual perturbation (virtual knockout), spatial diffusion ripples, target ranking.
*   **Output**: Mechanism insight, therapeutic targets.

---

## 2. 风格化 Prompts (Ready-to-Use)

### 风格 A：学术论文风 (Academic / Nature Biotechnology Style)
*适用场景：论文主图 (Figure 1)、严谨的技术文档*

**English Prompt:**
> A scientific diagram illustrating a computational biology framework named "HyperSCA". The composition is divided into three logical layers from left to right. **Left (Input)**: Abstract representations of single-cell sequencing data matrices and spatial tissue sections with cellular spots. **Middle (Core Algorithm)**: A 3D visualization of a hyperbolic Poincaré disk with curved grid lines, where cell clusters are embedded as hierarchical tree-like structures. Emanating from this disk are directed causal arrows forming a gene regulatory network. A specific node is highlighted showing a "virtual knockout" effect spreading like ripples in a pond. **Right (Output)**: A ranked list of therapeutic targets and a spatial heatmap showing intervention effects. **Style**: Clean vector art, high-tech scientific illustration, flat design with subtle 3D depth, professional color palette (teal, navy blue, soft coral), white background, precise data visualization style, highly detailed, 8k resolution. --ar 16:9 --v 6.0

**中文 Prompt (DALL·E 3 参考):**
> 一张科学示意图，展示名为“HyperSCA”的计算生物学框架。构图从左到右分为三层。**左侧（输入）**：单细胞测序数据矩阵和带有细胞点的空间组织切片的抽象表示。**中间（核心算法）**：双曲庞加莱盘的 3D 可视化，带有弯曲的网格线，细胞簇嵌入为层级树状结构。从该圆盘发出有向因果箭头，形成基因调控网络。突出显示特定节点，展示像池塘涟漪一样扩散的“虚拟敲除”效应。**右侧（输出）**：治疗靶点的排序列表和显示干预效果的空间热图。**风格**：简洁的矢量艺术，高科技科学插图，带有微妙 3D深度的扁平化设计，专业配色（蓝绿色、海军蓝、柔和珊瑚色），白色背景，精确的数据可视化风格，高细节，8k 分辨率。

---

### 风格 B：技术架构风 (Engineering / GitHub README Style)
*适用场景：GitHub README、技术博客、开发文档*

**English Prompt:**
> An isometric isometric system architecture diagram for "HyperSCA". The image features a modular pipeline layout. **Module 1**: A data ingestion funnel processing "scRNA-seq" and "Spatial" data cubes. **Module 2**: A central processing unit visualized as a glowing hyperbolic geometric manifold (saddle shape) representing the embedding space. **Module 3**: A network analysis module displaying a directed graph with nodes and edges representing causal discovery. **Module 4**: A simulation module showing a "What-if" perturbation scenario with diverging paths. **Style**: Isometric projection, blueprint schematic, clean lines, technical aesthetic, soft blue and purple gradients, glassmorphism UI elements, dark mode compatible background (dark grey), schematic labels, unreal engine render style. --ar 3:2 --v 6.0

**中文 Prompt (DALL·E 3 参考):**
> “HyperSCA”的等轴测系统架构图。图像采用模块化流水线布局。**模块 1**：处理“scRNA-seq”和“空间”数据立方体的数据摄入漏斗。**模块 2**：可视化为发光的双曲几何流形（马鞍形）的中央处理单元，代表嵌入空间。**模块 3**：网络分析模块，显示代表因果发现的带有节点和边的有向图。**模块 4**：模拟模块，显示带有分叉路径的“What-if”扰动场景。**风格**：等轴测投影，蓝图原理图，简洁的线条，技术美学，柔和的蓝紫色渐变，玻璃拟态 UI 元素，暗模式兼容背景（深灰色），原理图标签，虚幻引擎渲染风格。

---

### 风格 C：极简汇报风 (Abstract / Keynote Style)
*适用场景：PPT 封面、演讲背景、概念展示*

**English Prompt:**
> A conceptual abstract art piece representing "Hyperbolic Spatiotemporal Causal Analysis". A central hyperbolic disk (Poincaré model) acts as a lens focusing scattered data points into organized hierarchical structures. Through the lens, chaotic lines transform into orderly directed causal flows. The background suggests a biological tissue texture blending with mathematical grid lines. **Style**: Minimalist, abstract, futuristic, corporate Memphis style but more scientific, fluid shapes, gradient lighting, cinematic lighting, focus on the transformation from chaos to order, deep tech vibe. --ar 16:9 --v 6.0

**中文 Prompt (DALL·E 3 参考):**
> 代表“双曲时空因果分析”的概念抽象艺术作品。中央的双曲圆盘（庞加莱模型）作为一个透镜，将分散的数据点聚焦成有组织的层级结构。透过透镜，混乱的线条转变为有序的有向因果流。背景暗示了与数学网格线融合的生物组织纹理。**风格**：极简主义，抽象，未来主义，企业孟菲斯风格但更具科学感，流畅的形状，渐变照明，电影级照明，专注于从混乱到有序的转变，深度科技氛围。

---

### 风格 D：Banana Pro / Stable Diffusion 专用版 (SDXL / Flux Optimized)
*适用场景：开源模型生成，强调画质与构图描述，无 Midjourney 参数*

**Academic (论文风):**
> (masterpiece, best quality, highres:1.2), scientific diagram of "HyperSCA" framework, three-layer composition. Left side: abstract sequencing data matrix and spatial tissue spots. Center: 3D hyperbolic Poincaré disk with curved grid, hierarchical tree embedding, directed causal arrows, gene regulatory network, virtual knockout ripple effect. Right side: ranked target list, spatial heatmap. Style: clean vector art, Nature biotechnology illustration, flat design with depth, teal and navy blue color palette, white background, precise data visualization, 8k resolution, wide landscape aspect ratio.

**Architecture (架构风):**
> (masterpiece, best quality:1.2), isometric system architecture of "HyperSCA", modular pipeline. Module 1: data ingestion funnel for scRNA-seq and Spatial data. Module 2: glowing hyperbolic manifold saddle shape as central processing unit. Module 3: directed graph network analysis. Module 4: perturbation simulation paths. Style: isometric projection, blueprint schematic, clean lines, technical aesthetic, soft blue and purple gradients, glassmorphism, dark grey background, unreal engine render, sharp focus, wide landscape.

**Abstract (汇报风):**
> (masterpiece, best quality:1.2), conceptual abstract art for "Hyperbolic Spatiotemporal Causal Analysis". Central hyperbolic disk acting as a lens focusing scattered points into organized hierarchy. Chaotic lines transforming into orderly causal flows. Background: biological tissue texture blending with math grid. Style: minimalist, futuristic, corporate Memphis tech style, fluid shapes, gradient lighting, cinematic lighting, depth of field, 8k, wide screen.

---

## 3. 通用参数与约束 (Parameters & Constraints)

在使用上述 Prompt 时，建议附加以下设置以保证质量：

### Negative Prompt (负向提示词)
> **(Copy & Paste):** text, words, watermark, signature, blurry, low quality, pixelated, distorted geometry, messy lines, cartoon, anime style, face, human figures, messy biology, anatomical organs, blood, messy overlapping text.

### 推荐参数设置
*   **Aspect Ratio (纵横比)**: `--ar 16:9` (适合宽屏/PPT) 或 `--ar 3:2` (适合文档)
*   **Model Version**: Midjourney v6.0 或 DALL·E 3
*   **Stylize**: `--s 250` (Midjourney，保持适度艺术化)

## 4. 快速自定义模板

如果你需要微调内容，请使用以下模板替换关键词：

> A [STYLE: scientific / isometric / abstract] diagram for a biology AI model.
> **Input**: [INPUT_DATA: single-cell data / tissue slides].
> **Core**: [CORE_ALGORITHM: hyperbolic geometry / causal graph / neural network].
> **Action**: [ACTION: embedding / reasoning / perturbation].
> **Output**: [OUTPUT: target ranking / heatmap].
> **Visual Style**: [VISUALS: clean vector / blueprint / cinematic 3D], [COLORS: teal and blue / dark mode / vivid].
> --ar 16:9

---

## 5. CNS/Cell 子刊风格精细版（投稿主图）

本节用于高规范论文作图，强调面板结构、术语一致和可比性，弱化装饰风格。

### Prompt A：CNS 主图总览（A/B/C/D 四面板）

```text
(masterpiece, best quality, highres:1.2), publication-grade biomedical figure in CNS journal style, multi-panel layout with labels A, B, C, D, clean white background, vector-like sharp lines, consistent typography hierarchy.

Panel A (Data Inputs): three harmonized input modalities for HyperSCA, including scRNA-seq expression matrix, spatial transcriptomics spot map, and clinical stratification metadata (MSI/MMR). show standardized schema mapping arrows.
Panel B (Stage 1 Geometry): hyperbolic representation learning with Lorentz manifold and Poincare disk, H-VAE latent embedding, hierarchical cell-state organization, geodesic-aware structure preservation.
Panel C (Stage 2 Causality): latent disentanglement (z_int, z_ext), conditional independence pruning, bootstrap-supported directed causal graph, signaling flow inference with directional arrows and confidence encoding.
Panel D (Stage 3 Intervention and Output): counterfactual perturbation in latent space, spatial propagation decay map, intervention response heatmap, ranked therapeutic target list.

Overall logic left-to-right and top-to-bottom, minimal decorative elements, high information density, professional color system (blue, teal, neutral gray), clear legend area, figure prepared for manuscript main text, 16:9.
```

### Prompt B：方法学框架图（Mechanism-first）

```text
(masterpiece, best quality:1.2), mechanistic computational oncology framework figure for HyperSCA, strict academic visual language, no artistic metaphor.

Create a structured pipeline with three sequential stages:
Stage 1 Hyperbolic Embedding: manifold-aware encoder-decoder, Lorentz to Poincare representation, hierarchy-preserving latent space.
Stage 2 Causal Discovery: disentangled latent factors, conditional independence tests, directed acyclic causal graph, signaling path extraction.
Stage 3 Counterfactual Perturbation: virtual gene knockout in latent space, causal-constrained spatial diffusion, intervention effect quantification.
Add integration branch: multi-source MVP integration of scCRC_Neu, scCRC_IFNG, ST_CRC_MSS, with hyperbolic versus euclidean comparison and MSI/MMR stratified reporting.
Show outputs as: causal network, spatial effect map, target ranking, integrated report.
publication-style figure, compact labels, clean panels, white background, 3:2.
```

### Prompt C：双模式对比图（Hyperbolic vs Euclidean）

```text
(masterpiece, best quality, highres), comparative scientific figure for HyperSCA geometry modes, two-column layout.

Left column: Hyperbolic mode, show Poincare disk embedding with curved geodesics, stronger hierarchy separation, downstream causal graph and perturbation outputs.
Right column: Euclidean mode, show standard planar embedding, baseline separation, corresponding causal graph and perturbation outputs.
Bottom comparison row: metric summary placeholders (embedding quality, causal robustness, spatial propagation consistency, target ranking stability), MSI/MMR subgroup consistency indicators.
Design requirements: identical visual grammar across columns, strict comparability, neutral white background, restrained color palette with blue for hyperbolic and gray for euclidean, journal figure style, 16:9.
```

### Prompt D：结果证据图（Target Discovery Evidence）

```text
(masterpiece, best quality:1.2), manuscript-ready evidence synthesis figure for HyperSCA target discovery.

Compose four aligned modules:
1) candidate source aggregation from multi-modal data,
2) causal evidence from directed graph and signaling flow,
3) counterfactual and spatial propagation evidence maps,
4) final target prioritization table with confidence tiers.
Include MSI/MMR stratified evidence annotation and cross-sample consistency markers.
Use clean axis-aligned layout, high readability, no clutter, no decorative icons, scientific infographic style suitable for CNS supplementary figure, wide landscape.
```

### CNS 风格专用 Negative Prompt

```text
cartoon, anime, fantasy, painterly, abstract art, cinematic flare, excessive glow, noisy texture, photorealistic people, laboratory photo scene, messy overlap, distorted graph, illegible labels, random text blocks, watermark, logo, signature
```

### CNS 风格参数建议（Banana Pro）

- 分辨率：`2048x1152`（主图）或 `1792x1024`
- CFG：`5.5-7`
- Steps：`30-45`
- 先低 CFG 生成构图，再提高 Steps 做细节精修
- 可使用 `docs/pipeline_architectures.svg` 作为结构参考图
