# HyperSCA 评估指标体系

*版本: v1.0 | 日期: 2026-02-15*

---

本文档定义 HyperSCA 各阶段的评估指标，为工程验收与学术报告提供统一度量基准。

---

## 1. 阶段 1：嵌入质量 (Embedding Quality)

### 1.1 几何保真度

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Distortion Score** ($D$) | 原始距离结构在嵌入空间中的畸变程度 | $D = \frac{1}{\binom{N}{2}} \sum_{i<j} \left\| \frac{d_{\mathbb{H}}(\mathbf{z}_i, \mathbf{z}_j)}{d_{\text{orig}}(i,j)} - 1 \right\|$ | 越小越好 | `embedding_metrics.py` |
| **$\delta$-Hyperbolicity** | 数据内在双曲度，验证双曲嵌入的合理性 | Gromov 四点条件的 $\delta$ 值 | 越小表明数据越适合双曲嵌入 | `embedding_metrics.py` |
| **Triplet Accuracy** | 距离序保持能力 | 随机采样三元组 $(i,j,k)$，检查嵌入距离序是否与原始一致 | 越高越好（理想 > 0.9） | `embedding_metrics.py` |

### 1.2 生物学一致性

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **ARI** (Adjusted Rand Index) | 嵌入空间聚类与已知细胞类型注释的一致性 | Leiden 聚类 → 与 ground truth 标签计算 ARI | > 0.5（良好），> 0.7（优秀） | `embedding_metrics.py` |
| **NMI** (Normalized Mutual Information) | 互信息归一化版聚类一致性 | 同上，NMI 替代 ARI | > 0.5 | `embedding_metrics.py` |
| **Silhouette Score** | 簇内紧致度 vs 簇间分离度 | 在双曲距离矩阵上计算 | > 0.3 | `embedding_metrics.py` |
| **Branch Purity** | T 细胞分化分支中各状态的纯度 | 沿 Poincaré 径向方向分层，每层计算 dominant cell type 比例 | > 0.7 per branch | `embedding_metrics.py` |

### 1.3 与欧氏 baseline 对比

| Baseline | 对比指标 |
|----------|----------|
| PCA (50d) + UMAP (2d) | Distortion, ARI, NMI, Silhouette |
| scVI latent (10d) | Distortion, ARI, NMI |
| 原始 scDHMap (2d, 无 TopoLa) | Distortion, ARI |

---

## 2. 阶段 2：因果边可信度 (Causal Edge Reliability)

### 2.1 结构层面

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Bootstrap Edge Frequency** ($w_{ij}$) | 因果边在 bootstrap 样本中出现的频率 | $w_{ij} = \frac{1}{B}\sum_{b=1}^{B}\mathbb{1}[i \to j \in \hat{\mathcal{G}}^{(b)}]$ | 保留边 > 0.5，核心边 > 0.8 | `causal_metrics.py` |
| **Falsification p-value** | 因果图结构的可证伪性检验 | DoWhy `refute_causal_structure()` — 局部 Markov 条件独立性 | p > 0.05（不可拒绝） | `causal_metrics.py` |
| **Arrow Strength** ($\text{AS}_{i \to j}$) | 因果边的效应强度 | DoWhy `arrow_strength()` — KL 散度或 regression coefficient | 高优先排序依据 | `causal_metrics.py` |
| **Graph Sparsity** | 边数 / 最大可能边数 | $|\mathcal{E}| / (|\mathcal{V}| \cdot (|\mathcal{V}|-1))$ | < 0.1（稀疏） | `causal_metrics.py` |

### 2.2 生物学层面

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Known Axis Recovery** | 已知信号轴（如 CAF→TAM）在因果图中的召回率 | 从文献整理 ground truth 边集，计算 precision/recall | Recall > 0.6 | `causal_metrics.py` |
| **Direction Accuracy** | 因果方向与文献报道一致率 | 对 recovered edges 检查方向 | > 0.8 | `causal_metrics.py` |
| **Signaling Flow Completeness** | 完整信号流（4 层）被推断的比例 | 已知通路中成功推断出完整链路的比例 | > 0.3（P1 阶段） | `causal_metrics.py` |

### 2.3 解缠质量

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **HSIC(Z_int, Z_ext)** | 两分量之间的统计独立性 | Hilbert-Schmidt Independence Criterion | 越小越好 | `causal_metrics.py` |
| **Z_ext Neighbor Predictivity** | Z_ext 能否预测邻居组成 | 以邻居细胞类型比例为标签，Z_ext 回归 R² | > 0.3 | `causal_metrics.py` |
| **Z_int Neighbor Independence** | Z_int 不应预测邻居组成 | 同上，但使用 Z_int | R² < 0.1 或 p > 0.05 | `causal_metrics.py` |

---

## 3. 阶段 3：反事实质量 (Counterfactual Quality)

### 3.1 表达水平指标

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **R² (mean)** | 预测 vs 实际基因表达均值的决定系数 | 跨基因 mean(predicted) vs mean(ground_truth) 的 R² | > 0.8 | `cf_metrics.py` |
| **R² (var)** | 预测 vs 实际基因表达方差的决定系数 | 同上，var 替代 mean | > 0.5 | `cf_metrics.py` |
| **PCC** (Pearson Correlation) | 预测与实际的逐基因相关性 | 跨细胞 Pearson 相关系数的中位数 | > 0.7 | `cf_metrics.py` |
| **MSE** | 均方误差 | 全矩阵 MSE | 越小越好（与 baseline 对比） | `cf_metrics.py` |

### 3.2 生物学一致性指标

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Marker Direction Accuracy** | 关键 marker 基因表达变化方向的准确率 | 对预设 marker 列表，检查 KO 后表达变化方向（上/下调）是否与文献预期一致 | > 0.8 | `cf_metrics.py` |
| **Marker Magnitude Ranking** | Marker 基因变化幅度排序与预期一致性 | Spearman rank correlation | > 0.5 | `cf_metrics.py` |
| **DEG Overlap** | 反事实 DEGs 与文献报道 DEGs 的重叠 | Jaccard index | > 0.2 | `cf_metrics.py` |

### 3.3 跨方法一致性

| 对比方式 | 指标 |
|----------|------|
| CPA vs scGen (Latent Arithmetic) | Top-100 DEG Jaccard, PCC of fold changes |
| Latent Arithmetic vs Diffusion CF | Top-50 marker 方向 PCC, cosine similarity |
| 不同 bootstrap 种子 | 结果稳定性（std of PCC across seeds） |

### 3.4 不确定性量化

| 指标 | 定义 | 计算方式 | 实现位置 |
|------|------|----------|----------|
| **Prediction Uncertainty** | 模型对预测的置信度 | CPA: cosine/euclidean distance of latent 变分后验; Diffusion: 多次采样的 std | `cf_metrics.py` |
| **CRPS** | 连续分布概率评分 | DoWhy `gcm` 的 CRPS 实现 | `cf_metrics.py` |

---

## 4. 空间传播一致性 (Spatial Propagation Consistency)

### 4.1 空间自相关

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Moran's I (扰动前)** | 扰动基因表达的空间自相关 | `squidpy.gr.spatial_autocorr(method='moran')` | > 0（空间聚集） | `spatial_metrics.py` |
| **$\Delta$ Moran's I** | KO 前后 Moran's I 的变化 | $I^{\text{CF}} - I^{\text{obs}}$ | 方向与生物预期一致 | `spatial_metrics.py` |

### 4.2 传播梯度

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Gradient Decay R²** | 扰动效应随空间距离衰减的拟合优度 | 将 $|\Delta x_g|$ 对 $d_{\text{spatial}}$ 拟合指数衰减模型，计算 R² | > 0.3 | `spatial_metrics.py` |
| **Characteristic Length** ($\ell$) | 扰动信号的特征衰减尺度 | 拟合 $|\Delta x| \propto \exp(-d / \ell)$ 的 $\ell$ | 与分子扩散常数量级一致 | `spatial_metrics.py` |
| **Propagation Depth** | 扰动显著影响的最大跳数 | 沿因果图 BFS，每层检查效应是否 > 阈值 | 与因果图拓扑一致 | `spatial_metrics.py` |

### 4.3 空间-因果一致性

| 指标 | 定义 | 计算方式 | 期望方向 | 实现位置 |
|------|------|----------|----------|----------|
| **Spatial-Causal Correlation** | 空间距离与因果效应强度的关联 | Spearman correlation between $d_{\text{spatial}}$ and arrow strength | 显著负相关 | `spatial_metrics.py` |
| **Edge-Adjacency Overlap** | 因果边与空间邻接的重叠比例 | $|\mathcal{E}_{\text{causal}} \cap \mathcal{E}_{\text{spatial}}| / |\mathcal{E}_{\text{causal}}|$ | > 0.3（空间约束但非完全重叠） | `spatial_metrics.py` |

---

## 5. 综合评估报告模板

每次完整运行后自动生成 `results/evaluation_report.json`，结构如下：

```json
{
  "timestamp": "2026-02-15T12:00:00",
  "config": { "...HyperSCAConfig 全量..." },
  "step1_embedding": {
    "distortion": 0.15,
    "delta_hyperbolicity": 0.08,
    "ari": 0.72,
    "nmi": 0.68,
    "silhouette": 0.41,
    "branch_purity": 0.78,
    "training_loss_final": 1234.5,
    "training_epochs": 300
  },
  "step2_causal": {
    "n_edges": 45,
    "graph_sparsity": 0.03,
    "mean_bootstrap_freq": 0.71,
    "falsification_pvalue": 0.12,
    "hsic_z_int_z_ext": 0.002,
    "z_ext_neighbor_r2": 0.45,
    "z_int_neighbor_r2": 0.04,
    "known_axis_recall": 0.67,
    "direction_accuracy": 0.85,
    "signaling_flow_completeness": 0.40
  },
  "step3_perturbation": {
    "INHBA_ko": {
      "r2_mean": 0.85,
      "r2_var": 0.52,
      "pcc_median": 0.78,
      "mse": 0.032,
      "marker_direction_accuracy": 0.90,
      "deg_overlap_jaccard": 0.25
    },
    "POSTN_ko": { "..." },
    "MFAP2_ko": { "..." }
  },
  "spatial_consistency": {
    "INHBA_ko": {
      "morans_i_obs": 0.35,
      "morans_i_cf": 0.22,
      "delta_morans_i": -0.13,
      "gradient_decay_r2": 0.41,
      "characteristic_length_um": 150.0,
      "propagation_depth": 3
    }
  }
}
```

---

## 6. 评估优先级分配

| 阶段 | 指标类别 | P0 必须 | P1 必须 | P2 可选 |
|------|----------|---------|---------|---------|
| Step 1 | 几何保真度 | Distortion | Triplet Acc | $\delta$-Hyperbolicity |
| Step 1 | 生物一致性 | ARI | NMI, Silhouette | Branch Purity |
| Step 2 | 结构可信度 | Bootstrap Freq | Falsification, Arrow Strength | -- |
| Step 2 | 解缠质量 | HSIC | Z_ext/Z_int predictivity | -- |
| Step 2 | 生物验证 | -- | Known Axis Recovery | Direction Acc, Flow Completeness |
| Step 3 | 表达质量 | R² (mean) | PCC, MSE | R² (var), CRPS |
| Step 3 | 生物一致性 | Marker Direction | DEG Overlap | Marker Magnitude Ranking |
| Step 3 | 空间一致性 | -- | Moran's I | Gradient Decay, Propagation Depth |
