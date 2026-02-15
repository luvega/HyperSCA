# Repository Guidelines

## Project Overview

**HyperSCA**（Hyperbolic Spatiotemporal Causal Analysis）是一个融合双曲几何嵌入、因果结构学习与深度生成模型的计算框架，用于解构结直肠癌肿瘤免疫微环境（TME），整合 scRNA-seq 与空间转录组（ST）数据。项目处于早期开发阶段。

## Project Structure & Module Organization

```text
HyperSCA/
├── HyperSCA/                # 主项目 Git 仓库
│   ├── README.md
│   └── LICENSE
├── data/                    # 数据目录（已 gitignore）
│   ├── Chromium_HumanColon_Oliveira/
│   ├── Visium_HumanColon_Oliveira/
│   ├── VisiumHD_HumanColon_Oliveira/
│   ├── Xenium_HumanColon_Oliveira/
│   └── ...
├── docs/                    # 项目设计文档
│   ├── technical_roadmap.md       # 学术化技术路线（三阶段：双曲嵌入、因果网络、反事实扰动）
│   ├── engineering_blueprint.md   # 工程落地蓝图（模块目录、接口、P0/P1/P2 里程碑）
│   ├── evaluation_suite.md        # 评估指标体系（嵌入质量、因果可信度、反事实质量、空间一致性）
│   ├── priority_and_risks.md      # 执行优先级与风险预案
│   └── examples_guide.md          # Example 运行指南
├── references/              # 参考实现仓库（只读）；代码通过 adapter 模式重写至 src/，禁止直接 import
├── src/                     # 核心代码目录
│   ├── data/                # 数据加载、预处理、空间图构建
│   │   ├── loaders.py             # 统一数据加载（Chromium/Visium/VisiumHD/Xenium）
│   │   └── validators.py          # 数据字段校验
│   ├── examples/            # Example 分析模块
│   │   ├── config.py              # Example 公共配置
│   │   ├── metadata_qc.py         # Chromium 元数据分析
│   │   ├── spatial_graph.py       # Visium 空间图构建
│   │   ├── segmentation_stats.py  # VisiumHD 分割统计
│   │   └── gene_panel_summary.py  # Xenium 面板摘要
│   ├── models/hyperbolic/   # 双曲几何嵌入（HVAE、Lorentz、Poincaré、Wrapped Normal）
│   ├── causal/              # 因果解缠、CMI 剪枝、因果图、信号流推断
│   ├── perturbation/        # 潜空间扰动算术、扩散反事实、空间传播
│   ├── evaluation/          # 嵌入/因果/反事实/空间一致性指标
│   ├── pipeline/            # 阶段 1/2/3 流水线编排与配置
│   └── utils/               # IO、日志、可视化
├── scripts/                 # 工具脚本
│   ├── validate_env.py            # 环境验收
│   ├── run_step1.py               # 阶段 1 运行入口
│   ├── run_step2.py               # 阶段 2 运行入口
│   ├── run_step3.py               # 阶段 3 运行入口
│   ├── run_example_01.py          # Example 01：Chromium 元数据 QC
│   ├── run_example_02.py          # Example 02：Visium 空间图构建
│   ├── run_example_03.py          # Example 03：VisiumHD 分割统计
│   ├── run_example_04.py          # Example 04：Xenium 面板摘要
│   └── run_examples_all.py        # 批量运行所有 Example
├── tests/                   # 单元与集成测试
├── notebooks/               # 交互式实验（下游分析、消融、Example 可视化）
│   ├── 01_chromium_metadata_qc.ipynb
│   ├── 02_visium_spatial_graph.ipynb
│   ├── 03_visiumhd_segmentation.ipynb
│   └── 04_xenium_gene_panel.ipynb
├── results/                 # 运行产物输出目录（CSV/PNG/JSON/MD）
│   └── examples/
│       ├── 01_chromium_metadata_qc/
│       ├── 02_visium_spatial_graph/
│       ├── 03_visiumhd_segmentation/
│       └── 04_xenium_gene_panel/
├── requirements-core.txt
├── requirements-research.txt
└── environment-r.yml        # R 环境参考配置
```

## Environment

- **Conda 环境名**: `hypersca`
- **Conda 环境路径**: `E:\ProgramData\Anaconda3\envs\hypersca`
- **Python 解释器**: `E:\ProgramData\Anaconda3\envs\hypersca\python.exe`
- **Python 版本**: 3.10
- **CUDA**: 12.4
- **GPU**: NVIDIA GeForce RTX 3070
- **系统 Python**（仅有 torch，缺少其余依赖）: `C:\Python313\python.exe` (3.13)

> **重要**: 运行脚本时必须使用 conda 环境中的 Python，路径为:
> `E:\ProgramData\Anaconda3\envs\hypersca\python.exe`
> 或先激活环境: `conda activate hypersca`

## Build, Test, and Development Commands

```bash
# 激活 Conda 环境
conda activate hypersca

# 或直接使用完整路径
E:\ProgramData\Anaconda3\envs\hypersca\python.exe scripts/validate_env.py

# 安装依赖（按顺序）
pip install -r requirements-core.txt
pip install -r requirements-research.txt

# PyTorch + CUDA 需从官方源安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# PyG 需指定 wheel 源（与 torch 版本对齐）
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 验收环境
python scripts/validate_env.py
```

## Coding Style & Naming Conventions

- **语言**：Python 3.10，遵循 PEP 8
- **缩进**：4 空格
- **命名**：模块/函数 `snake_case`，类 `PascalCase`
- **文档字符串**：建议使用 NumPy/SciPy 风格 docstring
- **注释**：中文或英文均可

## Testing Guidelines

- 当前主要验收方式：`python scripts/validate_env.py`（检查核心包导入、GPU、数据目录）
- 后续将引入 pytest，测试命名约定：`test_<模块名>_<功能>.py`

## Commit & Pull Request Guidelines

- **Commit**：建议使用 Conventional Commits（如 `feat:`, `fix:`, `docs:`）
- **PR**：需描述变更动机、关联 Issue（如有）
- `data/` 已被 gitignore，**禁止提交大文件**；`references/` 仅作参考，不直接修改。

## Important Notes

- **docs/**：项目设计文档（技术路线、工程蓝图、评估指标、风险预案、Example 运行指南）
- **references/**：仅作参考，代码通过 adapter 模式重写至 `src/`，禁止直接 `import`
- **src/**：核心代码目录，包含 `data/`（数据加载与校验）、`examples/`（Example 分析模块）等子包
- **results/**：运行产物输出目录，按 Example 编号分子目录存放 CSV/PNG/JSON/MD 产物
- **notebooks/**：交互式 Notebook，4 个 Example 对应 4 个 `.ipynb` 文件
- 删除文件时使用 `mv` 移入 `.Trash`，避免使用 `rm`
- 主 Git 仓库位于 `HyperSCA/` 子目录
