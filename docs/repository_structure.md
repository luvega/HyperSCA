# HyperSCA Repository Structure (GitHub Scope)

本文件用于约束仓库中“必要源代码”和“示例入口”的边界，避免提交本地数据或中间产物。

## 1) 建议纳入版本控制（必要）

- `src/`：核心算法与流水线实现（HyperSCA 主体源码）
  - `src/models/`：双曲几何与模型实现
  - `src/causal/`：因果发现与信号流推断
  - `src/perturbation/`：反事实与空间传播
  - `src/pipeline/`：分阶段流程封装
  - `src/evaluation/`：评估指标
  - `src/visualization/`：可视化函数
- `scripts/`：可复现实验入口脚本
  - 阶段执行：`run_step1.py`, `run_step2.py`, `run_step3.py`
  - 数据整合：`build_canonical_schema.py`, `run_mvp_integration.py`
  - 发现流程：`run_target_discovery.py`
  - 作图脚本：`generate_step1_figures.py`, `generate_step2_figures.py`, `generate_step3_figures.py`, `generate_mvp_figures.py`
  - 示例运行：`run_example_01.py` ~ `run_example_04.py`, `run_examples_all.py`
- `tests/`：单元测试与流程测试
- `docs/`：技术文档与流程图
- `README.md`, `requirements.txt`, `LICENSE`

## 2) 示例与文档定位

- 轻量示例入口：优先使用 `scripts/run_example_*.py`
- 完整流程示例：
  - `scripts/run_mvp_integration.py`（多源 MVP）
  - `scripts/run_target_discovery.py`（开放靶点发现）
- 流程图文档：`docs/pipeline_architecture.mmd`

## 3) 不建议纳入版本控制（已忽略）

- 原始数据与大体量中间文件：`data/`, `results/`, `references/`
- 编辑器与本地环境：`.cursor/`, `.vscode/`, `venv/`, `.venv/`
- 缓存与临时文件：`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `nul`

## 4) GitHub 发布建议

- PR 或发布前，优先确认以下文件稳定：
  - `src/` 下新增/修改模块
  - `scripts/` 下新增入口与参数说明
  - `README.md` 的安装与快速运行命令
  - `tests/` 可运行且与代码一致
- 若包含新流程，建议在 `docs/` 同步更新一页结构/输入输出说明。
