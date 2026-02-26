# HyperSCA v0.2.0 (Draft)

> Release date: 2026-02-26  
> Type: Feature update (multi-source integration + repository cleanup)

## Overview

本版本新增多源结果级整合流程、靶点发现入口与跨样本评估模块，并同步完成仓库结构整理与文档更新。更新目标是提高流程可复现性与版本管理的可维护性。

## Highlights

- 新增多源整合主流程：支持 `scCRC_Neu + scCRC_IFNG + ST_CRC_MSS` 的 MVP 级联分析。
- 新增开放靶点发现流程：支持候选池构建、批量扰动与证据打分排序。
- 新增 canonical schema 构建脚本：统一样本/实体/特征/测量表结构，便于跨数据源对齐。
- 增加跨样本评估模块：包含 niche 聚类、边一致性、MMR 分层统计等能力。
- 提升训练与可视化稳定性：修复数值稳定细节并优化因果图可读性。
- 完成仓库边界清晰化：补充结构文档并强化忽略规则，减少本地产物误提交。

## What's Changed

### 1) New Pipelines and Scripts

- Added `scripts/build_canonical_schema.py`
- Added `scripts/run_mvp_integration.py`
- Added `scripts/run_target_discovery.py`
- Added stage entry scripts:
  - `scripts/run_step1.py`
  - `scripts/run_step2.py`
  - `scripts/run_step3.py`
- Added figure generation scripts:
  - `scripts/generate_mvp_figures.py`
  - `scripts/generate_step1_figures.py`
  - `scripts/generate_step2_figures.py`
  - `scripts/generate_step3_figures.py`
- Added data prep helper:
  - `scripts/prepare_h5ad.py`

### 2) Evaluation and Core Updates

- Added cross-sample evaluation module:
  - `src/evaluation/cross_sample_metrics.py`
- Exported new evaluation API in:
  - `src/evaluation/__init__.py`
- Improved hyperbolic wrapped normal numerical stability:
  - `src/models/hyperbolic/wrapped_normal.py`
- Improved config YAML loading encoding:
  - `src/pipeline/config.py`
- Improved Step1 robustness (NaN/Inf guard, gene inclusion, gradient clipping):
  - `src/pipeline/step1_embedding.py`
- Improved causal DAG label layout/readability:
  - `src/visualization/causal.py`

### 3) Docs and Repository Governance

- Updated project README with integration-focused workflow:
  - `README.md`
- Added architecture diagram source:
  - `docs/pipeline_architecture.mmd`
- Added repository scope guide:
  - `docs/repository_structure.md`
- Updated ignore rules to prevent local artifact leakage:
  - `.gitignore`

## Compatibility / Breaking Changes

- No API-level breaking changes announced in this release.
- 运行流程更偏向“多源结果级整合”场景；若使用旧流程，请继续调用 stage 脚本并核对参数。

## Recommended Upgrade Steps

```bash
git pull origin main
pip install -r requirements.txt
python scripts/validate_env.py
```

可选快速验证：

```bash
python scripts/build_canonical_schema.py
python scripts/run_mvp_integration.py --embedding-mode both --max-targets 10
```

## Notes

- 本仓库默认不提交 `data/`, `results/`, `references/` 等大体量目录。
- 仓库边界与保留策略请参考 `docs/repository_structure.md`。

## Full Changelog

- Commit: `9f74fe943fc6aa4a99b198c68649e5aa491bcde4`
- Compare: `https://github.com/luvega/HyperSCA/compare/67c443c...9f74fe9`
