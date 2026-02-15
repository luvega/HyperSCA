# HyperSCA 环境搭建进度记录

- 记录日期: 2026-02-14
- 记录范围: 依赖安装、参考仓库克隆、环境验收
- 执行环境: Windows 10, Conda (`E:\ProgramData\Anaconda3`), GPU RTX 3070

## 1) 已完成事项

### 1.1 Conda 环境

- 环境名: `hypersca`
- Python: `3.10.19`
- `requirements-core.txt` 头部环境说明已更新为 Python 3.10

### 1.2 A 层（基础运行依赖）

- A1 科学计算: 已完成  
  `numpy scipy pandas scikit-learn matplotlib seaborn h5py pyyaml tqdm leidenalg igraph`
- A2 单细胞生态: 已完成  
  `anndata scanpy squidpy scvi-tools`
- A3 深度学习: 已完成（无需重复升级）  
  当前 `torch` 为 `2.6.0+cu124`
- A4 图学习: 已完成  
  `torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv`
- A5 双曲几何: 已完成  
  `geoopt`

### 1.3 B 层（因果与通讯依赖）

- B1: 已完成 `dowhy econml`
- B2: 已完成 `networkx pgmpy`
- B3: 已完成 `statsmodels pingouin`

### 1.4 C 层（扰动与生成依赖）

- 已完成: `scgen diffusers accelerate pyarrow openpyxl jupyter ipywidgets`
- 兼容性修复: `scgen` 从 `2.1.0` 升级到 `2.1.1`（GitHub 版本）

### 1.5 R 参考环境配置

- 已创建文件: `environment-r.yml`
- 说明: 本次仅生成配置文件，未实际安装 R 包

### 1.6 E 层参考代码仓库

`E:\HyperSCA\references` 下已就位:

- TopoLa（原有）
- scDHMap
- flowsig
- celcomen
- scgen
- CPA
- Squidiff
- CausCell
- DynPerturb
- dowhy

## 2) 验收结果

使用 `scripts/validate_env.py` 验收，结果通过。

### 2.1 核心导入通过

- `scanpy 1.11.5`
- `squidpy 1.6.5`
- `torch 2.6.0+cu124`
- `torch_geometric 2.7.0`
- `geoopt 0.5.1`
- `dowhy 0.14`
- `anndata 0.11.4`
- `scvi-tools 1.3.3`
- `econml 0.16.0`
- `pgmpy 1.0.0`
- `pingouin 0.5.5`
- `scgen 2.1.1`
- `diffusers 0.36.0`

### 2.2 GPU 可用性通过

- `torch.cuda.is_available() = True`
- CUDA 版本: `12.4`
- GPU 设备: `NVIDIA GeForce RTX 3070`
- CUDA Tensor 测试通过

### 2.3 数据目录检查通过

检测到 `data/` 下 4 个目录:

- `Chromium_HumanColon_Oliveira`
- `VisiumHD_HumanColon_Oliveira`
- `Visium_HumanColon_Oliveira`
- `Xenium_HumanColon_Oliveira`

## 3) 产出文件清单

- `requirements-core.txt`（环境注释已更新）
- `environment-r.yml`（新增）
- `scripts/validate_env.py`（新增）
- `SETUP_PROGRESS_2026-02-14.md`（本记录）

## 4) 后续建议

- 建议在首次正式训练/推理前，补充一次真实样本文件的读取测试（`.h5ad` / 空间转录组格式）。
- 如后续需要复现实验，可将本记录与 `pip freeze` 输出一起归档。
