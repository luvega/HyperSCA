# HyperSCA 本机环境对齐记录（2026-06-12）

## 结论

本机 HyperSCA 主线环境已重建并验收通过。当前稳定执行入口为：

```powershell
C:\h\python.exe
```

不建议将主线环境升级到 Python 3.13。Python 3.13 可作为独立实验环境存在，但当前 HyperSCA 主干依赖 `torch`、`torch-geometric`、`scvi-tools`、`squidpy`、`dowhy`、`pgmpy` 与 `commot`，在 Windows 上采用 Python 3.10 更稳妥。仓库主线代码风格和历史验证口径也仍是 Python 3.10。

## 已安装和校准的核心栈

| 组件 | 当前版本或状态 |
|---|---|
| Python | 3.10.20, conda-forge |
| NumPy | 2.2.6 |
| Scanpy | 1.11.5 |
| Squidpy | 1.6.5 |
| AnnData | 0.11.4 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3070 |
| PyTorch Geometric | 2.8.0 |
| scvi-tools | 1.3.3 |
| DoWhy | 0.14 |
| EconML | 0.16.0 |
| pgmpy | 1.1.2 |
| diffusers | 0.38.0 |
| POT (`ot`) | 0.9.6.post1 |
| COMMOT | 已导入通过 |
| pytest | 9.0.3 |

`scgen` 未安装，保持为可选历史 baseline，不影响 HyperSCA 主线运行。

## COMMOT 兼容补丁

COMMOT 当前发布包中仍有 NumPy 1.x 时代的 `np.Inf` 调用；NumPy 2.x 已移除此别名。为避免降级 NumPy 并破坏 `dowhy/pgmpy` 依赖约束，本机环境只对安装包做最小补丁：

```text
C:\h\Lib\site-packages\commot\_optimal_transport\_usot.py
```

补丁内容为将 `np.Inf` 替换为 `np.inf`。该补丁不修改 HyperSCA 仓库源码。

## 验收命令

```powershell
C:\h\python.exe scripts\validate_env.py
```

验收结果：所有必需依赖通过，CUDA 与 PyG 扩展通过，数据目录可读；仅 `scgen` 为 optional warning。

COMMOT/OT 导入验收：

```powershell
@'
import commot as ct
import ot
print(hasattr(ct.tl, "spatial_communication"))
print(ot.__version__)
'@ | C:\h\python.exe -
```

结果：`commot.tl.spatial_communication` 可用，`ot` 可导入。

## 推荐使用方式

直接指定解释器最稳：

```powershell
C:\h\python.exe scripts\validate_env.py
C:\h\python.exe -m pytest tests\test_causal_metrics.py -q
```

或按路径激活：

```powershell
conda activate C:\h
python scripts\validate_env.py
```

旧的 `E:\ProgramData\Anaconda3\envs\hypersca\python.exe` 在本机不存在；旧的失败环境 `C:\Users\xsui\.conda\envs\hypersca` 与 `C:\hsca` 已清理，避免误激活。
