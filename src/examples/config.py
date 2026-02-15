"""HyperSCA example 公共配置"""
from pathlib import Path

# 项目根目录（自动检测）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
CHROMIUM_DIR = DATA_DIR / "Chromium_HumanColon_Oliveira"
VISIUM_DIR = DATA_DIR / "Visium_HumanColon_Oliveira"
VISIUMHD_DIR = DATA_DIR / "VisiumHD_HumanColon_Oliveira"
XENIUM_DIR = DATA_DIR / "Xenium_HumanColon_Oliveira"

# 输出路径
RESULTS_DIR = PROJECT_ROOT / "results" / "examples"

# 空间图参数
SPATIAL_K = 6  # kNN 邻居数
