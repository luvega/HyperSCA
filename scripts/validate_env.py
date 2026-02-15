"""HyperSCA 环境验收脚本"""
import sys

print("=" * 60)
print("HyperSCA Environment Validation")
print("=" * 60)

errors = []

# 1. Core imports
print("\n[1] Core package imports...")
try:
    import scanpy as sc; print(f"  scanpy {sc.__version__} ... OK")
except Exception as e: errors.append(f"scanpy: {e}")

try:
    import squidpy as sq; print(f"  squidpy {sq.__version__} ... OK")
except Exception as e: errors.append(f"squidpy: {e}")

try:
    import torch; print(f"  torch {torch.__version__} ... OK")
except Exception as e: errors.append(f"torch: {e}")

try:
    import torch_geometric; print(f"  torch_geometric {torch_geometric.__version__} ... OK")
except Exception as e: errors.append(f"torch_geometric: {e}")

try:
    import geoopt; print(f"  geoopt {geoopt.__version__} ... OK")
except Exception as e: errors.append(f"geoopt: {e}")

try:
    import dowhy; print(f"  dowhy {dowhy.__version__} ... OK")
except Exception as e: errors.append(f"dowhy: {e}")

try:
    import anndata; print(f"  anndata {anndata.__version__} ... OK")
except Exception as e: errors.append(f"anndata: {e}")

try:
    import scvi; print(f"  scvi-tools {scvi.__version__} ... OK")
except Exception as e: errors.append(f"scvi-tools: {e}")

try:
    import econml; print(f"  econml {econml.__version__} ... OK")
except Exception as e: errors.append(f"econml: {e}")

try:
    import pgmpy; print(f"  pgmpy ... OK")
except Exception as e: errors.append(f"pgmpy: {e}")

try:
    import pingouin; print(f"  pingouin {pingouin.__version__} ... OK")
except Exception as e: errors.append(f"pingouin: {e}")

try:
    import scgen; print(f"  scgen {scgen.__version__} ... OK")
except Exception as e: errors.append(f"scgen: {e}")

try:
    import diffusers; print(f"  diffusers {diffusers.__version__} ... OK")
except Exception as e: errors.append(f"diffusers: {e}")

# 2. GPU check
print("\n[2] GPU check...")
try:
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        t = torch.randn(3, 3).cuda()
        print(f"  GPU tensor test: {t.device} ... OK")
    else:
        errors.append("CUDA not available")
except Exception as e:
    errors.append(f"GPU test: {e}")

# 3. PyG extensions
print("\n[3] PyG extensions...")
try:
    import torch_scatter; print("  torch_scatter ... OK")
except Exception as e: errors.append(f"torch_scatter: {e}")

try:
    import torch_sparse; print("  torch_sparse ... OK")
except Exception as e: errors.append(f"torch_sparse: {e}")

try:
    import torch_cluster; print("  torch_cluster ... OK")
except Exception as e: errors.append(f"torch_cluster: {e}")

# 4. Data readability
print("\n[4] Data readability check...")
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
if os.path.exists(data_dir):
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"  Found {len(subdirs)} data directories: {subdirs}")
else:
    print(f"  Data directory not found at {data_dir}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"VALIDATION FAILED - {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All validations PASSED!")
    print("=" * 60)
