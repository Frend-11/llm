#!/usr/bin/env bash
# 在带 NVIDIA 显卡的本机上搭环境：在项目里建 .venv、装依赖、检查 CUDA、跑单元测试。
# 用法（在项目根目录）: bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  uv venv --allow-existing .venv
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -e ".[dev]"
fi
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, f"{p.total_memory / 2**30:.1f} GB")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    print("bf16 矩阵乘法正常:", tuple((x @ x).shape))
PY
.venv/bin/python -m pytest -q
