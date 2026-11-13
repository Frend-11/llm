#!/usr/bin/env bash
# 在带 NVIDIA 显卡的本机上搭环境：在项目里建 .venv、装依赖、检查 CUDA、跑单元测试。
# 用法（在项目根目录）: bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  # 优先用 Python 3.13（代码在这个版本上测试过），装不上就用 uv 找到的默认版本
  uv venv --allow-existing --python 3.13 .venv || uv venv --allow-existing .venv
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -e ".[dev]"
fi
# 驱动较旧时 PyPI 上默认的 CUDA 版本可能用不了：依次换成较老 CUDA 版本的 PyTorch 再试
for idx in https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu126; do
  .venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && break
  command -v nvidia-smi >/dev/null 2>&1 || break   # 没有显卡驱动就不用折腾了
  echo "当前 PyTorch 用不了显卡，改从 $idx 安装"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python --reinstall torch --index-url "$idx" || true
  else
    .venv/bin/pip install -q --force-reinstall torch --index-url "$idx" || true
  fi
done
.venv/bin/python - <<'PY'
import sys
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, f"{p.total_memory / 2**30:.1f} GB")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    print("bf16 矩阵乘法正常:", tuple((x @ x).shape))
PY
.venv/bin/python -m pytest -q
source scripts/_common.sh
need_cuda
echo "环境就绪。"
