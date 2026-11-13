# 被其它脚本 source：切到项目根目录，定位 Python，定义 brain 命令。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHONUNBUFFERED=1
# 大文件走普通 HTTP 下载（经代理时最稳妥）；想用 xet 加速就设 HF_HUB_DISABLE_XET=0
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
brain() { "$PYTHON" -m secondbrain.cli "$@"; }
need_cuda() {  # 没有可用显卡时直接报错，避免不知不觉在 CPU 上跑几个小时；确实要用 CPU 就设 ALLOW_CPU=1
  if [ "${ALLOW_CPU:-0}" = 1 ]; then return 0; fi
  "$PYTHON" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "错误：PyTorch 检测不到可用的 CUDA 显卡。先运行 bash scripts/setup_gpu.sh 检查环境；确实要用 CPU 请加 ALLOW_CPU=1。" >&2
    return 1
  }
}
