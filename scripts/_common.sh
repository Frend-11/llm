# 被其它脚本 source：定位 Python，定义 brain 命令。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHONUNBUFFERED=1
brain() { "$PYTHON" -m secondbrain.cli "$@"; }
