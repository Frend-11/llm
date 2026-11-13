#!/usr/bin/env bash
# 本机（显卡）一条命令跑完首轮全部实验，日志在 results/run_all.log。中断后重新运行会续跑。
#   bash scripts/run_all.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
SMALL=${SMALL:-Qwen/Qwen2.5-0.5B-Instruct}
LARGE=${LARGE:-Qwen/Qwen2.5-1.5B-Instruct}
mkdir -p results
need_cuda
{
  echo "==== 开始 $(date '+%F %T') ===="
  [ -f data/law_train.jsonl ] || "$PYTHON" scripts/prepare_data.py
  # A. 小模型、1 轮：与云端 CPU 试验同配置（旧结构，不做读入归一化），核对两边结果是否一致
  BASE=$SMALL OUT=results/gpu_0.5b_pilot LORA_R=56 GEN_N=50 \
    BRAIN_ARGS="--no-norm-in --abs-write --write-scale 1 --lr 1e-3" bash scripts/run_experiment.sh
  # B. 小模型上的结构 / 超参扫描
  BASE=$SMALL OUT=results/sweep_0.5b bash scripts/sweep.sh
  # C. 大一号的模型、3 轮：默认大脑 vs 参数量对齐的 LoRA
  BASE=$LARGE EPOCHS=3 ACCUM=2 LORA_R=32 GEN_N=50 OUT=results/gpu_1.5b bash scripts/run_experiment.sh
  echo "==== 全部完成 $(date '+%F %T') ===="
} 2>&1 | tee -a results/run_all.log
