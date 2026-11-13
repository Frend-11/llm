#!/usr/bin/env bash
# 本机（显卡）一条命令跑完首轮全部实验，日志在 results/run_all.log。中断后重新运行会续跑。
#   bash scripts/run_all.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
mkdir -p results
{
  date
  [ -f data/law_train.jsonl ] || "$PYTHON" scripts/prepare_data.py
  # A. 0.5B、1 轮：与云端 CPU 试验同配置（旧结构，不做读入归一化），核对两边结果是否一致
  OUT=results/gpu_0.5b_pilot BRAIN_ARGS="--no-norm-in" GEN_N=50 bash scripts/run_experiment.sh
  # B. 0.5B 上的结构 / 超参扫描
  OUT=results/sweep_0.5b bash scripts/sweep.sh
  # C. 1.5B、3 轮：默认大脑 vs 参数量对齐的 LoRA
  BASE=Qwen/Qwen2.5-1.5B-Instruct EPOCHS=3 ACCUM=2 LORA_R=32 GEN_N=50 OUT=results/gpu_1.5b bash scripts/run_experiment.sh
  date
} 2>&1 | tee -a results/run_all.log
