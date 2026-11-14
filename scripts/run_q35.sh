#!/usr/bin/env bash
# Qwen3.5 全流程：扩充数据 → 在最小的模型上两轮扫描（验证集）→ 用选出的配置在各个规模上做主对比（测试集）。
# 扫描都是单变量的：第一轮在默认结构上分别只改学习率、只改写入缩放；第二轮逐项关闭一个部件。
# 更细的单变量系列（层数、宽度、种子、样本数等）见 scripts/ofat_q35.sh。
# 中断后重新运行会续跑；日志在 results/run_q35.log。
#   12GB 显卡:  bash scripts/run_q35.sh
#   80GB 显卡:  MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B" bash scripts/run_q35.sh
set -uo pipefail
source "$(dirname "$0")/_common.sh"
MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"}
SWEEP_BASE=${SWEEP_BASE:-Qwen/Qwen3.5-0.8B}
SWEEP_LIMIT=${SWEEP_LIMIT:-3000}     # 扫描时每个试验只用这么多条训练样本
TRAIN_N=${TRAIN_N:-10000}
GEN_N=${GEN_N:-50}
SW=${SW:-results/q35_sweep}
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p results
need_cuda || exit 1
{
  echo "==== 开始 $(date '+%F %T') ===="
  if [ ! -f data/law_val.jsonl ] || [ "$(wc -l < data/law_train.jsonl)" -lt "$TRAIN_N" ]; then
    "$PYTHON" scripts/prepare_data.py --train "$TRAIN_N" --val 500 --test 1000 --general 1000 --general-val 500
  fi
  t() { BASE=$SWEEP_BASE OUT=$SW LIMIT=$SWEEP_LIMIT bash scripts/trial.sh "$@" || true; }

  echo "---- 扫描第一轮：学习率与写入缩放（第二大脑）、学习率（LoRA） ----"
  t b_lr1e-3        brain --lr 1e-3
  t l_lr2e-4        lora --lora-r 0 --lr 2e-4
  t b_lr3e-4        brain --lr 3e-4
  t b_lr3e-3        brain --lr 3e-3
  t b_lr1e-3_c4x    brain --lr 1e-3 --write-scale 0.25
  t b_lr1e-3_c025x  brain --lr 1e-3 --write-scale 0.015625
  t l_lr1e-4        lora --lora-r 0 --lr 1e-4
  t l_lr5e-4        lora --lora-r 0 --lr 5e-4
  t l_lr1e-3        lora --lora-r 0 --lr 1e-3
  B1=$("$PYTHON" scripts/pick_best.py "$SW" b_) || exit 1

  echo "---- 扫描第二轮：在第一轮选出的配置上逐项关闭一个部件（单变量；只报告，不参与选择） ----"
  # shellcheck disable=SC2086
  {
    t a_noinject  brain $B1 --no-inject
    t a_nologit   brain $B1 --no-logit
    t a_nonorm    brain $B1 --no-norm-in
  }
  "$PYTHON" scripts/summarize.py "$SW" | tee "$SW/summary.md"
  BRAIN_ARGS=${BRAIN_ARGS:-$B1}                # 主对比用第一轮选出的配置；也可以从外面指定
  LORA_ARGS=${LORA_ARGS:-$("$PYTHON" scripts/pick_best.py "$SW" l_)} || exit 1
  echo "第二大脑用: $BRAIN_ARGS | LoRA 用: $LORA_ARGS"

  echo "---- 主对比：各个规模，完整训练集，测试集评测 ----"
  for m in $MODELS; do
    tag=$(basename "$m" | tr 'A-Z' 'a-z')
    echo "==== $m $(date '+%T') ===="
    BASE=$m OUT=results/$tag LIMIT=0 GEN_N=$GEN_N BRAIN_ARGS="$BRAIN_ARGS" LORA_ARGS="$LORA_ARGS" \
      bash scripts/run_experiment.sh || echo "!! $m 有失败的步骤"
  done
  echo "==== 全部完成 $(date '+%F %T') ===="
} 2>&1 | tee -a results/run_q35.log
