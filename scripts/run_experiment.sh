#!/usr/bin/env bash
# 主对比实验：第二大脑 vs LoRA（参数量自动对齐）vs 角色提示词 vs 原模型，在测试集上评测并生成样例。
# 先运行 scripts/prepare_data.py 准备数据。可用环境变量改配置，例如
#   BASE=Qwen/Qwen3.5-4B OUT=results/q35_4b BRAIN_ARGS="--lr 3e-4" LORA_ARGS="--lr 2e-4" bash scripts/run_experiment.sh
# 已经存在的结果会跳过，中断后重新运行即可续跑；显存不够时自动减半批量重试。
set -uo pipefail
source "$(dirname "$0")/_common.sh"
BASE=${BASE:-Qwen/Qwen2.5-0.5B-Instruct}
EPOCHS=${EPOCHS:-1}; BS=${BS:-4}; ACCUM=${ACCUM:-1}; MAXLEN=${MAXLEN:-384}
LORA_R=${LORA_R:-0}          # 0 = 自动取与第二大脑参数量对齐的秩
GEN_N=${GEN_N:-20}; LIMIT=${LIMIT:-0}
OUT=${OUT:-results}
BRAIN_ARGS=${BRAIN_ARGS:-}   # 传给大脑训练的额外参数，如 "--n-taps 24 --d-side 96 --lr 3e-4"
LORA_ARGS=${LORA_ARGS:-}     # 传给 LoRA 训练的额外参数，如 "--lr 1e-4"
TRAIN=${TRAIN:-data/law_train.jsonl}; TEST=${TEST:-data/law_test.jsonl}; GTEST=${GTEST:-data/general_test.jsonl}
VAL_ARGS=""; [ -f data/law_val.jsonl ] && VAL_ARGS="--val data/law_val.jsonl"
SYSTEM="你是一名专业的中国法律咨询律师，请依据现行法律法规，准确、严谨地回答当事人的问题。"
mkdir -p "$OUT"
need_cuda || exit 1
FAILED=""

retry() {  # retry <日志> <命令...>：命令里的 @BS@ / @ACCUM@ 会被替换；显存不足时减半批量重试
  local log=$1; shift
  local bs=$BS accum=$ACCUM
  while :; do
    local cmd=("${@//@BS@/$bs}"); cmd=("${cmd[@]//@ACCUM@/$accum}")
    if "${cmd[@]}" > "$log" 2>&1; then return 0; fi
    if grep -qi "out of memory" "$log" && [ "$bs" -gt 1 ]; then
      bs=$((bs / 2)); accum=$((accum * 2)); echo "  显存不足，改用 bs=$bs accum=$accum 重试"
    else
      return 1
    fi
  done
}
ev() {  # ev <名字> <其余参数...>
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "跳过已完成的评测: $name"; return; fi
  echo "评测 $name"
  retry "$OUT/$name.eval.log" brain eval --base "$BASE" --role "$TEST" --general "$GTEST" --batch-size @BS@ \
    --max-len "$MAXLEN" --gen-out "$OUT/$name.gen.jsonl" --gen-n "$GEN_N" --max-new-tokens 160 \
    --report "$OUT/$name.json" "$@" || { echo "评测失败: $name"; tail -3 "$OUT/$name.eval.log"; FAILED="$FAILED eval:$name"; }
}
tr() {  # tr <名字> <输出文件> <其余参数...>
  local name=$1 out=$2; shift 2
  if [ -f "$out" ]; then echo "跳过已完成的训练: $out"; return 0; fi
  echo "训练 $name"
  # shellcheck disable=SC2086
  if retry "$OUT/$name.train.log" brain train --base "$BASE" --data "$TRAIN" $VAL_ARGS --out "$out" --limit "$LIMIT" \
      --epochs "$EPOCHS" --batch-size @BS@ --accum @ACCUM@ --max-len "$MAXLEN" "$@"; then
    grep -E "可训练参数|LoRA 秩|验证 loss|已保存" "$OUT/$name.train.log" | tail -4
  else
    echo "训练失败: $name"; tail -3 "$OUT/$name.train.log"; rm -f "$out"; FAILED="$FAILED train:$name"; return 1
  fi
}

ev base
ev prompt --system "$SYSTEM"
# shellcheck disable=SC2086
tr brain "$OUT/law.brain" $BRAIN_ARGS && ev brain --brain "$OUT/law.brain"
# shellcheck disable=SC2086
tr lora "$OUT/law.lora" --method lora --lora-r "$LORA_R" $LORA_ARGS && ev lora --lora "$OUT/law.lora"
"$PYTHON" scripts/summarize.py "$OUT"
if [ -n "$FAILED" ]; then echo "失败的步骤:$FAILED"; exit 1; fi
