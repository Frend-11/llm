#!/usr/bin/env bash
# 结构 / 超参扫描：每个配置训练后只算 loss（不生成），结果写到 $OUT/<名字>.json。
# 已有结果的配置会跳过，可随时中断续跑。用法: bash scripts/sweep.sh
#   ONLY="b_default l_r56_lr2e-4" bash scripts/sweep.sh   只跑其中几个
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BASE=${BASE:-Qwen/Qwen2.5-0.5B-Instruct}
EPOCHS=${EPOCHS:-1}
BS=${BS:-4}
ACCUM=${ACCUM:-1}
MAXLEN=${MAXLEN:-384}
OUT=${OUT:-results/sweep}
ONLY=${ONLY:-}   # 只跑指定的配置，空格分隔，如 ONLY="b_default b_t24_d96"
mkdir -p "$OUT"

one() {  # one <名字> <brain|lora> <训练参数...>
  local name=$1 method=$2; shift 2
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi
  if [ -f "$OUT/$name.json" ]; then echo "跳过已完成: $name"; return; fi
  echo "=== $name ==="
  local w="$OUT/$name.$method" flag=--brain
  if [ "$method" = lora ]; then flag=--lora; fi
  # 单个配置失败（比如显存不够）不中断整个扫描，完整日志在 $OUT/$name.log
  if brain train --base "$BASE" --data data/law_train.jsonl --out "$w" --method "$method" \
       --epochs "$EPOCHS" --batch-size "$BS" --accum "$ACCUM" --max-len "$MAXLEN" "$@" > "$OUT/$name.log" 2>&1 \
     && brain eval --base "$BASE" "$flag" "$w" --role data/law_test.jsonl --general data/general_test.jsonl \
       --batch-size "$BS" --max-len "$MAXLEN" --gen-n 0 --report "$OUT/$name.json" >> "$OUT/$name.log" 2>&1; then
    grep -E "可训练参数|验证 loss|已保存" "$OUT/$name.log" || true
  else
    echo "失败: $name（见 $OUT/$name.log）"; FAILED="$FAILED $name"
  fi
  rm -f "$w"   # 扫描只留指标，不留权重
}
FAILED=""
need_cuda

# --- 第二大脑：默认结构（6 层挂载，宽 256，读入归一化）与学习率 ---
one b_default        brain
one b_lr3e-4         brain --lr 3e-4
one b_lr3e-3         brain --lr 3e-3
# --- 消融：读入归一化、两条通路 ---
one b_nonorm         brain --no-norm-in
one b_noinject       brain --no-inject
one b_nologit        brain --no-logit
# --- 参数量大致不变，挂载更多层、旁路更窄 ---
one b_t12_d160       brain --n-taps 12 --d-side 160
one b_t24_d96        brain --n-taps 24 --d-side 96
one b_t24_d160_nomlp brain --n-taps 24 --d-side 160 --mlp-ratio 0
one b_t24_d96_noinj  brain --n-taps 24 --d-side 96 --no-inject
# --- LoRA 基线：学习率与秩 ---
one l_r56_lr2e-4     lora --lora-r 56
one l_r56_lr1e-4     lora --lora-r 56 --lr 1e-4
one l_r56_lr5e-4     lora --lora-r 56 --lr 5e-4
one l_r16_lr2e-4     lora --lora-r 16

"$PYTHON" scripts/summarize.py "$OUT"
if [ -n "$FAILED" ]; then echo "失败的配置:$FAILED"; fi
