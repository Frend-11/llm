#!/usr/bin/env bash
# 单次试验：训练 → 在验证集上评测（只算 loss，不生成）→ 删除权重。
# 结果写到 $OUT/<名字>.json 和 $OUT/<名字>.train.json，已有结果则跳过；显存不够时自动减半批量重试。
# 用法: OUT=results/s1 LIMIT=3000 bash scripts/trial.sh <名字> <brain|lora> [训练参数...]
set -uo pipefail
source "$(dirname "$0")/_common.sh"
BASE=${BASE:-Qwen/Qwen3.5-0.8B}
EPOCHS=${EPOCHS:-1}; BS=${BS:-4}; ACCUM=${ACCUM:-1}; MAXLEN=${MAXLEN:-384}; LIMIT=${LIMIT:-0}
TRAIN=${TRAIN:-data/law_train.jsonl}; VAL=${VAL:-data/law_val.jsonl}; GVAL=${GVAL:-data/general_val.jsonl}
OUT=${OUT:-results/trials}; KEEP=${KEEP:-0}
name=$1; method=$2; shift 2
mkdir -p "$OUT"
if [ -f "$OUT/$name.json" ]; then echo "跳过已完成: $name"; exit 0; fi
need_cuda || exit 1
w="$OUT/$name.$method"; flag=--brain; [ "$method" = lora ] && flag=--lora
log="$OUT/$name.log"; : > "$log"
echo "$*" > "$OUT/$name.args"   # 记下训练参数，供 pick_best.py 选出最优配置后复用
bs=$BS; accum=$ACCUM
while :; do
  echo "=== $name (bs=$bs accum=$accum) ===" | tee -a "$log"
  if brain train --base "$BASE" --data "$TRAIN" --val "$VAL" --out "$w" --method "$method" --limit "$LIMIT" \
       --epochs "$EPOCHS" --batch-size "$bs" --accum "$accum" --max-len "$MAXLEN" "$@" >> "$log" 2>&1 \
     && brain eval --base "$BASE" "$flag" "$w" --role "$VAL" --general "$GVAL" --batch-size "$bs" --max-len "$MAXLEN" \
       --gen-n 0 --report "$OUT/$name.json" >> "$log" 2>&1; then
    mv "$w.train.json" "$OUT/$name.train.json"
    grep -E "可训练参数|LoRA 秩|验证 loss|已保存" "$log" | tail -4
    [ "$KEEP" = 1 ] || rm -f "$w"
    exit 0
  fi
  if grep -qi "out of memory" "$log" && [ "$bs" -gt 1 ]; then
    bs=$((bs / 2)); accum=$((accum * 2)); echo "显存不足，改用 bs=$bs accum=$accum 重试" | tee -a "$log"
  else
    echo "失败: $name（见 $log）"; tail -5 "$log"; rm -f "$w"; exit 1
  fi
done
