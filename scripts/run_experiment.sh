#!/usr/bin/env bash
# 主对比实验：第二大脑 vs LoRA（参数量对齐）vs 角色提示词 vs 原模型。
# 先运行 scripts/prepare_data.py 准备数据。可用环境变量改配置，例如
#   BASE=Qwen/Qwen2.5-1.5B-Instruct EPOCHS=3 ACCUM=2 LORA_R=32 OUT=results/gpu_1.5b bash scripts/run_experiment.sh
# 已经存在的结果会跳过，所以中断后重新运行即可续跑。
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BASE=${BASE:-Qwen/Qwen2.5-0.5B-Instruct}
EPOCHS=${EPOCHS:-1}
BS=${BS:-4}
ACCUM=${ACCUM:-1}
MAXLEN=${MAXLEN:-384}
LORA_R=${LORA_R:-56}     # 0.5B 上 r=56 ≈ 7.6M 参数，与默认大脑 (7.7M) 对齐；1.5B 上用 r=32 ≈ 8.7M 对齐 9.7M
GEN_N=${GEN_N:-20}
OUT=${OUT:-results}
BRAIN_ARGS=${BRAIN_ARGS:-}   # 传给大脑训练的额外参数，如 "--n-taps 24 --d-side 96"
SYSTEM="你是一名专业的中国法律咨询律师，请依据现行法律法规，准确、严谨地回答当事人的问题。"
mkdir -p "$OUT"
need_cuda
ev() {  # ev <名字> <其余参数...>
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "跳过已完成的评测: $name"; return; fi
  brain eval --base "$BASE" --role data/law_test.jsonl --general data/general_test.jsonl --batch-size "$BS" \
    --max-len "$MAXLEN" --gen-out "$OUT/$name.gen.jsonl" --gen-n "$GEN_N" --max-new-tokens 160 \
    --report "$OUT/$name.json" "$@"
}
tr() {  # tr <输出文件> <其余参数...>
  local out=$1; shift
  if [ -f "$out" ]; then echo "跳过已完成的训练: $out"; return; fi
  brain train --base "$BASE" --data data/law_train.jsonl --out "$out" \
    --epochs "$EPOCHS" --batch-size "$BS" --accum "$ACCUM" --max-len "$MAXLEN" "$@"
}

# shellcheck disable=SC2086
tr "$OUT/law.brain" $BRAIN_ARGS
ev brain --brain "$OUT/law.brain"
tr "$OUT/law.lora" --method lora --lora-r "$LORA_R"
ev lora --lora "$OUT/law.lora"
ev prompt --system "$SYSTEM"
ev base
"$PYTHON" scripts/summarize.py "$OUT"
