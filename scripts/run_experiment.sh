#!/usr/bin/env bash
# 首轮对比实验：第二大脑 vs LoRA（参数量对齐）vs 角色提示词 vs 原模型。
# 先运行 python scripts/prepare_data.py 准备数据。
# 5070 上建议: BASE=Qwen/Qwen2.5-1.5B-Instruct EPOCHS=3 BS=8 bash scripts/run_experiment.sh
set -euo pipefail
BASE=${BASE:-Qwen/Qwen2.5-0.5B-Instruct}
EPOCHS=${EPOCHS:-1}
BS=${BS:-4}
ACCUM=${ACCUM:-1}
MAXLEN=${MAXLEN:-384}
LORA_R=${LORA_R:-56}     # 0.5B 上 r=56 ≈ 7.6M 参数，与默认大脑 (7.7M) 对齐
GEN_N=${GEN_N:-20}
OUT=${OUT:-results}
SYSTEM="你是一名专业的中国法律咨询律师，请依据现行法律法规，准确、严谨地回答当事人的问题。"
mkdir -p "$OUT"
brain() { python -m secondbrain.cli "$@"; }
ev() {  # ev <名字> <其余参数...>
  local name=$1; shift
  brain eval --base "$BASE" --role data/law_test.jsonl --general data/general_test.jsonl --batch-size "$BS" \
    --max-len "$MAXLEN" --gen-out "$OUT/$name.gen.jsonl" --gen-n "$GEN_N" --max-new-tokens 160 \
    --report "$OUT/$name.json" "$@"
}
tr() {  # tr <输出文件> <其余参数...>
  local out=$1; shift
  brain train --base "$BASE" --data data/law_train.jsonl --out "$out" \
    --epochs "$EPOCHS" --batch-size "$BS" --accum "$ACCUM" --max-len "$MAXLEN" "$@"
}

tr "$OUT/law.brain"
ev brain --brain "$OUT/law.brain"
tr "$OUT/law.lora" --method lora --lora-r "$LORA_R"
ev lora --lora "$OUT/law.lora"
ev prompt --system "$SYSTEM"
ev base
python scripts/summarize.py "$OUT"
