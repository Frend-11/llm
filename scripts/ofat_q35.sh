#!/usr/bin/env bash
# 单变量实验：固定一个基线，每个试验只改一个因素。结果与第一轮扫描放在同一目录，已有的试验会跳过。
#   模型 Qwen3.5-0.8B，训练集前 3000 条，1 轮，验证集评测（不碰测试集）。
#   第二大脑基线 B0：--lr 1e-3 --write-scale 0.25，其余默认（挂 6 层、宽 256、MLP×4、归一化读入、相对写入、两条写通路）
#                   = 已有试验 b_lr1e-3_c4x
#   LoRA 基线 L0：   --lora-r 0（自动取秩 27）--lr 2e-4 = 已有试验 l_lr2e-4
# 用法: bash scripts/ofat_q35.sh        只跑其中几个: ONLY="v_c1 v_seed1" bash scripts/ofat_q35.sh
set -uo pipefail
source "$(dirname "$0")/_common.sh"
export BASE=${BASE:-Qwen/Qwen3.5-0.8B} OUT=${OUT:-results/q35_sweep} HF_HUB_DISABLE_PROGRESS_BARS=1
N=${N:-3000}
ONLY=${ONLY:-}
B0="--lr 1e-3 --write-scale 0.25"
L0="--lr 2e-4"
need_cuda || exit 1
t() {  # t <训练样本数> <名字> <brain|lora> <参数...>
  local n=$1 name=$2; shift
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi
  LIMIT=$n bash scripts/trial.sh "$@" || true
}
{
  echo "==== 单变量实验开始 $(date '+%F %T') ===="
  # ---- 第二大脑：每行只有一个参数不同于 B0 ----
  t $N v_c1       brain --lr 1e-3 --write-scale 1.0       # 写入缩放 c
  t $N v_c4       brain --lr 1e-3 --write-scale 4.0
  t $N v_lr3e-4   brain --lr 3e-4 --write-scale 0.25      # 学习率
  t $N v_lr3e-3   brain --lr 3e-3 --write-scale 0.25
  t $N v_taps3    brain $B0 --n-taps 3                    # 挂载层数（参数量随之变化）
  t $N v_taps12   brain $B0 --n-taps 12
  t $N v_taps24   brain $B0 --n-taps 24
  t $N v_d128     brain $B0 --d-side 128                  # 旁路宽度（参数量随之变化）
  t $N v_d512     brain $B0 --d-side 512
  t $N v_mlp0     brain $B0 --mlp-ratio 0                 # 旁路块 MLP 扩张倍数
  t $N v_mlp1     brain $B0 --mlp-ratio 1
  t $N v_abs      brain $B0 --abs-write                   # 关掉相对写入
  t $N v_seed1    brain $B0 --seed 1                      # 随机种子（估计噪声大小）
  t $N v_seed2    brain $B0 --seed 2
  t 1000  v_n1000  brain $B0                              # 训练样本数
  t 10000 v_n10000 brain $B0
  # ---- LoRA：每行只有一个参数不同于 L0 ----
  t $N w_r8       lora $L0 --lora-r 8                     # 秩（参数量随之变化）
  t $N w_r16      lora $L0 --lora-r 16
  t $N w_r54      lora $L0 --lora-r 54
  t $N w_seed1    lora $L0 --lora-r 0 --seed 1            # 随机种子
  t $N w_seed2    lora $L0 --lora-r 0 --seed 2
  t 1000  w_n1000  lora $L0 --lora-r 0                    # 训练样本数
  t 10000 w_n10000 lora $L0 --lora-r 0
  "$PYTHON" scripts/ofat_report.py "$OUT" | tee "$OUT/ofat.md"
  echo "==== 单变量实验完成 $(date '+%F %T') ===="
} 2>&1 | tee -a results/ofat_q35.log
