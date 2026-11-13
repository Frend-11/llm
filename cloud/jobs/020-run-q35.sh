#!/bin/bash
# 80GB 显卡上的全流程：扩充数据 → 0.8B 上两轮扫描 → 0.8B 到 27B 的主对比，最后尝试 35B-A3B。
# 122B-A10B 和 397B-A17B 在单张 80GB 显卡上放不下，不在此列。
set -x
cd /opt/sb/llm
export HF_HUB_DISABLE_XET=0   # 机房网络直接用 xet 加速下载
MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B" \
  bash scripts/run_q35.sh
