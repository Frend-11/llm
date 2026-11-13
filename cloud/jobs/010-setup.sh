#!/bin/bash
# 服务器初始化：等显卡驱动就绪，解压代码，建虚拟环境并自检。
set -x
cd /opt/sb
for i in $(seq 1 120); do nvidia-smi > /dev/null 2>&1 && break; sleep 15; done
nvidia-smi || echo "!!! 没有可用的 NVIDIA 驱动"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /; free -g | sed -n 2p; nproc; head -2 /etc/os-release
/opt/sb/s3 get code/llm.tar.gz /opt/sb/llm.tar.gz && /opt/sb/s3 get code/REV /opt/sb/REV && cat /opt/sb/REV
mkdir -p /opt/sb/llm && tar -xzf /opt/sb/llm.tar.gz -C /opt/sb/llm
cd /opt/sb/llm && bash scripts/setup_gpu.sh
