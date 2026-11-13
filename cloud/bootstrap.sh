#!/bin/bash
# 云服务器首次开机脚本（作为 user_data 传入）。__占位符__ 由 cloud/ctl.py 在创建服务器时替换。
# 只做一件事：装好并启动任务代理 agent.py；其余工作都通过任务队列下发。
exec > /var/log/sb-bootstrap.log 2>&1
set -x
mkdir -p /opt/sb && cd /opt/sb
umask 077
cat > /opt/sb/env <<'ENV'
SB_S3_ENDPOINT=__ENDPOINT__
SB_S3_BUCKET=__BUCKET__
AWS_ACCESS_KEY_ID=__ACCESS__
AWS_SECRET_ACCESS_KEY=__SECRET__
AWS_DEFAULT_REGION=us-east-1
ENV
set -a; . /opt/sb/env; set +a
export HOME=/root PATH=/root/.local/bin:$PATH
upload_log() {  # 即使后面的步骤失败，也把开机日志传回去便于排查
  curl -sS --aws-sigv4 "aws:amz:us-east-1:s3" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
    -T /var/log/sb-bootstrap.log "$SB_S3_ENDPOINT/$SB_S3_BUCKET/logs/bootstrap.log" || true
}
trap upload_log EXIT
PY=""
for i in 1 2 3; do
  curl -LsSf https://astral.sh/uv/install.sh | sh && break
  sleep 15
done
if command -v uv >/dev/null 2>&1 && uv venv /opt/sb/agentenv --python 3.12 && uv pip install --python /opt/sb/agentenv/bin/python boto3; then
  PY=/opt/sb/agentenv/bin/python
else  # 备用方案：系统自带的 python3 + apt 里的 boto3
  apt-get -o DPkg::Lock::Timeout=900 update && apt-get -o DPkg::Lock::Timeout=900 install -y python3-boto3 && PY=/usr/bin/python3
fi
[ -n "$PY" ] || { echo "no python with boto3"; exit 1; }
cat > /opt/sb/s3 <<S3
#!/bin/bash
# 小工具: /opt/sb/s3 get <key> <file> | /opt/sb/s3 put <file> <key>
set -a; . /opt/sb/env; set +a
exec $PY - "\$@" <<'PYEOF'
import os, sys, boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ["SB_S3_ENDPOINT"], config=Config(s3={"addressing_style": "path"}))
op, a, b = sys.argv[1:4]
if op == "get":
    s3.download_file(os.environ["SB_S3_BUCKET"], a, b)
else:
    s3.upload_file(a, os.environ["SB_S3_BUCKET"], b)
PYEOF
S3
chmod 700 /opt/sb/s3
/opt/sb/s3 get ctl/agent.py /opt/sb/agent.py
cat > /etc/systemd/system/sb-agent.service <<UNIT
[Unit]
Description=secondbrain experiment agent
After=network-online.target
Wants=network-online.target
[Service]
EnvironmentFile=/opt/sb/env
Environment=HOME=/root
Environment=PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WorkingDirectory=/opt/sb
ExecStart=$PY -u /opt/sb/agent.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/sb-agent.log
StandardError=append:/var/log/sb-agent.log
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now sb-agent
echo "bootstrap done"
