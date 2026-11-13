"""跑在云服务器上的任务代理：从对象存储的 jobs/ 里按文件名顺序取 shell 脚本执行，
日志传到 logs/，退出码写到 done/，结果目录里的小文件同步到 results/。

需要的环境变量：SB_S3_ENDPOINT、SB_S3_BUCKET、AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY。
本机没有 SSH 通道时，用它代替远程登录：提交任务 = 往桶里放一个脚本。
"""
import json
import os
import signal
import subprocess
import time

import boto3
from botocore.config import Config

BUCKET = os.environ["SB_S3_BUCKET"]
WORK = os.environ.get("SB_WORK", "/opt/sb")
RESULTS = os.path.join(WORK, "llm", "results")
SYNC_EXT = (".json", ".jsonl", ".log", ".md", ".txt")
MAX_LOG = 4 * 1024 * 1024
s3 = boto3.client("s3", endpoint_url=os.environ["SB_S3_ENDPOINT"],
                  config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5}))
synced = {}


def keys(prefix):
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if not token:
            return out


def put_file(path, key):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > MAX_LOG and key.startswith("logs/"):  # 日志太长时只传末尾
            f.seek(size - MAX_LOG)
        s3.put_object(Bucket=BUCKET, Key=key, Body=f.read())


def sync_results():
    if not os.path.isdir(RESULTS):
        return
    for root, _, files in os.walk(RESULTS):
        for name in files:
            if not name.endswith(SYNC_EXT):
                continue
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
                sig = (st.st_mtime_ns, st.st_size)
                if synced.get(path) != sig and st.st_size <= 64 * 1024 * 1024:
                    put_file(path, "results/" + os.path.relpath(path, RESULTS))
                    synced[path] = sig
            except OSError:
                pass


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def heartbeat(job=None, started=None):
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_s": float(open("/proc/uptime").read().split()[0]),
        "job": job,
        "job_elapsed_s": round(time.time() - started) if started else None,
        "gpu": sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"),
        "disk": sh("df -h / | tail -1"),
        "mem": sh("free -g | sed -n 2p"),
    }
    s3.put_object(Bucket=BUCKET, Key="status/heartbeat.json", Body=json.dumps(info, ensure_ascii=False))


def run_job(key):
    name = os.path.basename(key)
    os.makedirs(f"{WORK}/jobs", exist_ok=True)
    os.makedirs(f"{WORK}/logs", exist_ok=True)
    script, log = f"{WORK}/jobs/{name}", f"{WORK}/logs/{name}.log"
    s3.download_file(BUCKET, key, script)
    started = time.time()
    with open(log, "wb") as lf:
        proc = subprocess.Popen(["bash", script], stdout=lf, stderr=subprocess.STDOUT, cwd=WORK,
                                start_new_session=True)
    killed = False
    while proc.poll() is None:
        time.sleep(20)
        try:
            put_file(log, f"logs/{name}.log")
            sync_results()
            heartbeat(name, started)
            if keys(f"ctl/kill/{name}"):
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(10)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                killed = True
        except Exception as e:  # noqa: BLE001  网络抖动不应中断任务
            print("loop error:", e, flush=True)
    code = 137 if killed else proc.returncode
    put_file(log, f"logs/{name}.log")
    sync_results()
    s3.put_object(Bucket=BUCKET, Key=f"done/{name}.exit",
                  Body=json.dumps({"exit": code, "seconds": round(time.time() - started)}))


def main():
    print("agent started", flush=True)
    while True:
        try:
            done = {os.path.basename(k)[: -len(".exit")] for k in keys("done/")}
            todo = sorted(k for k in keys("jobs/") if os.path.basename(k) and os.path.basename(k) not in done)
            if todo:
                run_job(todo[0])
                continue
            heartbeat()
        except Exception as e:  # noqa: BLE001
            print("agent error:", e, flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
