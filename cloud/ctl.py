#!/usr/bin/env python3
"""本地控制端：管理 Vultr 上的实验服务器，以及基于对象存储的任务队列。

密钥不进仓库：Vultr API 密钥放在 VULTR_KEY_FILE 指向的文件里，
对象存储的访问密钥保存在 SB_STATE 指向的状态文件里（两者都应在仓库之外）。

  ctl.py storage-create            创建对象存储并建桶
  ctl.py push                      上传 agent.py 和当前提交的代码包
  ctl.py instance-create           创建 GPU 服务器（开机后自动启动任务代理）
  ctl.py submit job.sh [--name N]  提交任务脚本
  ctl.py status                    心跳、任务完成情况、服务器状态、已花费用
  ctl.py log NAME [-n 40]          看某个任务的日志末尾
  ctl.py pull [DIR]                把结果同步到本地
  ctl.py kill NAME                 终止正在运行的任务
  ctl.py teardown                  销毁服务器、清空并删除对象存储
"""
import argparse
import base64
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.vultr.com/v2/"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("SB_STATE", os.path.expanduser("~/.sb_cloud_state.json"))
LABEL = "secondbrain-exp"


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(st):
    fd = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(st, f, indent=1)


def api(method, path, body=None):
    key = open(os.environ["VULTR_KEY_FILE"]).read().strip()
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Vultr API {method} {path} -> {e.code}: {e.read().decode()[:400]}")
    return json.loads(raw) if raw else {}


def s3():
    import boto3
    from botocore.config import Config
    st = load_state()
    return boto3.client("s3", endpoint_url="https://" + st["s3_hostname"], aws_access_key_id=st["s3_access_key"],
                        aws_secret_access_key=st["s3_secret_key"], region_name="us-east-1",
                        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5})), st["bucket"]


def keys(prefix):
    c, b = s3()
    out, token = [], None
    while True:
        kw = {"Bucket": b, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = c.list_objects_v2(**kw)
        out += r.get("Contents", [])
        token = r.get("NextContinuationToken")
        if not token:
            return out


def cmd_storage_create(a):
    st = load_state()
    if "storage_id" not in st:
        r = api("POST", "object-storage", {"cluster_id": a.cluster, "tier_id": a.tier, "label": LABEL})["object_storage"]
        st["storage_id"] = r["id"]
        save_state(st)
    for _ in range(60):
        r = api("GET", f"object-storage/{st['storage_id']}")["object_storage"]
        if r["status"] == "active":
            break
        time.sleep(5)
    st.update(s3_hostname=r["s3_hostname"], s3_access_key=r["s3_access_key"], s3_secret_key=r["s3_secret_key"])
    st.setdefault("bucket", "sb-exp-" + secrets.token_hex(5))
    save_state(st)
    print("对象存储:", r["status"], r["s3_hostname"], "桶:", st["bucket"])


def cmd_bucket_create(a):
    c, b = s3()
    c.create_bucket(Bucket=b)
    print("已建桶", b)


def cmd_push(a):
    c, b = s3()
    c.upload_file(os.path.join(HERE, "agent.py"), b, "ctl/agent.py")
    tar = subprocess.run(["git", "-C", os.path.dirname(HERE), "archive", "--format=tar.gz", "HEAD"], check=True,
                         capture_output=True).stdout
    c.put_object(Bucket=b, Key="code/llm.tar.gz", Body=tar)
    rev = subprocess.run(["git", "-C", os.path.dirname(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    c.put_object(Bucket=b, Key="code/REV", Body=rev.encode())
    print(f"已上传代码 {rev}（{len(tar) / 1024:.0f} KB）和 agent.py")


def cmd_instance_create(a):
    st = load_state()
    if st.get("instance_id"):
        raise SystemExit("状态文件里已有服务器 " + st["instance_id"])
    boot = open(os.path.join(HERE, "bootstrap.sh")).read()
    for k, v in (("__ENDPOINT__", "https://" + st["s3_hostname"]), ("__BUCKET__", st["bucket"]),
                 ("__ACCESS__", st["s3_access_key"]), ("__SECRET__", st["s3_secret_key"])):
        boot = boot.replace(k, v)
    r = api("POST", "instances", {
        "region": a.region, "plan": a.plan, "os_id": a.os_id, "label": LABEL, "hostname": "sb-gpu",
        "tags": [LABEL], "backups": "disabled", "activation_email": False,
        "user_data": base64.b64encode(boot.encode()).decode()})["instance"]
    st.update(instance_id=r["id"], instance_created=time.time(), plan=a.plan, hourly=a.hourly)
    save_state(st)
    print("已创建服务器", r["id"], r["status"], a.plan, a.region)


def cmd_submit(a):
    c, b = s3()
    name = a.name or os.path.basename(a.file)
    c.upload_file(a.file, b, "jobs/" + name)
    print("已提交", name)


def cmd_status(a):
    st = load_state()
    c, b = s3()
    try:
        hb = json.loads(c.get_object(Bucket=b, Key="status/heartbeat.json")["Body"].read())
        print("心跳:", json.dumps(hb, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        print("还没有心跳:", type(e).__name__)
    done = {}
    for o in keys("done/"):
        done[os.path.basename(o["Key"])[:-5]] = json.loads(c.get_object(Bucket=b, Key=o["Key"])["Body"].read())
    for o in keys("jobs/"):
        n = os.path.basename(o["Key"])
        print(f"  任务 {n}: {done.get(n, '未完成')}")
    if st.get("instance_id"):
        i = api("GET", f"instances/{st['instance_id']}")["instance"]
        hours = (time.time() - st["instance_created"]) / 3600
        print(f"服务器: {i['status']}/{i['power_status']}/{i['server_status']} ip={i['main_ip']} 已开机 {hours:.2f} 小时，"
              f"按 ${st.get('hourly', 0)}/小时估算约 ${hours * st.get('hourly', 0):.2f}")
    acc = api("GET", "account")["account"]
    print("账户: balance", acc["balance"], "pending_charges", acc["pending_charges"])


def cmd_log(a):
    c, b = s3()
    key = a.name if a.name.startswith("logs/") else f"logs/{a.name}.log"
    text = c.get_object(Bucket=b, Key=key)["Body"].read().decode("utf-8", "replace")
    lines = text.splitlines()
    if a.grep:
        lines = [l for l in lines if a.grep in l]
    print("\n".join(l[:a.width] for l in lines[-a.n:]))


def cmd_pull(a):
    c, b = s3()
    n = 0
    for o in keys("results/"):
        dst = os.path.join(a.dir, o["Key"][len("results/"):])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or os.path.getsize(dst) != o["Size"] or os.path.getmtime(dst) < o["LastModified"].timestamp():
            c.download_file(b, o["Key"], dst)
            n += 1
    print(f"已同步 {n} 个文件到 {a.dir}")


def cmd_kill(a):
    c, b = s3()
    c.put_object(Bucket=b, Key=f"ctl/kill/{a.name}", Body=b"1")
    print("已请求终止", a.name)


def cmd_instance_destroy(a):
    st = load_state()
    if st.get("instance_id"):
        api("DELETE", f"instances/{st['instance_id']}")
        hours = (time.time() - st["instance_created"]) / 3600
        print(f"已销毁服务器 {st['instance_id']}（开机 {hours:.2f} 小时）")
        st["last_instance"] = {k: st.pop(k) for k in ("instance_id", "instance_created", "plan", "hourly") if k in st}
        save_state(st)
    else:
        print("状态文件里没有服务器")


def cmd_teardown(a):
    cmd_instance_destroy(a)
    st = load_state()
    if st.get("storage_id"):
        c, b = s3()
        objs = keys("")
        for i in range(0, len(objs), 500):
            c.delete_objects(Bucket=b, Delete={"Objects": [{"Key": o["Key"]} for o in objs[i:i + 500]]})
        try:
            c.delete_bucket(Bucket=b)
        except Exception as e:  # noqa: BLE001
            print("删桶:", e)
        api("DELETE", f"object-storage/{st['storage_id']}")
        print("已删除对象存储", st["storage_id"])
        save_state({"last_instance": st.get("last_instance")})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("storage-create"); s.add_argument("--cluster", type=int, default=2); s.add_argument("--tier", type=int, default=2)
    sub.add_parser("bucket-create")
    sub.add_parser("push")
    s = sub.add_parser("instance-create")
    s.add_argument("--plan", default="vcg-a100-12c-120g-80vram"); s.add_argument("--region", default="ewr")
    s.add_argument("--os-id", type=int, default=2284); s.add_argument("--hourly", type=float, default=2.6)
    s = sub.add_parser("submit"); s.add_argument("file"); s.add_argument("--name")
    sub.add_parser("status")
    s = sub.add_parser("log"); s.add_argument("name"); s.add_argument("-n", type=int, default=40)
    s.add_argument("--grep"); s.add_argument("--width", type=int, default=220)
    s = sub.add_parser("pull"); s.add_argument("dir", nargs="?", default="results/vultr")
    s = sub.add_parser("kill"); s.add_argument("name")
    sub.add_parser("instance-destroy")
    sub.add_parser("teardown")
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
