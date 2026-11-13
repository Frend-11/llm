"""下载并切分实验数据（固定随机种子，可复现）。

岗位集：DISC-Law-SFT 的法律咨询问答（ShengbinYue/DISC-Law-SFT）
通用集：中文 Alpaca-GPT4（shibing624/alpaca-zh），只用来衡量通用能力是否下降

用法: python scripts/prepare_data.py [--train 1000] [--test 200] [--general 200]
"""
import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--out", default="data")
p.add_argument("--cache", default="data/raw")
p.add_argument("--train", type=int, default=1000)
p.add_argument("--test", type=int, default=200)
p.add_argument("--general", type=int, default=200)
p.add_argument("--max-chars", type=int, default=400, help="过滤掉过长的回答")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(a.seed)


def dump(name, rows):
    with open(out / name, "w", encoding="utf-8") as f:
        for q, ans in rows:
            f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]},
                               ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} 条")


law = hf_hub_download("ShengbinYue/DISC-Law-SFT", "DISC-Law-SFT-Pair-QA-released.jsonl", repo_type="dataset", local_dir=a.cache)
rows, seen = [], set()
with open(law, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        q, ans = r["input"].strip(), r["output"].strip()
        if q and q not in seen and 20 <= len(ans) <= a.max_chars:  # 按问题去重，避免训练/测试泄漏
            seen.add(q)
            rows.append((q, ans))
rng.shuffle(rows)
assert len(rows) >= a.train + a.test
dump("law_test.jsonl", rows[: a.test])
dump("law_train.jsonl", rows[a.test : a.test + a.train])

gen = hf_hub_download("shibing624/alpaca-zh", "alpaca_gpt4_data_zh.json", repo_type="dataset", local_dir=a.cache)
rows = [((r["instruction"] + ("\n" + r["input"] if r["input"] else "")).strip(), r["output"].strip())
        for r in json.load(open(gen, encoding="utf-8")) if 20 <= len(r["output"]) <= a.max_chars]
rng.shuffle(rows)
dump("general_test.jsonl", rows[: a.general])
