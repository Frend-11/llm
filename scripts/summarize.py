"""把 results/*.json 汇总成一张 Markdown 表。用法: python scripts/summarize.py results"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
order = ["base", "prompt", "lora", "brain"]
files = sorted(d.glob("*.json"), key=lambda f: (order.index(f.stem) if f.stem in order else 99, f.stem))
print("| 方法 | 新增参数(M) | 岗位 loss | 岗位 ppl | 通用 loss | 通用 ppl | tokens/s |")
print("|---|---|---|---|---|---|---|")
for f in files:
    r = json.loads(f.read_text(encoding="utf-8"))
    ro, ge = r.get("role", {}), r.get("general", {})
    print(f"| {f.stem} | {r['extra_params_M']} | {ro.get('loss', '-')} | {ro.get('ppl', '-')} | "
          f"{ge.get('loss', '-')} | {ge.get('ppl', '-')} | {r.get('tokens_per_s', '-')} |")
