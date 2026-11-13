"""把一个结果目录下的 *.json 汇总成 Markdown 表。用法: python scripts/summarize.py results/sweep"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
order = ["base", "prompt", "lora", "brain"]
rows = []
for f in sorted(d.glob("*.json")):
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if "extra_params_M" in r:
        rows.append((f.stem, r))
rows.sort(key=lambda x: (order.index(x[0]) if x[0] in order else 99, x[1].get("role", {}).get("loss", 0)))
print(f"\n{d}")
print("| 配置 | 新增参数(M) | 岗位 loss | 岗位 ppl | 通用 loss | 通用 ppl | ROUGE-L | tokens/s |")
print("|---|---|---|---|---|---|---|---|")
for name, r in rows:
    ro, ge = r.get("role", {}), r.get("general", {})
    print(f"| {name} | {r['extra_params_M']} | {ro.get('loss', '-')} | {ro.get('ppl', '-')} | "
          f"{ge.get('loss', '-')} | {ge.get('ppl', '-')} | {r.get('rouge_l', '-')} | {r.get('tokens_per_s', '-')} |")
