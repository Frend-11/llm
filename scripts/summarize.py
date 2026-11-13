"""把一个结果目录下的 *.json 汇总成 Markdown 表。用法: python scripts/summarize.py results/xxx
“Δ”列是相对原模型的 loss 变化：岗位集越负越好，通用集越接近 0 越好（正数表示通用能力下降）。"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
order = ["base", "prompt", "lora", "brain"]
rows, base = [], {}
for f in sorted(d.glob("*.json")):
    if f.name.endswith(".train.json"):
        continue
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if "extra_params_M" not in r:
        continue
    t = d / f"{f.stem}.train.json"
    if not t.exists():
        t = d / {"brain": "law.brain.train.json", "lora": "law.lora.train.json"}.get(f.stem, "_")
    r["_train"] = json.loads(t.read_text(encoding="utf-8")) if t.exists() else {}
    rows.append((f.stem, r))
    for k in ("role", "general"):  # 原模型的 loss：来自 base 这一行，或任一大脑评测里“关掉大脑”的那次
        if f.stem == "base" and k in r:
            base[k] = r[k]["loss"]
        elif "base_loss" in r.get(k, {}):
            base.setdefault(k, r[k]["base_loss"])
rows.sort(key=lambda x: (order.index(x[0]) if x[0] in order else 99, x[1].get("role", {}).get("loss", 0)))


def cell(r, k):
    v = r.get(k, {}).get("loss")
    if v is None:
        return "-", "-"
    return f"{v:.4f}", (f"{v - base[k]:+.4f}" if k in base else "-")


print(f"\n{d}" + (f"（原模型：岗位 {base.get('role', '-')}，通用 {base.get('general', '-')}）" if base else ""))
print("| 配置 | 新增参数(M) | 岗位 loss | Δ岗位 | 通用 loss | Δ通用 | ROUGE-L | 训练秒数 | 显存GB | tokens/s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, r in rows:
    (rl, rd), (gl, gd), t = cell(r, "role"), cell(r, "general"), r["_train"]
    print(f"| {name} | {r['extra_params_M']} | {rl} | {rd} | {gl} | {gd} | {r.get('rouge_l', '-')} | "
          f"{t.get('seconds', '-')} | {t.get('peak_vram_GB', '-')} | {r.get('tokens_per_s', '-')} |")
