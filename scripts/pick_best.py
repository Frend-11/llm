"""从扫描结果里选出验证集岗位 loss 最低的试验，打印它的训练参数。
用法: python scripts/pick_best.py results/q35_sweep b_      （只在名字以 b_ 开头的试验里选）"""
import json
import sys
from pathlib import Path

d, prefix = Path(sys.argv[1]), sys.argv[2]
best = None
for f in d.glob(f"{prefix}*.json"):
    if f.name.endswith(".train.json") or not (d / f"{f.stem}.args").exists():
        continue
    loss = json.loads(f.read_text(encoding="utf-8")).get("role", {}).get("loss")
    if loss is not None and loss == loss and (best is None or loss < best[0]):  # loss == loss 排除 NaN
        best = (loss, f.stem)
if best is None:
    sys.exit(f"没有找到以 {prefix} 开头的有效试验")
print(f"选中 {best[1]}（验证集岗位 loss {best[0]:.4f}）", file=sys.stderr)
print((d / f"{best[1]}.args").read_text(encoding="utf-8").strip())
