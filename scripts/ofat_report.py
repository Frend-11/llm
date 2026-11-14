"""按“单变量系列”汇总扫描结果：每个系列里只有一个因素在变，带 * 的是基线。
用法: python scripts/ofat_report.py results/q35_sweep"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q35_sweep")
B0, L0 = "b_lr1e-3_c4x", "l_lr2e-4"
SERIES = [
    ("第二大脑 · 写入缩放 c（lr=1e-3）", [("0.0156", "b_lr1e-3_c025x"), ("0.0625", "b_lr1e-3"), ("0.25 *", B0), ("1.0", "v_c1"), ("4.0", "v_c4")]),
    ("第二大脑 · 学习率（c=0.25）", [("3e-4", "v_lr3e-4"), ("1e-3 *", B0), ("3e-3", "v_lr3e-3")]),
    ("第二大脑 · 学习率（c=0.0625）", [("3e-4", "b_lr3e-4"), ("1e-3 *", "b_lr1e-3"), ("3e-3", "b_lr3e-3")]),
    ("第二大脑 · 挂载层数（宽度 256）", [("3", "v_taps3"), ("6 *", B0), ("12", "v_taps12"), ("24", "v_taps24")]),
    ("第二大脑 · 旁路宽度（挂 6 层）", [("128", "v_d128"), ("256 *", B0), ("512", "v_d512")]),
    ("第二大脑 · MLP 扩张倍数", [("0", "v_mlp0"), ("1", "v_mlp1"), ("4 *", B0)]),
    ("第二大脑 · 逐项关闭一个部件", [("全开 *", B0), ("关隐层回注", "a_noinject"), ("关解码融合", "a_nologit"),
                          ("关读入归一化", "a_nonorm"), ("关相对写入", "v_abs")]),
    ("第二大脑 · 随机种子", [("0 *", B0), ("1", "v_seed1"), ("2", "v_seed2")]),
    ("第二大脑 · 训练样本数", [("1000", "v_n1000"), ("3000 *", B0), ("10000", "v_n10000")]),
    ("LoRA · 学习率（秩 27）", [("1e-4", "l_lr1e-4"), ("2e-4 *", L0), ("5e-4", "l_lr5e-4"), ("1e-3", "l_lr1e-3")]),
    ("LoRA · 秩（lr=2e-4）", [("8", "w_r8"), ("16", "w_r16"), ("27 *", L0), ("54", "w_r54")]),
    ("LoRA · 随机种子", [("0 *", L0), ("1", "w_seed1"), ("2", "w_seed2")]),
    ("LoRA · 训练样本数", [("1000", "w_n1000"), ("3000 *", L0), ("10000", "w_n10000")]),
    ("方法对比（各自基线，参数量对齐）", [("LoRA *", L0), ("第二大脑", B0)]),
]


def load(name):
    f = d / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


base = {}
for f in d.glob("*.json"):  # 原模型的 loss：任一大脑评测里“关掉大脑”的那次
    if not f.name.endswith(".train.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        for k in ("role", "general"):
            if "base_loss" in r.get(k, {}):
                base.setdefault(k, r[k]["base_loss"])
print(f"{d}：原模型验证集 loss —— 岗位 {base.get('role', '-')}，通用 {base.get('general', '-')}")
print("“较原模型”越负越好；“较基线”是与本系列带 * 那一行的差，负数表示比基线好。\n")
for title, rows in SERIES:
    ref = next((load(n) for label, n in rows if label.endswith("*")), None)
    print(f"### {title}\n")
    print("| 取值 | 试验名 | 新增参数(M) | 岗位 loss | 较原模型 | 较基线 | 通用 loss | 通用较原模型 |")
    print("|---|---|---|---|---|---|---|---|")
    for label, name in rows:
        r = load(name)
        if r is None:
            print(f"| {label} | {name} | 未跑 | | | | | |")
            continue
        rl, gl = r["role"]["loss"], r.get("general", {}).get("loss")
        vs_base = f"{rl - base['role']:+.4f}" if "role" in base else "-"
        vs_ref = f"{rl - ref['role']['loss']:+.4f}" if ref else "-"
        g = f"{gl:.4f} | {gl - base['general']:+.4f}" if gl is not None and "general" in base else "- | -"
        print(f"| {label} | {name} | {r['extra_params_M']} | {rl:.4f} | {vs_base} | {vs_ref} | {g} |")
    print()
