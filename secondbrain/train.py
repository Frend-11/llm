from __future__ import annotations

import json
import math
import random
import time

import torch

from .brain import BrainConfig, BrainedModel, SideBrain, find_layers
from .common import add_lora, collate, dataset_loss, encode, load_base, lora_rank_for, lora_state, pick_device, read_jsonl


def run(a):
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = pick_device(a.device)
    tok, base = load_base(a.base, device)

    if a.method == "lora":
        if a.lora_r <= 0:  # 自动取秩：与同样参数下的第二大脑参数量对齐
            ref = BrainConfig(d_model=base.config.hidden_size, n_base_layers=len(find_layers(base)), d_side=a.d_side,
                              n_heads=a.n_heads, n_taps=a.n_taps, mlp_ratio=a.mlp_ratio, norm_in=not a.no_norm_in,
                              rel_write=not a.abs_write)
            a.lora_r = lora_rank_for(base, sum(p.numel() for p in SideBrain(ref).parameters()))
            print(f"LoRA 秩自动取 {a.lora_r}")
        base = add_lora(base, a.lora_r).to(device)
        model = BrainedModel(base, cfg=BrainConfig(n_taps=0))
        params = [p for n, p in base.named_parameters() if "lora_" in n]
        for p in params:
            p.requires_grad_(True)
    else:
        cfg = BrainConfig(d_side=a.d_side, n_heads=a.n_heads, n_taps=a.n_taps, mlp_ratio=a.mlp_ratio,
                          norm_in=not a.no_norm_in, rel_write=not a.abs_write,
                          write_scale=a.write_scale if a.write_scale > 0 else a.d_side ** -0.5,
                          taps=[int(x) for x in a.taps.split(",")] if a.taps else [],
                          use_inject=not a.no_inject, use_logit=not a.no_logit)
        model = BrainedModel(base, cfg=cfg)
        params = list(model.brain.parameters())
        print(f"taps = {model.brain.taps}")
    n_params = sum(p.numel() for p in params)
    print(f"可训练参数: {n_params / 1e6:.2f}M  (主模型 {sum(p.numel() for p in base.parameters()) / 1e6:.0f}M 冻结)")

    rows = read_jsonl(a.data)
    if a.limit:
        rows = rows[: a.limit]
    data = [encode(tok, r["messages"], a.max_len, a.system) for r in rows]
    if a.val:
        val = [encode(tok, r["messages"], a.max_len, a.system) for r in read_jsonl(a.val)]
    else:
        random.shuffle(data)
        k = max(1, int(len(data) * 0.05)) if len(data) >= 20 else 0
        val, data = data[:k], data[k:]
    print(f"训练 {len(data)} 条，验证 {len(val)} 条")

    lr = a.lr or (2e-4 if a.method == "lora" else 3e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = max(1, a.epochs * math.ceil(len(data) / (a.batch_size * a.accum)))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    def save():
        if a.method == "lora":
            torch.save({"r": a.lora_r, "state": lora_state(model)}, a.out)
        else:
            model.brain.save(a.out)

    step, best, t0 = 0, float("inf"), time.time()
    for ep in range(a.epochs):
        model.train()
        random.shuffle(data)
        batches = [data[i : i + a.batch_size] for i in range(0, len(data), a.batch_size)]
        for bi, batch in enumerate(batches):
            loss = model(*collate(batch, device))[1]
            (loss / a.accum).backward()
            if (bi + 1) % a.accum == 0 or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0:
                    print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        if val:
            model.eval()
            vl = dataset_loss(model, val, a.batch_size, device)
            print(f"== epoch {ep} 验证 loss {vl:.4f} ppl {math.exp(vl):.2f}")
            if vl < best:
                best = vl
                save()
        else:
            save()
    info = {"method": a.method, "base": a.base, "trainable_params_M": round(n_params / 1e6, 3), "lr": lr,
            "epochs": a.epochs, "steps": step, "batch_size": a.batch_size, "accum": a.accum, "train_examples": len(data),
            "best_val_loss": round(best, 4) if val else None, "seconds": round(time.time() - t0),
            "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "lora_r": a.lora_r if a.method == "lora" else None}
    with open(a.out + ".train.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    msg = f"已保存 -> {a.out}  训练用时 {info['seconds']}s"
    if info["peak_vram_GB"] is not None:
        msg += f"  显存峰值 {info['peak_vram_GB']} GB"
    print(msg)
