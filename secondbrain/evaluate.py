"""评测：岗位集 loss/ppl、通用集保持率、参数与速度开销，并导出生成结果供量表打分。"""
from __future__ import annotations

import json
import math
import time

import torch

from .common import dataset_loss, encode, load_model, read_jsonl, render_prompt


def rouge_l(hyp: str, ref: str) -> float:
    """字符级 ROUGE-L F1（最长公共子序列），对中文不需要分词。"""
    if not hyp or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for ch in hyp:
        cur = [0]
        for j, r in enumerate(ref):
            cur.append(prev[j] + 1 if ch == r else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    return 2 * lcs / (len(hyp) + len(ref))


def run(a):
    tok, model = load_model(a.base, a.brain, a.lora, a.device)
    device = next(model.base.parameters()).device
    model.alpha, model.gate_scale = a.alpha, a.gate
    report = {"base": a.base, "brain": a.brain, "lora": a.lora, "alpha": a.alpha, "gate": a.gate,
              "system": a.system, "extra_params_M": round(
                  (model.num_brain_params() + sum(p.numel() for n, p in model.base.named_parameters() if "lora_" in n)) / 1e6, 3)}

    def losses(path):
        data = [encode(tok, r["messages"], a.max_len, a.system) for r in read_jsonl(path)]
        with_ = dataset_loss(model, data, a.batch_size, device)
        out = {"loss": round(with_, 4), "ppl": round(math.exp(with_), 3)}
        if a.brain:  # 同一进程内关掉大脑，得到严格可比的原模型数字
            model.alpha, model.gate_scale = 0.0, 0.0
            off = dataset_loss(model, data, a.batch_size, device)
            model.alpha, model.gate_scale = a.alpha, a.gate
            out.update(base_loss=round(off, 4), delta=round(with_ - off, 4))
        return out

    if a.brain:
        report["brain_config"] = {k: v for k, v in vars(model.brain.cfg).items() if k not in ("base_name",)}
    if a.role:
        report["role"] = losses(a.role)
    if a.general:
        report["general"] = losses(a.general)  # delta > 0 表示通用能力下降

    if a.role and a.gen_out and a.gen_n > 0:
        rows, n_tok, t0, scores = read_jsonl(a.role)[: a.gen_n], 0, time.time(), []
        with open(a.gen_out, "w", encoding="utf-8") as f:
            for r in rows:
                ids = tok(render_prompt(tok, r["messages"][:-1], a.system), add_special_tokens=False,
                          return_tensors="pt")["input_ids"].to(device)
                out = list(model.generate(ids, a.max_new_tokens, eos_token_id=tok.eos_token_id))
                n_tok += len(out)
                text, ref = tok.decode(out), r["messages"][-1]["content"]
                scores.append(rouge_l(text, ref))
                f.write(json.dumps({"messages": r["messages"][:-1], "reference": ref, "output": text,
                                    "rouge_l": round(scores[-1], 4)}, ensure_ascii=False) + "\n")
        report["tokens_per_s"] = round(n_tok / max(time.time() - t0, 1e-9), 1)
        report["rouge_l"] = round(sum(scores) / max(len(scores), 1), 4)  # 贪心解码、截断到 max_new_tokens
        report["gen_n"] = len(scores)
        report["generations"] = a.gen_out
    if device.type == "cuda":
        report["peak_vram_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
