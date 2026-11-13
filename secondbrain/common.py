"""模型加载与数据处理。数据格式：每行一个 {"messages": [{"role":..., "content":...}, ...]}，
最后一条必须是 assistant，只对它计算损失。"""
from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .brain import BrainConfig, BrainedModel, SideBrain

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn"]


def pick_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_base(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # 旧版 transformers 用 torch_dtype
        base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return tok, base.to(device)


def add_lora(base, r: int, alpha: int | None = None):
    from peft import LoraConfig, inject_adapter_in_model
    names = {n.split(".")[-1] for n, _ in base.named_modules()}
    targets = [t for t in LORA_TARGETS if t in names]
    base = inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha or 2 * r, target_modules=targets), base)
    for n, p in base.named_parameters():
        if "lora_" in n:  # 主模型可以是 bf16，LoRA 权重始终用 float32（训练与推理一致）
            p.data = p.data.float()
    return base


def lora_state(model) -> dict:
    return {n: p.detach().cpu() for n, p in model.base.named_parameters() if "lora_" in n}


def load_model(base_name: str, brain: str | None = None, lora: str | None = None, device: str | None = None):
    """返回 (tokenizer, BrainedModel)。brain / lora 都不给时就是原模型。"""
    device = pick_device(device)
    tok, base = load_base(base_name, device)
    if lora:
        ckpt = torch.load(lora, map_location="cpu", weights_only=True)
        base = add_lora(base, ckpt["r"]).to(device)
        missing = set(ckpt["state"]) - set(dict(base.named_parameters()))
        if missing:
            raise ValueError(f"LoRA 权重与主模型不匹配: {sorted(missing)[:3]}")
        base.load_state_dict(ckpt["state"], strict=False)
    if brain:
        model = BrainedModel(base, brain=SideBrain.load(brain))
    else:
        model = BrainedModel(base, cfg=BrainConfig(n_taps=0))
    return tok, model.eval()


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_prompt(tok, messages: list[dict], system: str | None = None) -> str:
    if system and not (messages and messages[0]["role"] == "system"):
        messages = [{"role": "system", "content": system}] + messages
    if tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>"


def encode(tok, messages: list[dict], max_len: int, system: str | None = None):
    """返回 (input_ids, labels)；prompt 部分的 label 为 -100。过长时从左侧截断 prompt。"""
    assert messages[-1]["role"] == "assistant", "最后一条消息必须是 assistant"
    p = tok(render_prompt(tok, messages[:-1], system), add_special_tokens=False)["input_ids"]
    t = tok(messages[-1]["content"], add_special_tokens=False)["input_ids"][: max_len - 1]
    t = t + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    p = p[-(max_len - len(t)):] if max_len > len(t) else []
    return p + t, [-100] * len(p) + t


def collate(batch, device):
    n = max(len(i) for i, _ in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    for b, (i, l) in enumerate(batch):  # 右侧 padding
        ids[b, : len(i)] = torch.tensor(i)
        labels[b, : len(l)] = torch.tensor(l)
        mask[b, : len(i)] = 1
    return ids.to(device), mask.to(device), labels.to(device)


@torch.no_grad()
def dataset_loss(model, data, batch_size, device) -> float:
    """按 token 加权的平均交叉熵。"""
    total, count = 0.0, 0
    for i in range(0, len(data), batch_size):
        ids, mask, labels = collate(data[i : i + batch_size], device)
        n = (labels[:, 1:] != -100).sum().item()
        if n:
            total += model(ids, mask, labels)[1].item() * n
            count += n
    return total / max(count, 1)
