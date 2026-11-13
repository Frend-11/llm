"""第二大脑：挂在冻结主模型旁边的可插拔旁路网络。

三条通路：
  读   —— 在选定的层 (tap) 读取主模型隐状态  h_l（norm_in 时先做 LayerNorm）
  写   —— 门控回注  h_l' = h_l + gate_scale * c * r(h_l) * g_l * up_l(s_l)
  解码 —— logit 融合 logits = logits_base + alpha * c * lm_head(r(h_f) * up_out(s_final))
  c = write_scale，训练前定好的常数，控制写通路每一步更新的幅度（作用类似 LoRA 的 alpha/r）。
  r(h) 是该位置隐状态的均方根（rel_write 时启用，否则为 1；h_f 是送入 lm_head 的末层隐状态）：
  不同模型、不同层的隐状态数值尺度能差两个数量级，按相对幅度写入后同一套超参才能通用。

up_l 与 up_out 均为零初始化（g_l 是初值为 1 的可学习逐层标量），所以刚挂载时输出与原模型逐位一致。
alpha / gate_scale 是推理期旋钮，不参与训练，可在调用时随意调节。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BrainConfig:
    d_model: int = 0                 # 主模型隐层宽度（挂载时自动填）
    n_base_layers: int = 0           # 主模型层数（挂载时自动填）
    d_side: int = 256                # 旁路宽度
    n_heads: int = 4
    taps: list[int] = field(default_factory=list)  # 读写的主模型层下标；空 = 均匀取 n_taps 层
    n_taps: int = 6
    mlp_ratio: int = 4               # 旁路块里 MLP 的扩张倍数；0 = 不要 MLP（只留注意力）
    norm_in: bool = False            # 读入前先对主模型隐状态做 LayerNorm（train 命令默认打开；旧 .brain 文件没有它）
    write_scale: float = 1.0         # 写通路（回注与解码融合）的固定缩放，作用类似 LoRA 的 alpha/r
    rel_write: bool = False          # 写入量按所在位置隐状态的均方根缩放，并在写出前对旁路状态做 LayerNorm
    use_inject: bool = True          # 消融：隐层回注通路
    use_logit: bool = True           # 消融：logit 融合通路
    base_name: str = ""

    def resolve_taps(self) -> list[int]:
        if self.taps:
            return sorted(self.taps)
        n = min(self.n_taps, self.n_base_layers)
        if n <= 0:
            return []  # 不挂任何层：纯包装器（LoRA / 原模型基线用）
        # 均匀分布，且总是包含最后一层
        return sorted({round((i + 1) * self.n_base_layers / n) - 1 for i in range(n)})


class SideBlock(nn.Module):
    """带 KV 缓存的因果 pre-norm transformer 块。"""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        if mlp_ratio > 0:
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        else:
            self.mlp = None

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        if cache is not None:  # 增量解码：拼上历史 k/v，新 token 可看到全部历史
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        P = k.shape[2] - T  # 历史长度
        mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(diagonal=P)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class SideBrain(nn.Module):
    """只含可训练参数的旁路网络，本身不依赖主模型，可独立保存为 .brain 文件。"""

    def __init__(self, cfg: BrainConfig):
        super().__init__()
        self.cfg = cfg
        self.taps = cfg.resolve_taps()
        n = len(self.taps)
        self.ln_in = nn.ModuleList(nn.LayerNorm(cfg.d_model) for _ in range(n)) if cfg.norm_in else None
        self.down = nn.ModuleList(nn.Linear(cfg.d_model, cfg.d_side) for _ in range(n))
        self.blocks = nn.ModuleList(SideBlock(cfg.d_side, cfg.n_heads, cfg.mlp_ratio) for _ in range(n))
        self.ln_w = nn.ModuleList(nn.LayerNorm(cfg.d_side) for _ in range(n)) if cfg.rel_write else None
        self.up = nn.ModuleList(nn.Linear(cfg.d_side, cfg.d_model, bias=False) for _ in range(n))
        self.gates = nn.Parameter(torch.ones(n))           # 逐层可学习的回注强度
        self.ln_out = nn.LayerNorm(cfg.d_side)
        self.up_out = nn.Linear(cfg.d_side, cfg.d_model, bias=False)
        nn.init.zeros_(self.up_out.weight)                 # 零初始化 → 初始 logit 不变
        for u in self.up:
            nn.init.zeros_(u.weight)                       # 零初始化 → 初始不改动主模型

    def save(self, path: str):
        torch.save({"config": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> "SideBrain":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        brain = cls(BrainConfig(**ckpt["config"]))
        brain.load_state_dict(ckpt["state"])
        return brain


def _rms(h: torch.Tensor) -> torch.Tensor:
    """每个位置隐状态的均方根（不回传梯度，只当尺度用）。"""
    return h.detach().float().pow(2).mean(-1, keepdim=True).sqrt()


def find_layers(base: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = base
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError("找不到主模型的 decoder 层列表，请在 find_layers 里补充路径")


class BrainedModel(nn.Module):
    """冻结主模型 + 第二大脑。alpha / gate_scale 可在调用时修改。"""

    def __init__(self, base: nn.Module, brain: SideBrain | None = None, cfg: BrainConfig | None = None):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        layers = find_layers(base)
        if brain is None:
            cfg = cfg or BrainConfig()
            cfg.d_model = base.config.hidden_size
            cfg.n_base_layers = len(layers)
            cfg.base_name = getattr(base.config, "_name_or_path", "")
            brain = SideBrain(cfg)
        elif brain.cfg.d_model != base.config.hidden_size or brain.cfg.n_base_layers != len(layers):
            raise ValueError("该 .brain 文件与当前主模型的结构不匹配")
        p = next(base.parameters())
        self.brain = brain.to(p.device)       # 大脑保持 float32，主模型可为 bf16/fp16
        self.alpha = 1.0
        self.gate_scale = 1.0
        self._s = None
        self._cache_in = None
        self._cache_out = None
        self._hf = None
        for i, l in enumerate(brain.taps):
            layers[l].register_forward_hook(self._make_hook(i))
        if brain.taps and brain.cfg.rel_write:  # 记下送入 lm_head 的末层隐状态，用来给解码融合定尺度
            base.get_output_embeddings().register_forward_pre_hook(self._capture_head_input)

    def _capture_head_input(self, _module, args):
        if self._cache_out is not None and self._hf is None:  # 只记主模型自己的那一次调用
            self._hf = args[0]

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # 主模型永远处于 eval
        return self

    def _make_hook(self, i: int):
        def hook(_module, _inp, out):
            if self._cache_out is None:  # 未经 self.forward 调用主模型时不介入
                return None
            h = out[0] if isinstance(out, tuple) else out
            b = self.brain
            x = h.to(b.gates.dtype)
            if b.ln_in is not None:  # 主模型隐状态各层、各位置的模长差异很大，先归一化再读
                x = b.ln_in[i](x)
            x = b.down[i](x)
            if self._s is not None:
                x = x + self._s
            cache = self._cache_in[i] if self._cache_in is not None else None
            self._s, kv = b.blocks[i](x, cache)
            self._cache_out.append(kv)
            if not b.cfg.use_inject or self.gate_scale == 0:
                return None
            if b.cfg.rel_write:
                delta = b.up[i](b.ln_w[i](self._s)) * _rms(h)
            else:
                delta = b.up[i](self._s)
            h = h + (self.gate_scale * b.cfg.write_scale * b.gates[i] * delta).to(h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def forward(self, input_ids, attention_mask=None, labels=None, past=None):
        """past = (主模型 past_key_values, 旁路 KV 缓存)，用于增量解码。
        返回 (logits, loss, past)。训练时请使用右侧 padding。"""
        base_past, self._cache_in = past if past is not None else (None, None)
        self._s, self._cache_out, self._hf = None, [], None
        try:
            out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=base_past, use_cache=not self.training)
            logits = out.logits
            if self.brain.cfg.use_logit and self.alpha != 0 and self._s is not None:
                head = self.base.get_output_embeddings()
                delta = self.brain.up_out(self.brain.ln_out(self._s))
                if self.brain.cfg.rel_write:
                    delta = delta * _rms(self._hf)
                scale = self.alpha * self.brain.cfg.write_scale
                logits = logits + scale * head(delta.to(head.weight.dtype)).to(logits.dtype)
            new_past = (out.past_key_values, self._cache_out)
        finally:
            self._s, self._cache_in, self._cache_out, self._hf = None, None, None, None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)
        return logits, loss, new_past

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.0, top_p=1.0, eos_token_id=None):
        """单条序列的增量解码，逐 token yield。"""
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        ids, past = input_ids, None
        for _ in range(max_new_tokens):
            logits, _, past = self(ids, past=past)
            logits = logits[0, -1].float()
            if temperature <= 0:
                nxt = logits.argmax()
            else:
                probs = (logits / temperature).softmax(-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    sp[sp.cumsum(-1) - sp > top_p] = 0
                    probs = torch.zeros_like(probs).scatter(0, si, sp)
                nxt = torch.multinomial(probs / probs.sum(), 1)[0]
            if nxt.item() in eos:
                return
            yield nxt.item()
            ids = nxt.view(1, 1)

    def num_brain_params(self) -> int:
        if not self.brain.taps:  # 纯包装器（原模型 / LoRA 基线）里的大脑没有接入，不算新增参数
            return 0
        return sum(p.numel() for p in self.brain.parameters())
