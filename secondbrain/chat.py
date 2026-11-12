from __future__ import annotations

from .common import load_model, render_prompt

HELP = "命令: /alpha <x> 调解码融合强度 | /gate <x> 调隐层回注强度 | /reset 清空对话 | /quit"


def run(a):
    tok, model = load_model(a.base, a.brain, a.lora, a.device)
    device = next(model.base.parameters()).device
    model.alpha, model.gate_scale = a.alpha, a.gate
    history = []
    print(HELP)
    while True:
        try:
            text = input(f"[alpha={model.alpha:g} gate={model.gate_scale:g}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, *arg = text.split()
            try:
                if cmd == "/quit":
                    break
                elif cmd == "/reset":
                    history = []
                elif cmd == "/alpha":
                    model.alpha = float(arg[0])
                elif cmd == "/gate":
                    model.gate_scale = float(arg[0])
                else:
                    print(HELP)
            except (IndexError, ValueError):
                print(HELP)
            continue
        history.append({"role": "user", "content": text})
        ids = tok(render_prompt(tok, history, a.system), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        out, shown = [], ""
        for t in model.generate(ids, a.max_new_tokens, a.temperature, a.top_p, tok.eos_token_id):
            out.append(t)
            full = tok.decode(out)
            if not full.endswith("�"):  # 等多字节字符凑齐再输出
                print(full[len(shown):], end="", flush=True)
                shown = full
        print()
        history.append({"role": "assistant", "content": tok.decode(out)})
