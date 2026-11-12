import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="brain", description="第二大脑：可插拔旁路网络的训练 / 对话 / 评测")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, load=True):
        s.add_argument("--base", required=True, help="HF 模型名或本地路径，如 Qwen/Qwen2.5-1.5B-Instruct")
        s.add_argument("--device", default=None)
        s.add_argument("--system", default=None, help="岗位系统提示词（数据里没有 system 消息时自动加上）")
        if load:
            s.add_argument("--brain", default=None, help=".brain 文件")
            s.add_argument("--lora", default=None, help="LoRA 基线权重文件")
            s.add_argument("--alpha", type=float, default=1.0, help="logit 融合强度")
            s.add_argument("--gate", type=float, default=1.0, help="隐层回注强度")

    t = sub.add_parser("train", help="训练一个岗位大脑（或 LoRA 基线）")
    common(t, load=False)
    t.add_argument("--data", required=True)
    t.add_argument("--val", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--method", choices=["brain", "lora"], default="brain")
    t.add_argument("--d-side", type=int, default=256)
    t.add_argument("--n-heads", type=int, default=4)
    t.add_argument("--n-taps", type=int, default=6)
    t.add_argument("--taps", default="", help="手动指定层，如 3,7,11（覆盖 --n-taps）")
    t.add_argument("--no-inject", action="store_true", help="消融：关闭隐层回注")
    t.add_argument("--no-logit", action="store_true", help="消融：关闭 logit 融合")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--max-len", type=int, default=768)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("chat", help="终端对话，可实时调 alpha / gate")
    common(c)
    c.add_argument("--max-new-tokens", type=int, default=512)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-p", type=float, default=0.9)

    e = sub.add_parser("eval", help="岗位集 / 通用集评测")
    common(e)
    e.add_argument("--role", default=None, help="岗位测试集 jsonl")
    e.add_argument("--general", default=None, help="通用测试集 jsonl（衡量遗忘）")
    e.add_argument("--gen-out", default=None, help="把生成结果写到该 jsonl，供量表打分")
    e.add_argument("--gen-n", type=int, default=50)
    e.add_argument("--max-new-tokens", type=int, default=256)
    e.add_argument("--batch-size", type=int, default=4)
    e.add_argument("--max-len", type=int, default=768)
    e.add_argument("--report", default=None)

    a = p.parse_args(argv)
    if a.cmd == "train":
        from .train import run
    elif a.cmd == "chat":
        from .chat import run
    else:
        from .evaluate import run
    run(a)


if __name__ == "__main__":
    main()
