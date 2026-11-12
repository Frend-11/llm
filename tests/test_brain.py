import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from secondbrain import BrainConfig, BrainedModel, SideBrain
from secondbrain.cli import main

V = 64


def tiny():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=64, n_embd=32, n_layer=4, n_head=2)).eval()


def randomize(model):  # 让门控和输出投影离开零点，否则测不到大脑的作用
    torch.manual_seed(1)
    with torch.no_grad():
        for u in model.brain.up:
            u.weight.normal_(std=0.1)
        model.brain.up_out.weight.normal_(std=0.1)


def test_identity_at_init():
    base = tiny()
    ids = torch.randint(0, V, (2, 10))
    ref = base(ids).logits
    model = BrainedModel(base, cfg=BrainConfig(d_side=16, n_heads=2, n_taps=2)).eval()
    assert model.brain.taps == [1, 3]
    assert torch.equal(model(ids)[0], ref)


def test_knobs_switch_brain_off():
    base = tiny()
    ids = torch.randint(0, V, (1, 10))
    ref = base(ids).logits
    model = BrainedModel(base, cfg=BrainConfig(d_side=16, n_heads=2, n_taps=2)).eval()
    randomize(model)
    assert not torch.allclose(model(ids)[0], ref, atol=1e-4)
    model.alpha, model.gate_scale = 0.0, 0.0
    assert torch.equal(model(ids)[0], ref)


def test_incremental_matches_full():
    model = BrainedModel(tiny(), cfg=BrainConfig(d_side=16, n_heads=2, n_taps=3)).eval()
    randomize(model)
    ids = torch.randint(0, V, (1, 12))
    full = model(ids)[0]
    past, steps = None, []
    for chunk in (ids[:, :5], ids[:, 5:6], ids[:, 6:]):
        logits, _, past = model(chunk, past=past)
        steps.append(logits)
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-4)


def test_causal():
    model = BrainedModel(tiny(), cfg=BrainConfig(d_side=16, n_heads=2, n_taps=2)).eval()
    randomize(model)
    ids = torch.randint(0, V, (1, 10))
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % V
    assert torch.allclose(model(ids)[0][:, :-1], model(ids2)[0][:, :-1], atol=1e-5)


@pytest.mark.parametrize("flags", [{}, {"use_inject": False}, {"use_logit": False}])
def test_training_only_touches_brain(flags):
    base = tiny()
    before = [p.clone() for p in base.parameters()]
    model = BrainedModel(base, cfg=BrainConfig(d_side=16, n_heads=2, n_taps=2, **flags)).train()
    opt = torch.optim.AdamW(model.brain.parameters(), lr=1e-2)
    ids = torch.randint(0, V, (4, 12))
    first = None
    for _ in range(60):
        loss = model(ids, labels=ids)[1]
        first = first or loss.item()
        loss.backward(); opt.step(); opt.zero_grad()
    assert loss.item() < first - 0.3
    assert all(torch.equal(a, b) for a, b in zip(before, base.parameters()))


def test_save_load_roundtrip(tmp_path):
    model = BrainedModel(tiny(), cfg=BrainConfig(d_side=16, n_heads=2, n_taps=2)).eval()
    randomize(model)
    ids = torch.randint(0, V, (1, 8))
    ref = model(ids)[0]
    model.brain.save(tmp_path / "x.brain")
    again = BrainedModel(tiny(), brain=SideBrain.load(tmp_path / "x.brain")).eval()
    assert torch.allclose(again(ids)[0], ref, atol=1e-6)
    out = list(again.generate(ids, max_new_tokens=5))
    assert len(out) == 5


def test_cli_end_to_end(tmp_path, capsys):
    """用本地随机小模型把 train(brain/lora) -> eval 整条链路跑通。"""
    chars = sorted(set("abcdefghijklmnopqrstuvwxyz <>\n"))
    vocab = {c: i for i, c in enumerate(["<eos>", "<unk>"] + chars)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>", unk_token="<unk>").save_pretrained(tmp_path / "m")
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=32, n_layer=4, n_head=2,
                               eos_token_id=0, bos_token_id=0)).save_pretrained(tmp_path / "m")
    data = tmp_path / "d.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"say {w}"}, {"role": "assistant", "content": w * 3}]}
            for w in ["ab", "cd", "ef", "gh"] * 6]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    base = ["--base", str(tmp_path / "m"), "--device", "cpu"]

    for method, out in (("brain", "x.brain"), ("lora", "x.lora")):
        main(["train", *base, "--data", str(data), "--out", str(tmp_path / out), "--method", method,
              "--epochs", "8", "--accum", "1", "--d-side", "16", "--n-heads", "2", "--n-taps", "2", "--lr", "1e-2"])
        capsys.readouterr()
        flag = "--brain" if method == "brain" else "--lora"
        main(["eval", *base, flag, str(tmp_path / out), "--role", str(data), "--general", str(data),
              "--gen-out", str(tmp_path / "gen.jsonl"), "--gen-n", "2", "--max-new-tokens", "4",
              "--report", str(tmp_path / f"{method}.json")])
        capsys.readouterr()
    brain = json.loads((tmp_path / "brain.json").read_text())
    assert brain["role"]["delta"] < -0.3          # 比关掉大脑的原模型更好
    main(["eval", *base, "--role", str(data), "--report", str(tmp_path / "base.json")])
    base_loss = json.loads((tmp_path / "base.json").read_text())["role"]["loss"]
    assert abs(base_loss - brain["role"]["base_loss"]) < 1e-3   # alpha=gate=0 等价于原模型
    assert json.loads((tmp_path / "lora.json").read_text())["role"]["loss"] < base_loss - 0.3
    assert len((tmp_path / "gen.jsonl").read_text().strip().splitlines()) == 2
