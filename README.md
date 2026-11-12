# 第二大脑（secondbrain）

冻结开源主模型，在本地终端训练一个可插拔的旁路网络（"第二大脑"），调用时挂进 transformer 层并参与解码，让模型更胜任某个具体岗位。一个岗位对应一个 `.brain` 文件。

## 原理

| 通路 | 做法 |
|---|---|
| 读 | 在选定的若干层读取主模型隐状态 `h_l`，送入小型因果 transformer 得到 `s_l` |
| 写 | 回注：`h_l' = h_l + gate · g_l · up_l(s_l)` |
| 解码 | logit 融合：`logits = logits_主 + alpha · lm_head(up_out(s_末))`，复用主模型冻结的输出矩阵 |

- `up_l` 和 `up_out` 零初始化，刚挂载时输出与原模型逐位一致。
- `alpha` 和 `gate` 是调用时的旋钮，不参与训练；都设为 0 就严格退回原模型。
- 主模型参数全程冻结，大脑以 float32 训练，主模型在 GPU 上用 bf16。

## 安装

```bash
pip install -e ".[dev]"
pytest -q
```

## 用法

数据是 jsonl，每行 `{"messages": [...]}`，最后一条必须是 assistant（只对它算损失），见 `data/example.jsonl`。

```bash
# 训练岗位大脑
brain train --base Qwen/Qwen2.5-1.5B-Instruct --data legal_train.jsonl --out legal.brain

# 终端对话；对话中可用 /alpha 0.5、/gate 0 实时调节
brain chat --base Qwen/Qwen2.5-1.5B-Instruct --brain legal.brain

# 评测：岗位集、通用集（衡量遗忘）、速度，并导出生成结果
brain eval --base Qwen/Qwen2.5-1.5B-Instruct --brain legal.brain \
  --role legal_test.jsonl --general general_test.jsonl \
  --gen-out gen.jsonl --report legal_brain.json
```

### 基线

```bash
# LoRA
brain train --method lora --lora-r 16 --base ... --data legal_train.jsonl --out legal.lora
brain eval  --base ... --lora legal.lora --role legal_test.jsonl --general general_test.jsonl

# 原模型 / 角色提示词
brain eval --base ... --role legal_test.jsonl
brain eval --base ... --role legal_test.jsonl --system "你是一名劳动法律师助理。"
```

### 消融

```bash
brain train ... --no-inject          # 只保留 logit 融合
brain train ... --no-logit           # 只保留隐层回注
brain train ... --taps 20,23,27      # 指定挂载层
brain train ... --d-side 128 --n-taps 3
brain eval  ... --alpha 0.5 --gate 1 # 推理期扫旋钮
```

## 评测报告字段

- `role.loss / ppl`：岗位测试集上的表现。
- `role.delta`、`general.delta`：同一进程内与关掉大脑的原模型之差。`general.delta > 0` 表示通用能力下降。
- `extra_params_M`、`tokens_per_s`、`peak_vram_GB`：开销。
- `gen-out`：生成结果，供岗位适配度量表（专业性、语气、合规性）打分。

## 尚未实现

- 岗位适配度量表的自动打分（目前只导出生成结果）。
- ReFT 基线、RAG 基线。
- 多个大脑同时挂载与自动路由。
- 批量生成（`generate` 目前一次一条）。

## 硬件

RTX 5070（12 GB）建议用 Qwen2.5-0.5B / 1.5B 做主要实验。显存不够时先降 `--max-len` 和 `--batch-size`，用 `--accum` 补回等效批大小。
