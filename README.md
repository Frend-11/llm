# 第二大脑（secondbrain）

冻结开源主模型，在本地终端训练一个可插拔的旁路网络（"第二大脑"），调用时挂进 transformer 层并参与解码，让模型更胜任某个具体岗位。一个岗位对应一个 `.brain` 文件。

## 原理

| 通路 | 做法 |
|---|---|
| 读 | 在选定的若干层读取主模型隐状态 `h_l`（先做 LayerNorm），送入小型因果 transformer 得到 `s_l` |
| 写 | 回注：`h_l' = h_l + gate · c · r(h_l) · g_l · up_l(s_l)` |
| 解码 | logit 融合：`logits = logits_主 + alpha · c · lm_head(r(h_末) · up_out(s_末))`，复用主模型冻结的输出矩阵 |

- `r(h)` 是该位置隐状态的均方根，`c` 是固定常数（默认 `1/√旁路宽度`）。按相对幅度写入是必须的：不同模型、不同层的隐状态尺度能差两个数量级（Qwen3.5-0.8B 中间层每维只有 0.06–0.2），按绝对幅度写入时训练两步就会发散。
- `up_l` 和 `up_out` 零初始化，刚挂载时输出与原模型逐位一致。
- `alpha` 和 `gate` 是调用时的旋钮，不参与训练；都设为 0 就严格退回原模型。
- 主模型参数全程冻结，大脑以 float32 训练，主模型在 GPU 上用 bf16。

## 安装

```bash
bash scripts/setup_gpu.sh   # 在项目里建 .venv，装依赖，检查 CUDA，跑单元测试
source .venv/bin/activate   # 之后就可以直接用 brain 命令
```

不用 uv / 虚拟环境的话，`pip install -e ".[dev]" && pytest -q` 也可以。

## 复现实验

```bash
python scripts/prepare_data.py   # 下载并切分数据（固定种子）
bash scripts/run_all.sh          # 首轮全部实验，中断后重跑会续跑；日志在 results/run_all.log
```

`run_all.sh` 依次做三件事，也可以单独运行：

| 阶段 | 脚本 | 内容 | 结果目录 |
|---|---|---|---|
| A | `run_experiment.sh` | 0.5B、1 轮：大脑 / LoRA / 角色提示词 / 原模型，与云端 CPU 试验同配置 | `results/gpu_0.5b_pilot` |
| B | `sweep.sh` | 0.5B 上的结构与超参扫描（学习率、挂载层数、两条通路的消融、LoRA 的秩和学习率） | `results/sweep_0.5b` |
| C | `run_experiment.sh` | 1.5B、3 轮：默认大脑 vs 参数量对齐的 LoRA | `results/gpu_1.5b` |

数据来源：

- 岗位集：[DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) 的法律问答，按问题去重、过滤过长回答后取 1000 条训练、200 条测试。
- 通用集：[中文 Alpaca-GPT4](https://huggingface.co/datasets/shibing624/alpaca-zh) 的 200 条，只用来衡量通用能力是否下降。

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
brain train ... --mlp-ratio 0        # 旁路块只留注意力
brain train ... --no-norm-in         # 读入前不做 LayerNorm
brain train ... --abs-write --write-scale 1   # 旧结构：按绝对幅度写入
brain train --method lora --lora-r 0 ...      # LoRA 的秩自动取与第二大脑参数量对齐
brain eval  ... --alpha 0.5 --gate 1 # 推理期扫旋钮
```

## 评测报告字段

- `role.loss / ppl`：岗位测试集上的表现。
- `role.delta`、`general.delta`：同一进程内与关掉大脑的原模型之差。`general.delta > 0` 表示通用能力下降。
- `rouge_l`：生成结果与参考答案的字符级 ROUGE-L（贪心解码）。只是粗略的自动指标，不能代替量表打分。
- `extra_params_M`、`tokens_per_s`、`peak_vram_GB`：开销。
- `gen-out`：生成结果，供岗位适配度量表（专业性、语气、合规性）打分。

## 在云服务器上跑（Vultr）

本会话的网络只能走 HTTPS，没有 SSH，所以用对象存储当任务队列：`cloud/agent.py` 跑在服务器上，从桶里按文件名顺序取任务脚本执行，并把日志和结果传回桶里；`cloud/ctl.py` 是本地控制端，`cloud/bootstrap.sh` 是服务器的首次开机脚本。密钥都不进仓库（见 `cloud/ctl.py` 开头的说明）。

```bash
export VULTR_KEY_FILE=...  SB_STATE=...     # 仓库之外的两个文件
python cloud/ctl.py storage-create && python cloud/ctl.py bucket-create
python cloud/ctl.py push                    # 上传代码和 agent
python cloud/ctl.py instance-create         # 创建 GPU 服务器
python cloud/ctl.py submit job.sh           # 提交任务
python cloud/ctl.py status                  # 心跳、任务状态、花费
python cloud/ctl.py pull results/vultr      # 取回结果
python cloud/ctl.py teardown                # 销毁服务器和对象存储
```

## 尚未实现

- 岗位适配度量表的自动打分（目前只导出生成结果）。
- ReFT 基线、RAG 基线。
- 多个大脑同时挂载与自动路由。
- 批量生成（`generate` 目前一次一条）。

## 硬件

RTX 5070（12 GB）建议用 Qwen2.5-0.5B / 1.5B 做主要实验。显存不够时先降 `--max-len` 和 `--batch-size`，用 `--accum` 补回等效批大小。
