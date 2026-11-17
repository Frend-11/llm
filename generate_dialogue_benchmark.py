#!/usr/bin/env python3
"""
客服对话与动态政策推理基准生成器 (Dialogue + Dynamic Policy Benchmark Generator)

基于原 8 字段结构化售后审批任务升级：
1. 后台保留 8 字段隐状态与 7 动作优先级规则引擎，确保 100% 可验证的 ground-truth 标签。
2. 训练阶段随机采样政策阈值（申请期限、经理审批金额阈值），防止模型死记硬背单一阈值。
3. 表层渲染为多轮客服对话与系统日志，注入跨轮反转、隐式风险、实付金额陷阱与强干扰噪声，
   使纯正则/静态规则引擎无法直接读取明文状态。
4. 自动输出反事实测试集（原政策、更新政策、强干扰），并标记政策变更样本与禁止退款安全标记。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ACTIONS = {
    "A": "补充证据",
    "B": "拒绝",
    "C": "风险审核",
    "D": "经理审批",
    "E": "等待退货",
    "F": "退款",
    "G": "关闭",
}


@dataclass(frozen=True)
class OrderLatentState:
    paid: bool
    days_since_purchase: int
    order_amount: int  # 实付金额（元）
    risk_flag: bool
    evidence_complete: bool
    return_received: bool
    manager_approved: bool
    already_refunded: bool

    def key(self) -> Tuple:
        return (
            self.paid,
            self.days_since_purchase,
            self.order_amount,
            self.risk_flag,
            self.evidence_complete,
            self.return_received,
            self.manager_approved,
            self.already_refunded,
        )


@dataclass(frozen=True)
class PolicyConfig:
    policy_name: str
    return_window_days: int
    manager_threshold: int

    def to_natural_text(self) -> str:
        return (
            f"【售后审批政策（{self.policy_name}）】\n"
            f"请严格按以下自上而下的优先级顺序判断唯一动作，命中第一条即停止：\n"
            f"1. 若系统确认订单已完成退款，选 G（关闭）。\n"
            f"2. 若用户提交的凭证不完整或无法核验，选 A（补充证据）。\n"
            f"3. 若订单未实际支付成功，或申请距离下单/签收已超过 {self.return_window_days} 天，选 B（拒绝）。\n"
            f"4. 若订单或账号存在欺诈、异常关联、恶意拒付等风险信号，选 C（风险审核）。\n"
            f"5. 若订单实付金额严格大于 {self.manager_threshold} 元且系统内尚未记录经理审批通过，选 D（经理审批）。\n"
            f"6. 若仓库系统尚未确认收到退回商品，选 E（等待退货）。\n"
            f"7. 其余满足全部条件的订单，选 F（退款）。"
        )


def evaluate_oracle_policy(state: OrderLatentState, policy: PolicyConfig) -> str:
    """7 类动作的确定性优先级真值规则引擎（仅用于后台生成标签）。"""
    if state.already_refunded:
        return "G"
    if not state.evidence_complete:
        return "A"
    if (not state.paid) or (state.days_since_purchase > policy.return_window_days):
        return "B"
    if state.risk_flag:
        return "C"
    if (state.order_amount > policy.manager_threshold) and (not state.manager_approved):
        return "D"
    if not state.return_received:
        return "E"
    return "F"


def sample_random_state(rng: random.Random) -> OrderLatentState:
    return OrderLatentState(
        paid=rng.random() < 0.85,
        days_since_purchase=rng.randint(1, 50),
        order_amount=rng.choice([89, 149, 199, 220, 299, 450, 520, 680, 899, 1200]),
        risk_flag=rng.random() < 0.25,
        evidence_complete=rng.random() < 0.78,
        return_received=rng.random() < 0.65,
        manager_approved=rng.random() < 0.40,
        already_refunded=rng.random() < 0.14,
    )


def render_dialogue_and_logs(
    state: OrderLatentState,
    rng: random.Random,
    include_distractors: bool = False,
    reference_date: date = date(2026, 11, 18),
) -> Tuple[List[Dict[str, str]], str]:
    """
    将 8 个隐状态渲染为带有跨轮反转、隐式线索和金额/时间推理的多轮客服对话与系统日志。
    """
    purchase_date = reference_date - timedelta(days=state.days_since_purchase)
    coupon_discount = rng.choice([0, 30, 50, 100])
    original_price = state.order_amount + coupon_discount

    turns: List[Dict[str, str]] = []
    log_lines: List[str] = [f"当前核验日期：{reference_date.isoformat()}"]

    if coupon_discount > 0:
        amount_phrase = (
            f"这单原价 {original_price} 元，我用了 {coupon_discount} 元满减券，"
            f"实际付了 {state.order_amount} 元"
        )
    else:
        amount_phrase = f"这单实付 {state.order_amount} 元"

    turns.append({
        "role": "user",
        "content": (
            f"你好，我在 {purchase_date.strftime('%Y年%m月%d日')} 下了一笔订单，{amount_phrase}。"
            f"现在想申请售后退货退款，麻烦帮我处理一下。"
        ),
    })

    if not state.paid:
        turns.append({
            "role": "assistant",
            "content": "好的，我先帮您核对一下订单支付和物流状态，请问您这边付款成功了吗？",
        })
        turns.append({
            "role": "user",
            "content": "我刚才点过付款了，手机好像弹了个扣款提示。",
        })
        log_lines.append("支付网关日志：该订单最新交易流水状态为 CLOSED_UNPAID（银行扣款预授权已冲正，商户未入账）。")
    elif state.already_refunded:
        turns.append({
            "role": "assistant",
            "content": "请问您是还没收到退款，还是想确认一下退货进度？",
        })
        turns.append({
            "role": "user",
            "content": "我 App 里好像没看到钱回来，是不是还没给我退啊？",
        })
        log_lines.append("财务结算日志：原路退回通道已于昨日 16:40 结算成功，退款流水号 RF-992104。")
    else:
        log_lines.append("支付网关日志：交易已成功入账（PAID），暂无退款入账记录。")

    if state.return_received:
        turns.append({
            "role": "user",
            "content": "快递我前几天就寄回去了，单号也填在系统里了。",
        })
        log_lines.append("仓储签收日志：退货包裹已于今日上午到达华东售后仓并完成质检入库。")
    else:
        turns.append({
            "role": "user",
            "content": "商品我已经打包好，按退货流程寄出去了！",
        })
        turns.append({
            "role": "assistant",
            "content": "感谢配合，我帮您查一下退货物流轨迹。",
        })
        turns.append({
            "role": "user",
            "content": "呃……刚才看了一眼快递小程序，好像只是生成了面单号，快递员还没上门揽收。",
        })
        log_lines.append("仓储签收日志：尚未检测到该订单对应的退货入库扫描记录。")

    if state.evidence_complete:
        turns.append({
            "role": "user",
            "content": "开箱视频、商品正面和序列号照片我都按要求上传到附件了。",
        })
        log_lines.append("附件核验：已接收 3 张高清图片及 1 段开箱视频，要素齐全。")
    else:
        turns.append({
            "role": "user",
            "content": "照片我刚刚已经传进去了，你们直接退吧。",
        })
        log_lines.append("附件核验：仅上传 1 张快递外包装纸箱照片，缺少商品本体及瑕疵细节图。")

    if state.risk_flag:
        turns.append({
            "role": "user",
            "content": "你们要是今天不直接给我退到账，我就立刻打信用卡中心电话发起恶意拒付，再去平台投诉！",
        })
        log_lines.append("风控引擎提示：当前账号近 7 天内关联 3 个新设备高频退货，且本次收货地与常用登录地跨省异常。")

    if state.manager_approved:
        turns.append({
            "role": "user",
            "content": "之前联系过你们主管，他说我这种情况可以走特批。",
        })
        log_lines.append("工单审批流：值班经理已于工单 #MGR-408 中签署【同意特批】。")
    else:
        if rng.random() < 0.5:
            turns.append({
                "role": "user",
                "content": "刚才电话里你们张经理口头答应给我特批了，你直接点退款就行！",
            })
            log_lines.append("工单审批流：无经理签批记录（备注：用户声称口头沟通，系统内未见任何特批授权）。")
        else:
            log_lines.append("工单审批流：当前无经理审批记录。")

    if include_distractors:
        log_lines.append(
            "【非政策参考元数据（与审批规则无关）】"
            "华东 7 号仓今日已处理 200 件包裹；客服中心办公时间为 09:00-18:00；"
            "包装规范编号 PKG-500；新员工培训周期为 30 天；"
            "历史促销活动满 500 减 30、满 200 减 7 已于上月结束。"
        )

    return turns, "\n".join(log_lines)


def build_llm_paraphrase_prompt(turns: List[Dict[str, str]], system_logs: str) -> str:
    """
    可选增强：如果你希望用大模型把模板对话改写得更口语化、更自然，
    可将此 prompt 发给大模型，同时严格要求保留所有数字、日期和事实反转。
    """
    dialogue_text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    return (
        "请将以下客服对话改写得更加自然、口语化，但必须严格保留所有事实细节：\n"
        "1. 严禁修改任何日期、金额、优惠券数额和跨轮态度的反转；\n"
        "2. 严禁把系统日志里的隐含状态直接改成布尔值标签；\n"
        "3. 保持相同的说话人轮次。\n\n"
        f"原始对话：\n{dialogue_text}\n\n系统日志（仅供对齐参考，勿泄露给用户）：\n{system_logs}"
    )


def sample_balanced_split(
    n_per_class: int,
    policy_sampler,
    rng: random.Random,
    seen_keys: set,
) -> List[Tuple[OrderLatentState, PolicyConfig, str]]:
    buckets: Dict[str, List[Tuple[OrderLatentState, PolicyConfig, str]]] = {
        k: [] for k in ACTIONS
    }
    attempts = 0
    max_attempts = n_per_class * len(ACTIONS) * 200

    while any(len(v) < n_per_class for v in buckets.values()):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("拒绝采样超出最大尝试次数，请检查采样分布或减小 n_per_class。")
        state = sample_random_state(rng)
        if state.key() in seen_keys:
            continue
        policy = policy_sampler(rng)
        action = evaluate_oracle_policy(state, policy)
        if len(buckets[action]) < n_per_class:
            seen_keys.add(state.key())
            buckets[action].append((state, policy, action))

    items = [item for cls_list in buckets.values() for item in cls_list]
    rng.shuffle(items)
    return items


def save_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成客服对话与动态政策推理基准数据集")
    parser.add_argument("--out_dir", type=str, default="dialogue_benchmark_data")
    parser.add_argument("--train_per_class", type=int, default=64)
    parser.add_argument("--val_per_class", type=int, default=16)
    parser.add_argument("--test_per_class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen_keys = set()

    def random_train_policy(r: random.Random) -> PolicyConfig:
        days = r.randint(3, 45)
        thresh = r.choice([100, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
        return PolicyConfig(
            policy_name=f"动态训练政策-{days}d-{thresh}y",
            return_window_days=days,
            manager_threshold=thresh,
        )

    orig_test_policy = PolicyConfig("原政策-v1", return_window_days=30, manager_threshold=500)
    new_test_policy = PolicyConfig("更新政策-v2", return_window_days=7, manager_threshold=200)

    train_items = sample_balanced_split(args.train_per_class, random_train_policy, rng, seen_keys)
    val_items = sample_balanced_split(args.val_per_class, random_train_policy, rng, seen_keys)
    test_items = sample_balanced_split(
        args.test_per_class, lambda _: orig_test_policy, rng, seen_keys
    )

    def format_records(
        items: List[Tuple[OrderLatentState, PolicyConfig, str]],
        split_name: str,
        include_distractors: bool = False,
        override_policy: Optional[PolicyConfig] = None,
        orig_actions: Optional[List[str]] = None,
    ) -> List[dict]:
        records = []
        for idx, (state, policy, action) in enumerate(items):
            active_policy = override_policy if override_policy is not None else policy
            active_action = evaluate_oracle_policy(state, active_policy) if override_policy else action
            turns, sys_logs = render_dialogue_and_logs(
                state, rng, include_distractors=include_distractors
            )
            policy_changed = (
                (orig_actions is not None) and (active_action != orig_actions[idx])
            )
            records.append({
                "id": f"{split_name}-{idx:04d}",
                "split": split_name,
                "policy_text": active_policy.to_natural_text(),
                "policy_params": asdict(active_policy),
                "dialogue_turns": turns,
                "system_logs": sys_logs,
                "latent_state": asdict(state),
                "label_action": active_action,
                "label_action_name": ACTIONS[active_action],
                "should_not_refund": active_action != "F",
                "policy_changed_from_orig": policy_changed,
                "has_distractors": include_distractors,
            })
        return records

    out_path = Path(args.out_dir)
    train_records = format_records(train_items, "train", include_distractors=False)
    val_records = format_records(val_items, "val", include_distractors=False)

    test_orig_records = format_records(test_items, "test_orig", include_distractors=False)
    orig_actions = [r["label_action"] for r in test_orig_records]

    test_new_records = format_records(
        test_items,
        "test_new_policy",
        include_distractors=False,
        override_policy=new_test_policy,
        orig_actions=orig_actions,
    )
    test_dist_records = format_records(
        test_items,
        "test_distractor",
        include_distractors=True,
        override_policy=orig_test_policy,
        orig_actions=orig_actions,
    )

    save_jsonl(train_records, out_path / "train.jsonl")
    save_jsonl(val_records, out_path / "val.jsonl")
    save_jsonl(test_orig_records, out_path / "test_orig.jsonl")
    save_jsonl(test_new_records, out_path / "test_new_policy.jsonl")
    save_jsonl(test_dist_records, out_path / "test_distractor.jsonl")

    changed_count = sum(1 for r in test_new_records if r["policy_changed_from_orig"])
    print(f"生成完成，输出目录: {out_path.resolve()}")
    print(f"训练集: {len(train_records)} | 验证集: {len(val_records)} | 测试集(每条件): {len(test_orig_records)}")
    print(f"更新政策后标签发生变化的测试订单数: {changed_count} / {len(test_new_records)}")


if __name__ == "__main__":
    main()
