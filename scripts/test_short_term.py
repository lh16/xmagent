"""测试短期记忆"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import (
    Message,
    ShortTermMemory,
    DEFAULT_MAX_SUMMARY_ITEMS,
)

# 长会话下摘要的总字符上限。按 20 条 * (角色前缀 + 100 字) 估算，留足余量
MAX_SUMMARY_CHARS = 3000


def report(name: str, ok: bool, detail: str = "") -> bool:
    """打印单条用例结果，返回是否通过"""
    print(f">>> {'PASS' if ok else 'FAIL'}" + (f"：{detail}" if detail else ""))
    return bool(ok)


def main():
    print("=" * 60)
    print("短期记忆测试")
    print("=" * 60)

    passed = 0
    total = 0

    # ---- 基本对话与压缩 ----
    print("\n" + "─" * 60)
    print("测试：压缩后保留最近 max_turns 轮")

    memory = ShortTermMemory(max_turns=3, session_id="test_session")

    conversations = [
        ("user", "你好"),
        ("assistant", "你好！有什么可以帮你的吗？"),
        ("user", "我叫小明"),
        ("assistant", "你好小明！很高兴认识你。"),
        ("user", "我的订单12345到哪了？"),
        ("assistant", "订单12345已发货，正在运输中。"),
        ("user", "它什么时候能到？"),
        ("assistant", "预计明天送达。"),
    ]

    for role, content in conversations:
        if role == "user":
            memory.add_user_message(content)
        else:
            memory.add_assistant_message(content)

    print(f"记忆统计: {memory.get_stats()}")
    print("消息列表:")
    for msg in memory.get_messages():
        print(f"  [{msg['role']}]: {msg['content'][:50]}")

    keep = [m.content for m in memory.messages]
    expected_keep = [content for _, content in conversations[-6:]]

    total += 1
    passed += report(
        "压缩后保留最近 6 条消息（max_turns=3）",
        len(memory.messages) == 6 and keep == expected_keep,
        f"实际 {keep}",
    )

    total += 1
    passed += report(
        "被压缩的早期对话进入摘要",
        memory.summary is not None and "user: 你好 |" in memory.summary,
        f"摘要: {memory.summary}",
    )

    # ---- 长会话：messages 与摘要都要有界 ----
    print("\n" + "─" * 60)
    print("测试：长会话下 messages 与摘要均有上限")

    long_memory = ShortTermMemory(max_turns=3, session_id="long_session")
    for i in range(200):
        long_memory.add_user_message(f"用户第{i}轮")
        long_memory.add_assistant_message(f"助手第{i}轮")

    stats = long_memory.get_stats()

    total += 1
    passed += report(
        "长会话下 messages 限长",
        stats["message_count"] == 6,
        f"实际 {stats['message_count']} 条",
    )

    total += 1
    passed += report(
        "长会话下摘要条数不超上限",
        stats["summary_items"] <= DEFAULT_MAX_SUMMARY_ITEMS,
        f"实际 {stats['summary_items']} 条 / 上限 {DEFAULT_MAX_SUMMARY_ITEMS}",
    )

    total += 1
    passed += report(
        "长会话下摘要总字符有界",
        stats["summary_chars"] <= MAX_SUMMARY_CHARS,
        f"实际 {stats['summary_chars']} 字符 / 上限 {MAX_SUMMARY_CHARS}",
    )

    # ---- max_turns=0 边界 ----
    print("\n" + "─" * 60)
    print("测试：max_turns=0 不被静默替换，且消息进摘要而非丢失")

    zero = ShortTermMemory(max_turns=0, session_id="zero_session")
    zero.add_user_message("我的订单12345到哪了？")
    zero.add_assistant_message("订单12345已发货。")

    total += 1
    passed += report(
        "max_turns=0 保持为 0",
        zero.max_turns == 0,
        f"实际 {zero.max_turns}",
    )

    total += 1
    passed += report(
        "max_turns=0 时消息转入摘要而非静默丢失",
        len(zero.messages) == 0 and "12345" in (zero.summary or ""),
        f"messages={len(zero.messages)}, summary={zero.summary}",
    )

    # ---- to_dict ----
    print("\n" + "─" * 60)
    print("测试：to_dict 的字段控制")

    msg = Message(role="user", content="你好", metadata={"intent": "chitchat"})

    total += 1
    passed += report(
        "默认只返回 role 与 content",
        set(msg.to_dict()) == {"role", "content"},
        f"实际 {sorted(msg.to_dict())}",
    )

    total += 1
    passed += report(
        "include_metadata=True 时携带 metadata 与 timestamp",
        {"metadata", "timestamp"} <= set(msg.to_dict(include_metadata=True)),
        f"实际 {sorted(msg.to_dict(include_metadata=True))}",
    )

    # ---- 工作记忆 ----
    print("\n" + "─" * 60)
    print("测试：工作记忆读写")

    memory.set_working("current_order_id", "12345")
    memory.set_working("user_name", "小明")

    total += 1
    passed += report(
        "工作记忆可读写，缺失时返回默认值",
        memory.get_working("current_order_id") == "12345"
        and memory.get_working("missing_key", "默认值") == "默认值",
        f"实际 {memory.working_memory}",
    )

    # ---- clear ----
    print("\n" + "─" * 60)
    print("测试：clear 清空消息、摘要与工作记忆")

    memory.clear()

    total += 1
    passed += report(
        "clear 后消息、摘要、工作记忆均为空",
        memory.messages == []
        and memory.summary is None
        and memory.working_memory == {},
        f"messages={len(memory.messages)}, summary={memory.summary}, "
        f"working={memory.working_memory}",
    )

    print("\n" + "=" * 60)
    print(f"测试完成：{passed}/{total} 个用例通过")
    print("=" * 60)

    # 有失败用例时以非0退出，便于CI发现
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
