"""测试统一记忆管理器"""

import os
import shutil
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.long_term import LongTermMemory
from src.memory.memory_manager import MemoryManager


def report(name: str, ok: bool, detail: str = "") -> bool:
    """打印单条用例结果，返回是否通过"""
    print(f">>> {'PASS' if ok else 'FAIL'}" + (f"：{detail}" if detail else ""))
    return bool(ok)


def main():
    print("=" * 60)
    print("统一记忆管理器测试")
    print("=" * 60)

    passed = 0
    total = 0

    # 用临时库：长期记忆是持久化的，用真实库反复跑会累积数据导致断言失真
    temp_dir = tempfile.mkdtemp(prefix="memory_manager_test_")
    db_path = os.path.join(temp_dir, "test_memory.db")

    def new_manager(session_id: str) -> MemoryManager:
        """建一个指向临时库的管理器"""
        manager = MemoryManager(session_id=session_id)
        manager.long_term = LongTermMemory(db_path=db_path)
        return manager

    user_id = "user_001"
    memory = new_manager("test_session_001")

    try:
        # ---- 1. 模拟对话并提取记忆 ----
        print("\n" + "─" * 60)
        print("[1] 模拟对话并提取记忆")

        # 预算正则要求「我预算」，写成「，预算3000」则匹配不到
        conversation = "你好，我叫小明，我想买个手机，我预算3000左右"
        memory.add_user_message(conversation, user_id=user_id)
        memory.extract_and_remember(user_id, conversation)

        profile = memory.long_term.get_user(user_id)
        facts = memory.long_term.get_user_facts(user_id)

        total += 1
        passed += report(
            "从对话中提取到姓名与预算",
            profile is not None
            and profile["name"] == "小明"
            and any("3000" in fact["content"] for fact in facts),
            f"实际 profile={profile}, facts={facts}",
        )

        total += 1
        passed += report(
            "用户消息已进入短期记忆",
            len(memory.get_messages()) == 1
            and memory.get_messages()[0]["content"] == conversation,
            f"实际 {memory.get_messages()}",
        )

        # ---- 2. 构建上下文 ----
        print("\n" + "─" * 60)
        print("[2] 构建记忆上下文")

        context = memory.build_context(user_id)
        print(context)

        total += 1
        passed += report(
            "上下文包含画像与事实",
            "小明" in context and "3000" in context,
            f"实际 {context!r}",
        )

        empty_manager = new_manager("test_session_empty")

        total += 1
        passed += report(
            "无记忆时返回空字符串，不把兜底文案拼进上下文",
            empty_manager.build_context("user_999") == "",
            f"实际 {empty_manager.build_context('user_999')!r}",
        )

        # ---- 3. 新会话：验证跨会话记忆 ----
        print("\n" + "─" * 60)
        print("[3] 新会话 - 模拟用户下次来访")

        new_memory = new_manager("test_session_002")

        total += 1
        passed += report(
            "新会话仍能读到长期记忆中的画像",
            "小明" in new_memory.build_context(user_id),
            f"实际 {new_memory.build_context(user_id)!r}",
        )

        total += 1
        passed += report(
            "新会话的短期记忆是干净的",
            new_memory.get_messages() == [],
            f"实际 {new_memory.get_messages()}",
        )

        # ---- 4. 订单只在显式调用后记住 ----
        print("\n" + "─" * 60)
        print("[4] 订单记忆：不做文本盲提")

        memory.extract_and_remember(user_id, "帮我查一下订单999898984944")

        total += 1
        passed += report(
            "对话里提到不存在的订单号不会被记住",
            memory.long_term.get_user_orders(user_id) == [],
            f"实际 {memory.long_term.get_user_orders(user_id)}",
        )

        memory.remember_order(user_id, "12345", "星辰X10 Pro")
        orders = memory.long_term.get_user_orders(user_id)

        total += 1
        passed += report(
            "查询成功后显式记住的订单可写回",
            [order["order_id"] for order in orders] == ["12345"],
            f"实际 {orders}",
        )

        total += 1
        passed += report(
            "记住订单时同步写入工作记忆",
            memory.short_term.get_working("current_order_id") == "12345",
            f"实际 {memory.short_term.get_working('current_order_id')}",
        )

        # ---- 5. 姓名不再误提取 ----
        print("\n" + "─" * 60)
        print("[5] 姓名提取的边界")

        other = "user_002"
        memory.extract_and_remember(other, "我是来问订单的")

        total += 1
        passed += report(
            "「我是来问订单的」不产生姓名",
            memory.long_term.get_user(other) is None,
            f"实际 {memory.long_term.get_user(other)}",
        )

        # ---- 6. 统计 ----
        print("\n" + "─" * 60)
        print("[6] 记忆统计")

        stats = memory.get_stats(user_id)

        total += 1
        passed += report(
            "统计同时覆盖短期与长期记忆",
            "short_term" in stats
            and stats.get("long_term", {}).get("has_profile") is True
            and stats.get("long_term", {}).get("orders") == 1,
            f"实际 {stats}",
        )

        # ---- 7. 清空 ----
        print("\n" + "─" * 60)
        print("[7] 清空记忆")

        memory.clear(user_id)

        total += 1
        passed += report(
            "传入 user_id 时短期与长期记忆一并清除",
            memory.get_messages() == []
            and memory.long_term.get_user(user_id) is None
            and memory.long_term.get_user_orders(user_id) == [],
            f"实际 messages={memory.get_messages()}, "
            f"user={memory.long_term.get_user(user_id)}",
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"测试完成：{passed}/{total} 个用例通过")
    print("=" * 60)

    # 有失败用例时以非0退出，便于CI发现
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
