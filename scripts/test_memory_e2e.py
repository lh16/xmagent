"""带记忆的端到端测试：验证长期记忆能跨会话生效"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# settings 在导入时读取环境变量，建库路径必须在导入 config.settings 之前改掉。
# 否则会写进 ./data/long_term_memory.db，既污染真实数据，反复运行还会互相干扰。
TEMP_DIR = tempfile.mkdtemp(prefix="memory_e2e_")
os.environ["LONG_TERM_DB_PATH"] = os.path.join(TEMP_DIR, "long_term_memory.db")

from scripts.run_customer_service import build_system

USER_ID = "user_001"

# 会话1：首次咨询，留下姓名、预算、订单三条记忆
SESSION_1_QUESTIONS = [
    "你好，我叫小明",
    "我想买个手机，我预算3000左右",
    "有什么手机推荐？",
    "我的订单12345到哪了？",
]

# 会话2：新会话，只有长期记忆能答得上来
# 第一问走多轮，覆盖 chat_with_history 的记忆注入路径
SESSION_2_HISTORY = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "您好，我是智能客服小智，请问有什么可以帮您？"},
    {"role": "user", "content": "你还记得我叫什么吗？"},
]

SESSION_2_QUESTIONS = [
    "根据我的预算，有什么推荐？",
    "我上次看的那个订单到哪了？",
]


def check(failed: list, title: str, passed: bool, detail: str = ""):
    """记录一条断言结果"""
    if passed:
        print(f">>> 通过：{title}")
        return

    failed.append(title)
    print(f">>> 失败：{title}{detail}")


def ask(agent, question: str) -> str:
    """单轮提问，回复为空即记为失败"""
    print(f"\n{'─' * 60}")
    print(f"用户: {question}")
    print(f"{'─' * 60}")

    answer = agent.chat(question, user_id=USER_ID)
    print(f"小智: {answer}")

    return answer


def main():
    failed = []

    try:
        agent = build_system(session_id="session_001")

        print("=" * 60)
        print("会话1：用户首次咨询")
        print("=" * 60)

        for question in SESSION_1_QUESTIONS:
            answer = ask(agent, question)
            check(failed, f"会话1 有回复：{question}", bool(answer and answer.strip()))

        # 记忆落库的断言查库不查回复：回复措辞由模型自由生成，不稳定
        long_term = agent.memory.long_term
        profile = long_term.get_user(USER_ID)
        facts = long_term.get_user_facts(USER_ID)
        orders = long_term.get_user_orders(USER_ID)

        print(f"\n{'=' * 60}")
        print("会话1 结束：记忆是否落库")
        print("=" * 60)

        check(
            failed, "姓名已记住",
            bool(profile) and profile.get("name") == "小明",
            f" 实际: {profile}",
        )
        check(
            failed, "预算已记住",
            any("3000" in fact.get("content", "") for fact in facts),
            f" 实际: {facts}",
        )
        check(
            failed, "订单12345已记住",
            any(order.get("order_id") == "12345" for order in orders),
            f" 实际: {orders}",
        )
        stats = agent.memory.get_stats(USER_ID)
        check(failed, "统计含长期记忆", "long_term" in stats, f" 实际: {stats}")

        # ============================================================
        # 会话2：新会话
        # ============================================================
        print("\n\n" + "=" * 60)
        print("会话2：用户下次来访（新会话）")
        print("=" * 60)

        # 切会话而不是重建系统：短期记忆随会话重建，长期记忆不受影响
        agent.new_session("session_002")

        check(
            failed, "新会话短期记忆已清空",
            len(agent.memory.short_term.get_messages()) == 0,
            f" 实际: {len(agent.memory.short_term.get_messages())} 条",
        )

        print(f"\n{'─' * 60}")
        print("多轮对话：你还记得我叫什么吗？")
        print(f"{'─' * 60}")
        answer = agent.chat_with_history(SESSION_2_HISTORY, user_id=USER_ID)
        print(f"小智: {answer}")
        check(failed, "跨会话记得姓名", "小明" in answer, f" 实际: {answer[:100]}")

        for question in SESSION_2_QUESTIONS:
            answer = ask(agent, question)
            check(failed, f"会话2 有回复：{question}", bool(answer and answer.strip()))

        print(f"\n{'=' * 60}")
        print("记忆统计")
        print("=" * 60)
        print(agent.memory.get_stats(USER_ID))

        profile = agent.memory.long_term.get_user(USER_ID)
        check(
            failed, "长期记忆跨会话保留",
            bool(profile) and profile.get("name") == "小明",
            f" 实际: {profile}",
        )
        check(
            failed, "新会话已产生短期记忆",
            len(agent.memory.short_term.get_messages()) > 0,
        )

        print(f"\n{'=' * 60}")
        print("记忆端到端测试全部通过" if not failed else f"完成，{len(failed)} 项失败: {failed}")
        print("=" * 60)
    finally:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f"\n临时记忆库已清理: {TEMP_DIR}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
