"""测试订单查询子Agent"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.order_agent import OrderQueryAgent

# 订单字段只能来自 query_order 的返回，凭空出现即为编造
FABRICATED_FIELDS = [
    "📦 订单信息",
    "运单号",
    "快递公司",
    "预计送达",
    "下单时间",
    "订单状态",
]

# 数据库中不存在的订单号，用于验证 Agent 不会编造订单信息
NOT_FOUND_QUESTION = "我想查一下订单99999"

# 消息中直接带订单号（未出现"订单"二字），用于验证 Agent 不会反问索要订单号
ORDER_ID_QUESTION = "12346什么时候发货？"

# 反问索要订单号时的典型措辞
ASK_FOR_ORDER_ID = ["请提供", "提供订单号", "提供一下订单号", "告诉我订单号"]


def main():
    agent = OrderQueryAgent()
    failed = 0

    print("=" * 60)
    print("订单查询子Agent测试")
    print("=" * 60)
    
    test_cases = [
        "我的订单12345到哪了？",
        "12346什么时候发货？",
        "我想查一下订单99999",
        "我的快递单号是多少？",
        "我最近买了什么？",
    ]
    
    for question in test_cases:
        print(f"\n{'─' * 60}")
        print(f"用户: {question}")
        print(f"{'─' * 60}")
        
        answer = agent.chat(question)
        print(f"小智: {answer}")

        # 断言：订单不存在时不得输出任何订单字段，只能如实告知未找到
        if question == NOT_FOUND_QUESTION:
            hits = [field for field in FABRICATED_FIELDS if field in answer]
            if hits:
                failed += 1
                print(f">>> 失败：订单不存在却输出了 {hits}，属于编造")
            else:
                print(">>> 通过：未找到订单时如实告知，无编造字段")

        # 断言：消息已带订单号时，必须直接给出订单数据，不得反问索要订单号
        elif question == ORDER_ID_QUESTION:
            asked = [phrase for phrase in ASK_FOR_ORDER_ID if phrase in answer]
            if asked or "12346" not in answer:
                failed += 1
                reason = f"反问索要订单号 {asked}" if asked else "未用上订单号12346"
                print(f">>> 失败：{reason}")
            else:
                print(">>> 通过：识别到订单号并直接返回订单数据")

    # 多轮场景：历史里有其他订单的真实数据时，查无此单不得张冠李戴
    print(f"\n{'─' * 60}")
    print("多轮测试：先查12346拿到真实数据，再查不存在的999898984944")
    print(f"{'─' * 60}")

    history = [{"role": "user", "content": "我的订单12346到哪了？"}]
    reply_12346 = agent.chat_with_history(history)
    print(f"小智: {reply_12346.splitlines()[0]}...")

    history.append({"role": "assistant", "content": reply_12346})
    history.append({"role": "user", "content": "999898984944呢？帮我查一下"})
    reply_fake = agent.chat_with_history(history)
    print(f"小智: {reply_fake}")

    hits = [field for field in FABRICATED_FIELDS if field in reply_fake]
    if hits or "未找到" not in reply_fake:
        failed += 1
        reason = f"输出了 {hits}" if hits else "未如实告知未找到"
        print(f">>> 失败：多轮查无此单却{reason}，可能借用了历史订单数据")
    else:
        print(">>> 通过：多轮查无此单如实告知，未借用历史订单数据")

    print(f"\n{'=' * 60}")
    print("测试完成" if failed == 0 else f"测试完成，{failed} 个用例失败")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()