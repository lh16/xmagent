"""端到端测试"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_customer_service import build_system

# 确定性分支的强断言：这些回复必须由真实数据/固定文案产生，缺关键词即异常。
# 其余用例的回复由模型自由生成，措辞不固定，只校验非空。
REQUIRED_KEYWORDS = {
    "我的订单12345到哪了？": ["12345"],
    "12346什么时候发货？": ["12346"],
    "我要投诉！": ["人工客服"],
}

# 不存在的订单号：端到端层同样要防住编造（多轮时曾把历史订单数据张冠李戴）
NOT_FOUND_QUESTION = "帮我查一下订单999898984944"

# 订单字段只能来自 query_order 返回，凭空出现即为编造
FABRICATED_FIELDS = ["📦 订单信息", "运单号", "快递公司", "下单时间"]


def check(question: str, answer: str):
    """
    校验单轮回复

    Returns:
        失败原因，通过时返回 None
    """
    if not answer or not answer.strip():
        return "回复为空"

    if question == NOT_FOUND_QUESTION:
        hits = [field for field in FABRICATED_FIELDS if field in answer]
        if hits:
            return f"订单不存在却输出了 {hits}，属于编造"
        return None

    missing = [kw for kw in REQUIRED_KEYWORDS.get(question, []) if kw not in answer]
    if missing:
        return f"回复中缺少 {missing}"

    return None


def main():
    agent = build_system()
    
    print("=" * 60)
    print("端到端测试")
    print("=" * 60)
    
    test_cases = [
        # 闲聊
        ("你好", "闲聊"),
        ("谢谢", "闲聊"),
        # 产品咨询
        ("星辰X10 Pro多少钱？", "产品咨询"),
        ("有什么手机推荐？", "产品咨询"),
        # 订单查询
        ("我的订单12345到哪了？", "订单查询"),
        ("12346什么时候发货？", "订单查询"),
        (NOT_FOUND_QUESTION, "订单查询（订单不存在，须如实告知）"),
        # 售后政策
        ("怎么退货？", "售后政策（暂无子Agent）"),
        # 转人工
        ("我要投诉！", "转人工"),
    ]
    
    failed = 0
    
    for question, expected in test_cases:
        print(f"\n{'─' * 60}")
        print(f"用户: {question}")
        print(f"预期类型: {expected}")
        print(f"{'─' * 60}")
        
        try:
            answer = agent.chat(question)
            print(f"小智: {answer}")
            
            reason = check(question, answer)
            if reason:
                failed += 1
                print(f">>> 失败：{reason}")
            else:
                print(">>> 通过")
        except Exception as e:
            failed += 1
            print(f">>> 失败：抛出异常 {e}")
    
    # 多轮：历史里有其他订单的真实数据时，查无此单不得张冠李戴
    print(f"\n{'─' * 60}")
    print("多轮测试：先查12346拿到真实数据，再查不存在的999898984944")
    print(f"{'─' * 60}")
    
    try:
        history = [{"role": "user", "content": "我的订单12346到哪了？"}]
        first_reply = agent.chat_with_history(history)
        print(f"小智: {first_reply.splitlines()[0]}...")
        
        history.append({"role": "assistant", "content": first_reply})
        history.append({"role": "user", "content": "999898984944呢？帮我查一下"})
        second_reply = agent.chat_with_history(history)
        print(f"小智: {second_reply}")
        
        hits = [field for field in FABRICATED_FIELDS if field in second_reply]
        if hits:
            failed += 1
            print(f">>> 失败：多轮查无此单却输出了 {hits}，属于编造")
        else:
            print(">>> 通过")
    except Exception as e:
        failed += 1
        print(f">>> 失败：抛出异常 {e}")
    
    print(f"\n{'=' * 60}")
    print("端到端测试全部通过" if failed == 0 else f"端到端测试完成，{failed} 个用例失败")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()