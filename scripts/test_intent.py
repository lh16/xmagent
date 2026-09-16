"""测试意图识别模块"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.intent import Intent, IntentClassifier


def main():
    print("=" * 60)
    print("意图识别测试")
    print("=" * 60)

    classifier = IntentClassifier()

    # (消息, 期望意图, 是否应由规则命中)
    # 规则用例不依赖网络、结果确定；"你好/谢谢/再见"无关键词，必然走LLM兜底
    cases = [
        # 产品咨询
        ("星辰X10 Pro多少钱？", Intent.PRODUCT_INQUIRY, True),
        ("有什么手机推荐？", Intent.PRODUCT_INQUIRY, True),
        ("X10和X10 Pro有什么区别？", Intent.PRODUCT_INQUIRY, True),
        # 订单查询
        ("我的订单12345到哪了？", Intent.ORDER_QUERY, True),
        ("12346什么时候发货？", Intent.ORDER_QUERY, True),
        ("快递单号是多少？", Intent.ORDER_QUERY, True),
        # 售后政策
        ("怎么退货？", Intent.AFTER_SALES, True),
        ("保修期是多久？", Intent.AFTER_SALES, True),
        ("七天无理由退货怎么操作？", Intent.AFTER_SALES, True),
        # 闲聊
        ("你好", Intent.CHITCHAT, False),
        ("谢谢", Intent.CHITCHAT, False),
        ("再见", Intent.CHITCHAT, False),
        # 转人工
        ("我要投诉！", Intent.TRANSFER_HUMAN, True),
        ("转人工客服", Intent.TRANSFER_HUMAN, True),
    ]

    passed = 0
    total = 0

    for message, expected, by_rule in cases:
        total += 1
        result = classifier.classify_with_confidence(message)
        intent = result["intent"]

        # 规则用例额外校验命中方式：防止规则被改坏后静默退化到LLM
        ok = intent is expected and (result["method"] == "rule") == by_rule

        print("\n" + "─" * 60)
        print(f"消息: {message}")
        print(f"结果: {intent.value}（{result['method']}，置信度 {result['confidence']}）")
        expect_method = "rule" if by_rule else "llm"
        print(f">>> {'PASS' if ok else f'FAIL（期望：{expected.value} / {expect_method}）'}")
        passed += ok

    # classify() 应与 classify_with_confidence() 结果一致（只跑规则用例，避免额外网络调用）
    total += 1
    ok = all(
        classifier.classify(message) is classifier.classify_with_confidence(message)["intent"]
        for message, _, by_rule in cases if by_rule
    )
    print("\n" + "─" * 60)
    print("测试：classify() 与 classify_with_confidence() 结果一致")
    print(f">>> {'PASS' if ok else 'FAIL'}")
    passed += ok

    # 空白消息不应抛异常（走LLM兜底，允许数秒）
    total += 1
    print("\n" + "─" * 60)
    print("测试：空白消息兜底")
    try:
        intent = classifier.classify("   ")
        ok = isinstance(intent, Intent)
        print(f"结果: {intent.value}")
    except Exception as e:
        ok = False
        print(f"异常: {e}")
    print(f">>> {'PASS' if ok else 'FAIL'}")
    passed += ok

    print("\n" + "=" * 60)
    print(f"测试完成：{passed}/{total} 个用例通过")
    print("=" * 60)

    # 有失败用例时以非0退出，便于CI发现
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
