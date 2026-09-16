"""测试订单查询工具"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.order_tools import query_order, get_logistics, query_user_orders


def main():
    print("=" * 60)
    print("订单查询工具测试")
    print("=" * 60)

    # (用例说明, 调用, 结果必须包含的关键词, 结果不能包含的关键词)
    cases = [
        ("查询已发货订单 12345", lambda: query_order("12345"), "顺丰速运", None),
        ("查询不存在的订单 99999", lambda: query_order("99999"), "未找到订单号", None),
        ("查询物流 12345", lambda: get_logistics("12345"), "物流轨迹", None),
        ("查询待发货订单 12346（不应有物流段）",
         lambda: query_order("12346"), "待发货", "🚚 物流信息"),
        ("查询用户订单列表 user_001", lambda: query_user_orders("user_001"), "您最近有", None),
    ]

    passed = 0
    for name, call, expect, forbid in cases:
        print("\n" + "─" * 60)
        print(f"测试：{name}")
        print("─" * 60)

        result = call()
        print(result)

        # 工具内部已兜底异常，因此这里只需校验返回内容是否符合预期
        ok = expect in result and not (forbid and forbid in result)
        print(f">>> {'PASS' if ok else 'FAIL'}（期望包含：{expect}）")
        passed += ok

    print("\n" + "=" * 60)
    print(f"测试完成：{passed}/{len(cases)} 个用例通过")
    print("=" * 60)

    # 有失败用例时以非0退出，便于CI发现
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()