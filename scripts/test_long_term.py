"""测试长期记忆"""

import os
import shutil
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.long_term import LongTermMemory, EMPTY_CONTEXT


def report(name: str, ok: bool, detail: str = "") -> bool:
    """打印单条用例结果，返回是否通过"""
    print(f">>> {'PASS' if ok else 'FAIL'}" + (f"：{detail}" if detail else ""))
    return bool(ok)


def main():
    print("=" * 60)
    print("长期记忆测试")
    print("=" * 60)

    passed = 0
    total = 0

    # 用临时库：长期记忆是持久化的，用真实库反复跑会累积数据导致断言失真
    temp_dir = tempfile.mkdtemp(prefix="long_term_test_")
    memory = LongTermMemory(db_path=os.path.join(temp_dir, "test_memory.db"))
    user_id = "user_001"

    try:
        # ---- 1. 保存用户信息 ----
        print("\n" + "─" * 60)
        print("[1] 保存用户信息")

        memory.save_user(
            user_id=user_id,
            name="小明",
            preferences={"预算": "3000-5000", "品牌偏好": "星辰", "关注": "拍照"}
        )
        user = memory.get_user(user_id)

        total += 1
        passed += report(
            "用户画像可保存并读回",
            user is not None
            and user["name"] == "小明"
            and user["preferences"] == {
                "预算": "3000-5000", "品牌偏好": "星辰", "关注": "拍照"
            },
            f"实际 {user}",
        )

        # ---- 2. 记住订单 ----
        print("\n" + "─" * 60)
        print("[2] 记住订单")

        memory.remember_order(user_id, "12345", "星辰X10 Pro")
        memory.remember_order(user_id, "12346", "星辰Watch Pro")
        orders = memory.get_user_orders(user_id)

        total += 1
        passed += report(
            "两笔订单均可读回且顺序正确",
            [order["order_id"] for order in orders] == ["12345", "12346"]
            and orders[0]["product_name"] == "星辰X10 Pro",
            f"实际 {orders}",
        )

        # 同一订单重复记住应更新而非叠加
        memory.remember_order(user_id, "12345", "星辰X10 Pro（换新）")
        orders_again = memory.get_user_orders(user_id)

        total += 1
        passed += report(
            "重复记住同一订单不产生重复条目",
            len(orders_again) == 2
            and orders_again[0]["product_name"] == "星辰X10 Pro（换新）",
            f"实际 {orders_again}",
        )

        # ---- 3. 保存用户事实 ----
        print("\n" + "─" * 60)
        print("[3] 保存用户事实")

        memory.save_fact(user_id, "preference", "用户喜欢拍照好的手机")
        memory.save_fact(user_id, "context", "用户正在为家人选购生日礼物")
        facts = memory.get_user_facts(user_id)

        total += 1
        passed += report(
            "事实可按类别读回",
            len(facts) == 2
            and len(memory.get_user_facts(user_id, category="preference")) == 1
            and memory.get_user_facts(user_id, category="preference")[0]["content"]
            == "用户喜欢拍照好的手机",
            f"实际 {facts}",
        )

        memory.save_fact(user_id, "preference", "用户喜欢拍照好的手机")

        total += 1
        passed += report(
            "重复保存相同事实不产生冗余",
            len(memory.get_user_facts(user_id)) == 2,
            f"实际 {len(memory.get_user_facts(user_id))} 条",
        )

        # ---- 4. 保存交互记录 ----
        print("\n" + "─" * 60)
        print("[4] 保存交互记录")

        memory.save_interaction(
            user_id, "有什么手机推荐？", "产品咨询", "推荐了星辰X10 Pro"
        )
        recent = memory.get_recent_interactions(user_id)

        total += 1
        passed += report(
            "交互记录可保存并读回",
            len(recent) == 1 and recent[0]["intent"] == "产品咨询",
            f"实际 {recent}",
        )

        # ---- 5. 检索上下文 ----
        print("\n" + "─" * 60)
        print("[5] 检索用户记忆")

        context = memory.retrieve_context(user_id)
        print(context)

        required = [
            "小明",           # 姓名
            "3000-5000",      # 偏好
            "星辰",           # 品牌偏好
            "12345",          # 订单号
            "星辰X10 Pro（换新）",
            "12346",
            "用户喜欢拍照好的手机",   # 事实
            "为家人选购生日礼物",
            "有什么手机推荐？",       # 最近交互
        ]
        missing = [item for item in required if item not in context]

        total += 1
        passed += report(
            "检索上下文包含画像、订单、事实与最近交互",
            not missing,
            f"缺少 {missing}" if missing else "",
        )

        # ---- 6. 用户隔离 ----
        print("\n" + "─" * 60)
        print("[6] 用户隔离：不存在的用户不应读到他人数据")

        total += 1
        passed += report(
            "不存在的用户返回空而非报错",
            memory.get_user("user_999") is None
            and memory.get_user_orders("user_999") == []
            and memory.get_user_facts("user_999") == []
            and memory.retrieve_context("user_999") == EMPTY_CONTEXT,
            f"实际 user={memory.get_user('user_999')}, "
            f"orders={memory.get_user_orders('user_999')}",
        )

        # ---- 7. 偏好合并 ----
        print("\n" + "─" * 60)
        print("[7] 重复保存画像时偏好按 key 合并")

        other = "user_002"
        memory.save_user(other, name="小红", preferences={"预算": "2000-3000"})
        memory.save_user(other, preferences={"关注": "续航"})

        merged = memory.get_user(other)

        total += 1
        passed += report(
            "后写入的偏好与已有偏好合并，且姓名不被清空",
            merged["preferences"] == {"预算": "2000-3000", "关注": "续航"}
            and merged["name"] == "小红",
            f"实际 {merged}",
        )

        # ---- 8. 删除用户数据 ----
        print("\n" + "─" * 60)
        print("[8] 删除用户数据")

        memory.remember_order(other, "99999", "测试商品")
        memory.save_fact(other, "preference", "测试事实")
        memory.delete_user(other)

        total += 1
        passed += report(
            "删除后该用户的画像、订单、事实全部清除",
            memory.get_user(other) is None
            and memory.get_user_orders(other) == []
            and memory.get_user_facts(other) == [],
            f"实际 user={memory.get_user(other)}, "
            f"orders={memory.get_user_orders(other)}, "
            f"facts={memory.get_user_facts(other)}",
        )

        total += 1
        passed += report(
            "删除一个用户不影响其他用户",
            memory.get_user(user_id) is not None
            and len(memory.get_user_orders(user_id)) == 2,
            f"实际 {memory.get_user_orders(user_id)}",
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
