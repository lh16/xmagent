"""完整智能客服系统端到端测试

断言只覆盖数据层（长期记忆、订单记忆、转人工队列）：这些由代码路径决定，
不随 LLM 措辞变化，稳定可复现。

回复层（小智具体说了什么）暂不做断言：LLM 换个说法就会误报，需要先跑一次
拿到真实输出再按实际行为校准，否则容易写出永远通过或永远失败的假断言。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_full_system import build_system

# 断言结果：(用例名, 是否通过)
RESULTS = []


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print(f"{'─' * 60}")


def report(name: str, ok, detail: str = "") -> bool:
    """打印单条断言结果，风格与 scripts/test_memory_manager.py 保持一致"""
    ok = bool(ok)
    RESULTS.append((name, ok))
    print(f">>> {'PASS' if ok else 'FAIL'}" + (f"：{detail}" if detail else ""))
    return ok


def _profile_name(agent, user_id: str):
    """长期记忆中的姓名，无画像时返回 None"""
    profile = agent.memory.long_term.get_user(user_id)
    return (profile or {}).get("name")


def _remembered_order_ids(agent, user_id: str):
    """已写入长期记忆的订单号"""
    return [o["order_id"] for o in agent.memory.long_term.get_user_orders(user_id)]


def _fact_contents(agent, user_id: str):
    """长期记忆中的事实内容"""
    return [f["content"] for f in agent.memory.long_term.get_user_facts(user_id)]


def test_scenario_1_product_inquiry(agent, user_id):
    """场景1：产品咨询"""
    print_separator("场景1：产品咨询")

    # 第一句带上姓名：场景5、场景6 要靠这里记住的名字验证记忆能力，
    # 不说名字的话"你还记得我叫什么吗"必然答不上来
    questions = [
        "你好，我叫张三",
        "我想买个手机，预算3000左右",
        "有什么手机推荐？",
        "星辰X10和X10 Pro有什么区别？",
        "星辰X10 Pro的价格是多少？",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")

    name = _profile_name(agent, user_id)
    report("记住了用户姓名", name == "张三", f"实际 {name!r}")

    # 预算这句故意不带主语（"我想买个手机，预算3000左右"）。
    # 提取正则若退回要求"我/我的"就匹配不上，这里守住该回归
    facts = _fact_contents(agent, user_id)
    report(
        "记住了预算（不带主语也要能提取）",
        any("3000" in content for content in facts),
        f"实际 {facts}",
    )


def test_scenario_2_order_query(agent, user_id):
    """场景2：订单查询"""
    print_separator("场景2：订单查询")

    questions = [
        "我的订单12345到哪了？",
        "12346什么时候发货？",
        "我的快递单号是多少？",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")

    # 12345、12346 在 mock 库中都存在，查到才会写入长期记忆
    order_ids = _remembered_order_ids(agent, user_id)
    report(
        "查到的订单已写入长期记忆",
        "12345" in order_ids and "12346" in order_ids,
        f"实际 {order_ids}",
    )


def test_scenario_3_after_sales(agent, user_id):
    """场景3：售后政策"""
    print_separator("场景3：售后政策")

    questions = [
        "怎么退货？",
        "保修期是多久？",
        "退款多久能到账？",
        "发票丢了还能保修吗？",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")


def test_scenario_4_transfer_human(agent, user_id):
    """场景4：转人工"""
    print_separator("场景4：转人工")

    questions = [
        "我要投诉！",
        "转人工客服",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")

    queue = agent.get_transfer_queue()
    report("转人工工单已入队", len(queue) >= 1, f"实际 {len(queue)} 条")


def test_scenario_5_memory(agent, user_id):
    """场景5：记忆能力"""
    print_separator("场景5：记忆能力")

    questions = [
        "你还记得我叫什么吗？",
        "我上次看的那个订单到哪了？",
        "根据我的预算，有什么推荐？",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")


def test_scenario_6_cross_session(agent, user_id):
    """场景6：跨会话记忆"""
    print_separator("场景6：跨会话记忆")

    # 切会话而非重建系统：new_session 只重建短期记忆，长期记忆保留，
    # 这正是要验证的跨会话能力；重建系统会白白再跑一遍文档加载与子Agent装配
    agent.new_session("new_session_001")

    # 切会话后长期记忆必须还在，这是"跨会话"本身，先于提问验证
    name = _profile_name(agent, user_id)
    report("新会话仍读到长期记忆中的姓名", name == "张三", f"实际 {name!r}")

    questions = [
        "你还记得我吗？",
        "我之前问的那个手机现在有货吗？",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")


# 场景编号 -> (名称, 执行函数)
SCENARIOS = [
    ("1", "产品咨询", test_scenario_1_product_inquiry),
    ("2", "订单查询", test_scenario_2_order_query),
    ("3", "售后政策", test_scenario_3_after_sales),
    ("4", "转人工", test_scenario_4_transfer_human),
    ("5", "记忆能力", test_scenario_5_memory),
    ("6", "跨会话记忆", test_scenario_6_cross_session),
]

# 记忆类场景依赖场景1先留下姓名与预算，单跑会因没有数据而误报失败
DEPENDENCIES = {"5": ("1",), "6": ("1",)}


def _run_scenario(name: str, func, *args) -> bool:
    """执行单个场景，失败不中断后续场景"""
    try:
        func(*args)
        return True
    except Exception as exc:
        print(f"\n[场景失败] {name}: {type(exc).__name__}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="完整智能客服系统端到端测试")
    parser.add_argument(
        "--scenario",
        default="all",
        help="要运行的场景编号，逗号分隔，如 5,6；默认 all 跑全部",
    )
    args = parser.parse_args()

    known = [sid for sid, _, _ in SCENARIOS]
    if args.scenario.strip().lower() == "all":
        selected = list(known)
    else:
        selected = [s.strip() for s in args.scenario.split(",") if s.strip()]
        unknown = [s for s in selected if s not in known]
        if unknown:
            print(f"[警告] 未知场景编号，已忽略: {unknown}")
        for sid in list(selected):
            for dep in DEPENDENCIES.get(sid, ()):
                if dep not in selected:
                    selected.append(dep)
                    print(f"[提示] 场景{sid}依赖场景{dep}先留下记忆，已自动加入")
    # 按编号顺序执行，保证前置场景先跑
    selected = [sid for sid in known if sid in selected]

    if not selected:
        print("没有可运行的场景")
        sys.exit(1)

    user_id = "user_001"

    # 构建系统
    agent = build_system(session_id="test_session_001")

    # 长期记忆持久化在 SQLite，不清的话第二次跑会带着上次的姓名、预算、订单，
    # 记忆类场景的结果不可复现，还会掩盖"记不住名字"这类问题
    agent.memory.long_term.delete_user(user_id)
    print(f"\n已清理 {user_id} 的长期记忆，确保本轮结果可复现")

    errors = []
    for sid, name, func in SCENARIOS:
        if sid not in selected:
            continue
        if not _run_scenario(f"场景{sid} {name}", func, agent, user_id):
            errors.append(f"场景{sid} {name}")

    # 查看统计
    print_separator("系统统计")
    # get_stats 不带 user_id 时只统计短期记忆，看不到跨会话的长期记忆
    print(f"记忆统计: {agent.memory.get_stats(user_id=user_id)}")
    print(f"转人工队列: {len(agent.get_transfer_queue())} 条")

    print_separator("断言结果")
    failed_asserts = [name for name, ok in RESULTS if not ok]
    print(f"断言: {len(RESULTS) - len(failed_asserts)}/{len(RESULTS)} 通过")
    if failed_asserts:
        print(f"失败断言: {failed_asserts}")
    if errors:
        print(f"执行异常: {errors}")

    # 有失败断言或执行异常时以非0退出，便于CI发现
    sys.exit(1 if failed_asserts or errors else 0)


if __name__ == "__main__":
    main()
