"""完整智能客服系统端到端测试"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_full_system import build_system


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print(f"{'─' * 60}")


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
    
    questions = [
        "你还记得我吗？",
        "我之前问的那个手机现在有货吗？",
    ]
    
    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.chat(q, user_id=user_id)
        print(f"小智: {answer}")


def _run_scenario(name: str, func, *args) -> bool:
    """执行单个场景，失败不中断后续场景"""
    try:
        func(*args)
        return True
    except Exception as exc:
        print(f"\n[场景失败] {name}: {type(exc).__name__}: {exc}")
        return False


def main():
    user_id = "user_001"
    
    # 构建系统
    agent = build_system(session_id="test_session_001")
    
    # 长期记忆持久化在 SQLite，不清的话第二次跑会带着上次的姓名、预算、订单，
    # 记忆类场景的结果不可复现，还会掩盖"记不住名字"这类问题
    agent.memory.long_term.delete_user(user_id)
    print(f"\n已清理 {user_id} 的长期记忆，确保本轮结果可复现")
    
    # 运行所有测试场景
    scenarios = [
        ("场景1 产品咨询", test_scenario_1_product_inquiry),
        ("场景2 订单查询", test_scenario_2_order_query),
        ("场景3 售后政策", test_scenario_3_after_sales),
        ("场景4 转人工", test_scenario_4_transfer_human),
        ("场景5 记忆能力", test_scenario_5_memory),
        ("场景6 跨会话记忆", test_scenario_6_cross_session),
    ]
    
    failed = []
    for name, func in scenarios:
        if not _run_scenario(name, func, agent, user_id):
            failed.append(name)
    
    # 查看统计
    print_separator("系统统计")
    # get_stats 不带 user_id 时只统计短期记忆，看不到跨会话的长期记忆
    print(f"记忆统计: {agent.memory.get_stats(user_id=user_id)}")
    print(f"转人工队列: {len(agent.get_transfer_queue())} 条")
    
    if failed:
        print(f"\n失败场景: {failed}")


if __name__ == "__main__":
    main()
