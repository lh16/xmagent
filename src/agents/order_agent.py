"""
订单查询子Agent
专门处理订单相关问题
"""

import re

from deepagents import create_deep_agent
from config.prompts import ORDER_AGENT_PROMPT
from src.utils.model import create_model
from src.tools.order_tools import query_order, get_logistics, query_user_orders
from src.utils.logger import log

# 订单号特征：5 位以上连续数字
ORDER_ID_PATTERN = re.compile(r"\d{5,}")

# 预取到订单数据后注入的上下文。
# 仅靠提示词约束时，模型常漏调 query_order 而反问用户索要订单号，
# 因此这里直接把真实查询结果塞进上下文，模型只需基于它作答。
ORDER_CONTEXT_TEMPLATE = """以下是订单 {order_id} 的真实查询结果（来自 query_order 工具）：

{data}

请严格基于以上内容回答用户。若结果显示未找到订单，如实告知并请用户核对订单号，
不要补充或推测任何订单字段。"""

# 本轮没有订单号、也没有真实数据可注入时的约束。
# 此时没有任何数据兜底，模型会凭空编一个订单号来"询问确认"：实测记忆里只有姓名，
# 它回"您上次看的订单可能是12345，是否查询？"，用户一确认就查到不存在的单。
NO_ORDER_HINT = """当前用户没有提供订单号，上下文里也没有任何订单号。

- 直接请用户提供订单号
- 禁止输出任何具体订单号，包括"可能是xxxxx""是不是xxxxx"这类猜测
- 不要从训练数据里联想示例号码"""


class OrderQueryAgent:
    """订单查询子Agent"""
    
    def __init__(self):
        """初始化订单查询子Agent"""
        self.model = create_model(temperature=0.3)
        
        # 工具列表
        self.tools = [
            query_order,
            get_logistics,
            query_user_orders,
        ]
        
        # 创建Agent
        self.agent = self._create_agent()
        
        log.info("订单查询子Agent初始化完成")
    
    def _create_agent(self):
        """创建Deep Agent"""
        return create_deep_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=ORDER_AGENT_PROMPT,
        )
    
    def _prepare_messages(self, messages: list) -> tuple:
        """
        若最后一条用户消息含订单号，先取真实订单数据作为上下文注入。

        提示词已要求模型自行调用 query_order，但实测模型常漏调而反问用户索要
        订单号，因此这里兜底预取，保证带订单号的问法一定能拿到真实数据。

        Args:
            messages: 原始消息列表

        Returns:
            (messages, direct_reply)：预取到数据时注入上下文并返回
            (messages, None)；查无此单或查询失败时返回 (messages, 工具原文)，
            由调用方直接短路回复，不再经过LLM
        """
        prepared = list(messages)
        direct_reply = None

        if prepared and prepared[-1].get("role") == "user":
            match = ORDER_ID_PATTERN.search(prepared[-1].get("content", ""))
            if match:
                order_id = match.group()
                data = query_order(order_id)

                if data.startswith("📦 订单信息"):
                    log.info(f"[订单查询] 预取订单 {order_id} 并注入上下文")
                    prepared.insert(len(prepared) - 1, {
                        "role": "system",
                        "content": ORDER_CONTEXT_TEMPLATE.format(
                            order_id=order_id, data=data
                        ),
                    })
                else:
                    # 查无此单/查询失败：直接回复工具原文，不经过LLM。
                    # 多轮时历史里可能有其他订单的真实数据，LLM会张冠李戴，
                    # 给不存在的订单编出"已完成、金额xxx"这类假信息（实测复现）
                    log.info(f"[订单查询] 订单 {order_id} 未取到数据，直接回复工具结果")
                    direct_reply = data
            else:
                # 本轮没有订单号，无真实数据可注入。提示词里已有"禁止编造"，
                # 但实测仍会编一个号来问用户确认，因此在代码层补一条硬提醒。
                #
                # 上下文里已有订单号时不能注入：那种情况该由提示词里的
                # "记忆订单号只作候选、据此询问确认"接管，注入禁止类约束会让
                # 模型连上下文里真实存在的号都不敢提（实测：记忆里有 12345，
                # 却只回"请提供订单号"，把刚修好的询问确认行为又压掉了）
                has_known_order = any(
                    ORDER_ID_PATTERN.search(m.get("content") or "")
                    for m in prepared[:-1]
                    if isinstance(m.get("content"), str)
                )
                if not has_known_order:
                    log.info(
                        "[订单查询] 本轮无订单号且上下文无已知订单，"
                        "注入禁止编造订单号的约束"
                    )
                    prepared.insert(len(prepared) - 1, {
                        "role": "system",
                        "content": NO_ORDER_HINT,
                    })
                else:
                    log.info("[订单查询] 上下文已有订单号，交由提示词走询问确认")

        return prepared, direct_reply

    def chat(self, user_message: str) -> str:
        """处理订单查询"""
        log.info(f"[订单查询] 用户问题: {user_message}")

        messages, direct_reply = self._prepare_messages(
            [{"role": "user", "content": user_message}]
        )

        if direct_reply is not None:
            log.info(f"[订单查询] 回答: {direct_reply[:100]}...")
            return direct_reply

        result = self.agent.invoke({"messages": messages})
        
        reply = result["messages"][-1].content
        log.info(f"[订单查询] 回答: {reply[:100]}...")
        
        return reply
    
    def chat_with_history(self, messages: list) -> str:
        """
        带历史记录的多轮对话

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]

        Returns:
            回答
        """
        log.info(f"[订单查询] 多轮对话，历史消息数: {len(messages)}")

        messages, direct_reply = self._prepare_messages(messages)

        if direct_reply is not None:
            log.info(f"[订单查询] 回答: {direct_reply[:100]}...")
            return direct_reply

        result = self.agent.invoke({"messages": messages})
        reply = result["messages"][-1].content

        log.info(f"[订单查询] 回答: {reply[:100]}...")

        return reply