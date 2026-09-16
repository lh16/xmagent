"""
订单查询子Agent
专门处理订单相关问题
"""

from deepagents import create_deep_agent
from config.prompts import ORDER_AGENT_PROMPT
from src.utils.model import create_model
from src.tools.order_tools import query_order, get_logistics, query_user_orders
from src.utils.logger import log


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
    
    def chat(self, user_message: str) -> str:
        """处理订单查询"""
        log.info(f"[订单查询] 用户问题: {user_message}")
        
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
        
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

        result = self.agent.invoke({"messages": messages})
        reply = result["messages"][-1].content

        log.info(f"[订单查询] 回答: {reply[:100]}...")

        return reply