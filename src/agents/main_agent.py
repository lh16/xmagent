"""
主控Agent
智能客服系统的核心，负责协调所有子Agent和工具
"""

from deepagents import create_deep_agent
from config.prompts import MAIN_AGENT_PROMPT
from src.utils.model import create_model
from src.utils.logger import log


class CustomerServiceAgent:
    """智能客服主控Agent"""
    
    def __init__(self):
        """初始化Agent"""
        # 创建模型
        self.model = create_model(temperature=0.7)
        
        # 创建Deep Agent
        self.agent = self._create_agent()
        
        log.info("智能客服主控Agent初始化完成")
    
    def _create_agent(self):
        """
        创建Deep Agent实例
        
        当前版本：仅支持基础对话
        后续课程将逐步添加：
        - RAG工具
        - 订单查询工具
        - 子Agent
        - 记忆系统
        """
        return create_deep_agent(
            model=self.model,
            system_prompt=MAIN_AGENT_PROMPT,
            tools=[],  # 当前版本暂无工具
        )
    
    def chat(self, user_message: str, user_id: str = "test_user") -> str:
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
            user_id: 用户ID
        
        Returns:
            AI回复
        """
        log.info(f"[用户 {user_id}]: {user_message}")
        
        # 调用Agent
        result = self.agent.invoke({
            "messages": [
                {"role": "user", "content": user_message}
            ]
        })
        
        # 提取回复
        reply = result["messages"][-1].content
        
        log.info(f"[AI]: {reply[:100]}...")
        
        return reply
    
    def chat_with_history(self, messages: list) -> str:
        """
        带历史记录的多轮对话
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
        
        Returns:
            AI回复
        """
        result = self.agent.invoke({"messages": messages})
        return result["messages"][-1].content


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 创建Agent
    agent = CustomerServiceAgent()
    
    # 测试基础对话
    print("=" * 60)
    print("智能客服系统 - 测试模式")
    print("=" * 60)
    
    # 单轮对话测试
    print("\n--- 测试1：问候 ---")
    reply = agent.chat("你好")
    print(f"AI: {reply}")
    
    print("\n--- 测试2：产品咨询 ---")
    reply = agent.chat("你们有什么手机推荐？")
    print(f"AI: {reply}")
    
    print("\n--- 测试3：订单查询 ---")
    reply = agent.chat("我的订单12345到哪了？")
    print(f"AI: {reply}")
    
    # 多轮对话测试
    print("\n--- 测试4：多轮对话 ---")
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是智能客服小智，有什么可以帮你的吗？"},
        {"role": "user", "content": "我想问一下退货政策"},
    ]
    reply = agent.chat_with_history(messages)
    print(f"AI: {reply}")