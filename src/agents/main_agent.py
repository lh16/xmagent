"""
主控Agent
智能客服系统的核心，负责协调所有子Agent和工具
"""

import time

from deepagents import create_deep_agent
from config.prompts import MAIN_AGENT_PROMPT, PRODUCT_AGENT_PROMPT
from src.utils.model import create_model
from src.tools.rag_tools import RAGTool
from src.agents.product_agent import ProductConsultantAgent
from src.utils.logger import log


class CustomerServiceAgent:
    """智能客服主控Agent"""
    
    def __init__(self, rag_tool: RAGTool = None):
        """
        初始化Agent
        
        Args:
            rag_tool: RAG检索工具，传入后启用产品咨询子Agent
        """
        # 创建模型
        self.model = create_model(temperature=0.7)
        
        # 产品咨询子Agent（未提供rag_tool时不启用）
        self.product_agent = ProductConsultantAgent(rag_tool) if rag_tool else None
        if self.product_agent is None:
            log.warning("未提供RAG工具，产品咨询子Agent未启用")
        
        # 创建Deep Agent
        self.agent = self._create_agent()
        
        log.info("智能客服主控Agent初始化完成")
    
    def _create_agent(self):
        """
        创建Deep Agent实例
        
        已接入：
        - 产品咨询子Agent（需提供rag_tool）
        
        后续课程将逐步添加：
        - 订单查询工具
        - 记忆系统
        """
        subagents = []
        if self.product_agent:
            subagents.append({
                "name": "product-consultant",
                "description": (
                    "产品咨询专家，回答商品规格、功能、价格、库存、型号对比等问题。"
                    "当用户询问具体商品信息时，把任务派发给它。"
                ),
                "system_prompt": PRODUCT_AGENT_PROMPT,
                "tools": self.product_agent.tools,
            })
        
        return create_deep_agent(
            model=self.model,
            system_prompt=MAIN_AGENT_PROMPT,
            tools=[],  # 主控自身不直接持有工具，具体能力交由子Agent
            subagents=subagents or None,
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

        # 调用Agent（含子Agent派发与RAG检索，可能需要数十秒）
        start = time.time()
        result = self.agent.invoke({
            "messages": [
                {"role": "user", "content": user_message}
            ]
        })
        log.info(f"[Agent调用完成] 耗时 {time.time() - start:.1f}s")

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
    from config.settings import settings
    from src.rag.loader import DocumentLoader
    from src.rag.splitter import DocumentSplitter
    from src.rag.vectorstore import VectorStore
    
    # 构造RAG工具以启用产品咨询子Agent
    documents = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
    chunks = DocumentSplitter().split(documents)
    store = VectorStore(collection_name="product_knowledge").load()
    
    if store._collection.count() == 0:
        raise SystemExit("向量库为空，请先运行: uv run python -m scripts.build_knowledge_base")
    
    # 创建Agent
    agent = CustomerServiceAgent(
        rag_tool=RAGTool(vector_store=store, chunks=chunks)
    )
    
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