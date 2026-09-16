"""
主控Agent（带意图识别和路由）
智能客服系统的核心，负责协调所有子Agent和工具
"""

import time
from typing import Any, Dict, Optional

from deepagents import create_deep_agent
from config.prompts import MAIN_AGENT_PROMPT
from src.utils.model import create_model
from src.agents.intent import intent_classifier, Intent
from src.agents.order_agent import OrderQueryAgent
from src.agents.product_agent import ProductConsultantAgent
from src.tools.rag_tools import RAGTool
from src.utils.logger import log


class CustomerServiceAgent:
    """智能客服主控Agent（带意图路由）"""
    
    def __init__(self, rag_tool: RAGTool = None, subagents: Dict = None):
        """
        初始化Agent
        
        Args:
            rag_tool: RAG检索工具，传入后启用产品咨询子Agent
            subagents: 外部注入的子Agent {Intent: agent}，优先级高于自动装配
        """
        self.model = create_model(temperature=0.7)
        
        # 路由表：意图 → 子Agent
        self.subagents: Dict[Intent, Any] = dict(subagents or {})
        
        # 产品咨询子Agent（未提供rag_tool时不启用）
        self.product_agent = ProductConsultantAgent(rag_tool) if rag_tool else None
        if self.product_agent:
            self.subagents.setdefault(Intent.PRODUCT_INQUIRY, self.product_agent)
        else:
            log.warning("未提供RAG工具，产品咨询子Agent未启用")
        
        # 订单查询子Agent（无外部依赖，默认启用）
        self.order_agent = OrderQueryAgent()
        self.subagents.setdefault(Intent.ORDER_QUERY, self.order_agent)
        
        # 闲聊兜底Agent：未接入对应子Agent的意图也交给它
        self.chitchat_agent = self._create_chitchat_agent()
        
        log.info(
            f"主控Agent初始化完成，已接入意图: "
            f"{[intent.value for intent in self.subagents]}"
        )
    
    def _create_chitchat_agent(self):
        """创建闲聊Agent"""
        return create_deep_agent(
            model=self.model,
            system_prompt=MAIN_AGENT_PROMPT,
            tools=[],
        )
    
    def chat(self, user_message: str, user_id: str = "test_user") -> str:
        """
        处理用户消息（带意图路由）
        
        Args:
            user_message: 用户消息
            user_id: 用户ID
        
        Returns:
            AI回复
        """
        log.info(f"[用户 {user_id}]: {user_message}")
        
        start = time.time()
        intent = self._classify(user_message)
        reply = self._dispatch(intent, user_message)
        log.info(f"[Agent调用完成] 耗时 {time.time() - start:.1f}s")
        
        log.info(f"[AI]: {reply[:100]}...")
        return reply
    
    def chat_with_history(self, messages: list) -> str:
        """
        带历史记录的多轮对话
        
        同样按最后一条用户消息做意图路由，命中子Agent时把完整历史交给它，
        避免多轮追问时丢失上下文。
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
        
        Returns:
            AI回复
        """
        last_user_message = next(
            (
                message["content"]
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        
        intent = self._classify(last_user_message)
        return self._dispatch(intent, last_user_message, messages)
    
    def _classify(self, message: str) -> Intent:
        """识别意图，并记录判定方式与置信度"""
        result = intent_classifier.classify_with_confidence(message)
        
        log.info(
            f"[路由] 意图: {result['intent'].value} "
            f"(置信度: {result['confidence']}, 方法: {result['method']})"
        )
        
        return result["intent"]
    
    def _dispatch(self, intent: Intent, message: str, history: list = None) -> str:
        """按意图分派到对应的处理器"""
        if intent == Intent.ORDER_QUERY:
            return self._handle_order_query(message, history)
        elif intent == Intent.PRODUCT_INQUIRY:
            return self._handle_product_inquiry(message, history)
        elif intent == Intent.AFTER_SALES:
            return self._handle_after_sales(message, history)
        elif intent == Intent.TRANSFER_HUMAN:
            return self._handle_transfer_human()
        else:
            return self._handle_chitchat(message, history)
    
    def _call_subagent(
        self, intent: Intent, message: str, history: list = None
    ) -> Optional[str]:
        """
        调用意图对应的子Agent
        
        未接入对应子Agent时返回 None，由调用方交给主控兜底。
        
        Args:
            intent: 意图
            message: 当前用户消息
            history: 完整历史消息，传入且子Agent支持时优先走多轮
        
        Returns:
            子Agent回复，未接入时返回 None
        """
        agent = self.subagents.get(intent)
        if agent is None:
            log.warning(f"[路由] {intent.value} 没有对应的子Agent，交由主控处理")
            return None
        
        if history and hasattr(agent, "chat_with_history"):
            return agent.chat_with_history(history)
        
        return agent.chat(message)
    
    def _handle_order_query(self, message: str, history: list = None) -> str:
        """处理订单查询"""
        reply = self._call_subagent(Intent.ORDER_QUERY, message, history)
        return reply if reply is not None else self._handle_chitchat(message, history)
    
    def _handle_product_inquiry(self, message: str, history: list = None) -> str:
        """处理产品咨询"""
        reply = self._call_subagent(Intent.PRODUCT_INQUIRY, message, history)
        return reply if reply is not None else self._handle_chitchat(message, history)
    
    def _handle_after_sales(self, message: str, history: list = None) -> str:
        """处理售后政策（暂无专属子Agent，交由主控按其自身能力回答）"""
        reply = self._call_subagent(Intent.AFTER_SALES, message, history)
        return reply if reply is not None else self._handle_chitchat(message, history)
    
    def _handle_transfer_human(self, message: str = "", history: list = None) -> str:
        """处理转人工"""
        return (
            "非常抱歉给您带来不便！我已经为您记录问题，"
            "正在为您转接人工客服，请稍候...\n\n"
            "人工客服工作时间：9:00-21:00\n"
            "您也可以拨打客服热线：400-XXX-XXXX"
        )
    
    def _handle_chitchat(self, message: str, history: list = None) -> str:
        """处理闲聊"""
        messages = history if history else [{"role": "user", "content": message}]
        
        result = self.chitchat_agent.invoke({"messages": messages})
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
    
    print("=" * 60)
    print("智能客服系统 - 意图路由测试")
    print("=" * 60)
    
    test_cases = [
        ("问候", "你好"),
        ("产品咨询", "你们有什么手机推荐？"),
        ("订单查询", "我的订单12345到哪了？"),
        ("转人工", "我要投诉，帮我转人工"),
    ]
    
    for title, question in test_cases:
        print(f"\n--- {title} ---")
        print(f"AI: {agent.chat(question)}")
    
    # 多轮对话测试
    print("\n--- 多轮对话 ---")
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是智能客服小智，有什么可以帮你的吗？"},
        {"role": "user", "content": "我想问一下退货政策"},
    ]
    print(f"AI: {agent.chat_with_history(messages)}")
