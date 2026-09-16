"""
主控Agent（带意图识别和路由）
智能客服系统的核心，负责协调所有子Agent和工具
"""

import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from deepagents import create_deep_agent
from config.prompts import MAIN_AGENT_PROMPT
from src.utils.model import create_model
from src.agents.intent import intent_classifier, Intent
from src.agents.after_sales_agent import AfterSalesAgent
from src.agents.order_agent import OrderQueryAgent
from src.agents.product_agent import ProductConsultantAgent
from src.memory.memory_manager import MemoryManager
from src.tools.order_tools import query_order
from src.tools.rag_tools import RAGTool
from src.utils.logger import log

# 订单号特征：5 位以上连续数字，与订单查询子Agent的判定保持一致
ORDER_ID_PATTERN = re.compile(r"\d{5,}")

# query_order 取到数据时的开头，用于判定订单是否真实存在
ORDER_DATA_PREFIX = "📦 订单信息"


class CustomerServiceAgent:
    """智能客服主控Agent（带意图路由）"""
    
    def __init__(
        self,
        rag_tool: RAGTool = None,
        subagents: Dict = None,
        session_id: str = "default",
        memory: MemoryManager = None,
    ):
        """
        初始化Agent
        
        Args:
            rag_tool: RAG检索工具，传入后启用产品咨询子Agent
            subagents: 外部注入的子Agent {Intent: agent}，优先级高于自动装配
            session_id: 会话ID，短期记忆按会话隔离
            memory: 记忆管理器，注入后便于测试使用临时库，默认按会话新建
        """
        self.model = create_model(temperature=0.7)
        self.session_id = session_id
        self.memory = memory or MemoryManager(session_id=session_id)
        
        # 转人工队列（模拟）：工单号与用户问题一并留存，便于人工接手时追溯
        self.transfer_queue: list = []
        
        # 路由表：意图 → 子Agent（外部注入优先，未注入的按可用依赖自动装配）
        self.subagents: Dict[Intent, Any] = dict(subagents or {})
        
        # 知识库类子Agent：产品与售后共用同一个RAG工具。
        # 售后政策文档已并入 product_knowledge，检索时能命中 after_sales.md，
        # 因此不为售后单独建库，避免两套知识库内容重复打架。
        for intent, agent_cls in (
            (Intent.PRODUCT_INQUIRY, ProductConsultantAgent),
            (Intent.AFTER_SALES, AfterSalesAgent),
        ):
            if intent in self.subagents:
                continue
            if rag_tool:
                self.subagents[intent] = agent_cls(rag_tool)
            else:
                log.warning(
                    f"未提供RAG工具且未注入子Agent，{intent.value}能力未启用"
                )
        
        self.product_agent = self.subagents.get(Intent.PRODUCT_INQUIRY)
        self.after_sales_agent = self.subagents.get(Intent.AFTER_SALES)
        
        # 订单查询子Agent：无外部依赖，未注入时自动装配
        if Intent.ORDER_QUERY not in self.subagents:
            self.subagents[Intent.ORDER_QUERY] = OrderQueryAgent()
        self.order_agent = self.subagents[Intent.ORDER_QUERY]
        
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
        
        # 记录本轮消息，并从中提取可长期保存的信息（姓名、偏好等）
        self.memory.add_user_message(user_message, user_id=user_id)
        self.memory.extract_and_remember(user_id, user_message)
        
        start = time.time()
        intent = self._classify(user_message)
        memory_context = self._build_memory_context(user_id)
        reply = self._dispatch(
            intent, user_message, memory_context=memory_context, user_id=user_id
        )
        log.info(f"[Agent调用完成] 耗时 {time.time() - start:.1f}s")
        
        # 订单只在确实查到时才记忆，避免把查无此单的单号写进长期记忆
        if intent == Intent.ORDER_QUERY:
            self._remember_order_if_exists(user_id, user_message)
        
        self.memory.add_assistant_message(reply)
        self.memory.save_interaction(
            user_id, user_message, intent.value, reply[:200]
        )
        
        log.info(f"[AI]: {reply[:100]}...")
        return reply
    
    def chat_with_history(self, messages: list, user_id: str = "test_user") -> str:
        """
        带历史记录的多轮对话
        
        同样按最后一条用户消息做意图路由，命中子Agent时把完整历史交给它，
        避免多轮追问时丢失上下文。
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            user_id: 用户ID
        
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
        
        self.memory.add_user_message(last_user_message, user_id=user_id)
        self.memory.extract_and_remember(user_id, last_user_message)

        intent = self._classify(last_user_message)
        memory_context = self._build_memory_context(user_id)
        reply = self._dispatch(
            intent, last_user_message, messages,
            memory_context=memory_context, user_id=user_id,
        )

        # 订单只在确实查到时才记忆，避免把查无此单的单号写进长期记忆
        if intent == Intent.ORDER_QUERY:
            self._remember_order_if_exists(user_id, last_user_message)

        self.memory.add_assistant_message(reply)
        self.memory.save_interaction(
            user_id, last_user_message, intent.value, reply[:200]
        )
        return reply
    
    def _build_memory_context(self, user_id: str) -> str:
        """取长期记忆上下文，无记忆时为空字符串"""
        return self.memory.build_context(user_id)
    
    @staticmethod
    def _with_memory(message: str, memory_context: str) -> str:
        """单轮：把记忆上下文附到用户问题前"""
        if not memory_context:
            return message
        
        return f"{memory_context}\n\n用户问题：{message}"
    
    @staticmethod
    def _inject_memory(messages: list, memory_context: str) -> list:
        """
        把记忆作为独立 system 消息插到最后一条用户消息之前
        
        不能拼进用户消息：订单查询子Agent从最后一条用户消息里正则提取订单号，
        记忆中的历史订单号会被优先匹配到，导致查成另一笔订单。
        多轮子Agent只接收 history，因此记忆必须注入 history，否则带了历史就丢记忆。
        返回新列表，不修改原列表。
        """
        if not memory_context or not messages:
            return messages
        
        injected = list(messages)
        for index in range(len(injected) - 1, -1, -1):
            if injected[index].get("role") == "user":
                injected.insert(index, {
                    "role": "system",
                    "content": f"以下是该用户的历史记忆，仅供参考：\n{memory_context}",
                })
                break
        
        return injected
    
    def _remember_order_if_exists(self, user_id: str, message: str):
        """
        订单确实存在时才记住
        
        不做文本盲提：查无此单的单号被写进长期记忆后，后续模型会据此
        认定该订单存在，把不存在的订单编出完整信息。
        """
        match = ORDER_ID_PATTERN.search(message)
        if not match:
            return
        
        order_id = match.group()
        if query_order(order_id).startswith(ORDER_DATA_PREFIX):
            self.memory.remember_order(user_id, order_id)
    
    def _classify(self, message: str) -> Intent:
        """识别意图，并记录判定方式与置信度"""
        result = intent_classifier.classify_with_confidence(message)
        
        log.info(
            f"[路由] 意图: {result['intent'].value} "
            f"(置信度: {result['confidence']}, 方法: {result['method']})"
        )
        
        return result["intent"]
    
    def _dispatch(
        self,
        intent: Intent,
        message: str,
        history: list = None,
        memory_context: str = "",
        user_id: str = "test_user",
    ) -> str:
        """按意图分派到对应的处理器，并把记忆上下文一并传给处理器"""
        if intent == Intent.ORDER_QUERY:
            return self._handle_order_query(message, history, memory_context)
        elif intent == Intent.PRODUCT_INQUIRY:
            return self._handle_product_inquiry(message, history, memory_context)
        elif intent == Intent.AFTER_SALES:
            return self._handle_after_sales(message, history, memory_context)
        elif intent == Intent.TRANSFER_HUMAN:
            return self._handle_transfer_human(
                message, history, memory_context, user_id
            )
        else:
            return self._handle_chitchat(message, history, memory_context)
    
    def _call_subagent(
        self,
        intent: Intent,
        message: str,
        history: list = None,
        memory_context: str = "",
    ) -> Optional[str]:
        """
        调用意图对应的子Agent
        
        未接入对应子Agent时返回 None，由调用方交给主控兜底。
        
        Args:
            intent: 意图
            message: 当前用户消息
            history: 完整历史消息，传入且子Agent支持时优先走多轮
            memory_context: 长期记忆上下文，非空时附到用户消息前
        
        Returns:
            子Agent回复，未接入时返回 None
        """
        agent = self.subagents.get(intent)
        if agent is None:
            log.warning(f"[路由] {intent.value} 没有对应的子Agent，交由主控处理")
            return None
        
        # 有记忆时统一走多轮接口：记忆要以独立 system 消息注入，
        # 拼进用户消息会让订单子Agent提取到错误的订单号（实测复现）
        if memory_context and hasattr(agent, "chat_with_history"):
            messages = history or [{"role": "user", "content": message}]
            return agent.chat_with_history(
                self._inject_memory(messages, memory_context)
            )
        
        if history and hasattr(agent, "chat_with_history"):
            return agent.chat_with_history(history)
        
        return agent.chat(self._with_memory(message, memory_context))
    
    def _handle_order_query(
        self, message: str, history: list = None, memory_context: str = ""
    ) -> str:
        """处理订单查询"""
        reply = self._call_subagent(
            Intent.ORDER_QUERY, message, history, memory_context
        )
        return reply if reply is not None else self._handle_chitchat(
            message, history, memory_context
        )
    
    def _handle_product_inquiry(
        self, message: str, history: list = None, memory_context: str = ""
    ) -> str:
        """处理产品咨询"""
        reply = self._call_subagent(
            Intent.PRODUCT_INQUIRY, message, history, memory_context
        )
        return reply if reply is not None else self._handle_chitchat(
            message, history, memory_context
        )
    
    def _handle_after_sales(
        self, message: str, history: list = None, memory_context: str = ""
    ) -> str:
        """处理售后政策（未装配售后子Agent时交由主控兜底）"""
        reply = self._call_subagent(
            Intent.AFTER_SALES, message, history, memory_context
        )
        return reply if reply is not None else self._handle_chitchat(
            message, history, memory_context
        )
    
    def _handle_transfer_human(
        self,
        message: str = "",
        history: list = None,
        memory_context: str = "",
        user_id: str = "test_user",
    ) -> str:
        """处理转人工：生成工单号并入队，便于人工接手时追溯用户诉求"""
        transfer_id = uuid.uuid4().hex[:8]
        
        self.transfer_queue.append({
            "transfer_id": transfer_id,
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        
        log.info(f"[转人工] 用户 {user_id} 请求转人工，工单号: {transfer_id}")
        
        return (
            f"非常抱歉给您带来不便！\n\n"
            f"✅ 已为您创建工单：{transfer_id}\n"
            f"📞 人工客服将在 1-2 分钟内接入\n\n"
            f"人工客服工作时间：9:00-21:00\n"
            f"客服热线：400-XXX-XXXX"
        )
    
    def _handle_chitchat(
        self, message: str, history: list = None, memory_context: str = ""
    ) -> str:
        """处理闲聊"""
        messages = history if history else [{"role": "user", "content": message}]
        messages = self._inject_memory(messages, memory_context)
        
        result = self.chitchat_agent.invoke({"messages": messages})
        return result["messages"][-1].content
    
    def new_session(self, session_id: str):
        """切换新会话：短期记忆随会话重建，长期记忆不受影响"""
        self.session_id = session_id
        self.memory = MemoryManager(session_id=session_id)
        log.info(f"切换到新会话: {session_id}")
    
    def get_transfer_queue(self) -> list:
        """获取转人工工单队列"""
        return self.transfer_queue


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
