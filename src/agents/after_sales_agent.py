"""
售后政策子Agent
专门处理退换货、退款、保修问题
"""

from deepagents import create_deep_agent
from config.prompts import AFTER_SALES_AGENT_PROMPT
from src.utils.model import create_model
from src.tools.rag_tools import RAGTool
from src.utils.logger import log


class AfterSalesAgent:
    """售后政策子Agent"""
    
    def __init__(self, rag_tool: RAGTool):
        """
        初始化售后政策子Agent
        
        Args:
            rag_tool: RAG检索工具（售后知识库）
        """
        self.rag_tool = rag_tool
        self.model = create_model(temperature=0.3)
        
        # 创建工具
        self.tools = [self._create_search_tool()]
        
        # 创建Agent
        self.agent = self._create_agent()
        
        log.info("售后政策子Agent初始化完成")
    
    def _create_search_tool(self):
        """
        创建RAG搜索工具函数
        
        Returns:
            工具函数
        """
        def search_after_sales(query: str) -> str:
            """
            搜索售后政策知识库，获取退换货、退款、保修信息。
            
            当用户询问退货、换货、退款、保修相关问题时使用此工具。
            
            Args:
                query: 搜索查询词，如"退货流程"、"保修期限"
            
            Returns:
                检索到的售后政策信息
            """
            return self.rag_tool.search(query)

        return search_after_sales
    
    def _create_agent(self):
        """创建Deep Agent"""
        return create_deep_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=AFTER_SALES_AGENT_PROMPT,
        )
    
    def chat(self, user_message: str) -> str:
        """
        处理售后咨询
        
        Args:
            user_message: 用户消息
        
        Returns:
            回答
        """
        log.info(f"[售后咨询] 用户问题: {user_message}")
        
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
        
        reply = result["messages"][-1].content
        log.info(f"[售后咨询] 回答: {reply[:100]}...")
        
        return reply
    
    def chat_with_history(self, messages: list) -> str:
        """
        带历史记录的多轮对话

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]

        Returns:
            回答
        """
        log.info(f"[售后咨询] 多轮对话，历史消息数: {len(messages)}")

        result = self.agent.invoke({"messages": messages})
        reply = result["messages"][-1].content

        log.info(f"[售后咨询] 回答: {reply[:100]}...")

        return reply