"""
产品咨询子Agent
专门处理商品相关问题，使用RAG检索产品知识库
"""

from deepagents import create_deep_agent
from config.prompts import PRODUCT_AGENT_PROMPT
from src.utils.model import create_model
from src.tools.rag_tools import RAGTool
from src.utils.logger import log


class ProductConsultantAgent:
    """产品咨询子Agent"""
    
    def __init__(self, rag_tool: RAGTool):
        """
        初始化产品咨询子Agent
        
        Args:
            rag_tool: RAG检索工具
        """
        self.rag_tool = rag_tool
        self.model = create_model(temperature=0.3)
        
        # 创建工具函数
        self.tools = [self._create_search_tool()]
        
        # 创建Agent
        self.agent = self._create_agent()
        
        log.info("产品咨询子Agent初始化完成")
    
    def _create_search_tool(self):
        """
        创建RAG搜索工具函数
        
        Returns:
            工具函数
        """
        def search_product_knowledge(query: str) -> str:
            """
            搜索产品知识库，获取商品信息。
            
            当你需要回答关于产品功能、规格、价格、库存等问题时，
            使用此工具检索产品手册和FAQ。
            
            Args:
                query: 搜索查询词，如"星辰X10 Pro的价格"
            
            Returns:
                检索到的产品信息
            """
            return self.rag_tool.search(query)

        return search_product_knowledge
    
    def _create_agent(self):
        """创建Deep Agent"""
        return create_deep_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=PRODUCT_AGENT_PROMPT,
        )
    
    def chat(self, user_message: str) -> str:
        """
        处理产品咨询
        
        Args:
            user_message: 用户消息
        
        Returns:
            回答
        """
        log.info(f"[产品咨询] 用户问题: {user_message}")
        
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
        
        reply = result["messages"][-1].content
        log.info(f"[产品咨询] 回答: {reply[:100]}...")
        
        return reply
    
    def chat_with_history(self, messages: list) -> str:
        """
        带历史记录的多轮对话

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]

        Returns:
            回答
        """
        log.info(f"[产品咨询] 多轮对话，历史消息数: {len(messages)}")

        result = self.agent.invoke({"messages": messages})
        reply = result["messages"][-1].content

        log.info(f"[产品咨询] 回答: {reply[:100]}...")

        return reply
