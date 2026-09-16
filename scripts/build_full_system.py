"""
完整智能客服系统构建脚本
一键构建所有组件

知识库说明：
产品与售后共用同一个向量库 product_knowledge —— 售后政策文档已并入该库，
不单独建售后库，避免两套知识库内容重复打架。
向量库为空时自动构建，已有内容则直接复用。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from src.tools.rag_tools import RAGTool
from src.agents.product_agent import ProductConsultantAgent
from src.agents.order_agent import OrderQueryAgent
from src.agents.after_sales_agent import AfterSalesAgent
from src.agents.main_agent import CustomerServiceAgent
from src.agents.intent import Intent
from config.settings import settings

# 产品与售后共用的向量库
COLLECTION_NAME = "product_knowledge"


class SystemBuilder:
    """智能客服系统构建器"""
    
    def __init__(self):
        self.rag_tool = None
        self.subagents = {}
    
    def build_rag_tools(self):
        """构建RAG工具（产品与售后共用同一个知识库）"""
        print("\n[1/3] 构建RAG知识库...")
        
        docs = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
        if not docs:
            raise SystemExit(
                f"未从 {settings.KNOWLEDGE_BASE_DIR} 加载到任何文档，"
                f"请检查知识库目录配置"
            )
        
        chunks = DocumentSplitter().split(docs)
        print(f"  - 加载 {len(docs)} 个文档，切分为 {len(chunks)} 个文本块")
        
        store = VectorStore(collection_name=COLLECTION_NAME)
        store_inst = store.load()
        
        # 空库时直接构建，省去先跑 build_knowledge_base.py 的步骤；
        # 已有内容则复用，避免每次启动都重建
        if store_inst._collection.count() == 0:
            print("  - 向量库为空，正在构建...")
            store_inst = store.build(chunks)
        else:
            print(f"  - 复用已有向量库（{store_inst._collection.count()} 个向量）")
        
        self.rag_tool = RAGTool(vector_store=store_inst, chunks=chunks)
        print("  ✓ 知识库就绪（产品与售后共用）")
    
    def build_subagents(self):
        """构建所有子Agent"""
        print("\n[2/3] 构建子Agent...")
        
        print("  - 产品咨询子Agent...")
        self.subagents[Intent.PRODUCT_INQUIRY] = ProductConsultantAgent(
            rag_tool=self.rag_tool
        )
        
        print("  - 订单查询子Agent...")
        self.subagents[Intent.ORDER_QUERY] = OrderQueryAgent()
        
        # 售后复用产品库：售后政策文档已并入 product_knowledge，
        # 单独建库不仅两套内容打架，还容易建成空库导致售后检索不到任何内容
        print("  - 售后政策子Agent...")
        self.subagents[Intent.AFTER_SALES] = AfterSalesAgent(
            rag_tool=self.rag_tool
        )
        
        print(f"  ✓ 共构建 {len(self.subagents)} 个子Agent")
    
    def build_main_agent(self, session_id: str = "default"):
        """构建主控Agent（注入子Agent后自动具备交接能力）"""
        print("\n[3/3] 构建主控Agent...")
        
        agent = CustomerServiceAgent(
            subagents=self.subagents,
            session_id=session_id
        )
        
        print("  ✓ 主控Agent就绪")
        return agent
    
    def build(self, session_id: str = "default"):
        """构建完整系统"""
        print("=" * 60)
        print("构建智能客服系统")
        print("=" * 60)
        
        self.build_rag_tools()
        self.build_subagents()
        agent = self.build_main_agent(session_id)
        
        print("\n系统构建完成！")
        print("=" * 60)
        
        return agent


def build_system(session_id: str = "default"):
    """便捷函数：构建完整系统"""
    builder = SystemBuilder()
    return builder.build(session_id)


if __name__ == "__main__":
    agent = build_system()
    print("已接入意图:", [intent.value for intent in agent.subagents])
