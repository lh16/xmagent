"""智能客服系统完整启动脚本"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from src.tools.rag_tools import RAGTool
from src.agents.product_agent import ProductConsultantAgent
from src.agents.order_agent import OrderQueryAgent
from src.agents.main_agent import CustomerServiceAgent
from src.agents.intent import Intent
from config.settings import settings


def build_system(session_id: str = "default"):
    """
    构建智能客服系统

    Args:
        session_id: 会话ID，短期记忆按会话隔离
    """
    print("正在初始化智能客服系统...")
    
    # 1. 准备RAG工具（产品知识库）
    print("  [1/4] 加载产品知识库...")
    documents = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
    splitter = DocumentSplitter()
    chunks = splitter.split(documents)
    
    vector_store = VectorStore(collection_name="product_knowledge")
    store = vector_store.load()
    
    # 集合不存在时 Chroma 会静默创建空库，产品咨询会一路回答"没有相关信息"，
    # 很容易被误判成子Agent没接上，这里提前拦下来。
    if store._collection.count() == 0:
        raise SystemExit("向量库为空，请先运行: uv run python -m scripts.build_knowledge_base")
    
    rag_tool = RAGTool(
        vector_store=store,
        chunks=chunks,
        collection_name="product_knowledge"
    )
    
    # 2. 创建产品咨询子Agent
    print("  [2/4] 创建产品咨询子Agent...")
    product_agent = ProductConsultantAgent(rag_tool=rag_tool)
    
    # 3. 创建订单查询子Agent
    print("  [3/4] 创建订单查询子Agent...")
    order_agent = OrderQueryAgent()
    
    # 4. 创建主控Agent
    print("  [4/4] 创建主控Agent...")
    main_agent = CustomerServiceAgent(
        subagents={
            Intent.PRODUCT_INQUIRY: product_agent,
            Intent.ORDER_QUERY: order_agent,
        },
        session_id=session_id,
    )
    
    print("✓ 系统初始化完成！\n")
    return main_agent


def main():
    """交互式对话"""
    agent = build_system()
    
    print("=" * 60)
    print("智能客服系统（输入 'quit' 退出）")
    print("=" * 60)
    print()
    
    # 对话历史：保留上下文，否则追问"那它多少钱"这类指代无法解析
    messages = []
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break
            
            messages.append({"role": "user", "content": user_input})
            
            reply = agent.chat_with_history(messages)
            print(f"小智: {reply}\n")
            
            messages.append({"role": "assistant", "content": reply})
            
            # 限制历史长度，避免无限增长
            if len(messages) > 20:
                messages = messages[-20:]
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue


if __name__ == "__main__":
    main()