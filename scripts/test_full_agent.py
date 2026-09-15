"""测试集成产品咨询子Agent后的主控Agent"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from src.tools.rag_tools import RAGTool
from src.agents.main_agent import CustomerServiceAgent
from config.settings import settings


def main():
    # 1. 准备RAG工具
    print("正在初始化RAG工具...")
    documents = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
    if not documents:
        print(f"知识库目录没有可加载的文档: {settings.KNOWLEDGE_BASE_DIR}")
        sys.exit(1)

    splitter = DocumentSplitter()
    chunks = splitter.split(documents)

    vector_store = VectorStore(collection_name="product_knowledge")
    store = vector_store.load()

    # 集合不存在时 Chroma 会静默创建空库，主控会一路回答"没有相关信息"，
    # 很容易被误判成产品咨询子Agent没接上。这里提前拦下来。
    if store._collection.count() == 0:
        print("向量库为空，请先构建知识库：uv run python -m scripts.build_knowledge_base")
        sys.exit(1)

    rag_tool = RAGTool(
        vector_store=store,
        chunks=chunks,
        collection_name="product_knowledge"
    )

    # 2. 创建主控Agent（传入rag_tool后，内部会自动接入产品咨询子Agent）
    print("正在创建主控Agent...")
    main_agent = CustomerServiceAgent(rag_tool=rag_tool)

    # 3. 测试
    print("\n" + "=" * 60)
    print("智能客服系统测试（含产品咨询）")
    print("=" * 60)
    
    test_cases = [
        "你好",
        "有什么手机推荐？",
        "星辰X10 Pro多少钱？",
        "星辰Watch Pro能游泳时戴吗？",
        "谢谢",
    ]
    
    ok = 0
    for question in test_cases:
        print(f"\n{'─' * 60}")
        print(f"用户: {question}")
        print(f"{'─' * 60}")

        try:
            answer = main_agent.chat(question)
        except Exception as e:
            # 单次调用失败（网络抖动等）不应中断整轮测试
            print(f"小智: [调用失败] {e}")
            continue

        print(f"小智: {answer}")
        ok += 1

    print("\n" + "=" * 60)
    print(f"测试完成：{ok}/{len(test_cases)} 个用例成功")
    print("=" * 60)


if __name__ == "__main__":
    main()