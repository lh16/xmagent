"""测试产品咨询子Agent"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from src.tools.rag_tools import RAGTool
from src.agents.product_agent import ProductConsultantAgent
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

    # 集合不存在时 Chroma 会静默创建一个空库，Agent 会一路回答"未找到相关信息"，
    # 很容易被误判成模型或检索链路的问题。这里提前拦下来并给出修复指引。
    if store._collection.count() == 0:
        print("向量库为空，请先构建知识库：uv run python -m scripts.build_knowledge_base")
        sys.exit(1)


    rag_tool = RAGTool(
        vector_store=store,
        chunks=chunks,
        collection_name="product_knowledge"
    )
    
    # 2. 创建产品咨询子Agent
    print("正在创建产品咨询子Agent...")
    product_agent = ProductConsultantAgent(rag_tool=rag_tool)
    
    # 3. 测试
    print("\n" + "=" * 60)
    print("产品咨询子Agent测试")
    print("=" * 60)
    
    test_questions = [
        "星辰X10 Pro的价格是多少？",
        "有什么手机推荐？",
        "星辰Watch Pro续航多久？",
        "星辰X10和X10 Pro有什么区别？",
        "最便宜的手机是哪款？",
        "星辰Buds Pro支持多设备连接吗？",
    ]
    
    ok = 0
    for question in test_questions:
        print(f"\n{'─' * 60}")
        print(f"用户: {question}")
        print(f"{'─' * 60}")

        try:
            answer = product_agent.chat(question)
        except Exception as e:
            # 单次调用失败（网络抖动等）不应中断整轮测试
            print(f"小智: [调用失败] {e}")
            continue

        print(f"小智: {answer}")
        ok += 1

    print("\n" + "=" * 60)
    print(f"测试完成：{ok}/{len(test_questions)} 个问题成功回答")
    print("=" * 60)


if __name__ == "__main__":
    main()