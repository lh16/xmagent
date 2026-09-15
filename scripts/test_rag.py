"""测试RAG检索"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from src.tools.rag_tools import RAGTool
from config.settings import settings


def main():
    # 1. 加载并分块
    documents = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
    if not documents:
        print(f"知识库目录没有可加载的文档: {settings.KNOWLEDGE_BASE_DIR}")
        sys.exit(1)

    splitter = DocumentSplitter()
    chunks = splitter.split(documents)

    # 2. 加载向量存储
    vector_store = VectorStore(collection_name="product_knowledge")
    store = vector_store.load()

    # 集合不存在时 Chroma 会静默创建一个空库，随后每个查询都返回"未找到相关信息"，
    # 很容易被误判成检索链路坏了。这里提前拦下来并给出明确的修复指引。
    if store._collection.count() == 0:
        print("向量库为空，请先构建知识库：uv run python -m scripts.build_knowledge_base")
        sys.exit(1)

    # 3. 创建RAG工具
    rag_tool = RAGTool(
        vector_store=store,
        chunks=chunks,
        collection_name="product_knowledge"
    )
    
    # 4. 测试检索
    test_queries = [
        "星辰X10 Pro的价格是多少？",
        "有什么手机推荐？",
        "星辰Watch Pro续航多久？",
        "星辰X10和X10 Pro有什么区别？",
        "最便宜的手机是哪款？",
    ]
    
    hit = 0
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"查询: {query}")
        print("=" * 60)

        result = rag_tool.search(query, top_k=2)
        print(result)

        # 正常结果固定以【文档N】开头，其余为"未找到相关信息"/"检索失败："
        if result.startswith("【文档"):
            hit += 1

    print("\n" + "=" * 60)
    print(f"检索完成：{hit}/{len(test_queries)} 个查询有结果")
    print("=" * 60)


if __name__ == "__main__":
    main()