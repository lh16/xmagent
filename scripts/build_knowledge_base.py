"""
知识库构建脚本
一键构建产品知识库
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from config.settings import settings
from src.utils.logger import log


def build_product_knowledge_base():
    """构建产品知识库"""
    
    print("=" * 60)
    print("构建产品知识库")
    print("=" * 60)
    
    # 1. 加载文档
    print("\n[1/4] 加载文档...")
    documents = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
    print(f"  ✓ 加载了 {len(documents)} 个文档")
    
    # 2. 分块
    print("\n[2/4] 文档分块...")
    splitter = DocumentSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split(documents)
    print(f"  ✓ 切分为 {len(chunks)} 个文本块")
    
    # 3. 向量化并存储
    print("\n[3/4] 向量化并存储...")
    vector_store = VectorStore(collection_name="product_knowledge")
    vector_store.build(chunks)
    print(f"  ✓ 向量存储构建完成")
    
    # 4. 测试检索
    print("\n[4/4] 测试检索...")
    test_queries = [
        "星辰X10 Pro的价格是多少？",
        "有什么手机推荐？",
        "星辰Watch Pro续航多久？",
    ]
    
    for query in test_queries:
        print(f"\n  查询: {query}")
        results = vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"    结果{i+1}: {preview}...")
    
    print("\n" + "=" * 60)
    print("✓ 产品知识库构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    build_product_knowledge_base()