"""测试售后政策子Agent"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter
from src.rag.vectorstore import VectorStore
from src.tools.rag_tools import RAGTool
from src.agents.after_sales_agent import AfterSalesAgent
from config.settings import settings

# 售后文档（data/knowledge/after_sales.md）随主库一起构建，不再单独建库，
# 这里复用 product_knowledge：chunks 必须与库内内容一致，否则 BM25 与向量检索不对称
COLLECTION_NAME = "product_knowledge"

# 售后内容的来源文件，用于判定本问题是否真的检索到了政策文档
SOURCE_FILE = "after_sales.md"

# (问题, 答案中应出现的关键词，命中任一即可；仅作提示，不决定成败)
TEST_CASES = [
    ("怎么退货？", ["7天", "七天", "15天"]),
    ("保修期是多久？", ["1年", "一年"]),
    ("退款多久能到账？", ["3-5", "3～5", "3到5", "五个工作日"]),
    ("换货需要什么条件？", ["15天"]),
    ("发票丢了还能保修吗？", ["可以", "购买记录"]),
]


def main():
    # 1. 准备RAG工具
    print("正在初始化RAG工具...")
    documents = DocumentLoader.load_directory(settings.KNOWLEDGE_BASE_DIR)
    if not documents:
        print(f"知识库目录没有可加载的文档: {settings.KNOWLEDGE_BASE_DIR}")
        sys.exit(1)

    splitter = DocumentSplitter()
    chunks = splitter.split(documents)

    vector_store = VectorStore(collection_name=COLLECTION_NAME)
    store = vector_store.load()

    # 集合不存在时 Chroma 会静默创建一个空库，Agent 会凭常识作答，
    # 5 个问题看着"全部通过"其实全是编的，这里提前拦下来
    if store._collection.count() == 0:
        print("向量库为空，请先构建知识库：uv run python -m scripts.build_knowledge_base")
        sys.exit(1)

    rag_tool = RAGTool(
        vector_store=store,
        chunks=chunks,
        collection_name=COLLECTION_NAME
    )

    # 2. 创建售后子Agent
    print("正在创建售后政策子Agent...")
    agent = AfterSalesAgent(rag_tool=rag_tool)

    # 3. 测试
    print("\n" + "=" * 60)
    print("售后政策子Agent测试")
    print("=" * 60)

    ok = 0
    failed = 0
    for question, keywords in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"用户: {question}")
        print(f"{'─' * 60}")

        # 硬断言：必须召回售后文档，否则下面的回答没有依据，
        # 只看答案文本无法判断它是查到的还是编的
        hits = rag_tool.search_documents(question)
        sources = {doc.metadata.get("source_file") for doc in hits}
        if SOURCE_FILE not in sources:
            failed += 1
            print(f"[检索失败] 未召回 {SOURCE_FILE}，实际来源: {sources or '无'}")
            continue

        try:
            answer = agent.chat(question)
        except Exception as e:
            # 单次调用失败（网络抖动等）不应中断整轮测试
            failed += 1
            print(f"小智: [调用失败] {e}")
            continue

        print(f"小智: {answer}")

        # 软校验：模型表述不固定，关键词只提示不判失败
        hit = [k for k in keywords if k in answer]
        if hit:
            print(f"关键词: ✓ {'、'.join(hit)}")
        else:
            print(f"关键词: ⚠ 未命中 {'、'.join(keywords)}")

        ok += 1

    print("\n" + "=" * 60)
    print(f"测试完成：{ok}/{len(TEST_CASES)} 个问题成功回答，失败 {failed} 个")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
