"""
RAG检索工具
封装完整的RAG流程：检索 + 重排序
"""

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.rag.retriever import HybridRetriever
from src.rag.reranker import reranker
from config.settings import settings
from src.utils.logger import log


class RAGTool:
    """RAG检索工具"""

    def __init__(
        self,
        vector_store: Chroma,
        chunks: List[Document] = None,
        collection_name: str = "product_knowledge"
    ):
        """
        初始化RAG工具

        Args:
            vector_store: 向量存储
            chunks: 文档块（用于BM25）
            collection_name: 集合名称
        """
        self.collection_name = collection_name

        if not chunks:
            log.warning("未提供文档块，BM25 关键词检索不可用，将退化为纯向量检索")

        self.retriever = HybridRetriever(
            vector_store=vector_store,
            chunks=chunks
        )

    def _retrieve_and_rerank(self, query: str, top_k: int) -> List[Document]:
        """
        两阶段检索：混合检索召回候选，再重排序精排

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            重排序后的文档列表
        """
        candidates = self.retriever.retrieve(query)

        if not candidates:
            return []

        return reranker.rerank(query, candidates, top_k=top_k)

    def search(self, query: str, top_k: int = None) -> str:
        """
        检索知识库并返回格式化的结果

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            格式化的检索结果字符串
        """
        top_k = top_k or settings.RERANK_TOP_K

        log.info(f"[RAG检索] 查询: {query}")

        try:
            ranked = self._retrieve_and_rerank(query, top_k)
        except Exception as e:
            # 作为对外暴露的工具，检索失败不应中断整个 Agent
            log.error(f"[RAG检索] 检索失败: {e}")
            return f"检索失败：{e}"

        if not ranked:
            return "未找到相关信息。"

        # 格式化输出
        results = []
        for i, doc in enumerate(ranked):
            source = doc.metadata.get("source_file", "未知来源")
            # 重排分数集中在 0~1 小数区间，保留 4 位，避免低分全被显示成 0.00
            score = doc.metadata.get("rerank_score", 0)
            content = doc.page_content.strip()

            results.append(
                f"【文档{i+1}】来源: {source} | 相关度: {score:.4f}\n{content}"
            )

        output = "\n\n".join(results)
        log.info(f"[RAG检索] 返回 {len(ranked)} 个结果")

        return output

    def search_documents(self, query: str, top_k: int = None) -> List[Document]:
        """
        检索并返回Document对象列表

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            Document列表
        """
        top_k = top_k or settings.RERANK_TOP_K

        try:
            return self._retrieve_and_rerank(query, top_k)
        except Exception as e:
            log.error(f"[RAG检索] 检索失败: {e}")
            return []
