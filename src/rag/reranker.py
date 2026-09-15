"""
重排序模块
使用 SiliconFlow 云端 BGE 重排序模型（/v1/rerank，兼容 Cohere 接口）
避免本地部署 sentence-transformers / torch
"""

from typing import List, Tuple

import httpx
from langchain_core.documents import Document
from config.settings import settings
from src.utils.logger import log


class Reranker:
    """重排序器"""

    _instance = None
    _client = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化重排序客户端"""
        if self._client is None:
            log.info(f"正在初始化云端重排序模型: {settings.SILICONFLOW_RERANKER_MODEL}")

            self._client = httpx.Client(
                base_url=settings.MODEL_BASE_URL,
                headers={"Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}"},
                timeout=30,
            )

            log.info("云端重排序模型初始化完成")

    def _rank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[int, float]]:
        """
        调用云端 rerank 接口打分

        Args:
            query: 查询文本
            documents: 候选文档列表

        Returns:
            (原始下标, 相关性分数) 列表，按分数降序；
            接口异常或返回空时降级为原始顺序（分数记 0）
        """
        payload = {
            "model": settings.SILICONFLOW_RERANKER_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": len(documents),
            "return_documents": False,
        }

        try:
            resp = self._client.post("/rerank", json=payload)
            resp.raise_for_status()
            ranked = [
                (item["index"], float(item["relevance_score"]))
                for item in resp.json().get("results", [])
            ]
        except Exception as e:
            # 重排序是增强环节，失败不应中断整条检索链
            log.warning(f"云端重排序失败，保持原检索顺序: {e}")
            ranked = []

        if not ranked:
            ranked = [(i, 0.0) for i in range(len(documents))]

        return ranked

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        对检索结果重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前K个

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        top_k = top_k or settings.RERANK_TOP_K

        ranked = self._rank(query, documents)

        # 复制文档，避免直接改写传入的 Document 对象
        results = [
            Document(
                page_content=documents[index].page_content,
                metadata={**documents[index].metadata, "rerank_score": score},
            )
            for index, score in ranked[:top_k]
        ]

        log.debug(f"重排序完成，返回 {len(results)} 个文档")

        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        重排序并返回分数

        Returns:
            (文档, 分数) 列表
        """
        if not documents:
            return []

        top_k = top_k or settings.RERANK_TOP_K

        ranked = self._rank(query, documents)

        return [
            (
                Document(
                    page_content=documents[index].page_content,
                    metadata={**documents[index].metadata, "rerank_score": score},
                ),
                score,
            )
            for index, score in ranked[:top_k]
        ]


# 全局重排序器实例
reranker = Reranker()
