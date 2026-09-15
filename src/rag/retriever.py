"""
混合检索模块
结合向量检索和BM25关键词检索
"""

import re
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from config.settings import settings
from src.utils.logger import log

# 中文分词：优先使用 jieba，未安装时回退
try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def _tokenize(text: str) -> List[str]:
    r"""
    中文分词

    BM25 默认用正则 \w+ 切词，中文整句会被当成一个 token，
    关键词检索几乎失效。这里优先使用 jieba 切词。
    """
    if _HAS_JIEBA:
        return [token for token in jieba.lcut(text) if token.strip()]
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """混合检索器"""
    
    def __init__(
        self,
        vector_store: Chroma,
        chunks: List[Document] = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            chunks: 文档块（用于BM25）
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # 向量检索器
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVE_TOP_K}
        )
        
        # BM25检索器（需要原始文档块）
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(
                chunks,
                preprocess_func=_tokenize
            )
            self.bm25_retriever.k = settings.RETRIEVE_TOP_K
            
            # 集成检索器
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[self.bm25_weight, self.vector_weight]
            )
            
            log.info(f"混合检索器初始化完成（BM25分词: {'jieba' if _HAS_JIEBA else '正则回退'}）")
        else:
            self.ensemble_retriever = self.vector_retriever
            log.info("仅使用向量检索器")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        混合检索
        
        Args:
            query: 查询文本
        
        Returns:
            相关文档列表
        """
        results = self.ensemble_retriever.invoke(query)
        log.debug(f"检索到 {len(results)} 个文档")
        return results