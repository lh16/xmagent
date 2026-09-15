"""
嵌入模型模块
封装BGE中文嵌入模型
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
from src.utils.logger import log


class Embedder:
    """嵌入模型封装"""
    
    _instance = None
    _embeddings = None
    
    def __new__(cls):
        """单例模式，避免重复加载模型"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化嵌入模型"""
        if self._embeddings is None:
            log.info(f"正在加载嵌入模型: {settings.EMBEDDING_MODEL}")
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,  # 归一化，便于余弦相似度
                    "batch_size": 32
                }
            )
            
            log.info("嵌入模型加载完成")
    
    @property
    def embeddings(self):
        """获取嵌入模型实例"""
        return self._embeddings
    
    def embed_query(self, text: str):
        """嵌入单条查询"""
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, texts):
        """嵌入多条文档"""
        return self._embeddings.embed_documents(texts)


# 全局嵌入模型实例
embedder = Embedder()