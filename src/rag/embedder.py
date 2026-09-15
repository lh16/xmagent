"""
嵌入模型模块
使用 SiliconFlow 云端 BGE 中文嵌入模型（兼容 OpenAI 接口）
避免本地部署 sentence-transformers / torch
"""

from langchain_openai import OpenAIEmbeddings
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
            log.info(f"正在初始化云端嵌入模型: {settings.SILICONFLOW_EMBEDDING_MODEL}")
            
            self._embeddings = OpenAIEmbeddings(
                model=settings.SILICONFLOW_EMBEDDING_MODEL,
                base_url=settings.MODEL_BASE_URL,
                api_key=settings.SILICONFLOW_API_KEY,
                # 跳过 tiktoken 长度检查：中文及非 OpenAI 模型会误判 token 数
                check_embedding_ctx_length=False,
                # 分批调用，避免 SiliconFlow 单次请求 input 数组超限
                chunk_size=16,
            )
            
            log.info("云端嵌入模型初始化完成")
    
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