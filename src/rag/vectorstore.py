"""
向量存储模块
使用Chroma管理向量数据库
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config.settings import settings
from src.rag.embedder import embedder
from src.utils.logger import log


class VectorStore:
    """向量存储管理"""
    
    def __init__(
        self,
        collection_name: str = "product_knowledge",
        persist_dir: str = None
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_dir: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        
        self.store = None
    
    def build(self, chunks: List[Document]) -> Chroma:
        """
        从文档块构建向量存储
        
        Args:
            chunks: 文档块列表
        
        Returns:
            Chroma实例
        """
        log.info(f"正在构建向量存储，共 {len(chunks)} 个文本块...")
        
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=embedder.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )
        
        log.info(f"向量存储构建完成，持久化目录: {self.persist_dir}")
        return self.store
    
    def load(self) -> Chroma:
        """
        加载已有的向量存储
        
        Returns:
            Chroma实例
        """
        log.info(f"正在加载向量存储: {self.collection_name}")
        
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedder.embeddings,
            persist_directory=self.persist_dir
        )
        
        count = self.store._collection.count()
        log.info(f"向量存储加载完成，共 {count} 个向量")
        
        return self.store
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回结果数
            filter_dict: 元数据过滤条件
        
        Returns:
            相关文档列表
        """
        if self.store is None:
            self.load()
        
        return self.store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        带分数的相似度检索
        
        Returns:
            (文档, 分数) 列表
        """
        if self.store is None:
            self.load()
        
        return self.store.similarity_search_with_score(query, k=k)