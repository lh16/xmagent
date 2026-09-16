"""
向量存储模块
使用Chroma管理向量数据库
"""

from typing import List, Optional
from langchain_core.documents import Document
import chromadb
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
    
    def build(self, chunks: List[Document], rebuild: bool = True) -> Chroma:
        """
        从文档块构建向量存储

        Args:
            chunks: 文档块列表
            rebuild: 是否先删除同名集合再构建。默认 True，保证重复运行不会产生
                重复向量；需要往已有集合追加时才传 False

        Returns:
            Chroma实例
        """
        # Chroma.from_documents 对已存在的集合是追加语义（每次生成新的 id），
        # 直接重复构建会让向量成倍增长（实测 3 → 6）：重复块挤占 top_k，
        # 召回到的不同内容变少。所以默认走"先删后建"，让构建结果可重复。
        if rebuild:
            self.delete_collection()

        log.info(f"正在构建向量存储，共 {len(chunks)} 个文本块...")
        
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=embedder.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )
        
        log.info(f"向量存储构建完成，持久化目录: {self.persist_dir}")
        return self.store
    
    def delete_collection(self) -> bool:
        """
        删除当前集合（重建时调用，避免追加导致向量重复）
        
        Returns:
            是否实际删除了集合；集合本就不存在时返回 False
        """
        client = chromadb.PersistentClient(path=self.persist_dir)
        
        try:
            client.delete_collection(self.collection_name)
        except Exception as e:
            # 首次构建时集合不存在，属于正常情况，按未删除处理
            log.info(f"集合 {self.collection_name} 不存在，无需删除: {e}")
            return False
        
        # 断开可能持有的旧引用，避免指向已删除的集合
        self.store = None
        log.info(f"已删除旧集合: {self.collection_name}")
        return True
    
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