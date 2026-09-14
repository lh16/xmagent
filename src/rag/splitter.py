"""
文档分块模块
将长文档切分为适合检索的小块
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
from src.utils.logger import log


class DocumentSplitter:
    """文档分块器"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 每块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # 中文友好的分隔符
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n## ",      # 二级标题
                "\n### ",     # 三级标题
                "\n\n",       # 段落
                "\n",         # 换行
                "。",         # 中文句号
                "！",         # 中文感叹号
                "？",         # 中文问号
                "；",         # 中文分号
                "，",         # 中文逗号
                " ",          # 空格
                "",           # 字符
            ],
            length_function=len,
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        切分文档
        
        Args:
            documents: 原始文档列表
        
        Returns:
            切分后的文档块列表
        """
        chunks = self.splitter.split_documents(documents)
        
        # 过滤空块
        chunks = [c for c in chunks if c.page_content.strip()]
        
        # 为每个块添加索引
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        log.info(f"文档切分完成：{len(documents)} 个文档 → {len(chunks)} 个文本块")
        
        return chunks