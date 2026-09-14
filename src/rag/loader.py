"""
文档加载模块
支持多种格式的文档加载
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from src.utils.logger import log


class DocumentLoader:
    """文档加载器"""
    
    # 支持的文件类型与对应的加载器
    LOADERS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
    }
    
    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            Document列表
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in cls.LOADERS:
            raise ValueError(f"不支持的文件类型: {ext}")
        
        loader_class = cls.LOADERS[ext]
        
        # Markdown和文本文件需要指定编码
        if ext in [".md", ".txt"]:
            loader = loader_class(file_path, encoding="utf-8")
        else:
            loader = loader_class(file_path)
        
        documents = loader.load()
        log.info(f"加载文件 {file_path}，共 {len(documents)} 个文档")
        
        return documents
    
    @classmethod
    def load_directory(cls, dir_path: str) -> List[Document]:
        """
        加载目录下所有支持的文件
        
        Args:
            dir_path: 目录路径
        
        Returns:
            Document列表
        """
        all_documents = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in cls.LOADERS:
                    file_path = os.path.join(root, file)
                    try:
                        documents = cls.load_file(file_path)
                        # 添加来源元数据
                        for doc in documents:
                            doc.metadata["source_file"] = file
                        all_documents.extend(documents)
                    except Exception as e:
                        log.warning(f"加载文件 {file_path} 失败: {e}")
        
        log.info(f"从目录 {dir_path} 共加载 {len(all_documents)} 个文档")
        return all_documents