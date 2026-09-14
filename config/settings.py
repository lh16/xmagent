"""
全局配置
- 自动加载项目根目录下的 .env
- 使用 dataclass 集中管理所有配置项
- 敏感信息（API Key）必须从环境变量读取，不在代码里硬编码
"""
from pathlib import Path

from dotenv import load_dotenv
from dataclasses import dataclass
import os

# 自动加载项目根的 .env（本文件在 config/，往上两级即项目根）
# 这样 settings 一旦被 import，环境变量就已经准备好了
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    # ===== 模型 =====
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    MODEL_BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")

    # ===== 日志 =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ===== RAG（可选，后续课程会用到）=====
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RETRIEVE_TOP_K: int = int(os.getenv("RETRIEVE_TOP_K", "10"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))

    # ===== 存储 =====
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/orders.db")

    # ===== 会话 =====
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))


# 全局单例
settings = Settings()