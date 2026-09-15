"""
模型初始化模块
统一管理LLM实例
"""

from langchain_openai import ChatOpenAI
from config.settings import settings


def create_model(
    model_name: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> ChatOpenAI:
    """
    创建LLM实例
    
    Args:
        model_name: 模型名称，默认从配置读取
        temperature: 温度参数
        max_tokens: 最大生成Token数
    
    Returns:
        ChatOpenAI实例
    """
    return ChatOpenAI(
        model=model_name or settings.MODEL_NAME,
        api_key=settings.SILICONFLOW_API_KEY,
        base_url=settings.MODEL_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,    # 单次请求超时（秒），避免网络抖动时无限等待
        max_retries=2,  # 失败自动重试
    )