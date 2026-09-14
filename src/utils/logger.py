"""
日志配置模块
使用loguru统一管理日志
"""

import sys
from loguru import logger
from config.settings import settings


def setup_logger():
    """配置日志"""
    # 移除默认handler
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
               "<level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # 添加文件输出
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function} | {message}",
        level=settings.LOG_LEVEL,
        encoding="utf-8"
    )
    
    return logger


# 全局logger实例
log = setup_logger()